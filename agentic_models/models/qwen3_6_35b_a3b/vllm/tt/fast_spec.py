# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding verify path for the fast generator (QWEN_SPEC=1).

Verify graph: 32 rows = U users x T candidate columns (T = 1 + K drafts; row r = user * T + column), run through the
same batched decode as the ordinary path (one traced graph):
  - attention: per-row positions/pages; the KV of the T columns is written by T paged updates (other columns at -1)
    before the paged SDPA, so column j attends to columns < j of its own request;
  - MoE / norms / LM head: per row, unchanged;
  - GDN: conv inputs selected from persistent per-layer history buffers by exact 0/1 matmuls; recurrent state in a
    per-layer ring [T * U * 8, 128, 128] (native kernel ring mode: per-token states written in place, the next
    verify starts every head from block (accepted - 1) * U * 8 + head).
Users (requests) live in slots 0..U-1. The ordinary decode keeps its own per-slot state; a user moves into the ring
("enter", eager, at its first verify) and back ("exit", eager, before an ordinary decode or a slot remap/prefill).
Sampled ids per row come from the ordinary device sampler with per-row parameters remapped to the user
(spec_ids_are_sampled: drafts are accepted by id equality with the device draw = exact rejection sampling).
"""

import torch

import ttnn

from .fast_gdn import VH, FastGDN, write_row

DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS = 32


class SpecVerify:
    def __init__(self, gen, k):
        self.gen = gen
        self.K = k
        self.T = k + 1
        self.U = min(ROWS // self.T, 8)
        assert self.U * VH <= 110 and self.U * 3 <= 32
        up = gen._up
        T, U = self.T, self.U
        self.cur_pos = up(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32)
        self.upd_pos = [up(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32) for _ in range(T)]
        self.rope_idx = up(torch.zeros((1, ROWS), dtype=torch.int32), ttnn.uint32)
        self.page_table = up(torch.zeros((ROWS, gen.page_width), dtype=torch.int32), ttnn.int32)
        self.ring_idx = up(torch.arange(U * VH, dtype=torch.int32), ttnn.uint32)
        self.sel = up(torch.zeros(1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.tapmat = up(self._tapmat().reshape(1, 1, 128, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.active_f32 = gen.f_active_f32
        z = lambda shape, dtype: ttnn.zeros(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=gen.mesh, memory_config=DRAM
        )
        self.hidden = z((1, ROWS, 2048), ttnn.bfloat16)
        self.ring, self.vhist, self.xprev = {}, {}, {}
        for li in gen.fstate:
            self.ring[li] = z((T * U * VH, 128, 128), ttnn.float32)
            self.vhist[li] = z((1, 1, 32, 2048), ttnn.bfloat16)
            self.xprev[li] = z((1, 1, 32, 2048), ttnn.bfloat16)
        # host bookkeeping per user/slot (slot == user index for slots < U)
        self.in_ring = [False] * U  # state lives in the ring (True) or in the ordinary per-slot state
        # "fresh": history = E[0..2] (vhist rows), recurrent state in ring block 0;
        # "verified": the previous step verified this user -> history E[a..a+2], state in ring block a-1
        self.mode = [None] * U
        self.virtual_params = False
        self._host_pages = torch.zeros((ROWS, gen.page_width), dtype=torch.int32)
        self.stats = dict(verifies=0, enters=0, exits=0)

    # ------------------------------------------------------------------ constant / host-built selections
    def _tapmat(self):
        """[4*32, 64]: rows t*32 + (u*T + j) select the conv input of tap t for row (u, j) from X = [hist; xt],
        hist row u*3 + k = history element k (oldest first), xt row 32 + u*T + j."""
        T, U = self.T, self.U
        m = torch.zeros(4 * 32, 64)
        for u in range(U):
            for j in range(T):
                for t in range(4):
                    tau = j - 3 + t
                    col = u * 3 + (tau + 3) if tau < 0 else 32 + u * T + tau
                    m[t * 32 + u * T + j, col] = 1
        return m

    def _e_index(self, u, m):
        """Column in E = [vhist (hist layout); xprev (row layout)] of history element m (0..2 = old history,
        3.. = the previous verify's candidate inputs)."""
        return u * 3 + m if m < 3 else 32 + u * self.T + (m - 3)

    # ------------------------------------------------------------------ graph
    def buckets(self):
        """Verify trace row extents: users * T rounded up (MoE route plan / post-combine work only on these rows)."""
        out = sorted({min(ROWS, max(2, u * self.T)) for u in (1, 2, 4, self.U)})
        return out

    def bucket(self, live):
        need = (max(live) + 1) * self.T if live else 2
        return next(b for b in self.buckets() if b >= need)

    def forward(self, rows=ROWS):
        from .fast_generator import fast_norm

        g = self.gen
        m = g.model
        tok = ttnn.reshape(g.tokens, (1, ROWS))
        x = ttnn.reshape(m.embed(tok), (1, ROWS, 2048))
        cos = ttnn.reshape(ttnn.embedding(self.rope_idx, m.rope_cos, layout=ttnn.TILE_LAYOUT), (1, ROWS, 1, 64))
        sin = ttnn.reshape(ttnn.embedding(self.rope_idx, m.rope_sin, layout=ttnn.TILE_LAYOUT), (1, ROWS, 1, 64))
        for index, (layer, states) in enumerate(zip(m.layers, g.cache)):
            norm = fast_norm(layer, x, "input_layernorm.weight", sharded=True)
            if layer.kind == "linear_attention":
                partial = layer.fast.verify(
                    ttnn.reshape(norm, (1, 1, ROWS, 2048)), self, index, users=self.U, steps=self.T
                )
                partial = ttnn.reshape(partial, (1, ROWS, 2048))
            else:
                partial = layer.fast_attn.decode(
                    norm,
                    states[0],
                    cur_pos=self.cur_pos,
                    page_table=self.page_table,
                    cos=cos,
                    sin=sin,
                    attn_mask=g.f_attn_mask,
                    update_pos=self.upd_pos,
                )
            x = ttnn.add(x, layer._reduce(partial, batched=True), fast_and_approximate_mode=False)
            x = ttnn.to_memory_config(x, DRAM)
            norm = fast_norm(layer, x, "post_attention_layernorm.weight", sharded=True)
            moe = layer._moe(
                norm,
                batched=True,
                route_activity_mask=g.active_mask,
                route_slots=tuple(range(rows)),
                fuse_moe_route_plan=m.fuse_moe_route_plan,
            )
            x = ttnn.add(x, layer._reduce(moe, batched=True), fast_and_approximate_mode=False)
            x = ttnn.to_memory_config(x, DRAM)
        ttnn.copy(x, self.hidden)
        logits = m.terminal(x)
        ttnn.copy(ttnn.reshape(logits, (1, 1, ROWS, logits.shape[-1])), g.sampler_logits)

    # ------------------------------------------------------------------ transitions (eager; warmed before capture)
    def _ring_view(self, li):
        return ttnn.reshape(self.ring[li], (self.T * self.U, VH, 128, 128))  # user-block t * U + u

    def enter(self, u):
        """Move user u's state from the ordinary slot state into ring block 0 / the verify history rows."""
        g = self.gen
        consts = g.prefill_executor.consts
        for li, state in g.fstate.items():
            view, i = FastGDN.slot_view(state, u)
            ttnn.fill_cache(self._ring_view(li), view[i : i + 1], u)
            for k, c in enumerate(state["conv"]):
                write_row(self.vhist[li], c[:, :, u : u + 1, :], u * 3 + k, consts)
        self.in_ring[u] = True
        self.mode[u] = "fresh"
        self.stats["enters"] += 1

    def _src(self, u, accepted):
        """(history base index in E, ring block) of user u's current state."""
        if self.mode[u] == "fresh":
            return 0, 0
        a = int(accepted)
        assert 1 <= a <= self.T, a
        return a, a - 1

    def exit(self, u, accepted):
        """Move user u's state back from the ring (after a verify that committed `accepted` tokens)."""
        g = self.gen
        consts = g.prefill_executor.consts
        base, blk = self._src(u, accepted)
        for li, state in g.fstate.items():
            view, i = FastGDN.slot_view(state, u)
            src = self._ring_view(li)[blk * self.U + u : blk * self.U + u + 1]
            ttnn.fill_cache(view, src, i)
            e = ttnn.concat([self.vhist[li], self.xprev[li]], dim=2)
            for k, c in enumerate(state["conv"]):
                col = self._e_index(u, base + k)
                write_row(c, e[:, :, col : col + 1, :], u, consts)
        self.in_ring[u] = False
        self.mode[u] = None
        self.stats["exits"] += 1

    def drop(self, u):
        """User u's request is gone (prefill of a new request into the slot / release): discard its ring state."""
        if u < self.U:
            self.in_ring[u] = False
            self.mode[u] = None

    def exit_all(self, accepted=None):
        for u in range(self.U):
            if self.in_ring[u]:
                self.exit(u, 1 if accepted is None else accepted.get(u, 1))

    def warm(self):
        """Build every transition program before decode-trace capture (slot 0 round trip)."""
        self.enter(0)
        self.exit(0, 1)
        self.enter(0)
        self.mode[0] = "verified"
        self.exit(0, self.T)
        self.stats = dict(verifies=0, enters=0, exits=0)

    # ------------------------------------------------------------------ sampling parameters per row
    def _params(self, per_row_slot):
        from models.common.sampling.generator import SamplingParams

        g = self.gen
        sv = g.sampling_values
        limit = g.sampler.tt_sampling.max_top_k
        temps = [sv["temperature"][s] for s in per_row_slot]
        ks = [sv["top_k"][s] for s in per_row_slot]
        ps = [sv["top_p"][s] for s in per_row_slot]
        greedy = [t == 0 or k == 1 for t, k in zip(temps, ks)]
        pen = lambda name, d: [sv.get(name, [d] * 32)[s] for s in per_row_slot]
        return SamplingParams(
            temperature=torch.tensor([1.0 if q else 1.0 / t for q, t in zip(greedy, temps)]),
            top_k=torch.tensor([1 if q else min(limit, max(1, k)) for q, k in zip(greedy, ks)]),
            top_p=[0.0 if q else p for q, p in zip(greedy, ps)],
            presence_penalty=pen("presence_penalty", 0.0),
            frequency_penalty=pen("frequency_penalty", 0.0),
            repetition_penalty=pen("repetition_penalty", 1.0),
        )

    def use_virtual_params(self):
        rows = [min(r // self.T, self.U - 1) for r in range(ROWS)]
        self.gen.sampler.reset_sampling_params(self._params(rows))
        self.virtual_params = True

    def restore_params(self):
        if self.virtual_params:
            self.gen.sampler.reset_sampling_params(self._params(list(range(ROWS))))
            self.virtual_params = False

    def ensure_ordinary(self, accepted=None):
        """Before any ordinary decode / remap / prefill: state back in the slots, slot sampling params restored."""
        self.exit_all(accepted)
        self.restore_params()

    # ------------------------------------------------------------------ host step
    def prepare_step(self, tokens, positions, num_valid, accepted):
        """tokens/positions [B, T] (B >= live rows; live rows are slots < U with positions[:, 0] >= 0),
        num_valid [B], accepted [B] (count committed by the previous step for that row). Uploads inputs."""
        g = self.gen
        T, U = self.T, self.U
        tokens = torch.as_tensor(tokens).reshape(len(tokens), -1)
        positions = torch.as_tensor(positions).reshape(len(positions), -1)
        live = [int(r) for r in range(len(positions)) if int(positions[r, 0]) >= 0]
        if any(r >= U for r in live):
            raise ValueError(f"speculative verify serves slots < {U}; live rows {live}")
        ids = torch.zeros(1, 1, 1, ROWS, dtype=torch.int32)
        cur = torch.full((ROWS,), -1, dtype=torch.int32)
        upd = [torch.full((ROWS,), -1, dtype=torch.int32) for _ in range(T)]
        rope = torch.zeros((1, ROWS), dtype=torch.int32)
        sel = torch.zeros(32, 64)
        ring = torch.arange(U * VH, dtype=torch.int32)
        pages = self._host_pages.clone()
        active = []
        for u in range(U):
            if u in live and not self.in_ring[u]:
                self.enter(u)
            if not self.in_ring[u]:
                continue
            base, blk = self._src(u, accepted[u] if u < len(accepted) else 1)
            for k in range(3):
                sel[u * 3 + k, self._e_index(u, base + k)] = 1
            ring[u * VH : (u + 1) * VH] = blk * U * VH + torch.arange(u * VH, (u + 1) * VH)
            if u not in live:
                continue  # idle ring user: identity steps (inactive rows) carry its state to ring block 0
            nv = int(num_valid[u])
            for j in range(T):
                r = u * T + j
                pages[r] = g.f_host_pages[u]
                if j <= nv:
                    p = int(positions[u, j])
                    ids[..., r] = int(tokens[u, j])
                    cur[r] = p
                    upd[j][r] = p
                    rope[0, r] = p
                    active.append(r)
        g._set_active_slots(tuple(active))
        g._copy_host(ids, g.tokens)  # always: the ordinary path also writes g.tokens
        g._put(cur, self.cur_pos)
        for j in range(T):
            g._put(upd[j], self.upd_pos[j])
        g._put(rope, self.rope_idx)
        g._put(sel.reshape(1, 1, 32, 64).bfloat16(), self.sel)
        g._put(ring, self.ring_idx)
        if not torch.equal(pages, self._host_pages):
            g._copy_host(pages, self.page_table)
            self._host_pages = pages
        if not self.virtual_params:
            self.use_virtual_params()
        self._live = live
        return live

    def finish_step(self, live):
        """After the verify replay: every live user's next selection reads history E[a..] / ring block a-1."""
        for u in range(self.U):
            if self.in_ring[u]:
                self.mode[u] = "verified" if u in live else "fresh"
        self.stats["verifies"] += 1
