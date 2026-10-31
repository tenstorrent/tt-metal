# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batched DFlash speculative decoding for Laguna (many users per step, e.g. batch 32).

One round for U users, each checking K drafts:

  draft    every user's 16-row proposal block (its bonus token + 15 mask tokens, padded to 32 rows) through the draft
           layers in one pass (MultichipDecoder.dflash_query_forward_batched: matmuls over all U * 32 rows, attention
           batched over users against each user's own context ring), then the target LM head on rows 1..K of each
           block and an on-device greedy pick -> U x K drafts
  verify   U * (K + 1) target rows -- user u's bonus and its K drafts at positions p_u .. p_u + K -- in one target step
           (LagunaModel.verify_layers_batched: attention per 32-row chunk with tile-merging KV writes, the MoE once over
           every row), its LM head and on-device greedy picks; the states of the draft's six target layers are kept
  accept   per user: the leading drafts that match the target's picks, plus the target's own next token
  update   every checked row's target states -> the draft's fusion -> per-draft-layer K/V written into the users'
           context rings (ring_write_batched); rows past a user's accepted ones land in ring slots outside its window

Each user's draft context is the last 511 committed positions, kept as K/V rings [U, nkv, 544, head_dim] per draft
layer (slot of position p: (p - base_u) % 544), exactly as the batch-1 ring (dflash_tt.DFlashTTProposalCache).
"""

from __future__ import annotations

import math
import os
import time

import torch

import ttnn

from .ring_write import ring_write_batched

TILE = 32


class BatchedDFlash:
    # >= 511 context rows + 33 uncommitted checked rows; two users' rings side by side make 2 * 560 + 32 = 1152 keys,
    # 9 chunks of 128 for the draft attention
    # (the draft writes its 16 rows per user into the ring too: 576 >= 511 + 33 keeps every written slot outside the
    # window, and 2 * 576 = 1152 keys per user pair are 9 chunks of 128)
    RING = 576

    def __init__(self, gen, core, users: int = 32, drafts: int = 1, rows: int | None = None):
        """drafts (K): drafts proposed per user and the most a user checks. rows: the verify's fixed row count with
        confidence allocation -- every user checks its bonus token, and the rows - users remaining rows go to the
        (user, draft i) with the highest draft confidence (product of the top-1 probabilities of drafts 1..i), so a
        user checks 0..K drafts; None: every user checks K drafts (rows = users * (K + 1))."""
        self.gen, self.core = gen, core
        self.mesh, self.model = gen.mesh_device, gen.model
        self.U, self.K = int(users), int(drafts)
        self.T = int(rows) if rows else self.U * (self.K + 1)
        self.alloc = rows is not None
        if self.T % TILE or (self.U * self.K) % TILE or not self.U <= self.T <= self.U * (self.K + 1):
            raise ValueError(f"bad batched DFlash shape: U={self.U}, K={self.K}, rows={self.T}")
        layer0 = next(iter(core.layers.values()))
        self.nkv, self.hd = int(layer0.cfg.num_kv_heads), int(layer0.cfg.head_dim)
        self.window = int(core.config.sliding_window) - 1
        self.block = int(core.config.block_size)  # 16 proposal rows
        self.mask_id = int(core.config.mask_token_id)
        # the draft attends causally within a user's block, so rows past the last checked draft never reach rows
        # 1..K: each user runs only its first Q = K + 1 rows (rounded up to a power of two), G = 32 / Q users per tile
        self.Q = 1 << (self.K).bit_length()
        if self.Q > self.block or self.U % (TILE // self.Q):
            raise ValueError(f"batched draft: K={self.K} needs {self.Q} rows per user; U must be a multiple of {TILE // self.Q}")
        self.G = TILE // self.Q
        self.P = self.U // self.G  # user groups: group c holds users c * G .. in rows of Q each
        # per-user context rings [U, nkv, RING, hd] (the draft's attention runs per user: dflash_query_forward_users)
        zeros = torch.zeros((self.U, self.nkv, self.RING, self.hd), dtype=torch.bfloat16)
        self.rings = {i: (self._dev(zeros, ttnn.TILE_LAYOUT), self._dev(zeros, ttnn.TILE_LAYOUT)) for i in core.layers}
        self.base = [0] * self.U
        self.slot_pos = torch.full((self.U, self.RING), -1, dtype=torch.int64)
        self.ctx_start = [0] * self.U
        self.ctx_rows = [0] * self.U
        self.bonus = [0] * self.U
        self.stats = []
        self._traced = False
        self._gqa = int(layer0.cfg.num_heads) // self.nkv
        self._rp = -(-self._gqa * self.Q // TILE) * TILE  # per-user query block rows (dflash_q_users.to_users)
        # the draft's U * 16-row norms width-sharded over the grid (the interleaved norm used 16 cores)
        for layer in core.layers.values():
            layer._norm_max_rows = int(os.environ.get("TT_LAGUNA_DFLASH_DRAFT_NORM_ROWS", str(max(TILE, self.U * self.Q))))
            layer._dflash_rows_per_user = self.Q
            layer._glu_fused_silu = os.environ.get("TT_LAGUNA_DFLASH_DRAFT_GLU_FUSED", "1") == "1"

    # ---- helpers ------------------------------------------------------------------------------------------------ #
    def _dev(self, t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip

    def _rope_rows(self, positions):
        """(cos, sin) bf16 host tensors [n, head_dim] at the given absolute positions (NeoX, the draft's theta)."""
        phase = torch.outer(torch.as_tensor(positions, dtype=torch.float32), self.core._rope_inv_freq)
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos().to(torch.bfloat16), phase.sin().to(torch.bfloat16)

    def _rope_tables(self):
        """Device (cos, sin) tables [max_seq_len, head_dim] (bf16, row major; the same values as _rope_rows): a draft or
        verify gathers its rows' entries by position on device instead of uploading them every round."""
        if getattr(self, "_rope_tab", None) is None:  # made at the first (eager) pass, before any trace capture
            cos, sin = self._rope_rows(torch.arange(int(self.gen.max_seq_len)))
            self._rope_tab = (self._dev(cos), self._dev(sin))
        return self._rope_tab

    def _rope_gather(self, pos_tt, n):
        """(cos, sin) [1, 1, n, hd] TILE at the positions pos_tt ([1, n] uint32, row major)."""
        return tuple(ttnn.reshape(ttnn.embedding(pos_tt, tab, layout=ttnn.TILE_LAYOUT), (1, 1, n, self.hd))
                     for tab in self._rope_tables())  # fmt: skip

    @staticmethod
    def _unpack(pk_tt, n):
        """The three rows of a packed per-round input [1, 1, 3, n] uint32 (row major) as [1, n] uint32 tensors (one
        upload instead of three; ~0.2 ms per upload in a round)."""
        return [ttnn.reshape(ttnn.slice(pk_tt, [0, 0, i, 0], [1, 1, i + 1, n]), (1, n)) for i in range(3)]

    def _greedy(self, shards, rows):
        """On-device greedy picks of logit shards [1, 1, rows, V/D] (rows a multiple of 32) -> host list."""
        out = []
        for r0 in range(0, rows, TILE):
            part = shards if rows == TILE else ttnn.slice(shards, [0, 0, r0, 0], [1, 1, r0 + TILE, shards.shape[-1]])
            buf = self.gen._rep(torch.zeros([1, 1, 1, TILE], dtype=torch.int32), ttnn.uint32)
            self.gen._greedy_sample(part, TILE, buf)
            out += self.gen._read_token(buf, TILE)
        return out

    def _write_rows(self, x, positions, dests):
        """Draft K/V of the rows x [1, n, H] (n a multiple of 32, at most 128) at ``positions`` into the rings at
        ``dests`` (user * RING + slot per row, 0xFFFFFFFF = skip)."""
        cos, sin = self._rope_rows(positions)
        n = int(x.shape[-2])
        rope = (self._dev(cos.reshape(1, 1, n, self.hd), ttnn.TILE_LAYOUT),
                self._dev(sin.reshape(1, 1, n, self.hd), ttnn.TILE_LAYOUT))  # fmt: skip
        rings, srcs = [], []
        for i, layer in self.core.layers.items():
            k, v = layer.dflash_kv_rows(x, rope)
            rings += list(self.rings[i])
            srcs += [k, v]
        d = self._dev(torch.as_tensor(dests, dtype=torch.int64).to(torch.int32).reshape(1, n), dtype=ttnn.uint32)
        ring_write_batched(rings, srcs, d, rows_per_user=32)

    # ---- prefill ------------------------------------------------------------------------------------------------ #
    def prefill(self, prompts):
        """Prefill user u with prompts[u] (1-D token ids) into the target KV cache, fill its draft ring from the
        captured target states, and set its bonus token (the target's greedy next token). Returns TTFT seconds."""
        assert len(prompts) == self.U
        m, gen = self.model, self.gen
        start = time.perf_counter()
        for u, ids in enumerate(prompts):
            ids = torch.as_tensor(ids, dtype=torch.int64).reshape(1, -1)
            L = int(ids.shape[-1])
            x = m.embed_prefill(gen._tokens_to_device(ids))
            h, cap = m.prefill_layers_with_dflash_aux(x, gen._kv_cache, gen._page_table, user_id=u, start_pos=0,
                                                      enable_experimental=True)  # fmt: skip
            last = ttnn.slice(h, [0, L - 1, 0], [1, L, gen.hidden])
            self.bonus[u] = int(torch.argmax(m.logits_to_host(m.lm_head_shards_prefill(last)).reshape(-1)))
            rows, cstart = int(cap.row_count), int(cap.start_position)
            comb = self.core.combine_aux_hidden_states(cap.hidden_states)  # [1, rows, H]
            pad = -(-rows // TILE) * TILE
            if pad > rows:
                comb = ttnn.pad(comb, [(0, 0), (0, pad - rows), (0, 0)], value=0.0)
            self.base[u] = cstart
            for r0 in range(0, pad, 128):
                n = min(128, pad - r0)
                part = ttnn.slice(comb, [0, r0, 0], [1, r0 + n, gen.hidden])
                pos = [cstart + r0 + j for j in range(n)]
                dests = [u * self.RING + (r0 + j) if r0 + j < rows else 0xFFFFFFFF for j in range(n)]
                self._write_rows(part, pos, dests)
            self.slot_pos[u] = -1
            self.slot_pos[u, :rows] = torch.arange(cstart, cstart + rows)
            self.ctx_start[u], self.ctx_rows[u] = cstart, rows
        ttnn.synchronize_device(self.mesh)
        return time.perf_counter() - start

    # ---- one round ---------------------------------------------------------------------------------------------- #
    def _draft_inputs(self):
        """Host inputs of a draft with Q rows per user: [1, 1, 3, U * Q] int32 rows = token ids (user u: its bonus, then
        Q - 1 mask tokens), each row's position (its RoPE is gathered on device), the ring slots the rows' keys/values
        are written to (user * RING + slot); and the attention mask rows [U, 1, Q, RING] of each user's Q queries (row
        major; _mask_rows expands them on device to the query block's [U, 1, RP, RING], rows (h % G) * Q + i: row i of
        every query head): a row sees its user's slots holding a position in its causal sliding window -- committed
        context, and its block's earlier rows (written there by the draft layers). The block's padding rows see every
        slot (their outputs are dropped)."""
        U, R, Q = self.U, self.RING, self.Q
        ids = torch.full((U, Q), self.mask_id, dtype=torch.int32)
        ids[:, 0] = torch.tensor(self.bonus, dtype=torch.int32)
        ids = ids.reshape(1, U * Q)
        p0 = torch.tensor([self.ctx_start[u] + self.ctx_rows[u] for u in range(U)], dtype=torch.int64)
        base = torch.tensor(self.base, dtype=torch.int64)
        qpos = p0[:, None] + torch.arange(Q)[None, :]  # [U, Q]
        slots = (qpos - base[:, None]) % R  # [U, Q]
        dests = (torch.arange(U)[:, None] * R + slots).reshape(1, U * Q).to(torch.int32)
        pos = self.slot_pos.clone()
        pos.scatter_(1, slots, qpos)  # the block's rows at their slots
        pos = pos[:, None, :]  # [U, 1, R]
        start = torch.tensor(self.ctx_start, dtype=torch.int64)[:, None, None]
        q = qpos[:, :, None]
        own = (pos >= start) & (pos <= q) & (pos >= q - self.window)  # [U, Q, R]
        mask = torch.where(own, 0.0, -1e9).to(torch.bfloat16).reshape(U, 1, Q, R)
        pk = torch.stack([ids[0], qpos.reshape(-1).to(torch.int32), dests[0]]).reshape(1, 1, 3, U * Q)
        return pk, mask

    def _mask_rows(self, rows_tt):
        """[U, 1, Q, RING] mask rows (row major) -> the query blocks' [U, 1, RP, RING] TILE mask: the Q rows repeated
        for the G query heads of a KV head, padding rows 0 (see every slot). ~1.2 MB per chip uploaded every round
        before; now 1/8 of it, and no host tilize."""
        G, Q, RP = self._gqa, self.Q, self._rp
        m = ttnn.repeat(rows_tt, ttnn.Shape([1, 1, G, 1])) if G > 1 else rows_tt
        if RP > G * Q:
            m = ttnn.pad(m, [(0, 0), (0, 0), (0, RP - G * Q), (0, 0)], 0.0)
        return ttnn.to_layout(m, ttnn.TILE_LAYOUT)

    def _draft_body(self, pk_tt, mask_tt, bufs):
        """Device work of a draft (traceable): proposal blocks through the draft layers, rows 1..K of each block
        through the target LM head, greedy picks into ``bufs`` (one [1, 1, 1, U * Q] uint32). pk_tt / mask_tt: the
        uploaded _draft_inputs."""
        m = self.model
        ids_tt, qpos_tt, dests_tt = self._unpack(pk_tt, self.U * self.Q)
        rope = self._rope_gather(qpos_tt, self.U * self.Q)
        x = m.embed_prefill(ids_tt)  # [1, U * Q, H]
        mask_tt = self._mask_rows(mask_tt)
        for i in sorted(self.core.layers):
            x = self.core.layers[i].dflash_query_forward_users(x, *self.rings[i], rope, mask_tt, dests_tt, self.Q)
        x = self.core.apply_final_norm(x)  # [1, U * Q, H]
        # the target LM head over every row (the host takes rows 1..K of each user's Q)
        shards = m.lm_head_shards_dflash(x, enable_experimental=True)
        shards = ttnn.reshape(shards, (1, 1, self.U * self.Q, shards.shape[-1]))
        self._greedy_into(shards, bufs)
        if self.alloc:
            self._draft_conf(shards)

    def _draft_conf(self, shards):
        """Each draft row's (max logit, sum of exp(logit - max)) over this chip's vocabulary shard, all-gathered over
        the chips into self._conf [1, 1, rows, 4 * 32] (columns 32 c, 32 c + 1: chip c's pair). A row reduction over one
        32-row tile row runs on one core, so each tile row's columns are regrouped into C chunks reduced on their own
        cores, then the C partials."""
        from .tile_regroup import regroup

        R, V = int(shards.shape[-2]), int(shards.shape[-1])
        C = next((c for c in (56, 49, 28, 16, 8) if V % (c * 32) == 0), 1)
        parts = []
        for r0 in range(0, R, TILE):
            lg = shards if R == TILE else ttnn.slice(shards, [0, 0, r0, 0], [1, 1, r0 + TILE, V])
            if C > 1:
                lg = regroup(lg, [((1, C, TILE, V // C), 0)], ttnn.DRAM_MEMORY_CONFIG)[0]
            mx = ttnn.max(ttnn.max(lg, dim=-1, keepdim=True), dim=1, keepdim=True)
            z = ttnn.sum(ttnn.sum(ttnn.exp(ttnn.subtract(lg, mx)), dim=-1, keepdim=True), dim=1, keepdim=True)
            parts.append(ttnn.concat([mx, z], dim=-1))
        mz = ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0]
        mz = ttnn.pad(mz, [(0, 0), (0, 0), (0, 0), (0, 30)], 0.0)
        layer0 = next(iter(self.core.layers.values()))
        g = ttnn.all_gather(mz, dim=3, cluster_axis=layer0.tp_axis, topology=layer0.ccl_topology,
                            num_links=layer0.num_links)  # fmt: skip
        if getattr(self, "_conf", None) is None:  # allocated at the first (eager) draft, before any trace capture
            self._conf = ttnn.allocate_tensor_on_device(g.shape, g.dtype, g.layout, self.mesh, g.memory_config())
        ttnn.copy(g, self._conf)  # every pass copies (a first copy inside a capture would compile its program)

    def _read_conf(self):
        """Top-1 probability [U, K] of each draft from the per-chip (max, sum exp) pairs."""
        c = ttnn.to_torch(ttnn.get_device_tensors(self._conf)[0]).float().reshape(self.U * self.Q, -1)
        mx, z = c[:, 0::32], c[:, 1::32]
        top = mx.max(dim=-1, keepdim=True).values
        return (1.0 / (z * torch.exp(mx - top)).sum(dim=-1)).reshape(self.U, self.Q)[:, 1 : 1 + self.K]

    def _allocate(self, conf):
        """Drafts checked per user: the rows - users extra rows go to the highest products of draft top-1
        probabilities (prefix products decrease along a user's drafts, so the chosen ones are leading drafts)."""
        value = torch.cumprod(conf.clamp(min=0.0, max=1.0), dim=1)  # [U, K]
        extra = self.T - self.U
        n = [0] * self.U
        if extra > 0:
            order = torch.argsort(value.reshape(-1), descending=True, stable=True)[:extra]
            for flat in order.tolist():
                n[flat // self.K] += 1
        return n

    def _greedy_into(self, shards, bufs):
        rows = int(shards.shape[-2])
        if len(bufs) == 1 and int(bufs[0].shape[-1]) == rows:  # every row's pick in one buffer (one read)
            self.gen._greedy_sample(shards, rows, bufs[0])
            return
        for c, buf in enumerate(bufs):
            r0 = c * TILE
            part = shards if rows == TILE else ttnn.slice(shards, [0, 0, r0, 0], [1, 1, r0 + TILE, shards.shape[-1]])
            self.gen._greedy_sample(part, TILE, buf)

    def _verify_body(self, pk_tt, pt_tt, bufs):
        """Device work of a verify + ring update (traceable): the target over every checked row, greedy picks into
        ``bufs``, then the checked rows' target states -> draft fusion -> K/V of each draft layer -> the rings.
        pk_tt / pt_tt: the uploaded _verify_host inputs (tokens, positions, ring destinations; page-table rows)."""
        m, gen = self.model, self.gen
        T = self.T
        tok_tt, ridx_tt, dests_tt = self._unpack(pk_tt, T)
        cur_tt = ttnn.typecast(ttnn.reshape(ridx_tt, (T,)), ttnn.int32)
        rope_rows = self._rope_gather(ridx_tt, T)
        h = m.embed_decode(tok_tt)
        h, aux = m.verify_layers_batched(h, m.verify_inputs_batched(cur_tt, ridx_tt, pt_tt), gen._kv_cache, capture=True)
        self._greedy_into(m.lm_head_shards_decode(h), bufs)
        x = self.core.combine_aux_hidden_states(aux)  # [1, T, H]
        rings, srcs = [], []
        for i, layer in self.core.layers.items():
            k, v = layer.dflash_kv_rows(x, rope_rows)
            rings += list(self.rings[i])
            srcs += [k, v]
        ring_write_batched(rings, srcs, dests_tt, rows_per_user=self.T if self.alloc else self.K + 1)

    def _verify_host(self, drafts, n=None):
        """Host inputs of a verify where user u checks n[u] drafts (rows: its bonus, then the drafts; users in
        order): [1, 1, 3, T] int32 rows = tokens, positions, ring destinations (every checked row is written into its
        user's ring); page-table rows [T, blocks]."""
        U, T = self.U, self.T
        n = [self.K] * U if n is None else n
        assert sum(n) + U == T, (sum(n), U, T)
        # plain lists, one tensor conversion (per-element tensor writes cost ~0.7 ms per round)
        toks, pos, dests, user = [], [], [], []
        for u in range(U):
            p0 = self.ctx_start[u] + self.ctx_rows[u]
            k = 1 + n[u]
            toks += [self.bonus[u]] + list(drafts[u][: n[u]])
            pos += range(p0, p0 + k)
            dests += [u * self.RING + (p0 + i - self.base[u]) % self.RING for i in range(k)]
            user += [u] * k
        pt_rows = self._pt_host()[torch.tensor(user)].to(torch.int32)
        pk = torch.tensor([toks, pos, dests], dtype=torch.int64).to(torch.int32).reshape(1, 1, 3, T)
        return pk, pt_rows

    def capture_traces(self):
        """Persistent inputs and two traces: the draft, and the verify with its ring update. Call after prefill (the
        first eager pass of each body compiles its programs; nothing is allocated between the captures and replay)."""
        U, K = self.U, self.K
        T = self.T
        g = self.gen
        pk, mask = self._draft_inputs()
        self._d_in = (self._dev(pk, dtype=ttnn.uint32), self._dev(mask))
        self._d_out = [g._rep(torch.zeros([1, 1, 1, U * self.Q], dtype=torch.int32), ttnn.uint32)]
        pk, pt_rows = self._verify_host([[0] * K for _ in range(U)], self._uniform_n())
        # the ring update of the warm-up / capture passes writes zero-token rows: keep the rings intact by pointing
        # every destination at "skip"
        pk[0, 0, 2] = -1
        self._v_in = (self._dev(pk, dtype=ttnn.uint32), g._rep(pt_rows, ttnn.int32))
        self._v_out = [g._rep(torch.zeros([1, 1, 1, T], dtype=torch.int32), ttnn.uint32)]

        def draft():
            self._draft_body(*self._d_in, self._d_out)

        def verify():
            self._verify_body(*self._v_in, self._v_out)

        mesh = self.mesh
        draft()
        verify()
        ttnn.synchronize_device(mesh)
        print("[batched dflash] capturing the draft trace", flush=True)
        self._tid_d = ttnn.begin_trace_capture(mesh, cq_id=0)
        draft()
        ttnn.end_trace_capture(mesh, self._tid_d, cq_id=0)
        print("[batched dflash] capturing the verify trace", flush=True)
        self._tid_v = ttnn.begin_trace_capture(mesh, cq_id=0)
        verify()
        ttnn.end_trace_capture(mesh, self._tid_v, cq_id=0)
        print("[batched dflash] traces captured", flush=True)
        ttnn.synchronize_device(mesh)
        self._traced = True

    def _uniform_n(self):
        extra = self.T - self.U
        return [extra // self.U + (1 if u < extra % self.U else 0) for u in range(self.U)]

    def _host(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def _read(self, bufs, n):
        """The picks of n rows (replicated on every chip: chip 0's copy)."""
        out = []
        for buf in bufs:
            out += ttnn.to_torch(ttnn.get_device_tensors(buf)[0]).flatten().to(torch.int64).tolist()
        return out[:n]

    def draft(self):
        """U x K draft tokens (host lists)."""
        pk, mask = self._draft_inputs()
        if self._traced:
            ttnn.copy_host_to_device_tensor(self._host(pk, ttnn.uint32), self._d_in[0])
            ttnn.copy_host_to_device_tensor(self._host(mask, ttnn.bfloat16), self._d_in[1])
            ttnn.execute_trace(self.mesh, self._tid_d, cq_id=0, blocking=False)
            picks = self._read(self._d_out, self.U * self.Q)
        else:
            bufs = [self.gen._rep(torch.zeros([1, 1, 1, self.U * self.Q], dtype=torch.int32), ttnn.uint32)]
            self._draft_body(self._dev(pk, dtype=ttnn.uint32), self._dev(mask), bufs)
            picks = self._read(bufs, self.U * self.Q)
        drafts = [picks[u * self.Q + 1 : u * self.Q + 1 + self.K] for u in range(self.U)]
        self._last_n = self._allocate(self._read_conf()) if self.alloc else None
        return drafts

    def verify(self, drafts, n=None):
        """Target picks for every checked row (host list, rows in _verify_host order); the ring update runs in the
        same pass."""
        T = self.T
        pk, pt_rows = self._verify_host(drafts, n)
        if self._traced:
            ttnn.copy_host_to_device_tensor(self._host(pk, ttnn.uint32), self._v_in[0])
            if self.alloc or not getattr(self, "_pt_uploaded", False):  # uniform rows: the page-table rows never change
                ttnn.copy_host_to_device_tensor(self._host(pt_rows, ttnn.int32), self._v_in[1])
            self._pt_uploaded = True
            ttnn.execute_trace(self.mesh, self._tid_v, cq_id=0, blocking=False)
            return self._read(self._v_out, T)
        g = self.gen
        bufs = [g._rep(torch.zeros([1, 1, 1, T], dtype=torch.int32), ttnn.uint32)]
        self._verify_body(self._dev(pk, dtype=ttnn.uint32), g._rep(pt_rows, ttnn.int32), bufs)
        return self._read(bufs, T)

    def _pt_host(self):
        if getattr(self, "_pt_cache", None) is None:
            pt = self.gen._page_table
            self._pt_cache = ttnn.to_torch(pt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))[: pt.shape[0]]
        return self._pt_cache[: self.U]

    def round(self):
        """One draft + verify (with its ring update) + accept round. Returns the committed tokens per user."""
        U = self.U
        t0 = time.perf_counter()
        drafts = self.draft()
        n = self._last_n if self.alloc else [self.K] * U
        t1 = time.perf_counter()
        picks = self.verify(drafts, n)
        t2 = time.perf_counter()
        committed = []
        r = 0
        for u in range(U):
            k = n[u]
            tgt = picks[r : r + k + 1]
            r += k + 1
            a = 0
            while a < k and drafts[u][a] == tgt[a]:
                a += 1
            committed.append(list(drafts[u][:a]) + [tgt[a]])
            p0 = self.ctx_start[u] + self.ctx_rows[u]
            for i in range(k + 1):
                self.slot_pos[u, (p0 + i - self.base[u]) % self.RING] = p0 + i if i <= a else -1
            rows = min(self.window, self.ctx_rows[u] + a + 1)
            self.ctx_start[u] = p0 + a + 1 - rows
            self.ctx_rows[u] = rows
            self.bonus[u] = tgt[a]
        self.stats.append({"draft_s": t1 - t0, "verify_s": t2 - t1, "accepted": [len(c) - 1 for c in committed],
                           "checked": list(n)})  # fmt: skip
        return committed
