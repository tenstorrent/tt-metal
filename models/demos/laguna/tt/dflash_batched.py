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

    def __init__(self, gen, core, users: int = 32, drafts: int = 1):
        self.gen, self.core = gen, core
        self.mesh, self.model = gen.mesh_device, gen.model
        self.U, self.K = int(users), int(drafts)
        if (self.U * (self.K + 1)) % TILE or (self.U * self.K) % TILE:
            raise ValueError(f"U * K and U * (K + 1) must be multiples of {TILE}: U={self.U}, K={self.K}")
        layer0 = next(iter(core.layers.values()))
        self.nkv, self.hd = int(layer0.cfg.num_kv_heads), int(layer0.cfg.head_dim)
        self.window = int(core.config.sliding_window) - 1
        self.block = int(core.config.block_size)  # 16 proposal rows
        self.mask_id = int(core.config.mask_token_id)
        if self.U % 2:
            raise ValueError("the batched draft packs two users per 32-row tile: U must be even")
        self.P = self.U // 2  # user pairs: pair c holds user 2c in rows 0..15 and user 2c + 1 in rows 16..31
        # ring of pair c: [nkv, 2 * RING, hd], user 2c's slots first (dest u * RING + slot addresses both layouts)
        zeros = torch.zeros((self.P, self.nkv, 2 * self.RING, self.hd), dtype=torch.bfloat16)
        self.rings = {i: (self._dev(zeros, ttnn.TILE_LAYOUT), self._dev(zeros, ttnn.TILE_LAYOUT)) for i in core.layers}
        self.base = [0] * self.U
        self.slot_pos = torch.full((self.U, self.RING), -1, dtype=torch.int64)
        self.ctx_start = [0] * self.U
        self.ctx_rows = [0] * self.U
        self.bonus = [0] * self.U
        self.stats = []
        self._traced = False
        self._gqa = int(layer0.cfg.num_heads) // self.nkv
        self._gqa_fold = os.environ.get("TT_LAGUNA_DFLASH_DRAFT_GQA_FOLD", "1") == "1" and self._gqa > 1
        # the draft's U * 16-row norms width-sharded over the grid (the interleaved norm used 16 cores)
        for layer in core.layers.values():
            layer._norm_max_rows = int(os.environ.get("TT_LAGUNA_DFLASH_DRAFT_NORM_ROWS", str(self.U * self.block)))
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
        """Host inputs of a draft with two users per 32-row tile: token ids [1, U * 16] (pair c: user 2c's bonus +
        15 mask tokens, then user 2c + 1's), (cos, sin) [P, 1, 32, hd] at each row's position, the ring slots
        [1, U * 16] the rows' keys/values are written to (user * RING + slot), and the attention mask [P, 1, 32,
        2 * RING] over the pair's ring slots: a row sees its own user's slots holding a position in its causal
        sliding window -- committed context, and its own block's earlier rows (written there by the draft layers)."""
        U, P, R, B = self.U, self.P, self.RING, self.block
        ids = torch.full((1, U * B), self.mask_id, dtype=torch.int32)
        for u in range(U):
            ids[0, u * B] = self.bonus[u]
        p0 = torch.tensor([self.ctx_start[u] + self.ctx_rows[u] for u in range(U)], dtype=torch.int64)
        base = torch.tensor(self.base, dtype=torch.int64)
        qi = torch.arange(B)
        qpos = p0[:, None] + qi[None, :]  # [U, 16]
        cos, sin = self._rope_rows(qpos.reshape(-1))
        slots = (qpos - base[:, None]) % R  # [U, 16]
        dests = (torch.arange(U)[:, None] * R + slots).reshape(1, U * B).to(torch.int32)
        pos = self.slot_pos.clone()
        pos.scatter_(1, slots, qpos)  # the block's rows at their slots
        pos = pos[:, None, :]  # [U, 1, R]
        start = torch.tensor(self.ctx_start, dtype=torch.int64)[:, None, None]
        q = qpos[:, :, None]
        own = (pos >= start) & (pos <= q) & (pos >= q - self.window)  # [U, 16, R]
        allowed = torch.zeros((P, 2, B, 2 * R), dtype=torch.bool)
        own = own.reshape(P, 2, B, R)
        allowed[:, 0, :, :R] = own[:, 0]
        allowed[:, 1, :, R:] = own[:, 1]
        mask = torch.where(allowed, 0.0, -1e9).to(torch.bfloat16).reshape(P, 1, 2 * B, 2 * R)
        return (ids, cos.reshape(P, 1, 2 * B, self.hd), sin.reshape(P, 1, 2 * B, self.hd), mask, dests)

    def _draft_body(self, ids_tt, rope, mask_tt, bufs, dests_tt=None):
        """Device work of a draft (traceable): proposal blocks through the draft layers, rows 1..K of each block
        through the target LM head, greedy picks into ``bufs`` (one [1, 1, 1, 32] uint32 per 32 picks)."""
        m = self.model
        if self._gqa_fold:  # the 32-row mask repeated once per query head of a KV group (dflash_query_forward_batched)
            mask_tt = ttnn.repeat(mask_tt, ttnn.Shape((1, 1, self._gqa, 1)))
        x = m.embed_prefill(ids_tt)  # [1, U * 16, H]: two users per 32-row tile
        for i in sorted(self.core.layers):
            x = self.core.layers[i].dflash_query_forward_batched(x, *self.rings[i], rope, mask_tt, dests_tt)
        x = self.core.apply_final_norm(x)
        H, B, K = self.gen.hidden, self.block, self.K
        xp = ttnn.reshape(x, (self.P, TILE, H))
        # rows 1..K of each user's block: user 2c at rows 1.., user 2c + 1 at rows 17.. of pair c
        sel = ttnn.concat([ttnn.slice(xp, [0, 1, 0], [self.P, 1 + K, H]),
                           ttnn.slice(xp, [0, B + 1, 0], [self.P, B + 1 + K, H])], dim=1)  # fmt: skip
        sel = ttnn.reshape(sel, (1, self.U * K, H))
        shards = m.lm_head_shards_dflash(sel, enable_experimental=True)
        shards = ttnn.reshape(shards, (1, 1, self.U * self.K, shards.shape[-1]))
        self._greedy_into(shards, bufs)

    def _greedy_into(self, shards, bufs):
        rows = int(shards.shape[-2])
        for c, buf in enumerate(bufs):
            r0 = c * TILE
            part = shards if rows == TILE else ttnn.slice(shards, [0, 0, r0, 0], [1, 1, r0 + TILE, shards.shape[-1]])
            self.gen._greedy_sample(part, TILE, buf)

    def _verify_body(self, tok_tt, cur_tt, ridx_tt, pt_tt, rope_rows, dests_tt, bufs):
        """Device work of a verify + ring update (traceable): the target over every checked row, greedy picks into
        ``bufs``, then the checked rows' target states -> draft fusion -> K/V of each draft layer -> the rings."""
        m, gen = self.model, self.gen
        h = m.embed_decode(tok_tt)
        h, aux = m.verify_layers_batched(h, m.verify_inputs_batched(cur_tt, ridx_tt, pt_tt), gen._kv_cache, capture=True)
        self._greedy_into(m.lm_head_shards_decode(h), bufs)
        x = self.core.combine_aux_hidden_states(aux)  # [1, T, H]
        rings, srcs = [], []
        for i, layer in self.core.layers.items():
            k, v = layer.dflash_kv_rows(x, rope_rows)
            rings += list(self.rings[i])
            srcs += [k, v]
        ring_write_batched(rings, srcs, dests_tt, rows_per_user=self.K + 1)

    def _verify_host(self, drafts):
        """Host inputs of a verify: tokens [1, T], positions [T], page-table rows [T, blocks], ring rope (cos, sin)
        [1, 1, T, hd] and destinations [1, T] (every checked row is written into its user's ring)."""
        U, K = self.U, self.K
        T = U * (K + 1)
        toks = torch.zeros((1, T), dtype=torch.int32)
        pos = torch.zeros((T,), dtype=torch.int32)
        dests = torch.zeros((1, T), dtype=torch.int64)
        for u in range(U):
            p0 = self.ctx_start[u] + self.ctx_rows[u]
            toks[0, u * (K + 1)] = self.bonus[u]
            for i in range(K):
                toks[0, u * (K + 1) + 1 + i] = drafts[u][i]
            for i in range(K + 1):
                pos[u * (K + 1) + i] = p0 + i
                dests[0, u * (K + 1) + i] = u * self.RING + (p0 + i - self.base[u]) % self.RING
        cos, sin = self._rope_rows(pos)
        pt_rows = self._pt_host().repeat_interleave(K + 1, dim=0).to(torch.int32)
        return toks, pos, pt_rows, cos.reshape(1, 1, T, self.hd), sin.reshape(1, 1, T, self.hd), dests.to(torch.int32)

    def capture_traces(self):
        """Persistent inputs and two traces: the draft, and the verify with its ring update. Call after prefill (the
        first eager pass of each body compiles its programs; nothing is allocated between the captures and replay)."""
        U, K = self.U, self.K
        T = U * (K + 1)
        g = self.gen
        ids, cos, sin, mask, dd = self._draft_inputs()
        self._d_in = (self._dev(ids, dtype=ttnn.uint32), self._dev(cos, ttnn.TILE_LAYOUT),
                      self._dev(sin, ttnn.TILE_LAYOUT), self._dev(mask, ttnn.TILE_LAYOUT),
                      self._dev(dd, dtype=ttnn.uint32))  # fmt: skip
        self._d_out = [g._rep(torch.zeros([1, 1, 1, TILE], dtype=torch.int32), ttnn.uint32) for _ in range(U * K // TILE)]
        toks, pos, pt_rows, rc, rs, dests = self._verify_host([[0] * K for _ in range(U)])
        self._v_in = (g._rep(toks, ttnn.uint32), g._rep(pos, ttnn.int32), g._rep(pos.reshape(1, T), ttnn.uint32),
                      g._rep(pt_rows, ttnn.int32), self._dev(rc, ttnn.TILE_LAYOUT), self._dev(rs, ttnn.TILE_LAYOUT),
                      self._dev(dests, dtype=ttnn.uint32))  # fmt: skip
        self._v_out = [g._rep(torch.zeros([1, 1, 1, TILE], dtype=torch.int32), ttnn.uint32) for _ in range(T // TILE)]
        # the ring update of the warm-up / capture passes writes zero-token rows: keep the rings intact by
        # pointing every destination at "skip"
        skip = torch.full((1, T), -1, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(self._host(skip, ttnn.uint32), self._v_in[6])

        def draft():
            ids_tt, c, s_, mk, dd_tt = self._d_in
            self._draft_body(ids_tt, (c, s_), mk, self._d_out, dd_tt)

        def verify():
            t, p, r, pt, c, s_, d = self._v_in
            self._verify_body(t, p, r, pt, (c, s_), d, self._v_out)

        mesh = self.mesh
        draft()
        verify()
        ttnn.synchronize_device(mesh)
        self._tid_d = ttnn.begin_trace_capture(mesh, cq_id=0)
        draft()
        ttnn.end_trace_capture(mesh, self._tid_d, cq_id=0)
        self._tid_v = ttnn.begin_trace_capture(mesh, cq_id=0)
        verify()
        ttnn.end_trace_capture(mesh, self._tid_v, cq_id=0)
        ttnn.synchronize_device(mesh)
        self._traced = True

    def _host(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def _read(self, bufs, n):
        out = []
        for buf in bufs:
            out += self.gen._read_token(buf, TILE)
        return out[:n]

    def draft(self):
        """U x K draft tokens (host lists)."""
        ids, cos, sin, mask, dd = self._draft_inputs()
        if self._traced:
            for dst, (t, dt, lay) in zip(self._d_in, ((ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                                                      (cos, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                                                      (sin, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                                                      (mask, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                                                      (dd, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))):  # fmt: skip
                ttnn.copy_host_to_device_tensor(self._host(t, dt, lay), dst)
            ttnn.execute_trace(self.mesh, self._tid_d, cq_id=0, blocking=False)
            picks = self._read(self._d_out, self.U * self.K)
        else:
            bufs = [self.gen._rep(torch.zeros([1, 1, 1, TILE], dtype=torch.int32), ttnn.uint32)
                    for _ in range(self.U * self.K // TILE)]  # fmt: skip
            self._draft_body(self._dev(ids, dtype=ttnn.uint32),
                             (self._dev(cos, ttnn.TILE_LAYOUT), self._dev(sin, ttnn.TILE_LAYOUT)),
                             self._dev(mask, ttnn.TILE_LAYOUT), bufs, self._dev(dd, dtype=ttnn.uint32))  # fmt: skip
            picks = self._read(bufs, self.U * self.K)
        return [picks[u * self.K : (u + 1) * self.K] for u in range(self.U)]

    def verify(self, drafts):
        """Target picks for every checked row (host list, user-major); the ring update runs in the same pass."""
        U, K = self.U, self.K
        T = U * (K + 1)
        toks, pos, pt_rows, rc, rs, dests = self._verify_host(drafts)
        if self._traced:
            srcs = ((toks, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), (pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    (pos.reshape(1, T), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), (pt_rows, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    (rc, ttnn.bfloat16, ttnn.TILE_LAYOUT), (rs, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    (dests, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))  # fmt: skip
            for i, (t, dt, lay) in enumerate(srcs):
                if i == 3 and getattr(self, "_pt_uploaded", False):
                    continue  # the page-table rows never change
                ttnn.copy_host_to_device_tensor(self._host(t, dt, lay), self._v_in[i])
            self._pt_uploaded = True
            ttnn.execute_trace(self.mesh, self._tid_v, cq_id=0, blocking=False)
            return self._read(self._v_out, T)
        g = self.gen
        bufs = [g._rep(torch.zeros([1, 1, 1, TILE], dtype=torch.int32), ttnn.uint32) for _ in range(T // TILE)]
        self._verify_body(g._rep(toks, ttnn.uint32), g._rep(pos, ttnn.int32), g._rep(pos.reshape(1, T), ttnn.uint32),
                          g._rep(pt_rows, ttnn.int32),
                          (self._dev(rc, ttnn.TILE_LAYOUT), self._dev(rs, ttnn.TILE_LAYOUT)),
                          self._dev(dests, dtype=ttnn.uint32), bufs)  # fmt: skip
        return self._read(bufs, T)

    def _pt_host(self):
        if getattr(self, "_pt_cache", None) is None:
            pt = self.gen._page_table
            self._pt_cache = ttnn.to_torch(pt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))[: pt.shape[0]]
        return self._pt_cache[: self.U]

    def round(self):
        """One draft + verify (with its ring update) + accept round. Returns the committed tokens per user."""
        U, K = self.U, self.K
        t0 = time.perf_counter()
        drafts = self.draft()
        t1 = time.perf_counter()
        picks = self.verify(drafts)
        t2 = time.perf_counter()
        committed = []
        for u in range(U):
            tgt = picks[u * (K + 1) : (u + 1) * (K + 1)]
            a = 0
            while a < K and drafts[u][a] == tgt[a]:
                a += 1
            committed.append(list(drafts[u][:a]) + [tgt[a]])
            p0 = self.ctx_start[u] + self.ctx_rows[u]
            for i in range(K + 1):
                self.slot_pos[u, (p0 + i - self.base[u]) % self.RING] = p0 + i if i <= a else -1
            rows = min(self.window, self.ctx_rows[u] + a + 1)
            self.ctx_start[u] = p0 + a + 1 - rows
            self.ctx_rows[u] = rows
            self.bonus[u] = tgt[a]
        self.stats.append({"draft_s": t1 - t0, "verify_s": t2 - t1, "accepted": [len(c) - 1 for c in committed]})
        return committed
