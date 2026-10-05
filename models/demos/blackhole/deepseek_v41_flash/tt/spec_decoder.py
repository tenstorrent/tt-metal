# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative-decoding verify step on the device: one traced step over ``T = U * n`` token rows per mesh row (n = 1 + k, k = 1..5 at runtime).

``SpecVerifier`` = the existing decode step (embedding -> 40 layers with Engram -> head -> greedy sampling of EVERY row) over virtual-user rows,
with the paged spec attention, plus the ratio-2 compressor state commit at the end of the step.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder


class SpecVerifier(DSV41Decoder):
    def _engram_rows(self, T):
        """Engram rows of every token row from the packed host upload (same layout as ``DSV41Decoder.forward``: v2 = one [T,Kin] tile block)."""
        self.rows, off = {}, 0
        if not self.engram or self.rows_cat is None:
            return
        kin = {l: e.kin for l, e in self.engram.items()}
        v2 = os.environ.get("DSV41_ENGRAM_V2") != "0"
        rc = ttnn.reshape(self.rows_cat, [1, 1, T, self.rows_cat.shape[-1]]) if v2 else self.rows_cat
        for l in self.engram_order:
            sl = (
                ttnn.slice(rc, [0, 0, 0, off], [1, 1, T, off + kin[l]])
                if v2
                else ttnn.slice(rc, [0, 0, 0, off], [T, 1, 1, off + kin[l]])
            )
            self.rows[l] = ttnn.to_layout(sl, ttnn.TILE_LAYOUT)
            off += kin[l]

    def _engram_fwd(self, lid, x):
        eng = self.engram[lid]
        return (
            eng.forward(x, self.rows[lid])
            if os.environ.get("DSV41_ENGRAM_V2") == "0"
            else eng.forward_v2(x, self.rows[lid])
        )

    def forward(self):
        """-> logits shard [1,1,T,vocab/cols] fp32 for every token row; ``self.sampled`` = per-row greedy (max, idx) pairs (see DSV41DeviceHead.sample)."""
        T = self.ctrl.shape[0]
        tokens = ttnn.typecast(ttnn.slice(self.ctrl, [0, 0], [T, 1]), ttnn.uint32)
        self.pos = ttnn.reshape(ttnn.slice(self.ctrl, [0, 1], [T, 2]), [T])
        self._engram_rows(T)
        x, pre = self.embedding.forward(tokens)
        states = {k: ss.build(self.pos) for k, ss in self.step_states.items()}
        dbg = os.environ.get("DSV41_DEBUG_LAYERS") == "1" and not getattr(self, "_dbg_done", False)
        for lid, layer, st in self.layers:
            if lid in self.engram:
                x = self._engram_fwd(lid, x)
                if dbg:
                    self._dbg(f"engram {lid}", x)
            x, pre = layer.forward(x, pre, states[st], profile=getattr(self, "profile", None))
            if dbg:
                self._dbg(f"layer {lid}", x)
        self._dbg_done = True
        logits = self.head.forward(x, pre)
        if self.mesh_config is not None:
            self.sampled = self.head.sample(logits, self.mesh_config, self.ccl)
            t2 = ttnn.topk(logits, k=2, dim=-1, largest=True, sorted=True)[0]
            self.top2 = self.mesh_config.allgather(
                t2, self.ccl, axis=1, dim=3
            )  # near-tie evidence (see SpecDecoder.forward)
        self.commit()
        self.logits = logits
        return logits

    def _dbg(self, tag, x):
        ttnn.synchronize_device(self.md)
        t = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
        print(
            f"LAYERDBG {tag:12s} absmax {float(t.abs().max()):.4g} mean|x| {float(t.abs().mean()):.4g} rows absmax {[round(float(t[i].abs().max()), 1) for i in range(min(t.shape[0], 4))]}",
            flush=True,
        )

    def commit(self, m=None):
        for _, layer, _ in self.layers:
            layer.attention.commit(m)

    def snapshot_states(self):
        return [
            (layer.attention, layer.attention.snapshot_state())
            for _, layer, _ in self.layers
            if getattr(layer.attention, "prev_cs", None) is not None
        ]


TAP_LAYERS = (37, 38, 39)  # dspark_target_layer_ids


class SpecDecoder(SpecVerifier):
    """One whole speculative ROUND as a single traced step (verify block -> greedy argmax of every row -> accept -> drafter), n = 1 + k rows per user.

    Host inputs per round (``set_packed_inputs``): block tokens ``[t, d_1..d_k]`` per user (user-major rows), positions ``base + j``, Engram rows.
    Device: verify with hidden taps, ``a_j = argmax`` per row, m = #leading drafts with ``x_{j+1} == a_j``, ``prev_cs`` commit (m), write of ``main_kv`` of all
    n rows into the drafter rings, draft of 5 tokens from (``a_m``, frontier ``base + m``). ``self.pack`` = uint32 [1, T + U + 5U] per mesh row:
    ``a`` (T), ``m`` (U), drafts d_1..d_5 block-index-major (5U); ``self.draft_out['conf']`` stays on the device (diagnostics).
    """

    def __init__(self, md, layers, embedding, head, drafter, engram=None, step_states=None, n=2):
        super().__init__(md, layers, embedding, head, engram, step_states=step_states)
        self.drafter, self.n = drafter, n
        self.U = drafter.U
        rep = ttnn.ReplicateTensorToMesh(md)
        c = lambda t, dt, lay: ttnn.from_torch(
            t, device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
        )
        U = self.U
        self.noise = c(torch.full((4 * U, 1), 128799, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.arange_n = c(torch.arange(n, dtype=torch.float32).reshape(1, n), ttnn.float32, ttnn.TILE_LAYOUT)
        self.zero_tok = None
        self.keff = (
            int(os.environ.get("DSV41_KEFF", "0")) or None
        )  # valid drafts per block when the block is padded (n = 1 + k rows, k > keff)
        self.stop_after = None  # None | 'verify' | 'accept': truncate the traced round (timing breakdown)
        # runtime override of the accept count (prompt feeding / teacher forcing): m = force if force >= 0 else the computed one
        rows, cols = tuple(md.shape)
        self._row_map = ttnn.ShardTensor2dMesh(
            md, dims=(0, None), mesh_shape=(rows, cols)
        )  # users sharded over the mesh rows
        self.force = ttnn.from_torch(
            torch.full((rows * U, 1), -1.0),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._row_map,
        )

    def set_force(self, value):
        """Host: force m (accepted drafts) for the next replays; ``value`` a scalar (all users) or a [B] tensor (one per user, global order). value < 0 = normal operation."""
        rows = self.md.shape[0]
        v = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        v = v.expand(rows * self.U) if v.numel() == 1 else v
        host = ttnn.from_torch(
            v.reshape(rows * self.U, 1).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._row_map,
        )
        ttnn.copy_host_to_device_tensor(host, self.force)

    def _tap(self, x):
        m = ttnn.mean(x, dim=2, keepdim=True)  # [T,1,1,D] fp32 mean over the 4 streams
        return ttnn.reshape(m, [1, 1, x.shape[0], x.shape[3]])

    def _chunk_sizes(self, T):
        """Row chunks (<= 32 rows: the mHC kernels / router / shared expert limit); only with DSV41_SPEC_ROWS=1 and T > 32."""
        if T <= 32 or os.environ.get("DSV41_SPEC_ROWS") != "1":
            return None
        assert T % 32 == 0, f"T={T} rows per mesh row: row-chunked verify needs a multiple of 32"
        return [32] * (T // 32)

    def _forward_chunked(self, T, tokens, sizes):
        """Verify body for T > 32 rows: streams as a list of 32-row chunks (see DSV41Layer.forward_chunks)."""
        offs = [32 * i for i in range(len(sizes))]
        xs, pres, taps = [], [], []
        for o in offs:
            x, pre = self.embedding.forward(ttnn.slice(tokens, [o, 0], [o + 32, 1]))
            xs.append(x)
            pres.append(ttnn.slice(self.embedding.pre, [o, 0, 0, 0], [o + 32, 1, 1, 4]))
        for lid, layer, st in self.layers:
            if lid in self.engram:
                xs = [
                    self.engram[lid].forward_v2(
                        xs[c], ttnn.slice(self.rows[lid], [0, 0, o, 0], [1, 1, o + 32, self.rows[lid].shape[3]])
                    )
                    for c, o in enumerate(offs)
                ]
            if lid in TAP_LAYERS:
                taps.append(ttnn.concat([self._tap(x) for x in xs], dim=2))
            xs, pres = layer.forward_chunks(xs, pres, self._states[st])
        logits = ttnn.concat([self.head.forward(xs[c], pres[c]) for c in range(len(xs))], dim=2)
        return logits, taps, xs

    def forward(self):
        U, n = self.U, self.n
        T = U * n
        tokens = ttnn.typecast(ttnn.slice(self.ctrl, [0, 0], [T, 1]), ttnn.uint32)
        self.pos = ttnn.reshape(ttnn.slice(self.ctrl, [0, 1], [T, 2]), [T])
        self._engram_rows(T)
        states = {k: ss.build(self.pos) for k, ss in self.step_states.items()}
        taps = []
        sizes = self._chunk_sizes(T)
        if sizes is not None:
            self._states = states
            logits, taps, xs = self._forward_chunked(T, tokens, sizes)
            x = None
        else:
            x, pre = self.embedding.forward(tokens)
            dbg = os.environ.get("DSV41_DEBUG_LAYERS") == "1" and not getattr(self, "_dbg_done", False)
            for lid, layer, st in self.layers:
                if lid in self.engram:
                    x = self._engram_fwd(lid, x)
                    if dbg:
                        self._dbg(f"engram {lid}", x)
                if lid in TAP_LAYERS:
                    taps.append(self._tap(x))
                x, pre = layer.forward(x, pre, states[st], profile=getattr(self, "profile", None))
                if dbg:
                    self._dbg(f"layer {lid}", x)
            self._dbg_done = True
            logits = self.head.forward(x, pre)
        a = self.head.sample_global(logits, self.mesh_config, self.ccl)  # [T,1] uint32 RM: argmax of every row
        # top-2 logits of every row (per column shard, all-gathered): near-tie evidence for exactness analysis, read only on request
        t2 = ttnn.topk(logits, k=2, dim=-1, largest=True, sorted=True)[0]  # [1,1,T,2] fp32 per column shard
        self.top2 = self.mesh_config.allgather(t2, self.ccl, axis=1, dim=3)  # [1,1,T,2*cols]
        if self.stop_after == "verify":  # timing breakdown only
            return a
        # ---- accept ----
        rm = ttnn.ROW_MAJOR_LAYOUT
        f32 = lambda t, shape: ttnn.to_layout(ttnn.typecast(ttnn.reshape(t, shape), ttnn.float32), ttnn.TILE_LAYOUT)
        a2 = ttnn.typecast(ttnn.reshape(a, [U, n]), ttnn.float32)  # RM [U,n]
        x2 = ttnn.typecast(ttnn.reshape(tokens, [U, n]), ttnn.float32)
        col = lambda t, j: ttnn.to_layout(ttnn.slice(t, [0, j], [U, j + 1]), ttnn.TILE_LAYOUT)  # [U,1] tile
        c, mcount = None, None
        for j in range(n - 1):
            e = ttnn.eq(col(a2, j), col(x2, j + 1))  # draft d_{j+1} (input of row j+1) == argmax of row j
            c = e if c is None else ttnn.multiply(c, e)
            mcount = c if mcount is None else ttnn.add(mcount, c)
        if mcount is None:
            mcount = ttnn.multiply(col(a2, 0), 0.0)
        if (
            self.keff is not None
        ):  # padded verify block (rows beyond keff are filler on the T=16/32 fast paths): never accept more than keff drafts
            mcount = ttnn.minimum(mcount, float(self.keff))
        mcount = ttnn.where(ttnn.ge(self.force, 0.0), self.force, mcount)
        onehot = ttnn.eq(
            ttnn.repeat(mcount, [1, n]), ttnn.repeat(self.arange_n, [U, 1])
        )  # [U,n] fp32 tile: 1 at j == m
        t_next = ttnn.sum(
            ttnn.multiply(onehot, ttnn.to_layout(a2, ttnn.TILE_LAYOUT)), dim=1, keepdim=True
        )  # [U,1] = a_m
        pos2 = ttnn.typecast(ttnn.reshape(self.pos, [U, n]), ttnn.float32)
        base = col(pos2, 0)
        f_next = ttnn.add(base, mcount)  # frontier = position of the input row whose argmax is the new bonus token
        if self.stop_after == "accept":
            return t_next
        # ---- commit compressor state, write main_kv of the n rows, draft ----
        oh3 = ttnn.reshape(ttnn.to_layout(onehot, rm), [U, n, 1])
        self.commit(oh3)
        hidden = ttnn.typecast(
            ttnn.concat(
                taps or [(self._tap(x) if x is not None else ttnn.concat([self._tap(c) for c in xs], dim=2))] * 3, dim=3
            ),
            ttnn.bfloat16,
        )  # [1,1,T,15360]; partial-layer debug runs have no tap layers
        dr = self.drafter
        dr.write_main_full(hidden, self.pos)
        t_u32 = ttnn.typecast(ttnn.to_layout(t_next, rm), ttnn.uint32)  # [U,1]
        f_i32 = ttnn.reshape(ttnn.typecast(ttnn.to_layout(f_next, rm), ttnn.int32), [U, 1])
        d, drafts = dr.draft_full(t_u32, f_i32)  # drafts [5U,1] uint32 block-index-major
        self.draft_out = d
        m_u32 = ttnn.typecast(ttnn.to_layout(mcount, rm), ttnn.uint32)  # [U,1]
        self.pack = ttnn.concat(
            [ttnn.reshape(a, [1, T]), ttnn.reshape(m_u32, [1, U]), ttnn.reshape(drafts, [1, 5 * U])], dim=1
        )
        self.logits = logits
        return self.pack

    def commit(self, onehot=None):
        for _, layer, _ in self.layers:
            layer.attention.commit(onehot)
