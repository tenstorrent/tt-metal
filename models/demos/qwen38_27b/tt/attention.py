# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gated full attention (the `attention` step of the `attn` block) of Qwen3.8-27B on the 8 x 4 mesh.

Per chip (SP row r, TP column c), on its own 640 rows of the chunk ([1, 1, 640, 5120] bf16) in the
update_padded_kv_cache writer's order (chip r row i holds position p(r, i): start + r * 640 + i when start is a
multiple of the chunk; KV-pad rotation otherwise, see rope_offsets):
  1. one column-parallel matmul (fp32 out; plus one with the bfp8 residual weight, ATTN_QKV_HILO): [q 6 x 256 | k 256 | v 256 | gate 6 x 256] of q heads 6c..6c+5 and
     KV head c; nlp_create_qkv_heads splits q [1, 6, 640, 256], k / v [1, 1, 640, 256].
  2. q_norm / k_norm per head ((1 + w), fp32), partial RoPE on dims 0-63 (rotate-half by slices, cos / sin fp32 rows
     of this chip's positions sliced from load-time block-cyclic tables).
  3. K (after RoPE, HF order) and V written to the served bfp8 cache rows [start, start + T) with
     update_padded_kv_cache; ring_joint SDPA over the SP ring (axis 0) reads the cache prefix [0, start + T): GQA 6 q
     heads on 1 KV head, scale 1/16, causal, kv_actual_isl = start.
  4. o * sigmoid(gate), o_proj row-parallel (fp32 partials) -> all_reduce over axis 1 -> bf16 replicated residual.
"""

import torch

import ttnn

from .settings import settings


def compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("ATTN_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("ATTN_FP32_ACC"),
        packer_l1_acc=False,
    )


def sdpa_compute_config():
    # fp32_dest_acc_en must be False: kv_actual_isl (the chunked cache read) runs only on the streaming kernel
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("SDPA_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def rope_offsets(start: int, sp: int, chunk_local: int) -> list[int]:
    """Per chip, the first cache-local row this chunk's rows occupy (update_padded_kv_cache's update_idxt)."""
    split_chunk, split_chip, split_off = start // (sp * chunk_local), (start // chunk_local) % sp, start % chunk_local
    return [
        (split_chunk + 1) * chunk_local
        if c < split_chip
        else split_chunk * chunk_local + (split_off if c == split_chip else 0)
        for c in range(sp)
    ]


class TtGatedAttention:
    """self_attn of one layer. Weights from the checkpoint dict ``w`` (``self_attn.*`` keys, bf16)."""

    def __init__(self, mesh, cfg, w: dict, max_seq: int, chunk: int, prefix: str = "self_attn."):
        self.mesh, self.cfg = mesh, cfg
        self.sp, self.tp = mesh.shape[0], mesh.shape[1]
        self.D, self.R = cfg.head_dim, cfg.rope_dim
        self.nq, self.nkv = cfg.n_heads // self.tp, cfg.n_kv // self.tp  # 6 q heads, 1 KV head per column
        assert self.nkv == 1, "one KV head per TP column"
        self.max_seq, self.chunk = max_seq, chunk
        self.ckc, self.sdpa_ckc = compute_config(), sdpa_compute_config()
        self.eps = cfg.eps
        p, D, nq = prefix, self.D, self.nq

        # column-parallel [q | k | v | gate] weight [3584, hidden] per column, [out, in] with transpose_b (bfp8 blocks
        # along the reduction dim, as gdn.py)
        qg = w[p + "q_proj.weight"].float().view(cfg.n_heads, 2, D, -1)  # head: [q 256 | gate 256]
        kw, vw = (w[p + n].float().view(cfg.n_kv, D, -1) for n in ("k_proj.weight", "v_proj.weight"))
        cols = []
        for c in range(self.tp):
            heads = slice(c * nq, (c + 1) * nq)
            cols += [qg[heads, 0].reshape(nq * D, -1), kw[c], vw[c], qg[heads, 1].reshape(nq * D, -1)]
        qkvg = torch.cat(cols, 0)
        self.w_qkvg = self._shard(qkvg, ttnn.bfloat8_b, dim=2)
        # ATTN_QKV_HILO: also the bfp8 residual W - bfp8(W) (exactly as the device rounds W). One attn_norm channel
        # reaches |x| = 92 (median 1.9); with one bfp8 weight the output row norms come out 0.5 % low on average
        self.w_qkvg_lo = None
        if settings.get("ATTN_QKV_HILO"):
            hi = ttnn.to_torch(ttnn.from_torch(qkvg, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
            self.w_qkvg_lo = self._shard(qkvg - hi, ttnn.bfloat8_b, dim=2)
        # row-parallel o_proj [hidden, n_heads * D]: column c takes inputs of its heads [1536 c, 1536 (c + 1))
        self.w_o = self._shard(w[p + "o_proj.weight"].float(), ttnn.bfloat8_b, dim=3)
        self.q_norm = self._replicated((1.0 + w[p + "q_norm.weight"].float()).reshape(1, 1, 1, D))
        self.k_norm = self._replicated((1.0 + w[p + "k_norm.weight"].float()).reshape(1, 1, 1, D))

        # cos / sin [max_seq / sp, rope_dim] fp32 per SP row in cache-local row order (row lr of chip r holds
        # position (lr // 640) * chunk + r * 640 + lr % 640), rotate-half layout as the reference
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.R, 2, dtype=torch.int64).float() / self.R))
        chunk_local = chunk // self.sp
        local = max_seq // self.sp
        r, lr = torch.arange(self.sp).repeat_interleave(local), torch.arange(local).repeat(self.sp)
        pos = ((lr // chunk_local) * chunk + r * chunk_local + lr % chunk_local).float()
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        mm = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))
        self.cos, self.sin = (
            ttnn.from_torch(
                t.reshape(1, 1, *t.shape),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mm,
            )
            for t in (emb.cos(), emb.sin())
        )
        # this chip's SP row index [1, 1, 1, 1] fp32 (selects its rope rows when the chips' offsets differ)
        self.row_id = ttnn.from_torch(
            torch.arange(self.sp, dtype=torch.float32).reshape(self.sp, 1, 1, 1),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
        )

        # ring SDPA: CCL workers in the last compute column, SDPA on the rest; persistent gather scratch per K / V
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.sems = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        self.ccl_offset = ttnn.CoreCoord(grid.x - 1, 0)
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=settings.get("SDPA_Q_CHUNK"),
            k_chunk_size=settings.get("SDPA_K_CHUNK"),
            exp_approx_mode=False,
            segmented_accumulation=settings.get("SDPA_SEGMENTED"),
        )
        self.num_links = settings.get("SDPA_NUM_LINKS")
        self.topology = getattr(ttnn.Topology, settings.get("SDPA_TOPOLOGY"))
        self.gather_k, self.gather_v = (
            ttnn.from_torch(
                torch.zeros(1, cfg.n_kv, max_seq, D),
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 1)),
            )
            for _ in range(2)
        )

    # ---------------------------------------------------------------- load-time helpers
    def _shard(self, t: torch.Tensor, dtype, dim: int) -> ttnn.Tensor:
        """2-D t as [1, 1, *t.shape], split in tp equal blocks along dim (2 = rows, 3 = columns): block c on TP
        column c, replicated over the SP rows."""
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, dim)),
        )

    def _replicated(self, t: torch.Tensor) -> ttnn.Tensor:
        # TILE fp32 gamma (a ROW_MAJOR gamma disables rms_norm's streaming kernel)
        return ttnn.from_torch(
            t,
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    # ---------------------------------------------------------------- forward helpers
    def _rope_rows(self, start: int, rows: int):
        """cos / sin [1, 1, rows, rope_dim] fp32 of this chip's rows: the table rows from rope_offsets(start)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        offs = rope_offsets(start, self.sp, rows)
        out = []
        for table in (self.cos, self.sin):
            parts = {
                o: ttnn.slice(table, (0, 0, o, 0), (1, 1, o + rows, self.R), memory_config=dram)
                for o in sorted(set(offs))
            }
            if len(parts) == 1:
                out.append(next(iter(parts.values())))
                continue
            # chips differ (start inside a chunk): sum of each offset's rows times the mask of the chips using it
            acc = None
            for o, t in parts.items():
                chips = [c for c in range(self.sp) if offs[c] == o]
                lo, hi = chips[0], chips[-1]
                m = ttnn.logical_and(ttnn.ge(self.row_id, float(lo)), ttnn.le(self.row_id, float(hi)))
                term = ttnn.multiply(t, m, memory_config=dram)
                for u in (t, m):
                    ttnn.deallocate(u)
                acc = term if acc is None else ttnn.add(acc, term, memory_config=dram)
            out.append(acc)
        return out

    def _norm_rope(self, t: ttnn.Tensor, gamma: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """(1 + w) RMSNorm per head, then rotate-half RoPE on dims [0, rope_dim) (fp32)."""
        dram, R, D = ttnn.DRAM_MEMORY_CONFIG, self.R, self.D
        n = ttnn.rms_norm(t, epsilon=self.eps, weight=gamma, memory_config=dram, compute_kernel_config=self.ckc)
        ttnn.deallocate(t)
        H, S = n.shape[1], n.shape[2]
        x1 = ttnn.slice(n, (0, 0, 0, 0), (1, H, S, R // 2), memory_config=dram)
        x2 = ttnn.slice(n, (0, 0, 0, R // 2), (1, H, S, R), memory_config=dram)
        tail = ttnn.slice(n, (0, 0, 0, R), (1, H, S, D), memory_config=dram)
        xr = ttnn.concat([x1, x2], dim=-1, memory_config=dram)
        rot = ttnn.concat([ttnn.neg(x2, memory_config=dram), x1], dim=-1, memory_config=dram)
        ttnn.deallocate(n)
        a = ttnn.multiply(xr, cos, memory_config=dram)
        b = ttnn.multiply(rot, sin, memory_config=dram)
        roped = ttnn.add(a, b, memory_config=dram)
        out = ttnn.concat([roped, tail], dim=-1, memory_config=dram)
        for u in (x1, x2, tail, xr, rot, a, b, roped):
            ttnn.deallocate(u)
        return out

    def __call__(self, x: ttnn.Tensor, start: int, cache, slot: int = 0, cache_layer: int = 0) -> ttnn.Tensor:
        """x [1, 1, T / sp, hidden] bf16 (this SP row's rows, replicated over TP); cache a TtKVCache (written in
        place at rows [start, start + T)). Returns [1, 1, T / sp, hidden] bf16 replicated over TP."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        rows = x.shape[-2]
        T = rows * self.sp
        D, nq = self.D, self.nq
        qkv_w = (nq + 2) * D

        # 1. projections (fp32 out) and heads
        qkvg = ttnn.linear(
            x, self.w_qkvg, transpose_b=True, compute_kernel_config=self.ckc, dtype=ttnn.float32, memory_config=dram
        )
        if self.w_qkvg_lo is not None:
            lo = ttnn.linear(
                x,
                self.w_qkvg_lo,
                transpose_b=True,
                compute_kernel_config=self.ckc,
                dtype=ttnn.float32,
                memory_config=dram,
            )
            qkvg_lo = qkvg
            qkvg = ttnn.add(qkvg_lo, lo, memory_config=dram)
            for u in (qkvg_lo, lo):
                ttnn.deallocate(u)
        qkv = ttnn.slice(qkvg, (0, 0, 0, 0), (1, 1, rows, qkv_w), memory_config=dram)
        gate = ttnn.slice(qkvg, (0, 0, 0, qkv_w), (1, 1, rows, qkv_w + nq * D), memory_config=dram)
        ttnn.deallocate(qkvg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nq, num_kv_heads=self.nkv, transpose_k_heads=False, memory_config=dram
        )
        ttnn.deallocate(qkv)

        # 2. q / k norm and partial RoPE at this chip's positions
        cos, sin = self._rope_rows(start, rows)
        q = self._norm_rope(q, self.q_norm, cos, sin)
        k = self._norm_rope(k, self.k_norm, cos, sin)
        for u in (cos, sin):
            ttnn.deallocate(u)

        # 3. cache write [start, start + T), then the ring SDPA over the cache prefix
        for c_, t in ((cache.k, k), (cache.v, v)):
            t8 = ttnn.typecast(t, cache.dtype)
            ttnn.deallocate(t)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                c_,
                t8,
                slot_idx=slot,
                layer_idx=cache_layer,
                num_layers=cache.num_layers,
                kv_actual_global=start,
                cluster_axis=0,
            )
            ttnn.deallocate(t8)
        qb = ttnn.typecast(q, ttnn.bfloat16)
        ttnn.deallocate(q)
        o, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            qb,
            cache.k,
            cache.v,
            None,
            None,
            None,
            persistent_output_buffer_k=self.gather_k,
            persistent_output_buffer_v=self.gather_v,
            joint_strategy="rear",
            logical_n=start + T,
            program_config=self.sdpa_pc,
            compute_kernel_config=self.sdpa_ckc,
            dim=2,
            multi_device_global_semaphore=self.sems,
            num_links=self.num_links,
            cluster_axis=0,
            mesh_device=self.mesh,
            topology=self.topology,
            ccl_core_grid_offset=self.ccl_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=float(torch.tensor(D**-0.5, dtype=torch.float32)),
            is_balanced=False,
            kv_cache_batch_idx=slot,
            kv_cache_num_layers=cache.num_layers,
            kv_cache_layer_idx=cache_layer,
            kv_actual_isl=start,
        )  # [1, nq, rows, D] bf16
        ttnn.deallocate(qb)

        # 4. gate, o_proj, all_reduce over TP
        oc = ttnn.experimental.nlp_concat_heads(o, memory_config=dram)  # [1, 1, rows, nq * D]
        ttnn.deallocate(o)
        sg = ttnn.sigmoid(gate, memory_config=dram)
        ttnn.deallocate(gate)
        oc32 = ttnn.typecast(oc, ttnn.float32)
        ttnn.deallocate(oc)
        gated = ttnn.multiply(oc32, sg, dtype=ttnn.bfloat16, memory_config=dram)
        for u in (oc32, sg):
            ttnn.deallocate(u)
        part = ttnn.linear(
            gated, self.w_o, transpose_b=True, compute_kernel_config=self.ckc, dtype=ttnn.float32, memory_config=dram
        )
        ttnn.deallocate(gated)
        red = ttnn.all_reduce(part, cluster_axis=1, memory_config=dram)
        ttnn.deallocate(part)
        out = ttnn.typecast(red, ttnn.bfloat16)
        ttnn.deallocate(red)
        return out
