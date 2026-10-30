# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (the `attention` step of the `linear` block) of Qwen3.8-27B on the 8 x 4 mesh.

Per chip (SP row r, TP column c), on its own 640 rows of the chunk ([1, 1, 640, 5120] bf16, block-cyclic):
  1. column-parallel projections: [q 512 | k 512 | v 1536] (key heads 4c..4c+3, value heads 12c..12c+11) and z 1536
     in one matmul (fp32 out); a / b (12 heads each) in one fp32 matmul. g = -exp(A_log) * softplus(a + dt_bias) and
     beta = sigmoid(b) in fp32.
  2. all_gather over axis 0 of the raw qkv, g and beta to the whole chunk, rotated from the engine's chip order to
     natural order on the device (slices + concat; identity when start is a multiple of the chunk).
  3. qkv_causal_conv1d_silu on the whole natural chunk with the conv state as history (redundant over the 8 SP
     rows; actual_start 0 and predecessor_carry aliased to history, so every chip reads the history).
  4. chunk_gated_delta_rule from the fp32 recurrent state: chunk 64 on the rank-4 path (q / k l2-normalised in fp32
     here; GDN_CHUNK) or chunk 32 on the rank-3 flat path (in-kernel l2 norm).
  5. back to chip order, mesh_partition keeps this row's 640 rows; gated RMSNorm per head (plain w) * silu(z);
     out_proj row-parallel (fp32 partials) -> all_reduce over axis 1 -> bf16 replicated residual layout.
"""

import functools

import torch

import ttnn

from .settings import settings

CHUNK_FLAT = 32  # the rank-3 (flat q/k/v, in-kernel l2 norm) path of chunk_gated_delta_rule runs only at chunk 32


def compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("GDN_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("GDN_FP32_ACC"),
        packer_l1_acc=False,
    )


@functools.lru_cache(maxsize=None)
def chip_order_runs(start: int, T: int, sp: int, to_natural: bool) -> list[tuple[int, int]]:
    """Row runs [lo, hi) that, concatenated, reorder a chunk of T rows between the engine's chip order (chip q's
    T / sp local rows at [q * T / sp, (q + 1) * T / sp), token g on chip (g / rows) % sp at local row g % rows) and
    natural order. to_natural: runs index the chip-order tensor; else they index the natural-order tensor."""
    rows = T // sp
    # chip-order index of each natural row n (token start + n)
    pos = [((start + n) // rows % sp) * rows + (start + n) % rows for n in range(T)]
    if not to_natural:
        inv = [0] * T
        for n, i in enumerate(pos):
            inv[i] = n
        pos = inv
    runs, lo = [], pos[0]
    for a, b in zip(pos, pos[1:] + [None]):
        if b != a + 1:
            runs.append((lo, a + 1))
            lo = b
    return runs


class TtGatedDeltaNet:
    """linear_attn of one layer. Weights from the checkpoint dict ``w`` (``linear_attn.*`` keys, bf16)."""

    def __init__(self, mesh, cfg, w: dict, prefix: str = "linear_attn."):
        self.mesh, self.cfg = mesh, cfg
        self.tp, self.sp = mesh.shape[1], mesh.shape[0]
        tp = self.tp
        self.nk, self.nv = cfg.lk_heads // tp, cfg.lv_heads // tp  # 4 key heads, 12 value heads per column
        self.dk, self.dv = cfg.lk_dim, cfg.lv_dim
        self.kd, self.vd = self.nk * self.dk, self.nv * self.dv  # 512, 1536
        self.cd = 2 * self.kd + self.vd  # 2560 conv channels per column
        self.K = cfg.conv_k
        self.ckc = compute_config()
        p = prefix

        qkv_w = w[p + "in_proj_qkv.weight"].float()
        conv_w = w[p + "conv1d.weight"].float()[:, 0, :]  # [conv_dim, K]
        self.qkv_rows = [self._qkv_rows(c) for c in range(tp)]  # conv channel index of each column's channels
        z_w, a_w, b_w = (w[p + n].float() for n in ("in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight"))
        nv, vd = self.nv, self.vd

        # Weights stay [out, in] (HF layout) and run with transpose_b: bfloat8_b shares one exponent per 16 values
        # along the last dim, so the blocks run along the reduction dim. Blocks along the output dim let a small
        # output channel share the exponent of 15 larger neighbours (out_proj column 3993 rel 0.50).
        # column-parallel [qkv | z] weight: [4096, hidden] per column
        qkvz = torch.cat([torch.cat([qkv_w[self.qkv_rows[c]], z_w[c * vd : (c + 1) * vd]]) for c in range(tp)], 0)
        self.w_qkvz = self._shard(qkvz, ttnn.bfloat8_b, dim=2)
        # a / b in one [64, hidden] weight per column: a in rows 0..11, b in 32..43 (tile-aligned slices)
        ab = torch.zeros(64 * tp, cfg.hidden)
        for c in range(tp):
            ab[64 * c : 64 * c + nv] = a_w[c * nv : (c + 1) * nv]
            ab[64 * c + 32 : 64 * c + 32 + nv] = b_w[c * nv : (c + 1) * nv]
        self.w_ab = self._shard(ab, ttnn.bfloat8_b, dim=2)
        # row-parallel out_proj [hidden, value_dim]: column c takes inputs [1536 c, 1536 (c + 1)): [hidden, 1536]
        self.w_out = self._shard(w[p + "out_proj.weight"].float(), ttnn.bfloat8_b, dim=3)

        # per-head constants [1, 1, 1, 32] fp32 per column (12 heads, zero padded to a tile: g and beta are computed
        # and gathered 32 wide, then sliced to the 12 heads)
        def per_head(t):
            out = torch.zeros(1, 32 * tp)
            for c in range(tp):
                out[0, 32 * c : 32 * c + nv] = t[c * nv : (c + 1) * nv]
            return self._shard(out, ttnn.float32, dim=3)

        self.neg_exp_a = per_head(-w[p + "A_log"].float().exp())
        self.dt_bias = per_head(w[p + "dt_bias"].float())
        # conv taps: four [1, 1, 2560] bf16 TILE per column, tap j multiplies row t - 3 + j
        self.taps = [
            self._shard(
                torch.cat([conv_w[self.qkv_rows[c], j] for c in range(tp)]).reshape(1, -1), ttnn.bfloat16, dim=3
            )
            for j in range(self.K)
        ]
        # gated norm weight (plain w), TILE fp32 [1, 1, 1, 128] (a ROW_MAJOR gamma disables the streaming kernel)
        self.norm_w = ttnn.from_torch(
            w[p + "norm.weight"].float().reshape(1, 1, 1, -1),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.eps = cfg.eps
        self.actual_start = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int64),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.conv_pc = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self._channel_chunk(self.cd))
        self.consts = {C: self._gdn_consts(C) for C in (32, 64)}

    # ---------------------------------------------------------------- load-time helpers
    def _qkv_rows(self, c: int) -> torch.Tensor:
        cfg, kd, vd = self.cfg, self.kd, self.vd
        q = torch.arange(c * kd, (c + 1) * kd)
        k = cfg.key_dim + q
        v = 2 * cfg.key_dim + torch.arange(c * vd, (c + 1) * vd)
        return torch.cat([q, k, v])

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

    @staticmethod
    def _channel_chunk(channels: int, cap: int = 512) -> int:
        for c in range(min(cap, channels) - min(cap, channels) % 32, 0, -32):
            if channels % c == 0:
                return c
        raise ValueError(f"no tile-aligned channel chunk divides {channels}")

    def _gdn_consts(self, C: int):
        """chunk_gated_delta_rule's constant tiles (eye, tril, ones [C, C], quadrant masks [32, 96]), built once."""
        ii, jj = torch.arange(32)[:, None], torch.arange(32)[None, :]
        lo_i, lo_j = ii < 16, jj < 16
        masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)
        return tuple(
            ttnn.from_torch(
                t.reshape(1, 1, *t.shape),
                device=self.mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            for t in (torch.eye(C), torch.tril(torch.ones(C, C)), torch.ones(C, C), masks)
        )

    def state_to_device(self, conv: torch.Tensor, recurrent: torch.Tensor) -> dict:
        """The reference's state (conv [conv_dim, K] oldest first, recurrent [48, 128, 128] fp32) in the device
        layout: conv history [1, 3, 2560] bf16 ROW_MAJOR (the last K - 1 inputs) and recurrent [1, 12, 128, 128]
        fp32 TILE, per TP column, replicated over the SP rows. Load time / harness boundary only."""
        hist = torch.cat([conv[self.qkv_rows[c], 1:].T.float() for c in range(self.tp)], -1)  # [3, tp * 2560]
        mm = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 2))
        h = ttnn.from_torch(
            hist.reshape(1, *hist.shape),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        mm = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 1))
        r = ttnn.from_torch(
            recurrent.float().reshape(1, *recurrent.shape),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        return {"conv": h, "recurrent": r}

    # ---------------------------------------------------------------- forward helpers
    def _gather_natural(self, t: ttnn.Tensor, start: int, T: int) -> ttnn.Tensor:
        """[1, 1, T / sp, W] per chip -> [1, 1, T, W] whole chunk in natural order on every chip."""
        g = ttnn.all_gather(t, dim=2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._reorder(g, chip_order_runs(start, T, self.sp, to_natural=True), dim=2)

    def _linear(self, x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        """x @ w^T (w stored [out, in]), fp32 out."""
        return ttnn.linear(
            x,
            w,
            transpose_b=True,
            compute_kernel_config=self.ckc,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _split_heads(t: ttnn.Tensor, H: int, D: int, T: int) -> ttnn.Tensor:
        """flat [1, T, H * D] TILE -> [1, T, H, D] TILE (through ROW_MAJOR: a TILE reshape would pad H)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        r = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        ttnn.deallocate(t)
        r = ttnn.reshape(r, (1, T, H, D))
        out = ttnn.to_layout(r, ttnn.TILE_LAYOUT, memory_config=dram)
        ttnn.deallocate(r)
        return out

    def _l2norm(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """x * rsqrt(sum(x^2) + 1e-6) over the head dim, in fp32 (the reference's l2norm)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.typecast(t, ttnn.float32)
        ttnn.deallocate(t)
        sq = ttnn.multiply(x, x, memory_config=dram)
        ss = ttnn.sum(sq, dim=-1, keepdim=True, memory_config=dram, compute_kernel_config=self.ckc)
        ttnn.deallocate(sq)
        inv = ttnn.rsqrt(ttnn.add(ss, 1e-6, memory_config=dram), memory_config=dram)
        ttnn.deallocate(ss)
        out = ttnn.multiply(x, inv, memory_config=dram)
        for u in (x, inv):
            ttnn.deallocate(u)
        return out

    def _heads(self, t: ttnn.Tensor, T: int) -> ttnn.Tensor:
        """[1, 1, T, 32] (12 heads + zero padding) -> [1, T, 12] fp32."""
        h = ttnn.slice(t, (0, 0, 0, 0), (1, 1, T, self.nv), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t)
        return ttnn.reshape(h, (1, T, self.nv))

    @staticmethod
    def _reorder(t: ttnn.Tensor, runs, dim: int) -> ttnn.Tensor:
        if len(runs) == 1:
            return t
        shape = list(t.shape)
        parts = []
        for lo, hi in runs:
            s, e = [0] * len(shape), list(shape)
            s[dim], e[dim] = lo, hi
            parts.append(ttnn.slice(t, s, e, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        out = ttnn.concat(parts, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p_ in parts:
            ttnn.deallocate(p_)
        ttnn.deallocate(t)
        return out

    def __call__(self, x: ttnn.Tensor, start: int, state: dict):
        """x [1, 1, T / sp, hidden] bf16 (this SP row's rows, replicated over TP); state from state_to_device.
        Returns (out [1, 1, T / sp, hidden] bf16 replicated over TP, new state)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        rows = x.shape[-2]
        T = rows * self.sp
        kd, vd, cd, nv = self.kd, self.vd, self.cd, self.nv

        # 1. projections on the local rows
        # fp32 out: z gates in fp32 (as the reference); the conv input is cast to bf16 (the op's contract)
        qkvz = self._linear(x, self.w_qkvz)
        qkv32 = ttnn.slice(qkvz, (0, 0, 0, 0), (1, 1, rows, cd), memory_config=dram)
        qkv = ttnn.typecast(qkv32, ttnn.bfloat16)
        ttnn.deallocate(qkv32)
        z = ttnn.slice(qkvz, (0, 0, 0, cd), (1, 1, rows, cd + vd), memory_config=dram)
        ttnn.deallocate(qkvz)
        ab = self._linear(x, self.w_ab)
        a = ttnn.slice(ab, (0, 0, 0, 0), (1, 1, rows, 32), memory_config=dram)
        b = ttnn.slice(ab, (0, 0, 0, 32), (1, 1, rows, 64), memory_config=dram)
        ttnn.deallocate(ab)
        sp_a = ttnn.softplus(
            ttnn.add(a, self.dt_bias, memory_config=dram), beta=1.0, threshold=20.0, memory_config=dram
        )
        g = ttnn.multiply(sp_a, self.neg_exp_a, memory_config=dram)
        beta = ttnn.sigmoid(b, memory_config=dram)
        for t in (a, b, sp_a):
            ttnn.deallocate(t)

        # 2. the whole chunk in natural order on every chip of the column
        qkv_all = self._gather_natural(qkv, start, T)
        g_all = self._heads(self._gather_natural(g, start, T), T)
        beta_all = self._heads(self._gather_natural(beta, start, T), T)
        for t in (qkv, g, beta):
            ttnn.deallocate(t)

        # 3. causal conv + silu + q/k/v split on the whole chunk, history = the conv state
        xin = ttnn.to_layout(ttnn.reshape(qkv_all, (1, T, cd)), ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        ttnn.deallocate(qkv_all)
        hist = state["conv"]
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            xin,
            hist,
            *self.taps,
            kd,
            kd,
            vd,
            program_config=self.conv_pc,
            actual_start=self.actual_start,
            predecessor_carry=hist,
            memory_config=dram,
        )
        new_conv = ttnn.slice(xin, (0, T - (self.K - 1), 0), (1, T, cd), memory_config=dram)
        ttnn.deallocate(xin)

        # 4. the gated delta rule over the chunk from the recurrent state
        chunk = settings.get("GDN_CHUNK")
        if chunk != CHUNK_FLAT:
            # rank-4 head-split path: q / k l2-normalised here (fp32), the op casts q / k / v to bf16
            q, k = (self._l2norm(self._split_heads(t, self.nk, self.dk, T)) for t in (q, k))
            v = self._split_heads(v, self.nv, self.dv, T)
        eye, tril, ones, masks = self.consts[chunk]
        o, new_rec = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g_all,
            beta_all,
            scale=self.dk**-0.5,
            initial_state=state["recurrent"],
            output_final_state=True,
            chunk_size=chunk,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )  # o [nv, T, dv] fp32 TILE
        for t in (q, k, v, g_all, beta_all):
            ttnn.deallocate(t)

        # 5. this row's rows, gated norm, out_proj, all_reduce over TP
        o = self._reorder(o, chip_order_runs(start, T, self.sp, to_natural=False), dim=1)
        o = ttnn.reshape(o, (1, nv, T, self.dv))
        mine = ttnn.mesh_partition(o, dim=2, cluster_axis=0, memory_config=dram)  # [1, nv, rows, dv]
        ttnn.deallocate(o)
        n = ttnn.rms_norm(
            mine, epsilon=self.eps, weight=self.norm_w, memory_config=dram, compute_kernel_config=self.ckc
        )
        ttnn.deallocate(mine)
        n = ttnn.experimental.nlp_concat_heads(n, memory_config=dram)  # [1, 1, rows, vd]
        sz = ttnn.silu(z, memory_config=dram)
        ttnn.deallocate(z)
        gated = ttnn.multiply(n, sz, dtype=ttnn.bfloat16, memory_config=dram)
        for t in (n, sz):
            ttnn.deallocate(t)
        part = self._linear(gated, self.w_out)
        ttnn.deallocate(gated)
        red = ttnn.all_reduce(part, cluster_axis=1, memory_config=dram)
        ttnn.deallocate(part)
        out = ttnn.typecast(red, ttnn.bfloat16)
        ttnn.deallocate(red)
        return out, {"conv": new_conv, "recurrent": new_rec}
