# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 shared expert split over each mesh row's chips (TP = the mesh's columns), for the token-split MoE block's
folded shared expert: the block's dispatch already put the row's C S tokens on every chip of the row, this computes
the chip's slice of the clamped SwiGLU MLP on them, and the combine carries each partial row to its token's owner.

    chip (r, c) holds intermediate columns [c I/C, (c + 1) I/C) (the same slice on every row)
    g, u = x @ Wg_c, x @ Wu_c             [C S, I/C] fp32 (one matmul, gate and up side by side)
    h    = silu(min(g, L)) * clamp(u, -L, L)
    o    = h @ Wd_c                        [C S, H] bf16, the partial (summed over the row by the combine)

Weights bf16, HiFi4 + fp32 accumulation, as the model's TtDenseMLP (mlp.py). It runs while the token-split block's
arena holds most of L1: the matmuls' circular buffers are sized to l1_budget (bytes per core), and the block calls it
per owner block of S rows."""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, mm_config
from models.demos.glm53_flash_d_p.tt.mm_configs import TILE_BYTES, _largest_div


def mm2d_fit(device, M, K, N, in0_dtype, w_dtype, out_dtype, budget):
    """A 2D-multicast config whose circular buffers fit budget bytes per core (mm_configs.mm2d with a budget)."""
    grid = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = math.ceil(M / 32), K // 32, N // 32
    pm = math.ceil(Mt / grid.y)
    gy = math.ceil(Mt / pm)
    pn = math.ceil(Nt / grid.x)
    gx = math.ceil(Nt / pn)
    sw = _largest_div(pn, 4)
    sh = _largest_div(pm, max(1, 4 // sw))
    a, b, o = TILE_BYTES[in0_dtype], TILE_BYTES[w_dtype], TILE_BYTES[out_dtype]
    for bw in (8, 4, 2, 1):
        if Kt % bw == 0 and 2 * pm * bw * a + 2 * bw * pn * b + pm * pn * (o + 4096) <= budget:
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=bw,
                out_subblock_h=sh,
                out_subblock_w=sw,
                out_block_h=pm,
                out_block_w=pn,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
    raise AssertionError(f"shared expert: no matmul config for {M} x {K} x {N} within {budget} B of L1")


class TtSharedTpRow:
    def __init__(
        self,
        mesh,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        limit: float = 10.0,
        l1_budget: int = 160 * 1024,
        axis: int = 1,
    ):
        """HF weights: w_gate, w_up [I, H]; w_down [H, I]. axis: the group is each mesh row (1: slice c on chip (r,
        c)) or each column (0: slice r)."""
        self.mesh = mesh
        self.limit = float(limit)
        self.axis = axis
        R, C = tuple(mesh.shape)
        G = C if axis == 1 else R
        inter, hidden = w_gate.shape
        assert inter % (G * 32) == 0, f"intermediate {inter} not tile aligned over {G} chips"
        self.per = per = inter // G
        gt, ut = w_gate.float().T, w_up.float().T  # [H, I]
        gu = torch.cat(
            [torch.cat([gt[:, c * per : (c + 1) * per], ut[:, c * per : (c + 1) * per]], -1) for c in range(G)], -1
        )
        self.w_gu = self._shard(gu.reshape(1, 1, hidden, 2 * inter), -1)
        self.w_down = self._shard(w_down.float().T.reshape(1, 1, inter, hidden), -2)
        self.cfg = hifi4_config()
        self.l1_budget = l1_budget

    def _shard(self, t, dim):
        R, C = tuple(self.mesh.shape)
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh, mesh_shape=(R, C), dims=(None, dim) if self.axis == 1 else (dim, None)
            ),
        )

    def _pc(self, x, w, out_dtype, budget):
        return mm2d_fit(x.device(), x.shape[-2], x.shape[-1], w.shape[-1], x.dtype, w.dtype, out_dtype, budget)

    def __call__(self, x: ttnn.Tensor, l1_budget: int | None = None) -> ttnn.Tensor:
        """x: [1, 1, M, H] bf16 TILE (rows of the row's tokens) -> the chip's partial [1, 1, M, H] bf16 TILE.
        l1_budget: the circular-buffer bytes per core the matmuls may take (default: the constructor's)."""
        budget = l1_budget or self.l1_budget
        mc = ttnn.DRAM_MEMORY_CONFIG
        lim = self.limit
        gu = ttnn.linear(
            x,
            self.w_gu,
            dtype=ttnn.float32,
            program_config=self._pc(x, self.w_gu, ttnn.float32, budget),
            compute_kernel_config=mm_config(self.cfg),
            memory_config=mc,
        )
        s_, per = gu.shape[-2], self.per
        g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, s_, per), memory_config=mc)
        u = ttnn.slice(gu, (0, 0, 0, per), (1, 1, s_, 2 * per), memory_config=mc)
        ttnn.deallocate(gu)
        gc = ttnn.minimum(g, lim, memory_config=mc)
        ttnn.deallocate(g)
        a = ttnn.silu(gc, memory_config=mc)
        ttnn.deallocate(gc)
        uc = ttnn.clamp(u, min=-lim, max=lim, memory_config=mc)
        ttnn.deallocate(u)
        h = ttnn.multiply(a, uc, dtype=ttnn.float32, memory_config=mc)
        ttnn.deallocate(a)
        ttnn.deallocate(uc)
        o = ttnn.linear(
            h,
            self.w_down,
            dtype=ttnn.bfloat16,
            program_config=self._pc(h, self.w_down, ttnn.bfloat16, budget),
            compute_kernel_config=mm_config(self.cfg),
            memory_config=mc,
        )
        ttnn.deallocate(h)
        return o
