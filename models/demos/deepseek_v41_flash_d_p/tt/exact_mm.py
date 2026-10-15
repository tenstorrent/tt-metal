# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Exact fp32 matmul of fp8-grid operands on the FPU (DS41-16).

Blackhole's FPU rounds every in-tile partial dot product (the sum of the 32 products one ``matmul_tiles`` adds into
an output element) to about TF32 (11 significant bits) before it adds it to DEST: 1 + 2^-12 in one tile -> 1.0,
1 + 2^-11 -> 1 + 2^-10, at every fidelity, with fp32 DEST and bf16 operands. The sum across K tiles in fp32 DEST
is exact fp32. A plain ``ttnn.linear`` of q(x) [M, K] by an fp8 weight is therefore off by rel ~1.8e-4 (torch
fp32: 4e-8), and an fp8 quantizer after it (the window KV's act_quant, wq_b's input quant) flips ~0.8 % of its
values (DS41-1).

With one nonzero product per tile-dot the TF32 rounding is exact (an e4m3 x e4m3 product has 8 significant bits),
so: the 32 masked copies x_u = x * [k % 32 == u] (u = 0..31) go through one batched matmul (each tile of x_u holds one
column per row), and the 32 fp32 partials are added on the SFPU (ttnn.add, IEEE fp32). Result: an fp32 sum of exact
products, as torch's fp32 GEMM (only the summation order differs: rel 5e-8 vs fp64 on [1024, 2560] x [2560, 1792]).

Cost: 32x the matmul work, a [32, M, K] bf16 stack and a [32, M, N] fp32 partial (~10 ms vs 0.4 ms for [1024, 2560]
x [2560, 1792] on one p300c chip). tt/settings.py ATTN_IN_MM = "fpu" keeps the plain matmul (a perf pick).

Operands: x bf16 TILE [1, 1, M, K], w bf16 TILE [1, 1, K, N] whose products a * b fit 11 significant bits (fp8
e4m3 x 2^k by fp8 e4m3 x 2^k: 8 bits); M, K, N multiples of 32. No host work in __call__: the mask is built at load,
the program config once per shape.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM

LANES = 32  # products per in-tile dot (tile width)


def _divisor_at_most(n: int, cap: int) -> int:
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


class ExactMatmul:
    """``__call__(x, w)``: x [1, 1, M, K] bf16 TILE @ w [1, 1, K, N] bf16 TILE -> [1, 1, M, N] fp32 TILE, exact
    products summed in fp32. ``k`` is the (per-chip) K. Does not consume its inputs."""

    def __init__(self, mesh, k: int, compute_config, groups: int = 2):
        assert k % LANES == 0, k
        assert LANES % groups == 0 and (LANES // groups) & (LANES // groups - 1) == 0, groups  # pairwise tree
        self.mesh, self.k, self.ckc, self.groups = mesh, k, compute_config, groups
        self.batch = LANES // groups
        lane = torch.arange(k) % LANES
        self.masks = []
        for g in range(groups):
            m = torch.zeros(1, self.batch, 1, k)
            for b in range(self.batch):
                m[0, b, 0] = (lane == g * self.batch + b).float()
            self.masks.append(
                ttnn.from_torch(
                    m,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            )
        self._pc = {}

    def _program_config(self, rows: int, n: int):
        key = (rows, n)
        if key not in self._pc:
            grid = self.mesh.compute_with_storage_grid_size()
            mt, nt, kt = rows // 32, -(-n // 32), self.k // 32
            pcn = -(-nt // min(grid.x, nt))
            gx = -(-nt // pcn)
            pcm = -(-mt // min(grid.y, mt))
            obh = min(8, pcm)
            pcm += -pcm % obh
            gy = -(-mt // pcm)
            sw = _divisor_at_most(pcn, 4)
            sh = _divisor_at_most(obh, max(1, 4 // sw))  # fp32 DEST: at most 4 tiles per subblock
            self._pc[key] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=_divisor_at_most(kt, 8),
                out_subblock_h=sh,
                out_subblock_w=sw,
                out_block_h=obh,
                out_block_w=pcn,
                per_core_M=pcm,
                per_core_N=pcn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
        return self._pc[key]

    def __call__(self, x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        m, k = x.shape[-2], x.shape[-1]
        n = w.shape[-1]
        assert k == self.k and m % 32 == 0, (x.shape, self.k)
        b = self.batch
        pc = self._program_config(b * m, n)
        acc = None
        for mask in self.masks:  # groups: [b, M, N] partials, added without slicing
            st = ttnn.multiply(x, mask, memory_config=DRAM)  # [1, b, M, K]: exact (x 0 / 1)
            st2 = ttnn.reshape(st, [1, 1, b * m, k])
            part = ttnn.matmul(
                st2, w, dtype=ttnn.float32, compute_kernel_config=self.ckc, program_config=pc, memory_config=DRAM
            )
            ttnn.deallocate(st)
            part = ttnn.reshape(part, [1, b, m, n])
            if acc is None:
                acc = part
            else:
                nxt = ttnn.add(acc, part, dtype=ttnn.float32, memory_config=DRAM)
                ttnn.deallocate(acc)
                ttnn.deallocate(part)
                acc = nxt
        while b > 1:  # pairwise tree over the batch, fp32 SFPU adds
            h = b // 2
            lo = ttnn.slice(acc, [0, 0, 0, 0], [1, h, m, n], memory_config=DRAM)
            hi = ttnn.slice(acc, [0, h, 0, 0], [1, b, m, n], memory_config=DRAM)
            ttnn.deallocate(acc)
            acc = ttnn.add(lo, hi, dtype=ttnn.float32, memory_config=DRAM)
            ttnn.deallocate(lo)
            ttnn.deallocate(hi)
            b = h
        return acc
