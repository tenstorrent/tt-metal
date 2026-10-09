# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Explicit matmul program configs (GLM_MM_CONFIGS=0: ttnn's auto configs).

2D multicast for [M, K] x [K, N] (M = rows per chip): M tiles over the grid's rows, N tiles over its columns
(ceil(Nt / grid.x) per core), the widest in0 block that fits L1, the largest out subblock within the 4-tile fp32 DEST.
The rule of MiMo's mm_configs.mm_2d_config (branch mstaletovic/mimo-v2-dp); on the model's shapes it is what the sweep
(tests/test_matmul_tune.py, one chip, chunk 5120) measured best: MLA o_proj 0.763 -> 0.488 ms, router 0.328 -> 0.203,
shared expert gate+up (one N=512 matmul) 2 x 0.297 -> 0.261, MLA kv_a 0.339 -> 0.230, q_a 0.137 -> 0.076, KDA o_proj
0.504 -> 0.450; numerics identical to the auto config (same kernel family). KDA_IN_MINIMAL: minimal_matmul blocking for
the KDA input projection (1.511 -> 1.260 ms, 91% of the HiFi4 peak; bf16 output, same error as ttnn.linear)."""

from __future__ import annotations

import math
import os

import ttnn

TILE = 32
L1_BUDGET = (
    1_200_000  # per-core CB bytes; the model keeps other L1 buffers (1.4 MB clashed at 1.46 MB in the full model)
)
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
ENABLED = os.environ.get("GLM_MM_CONFIGS", "1") != "0"
_cache: dict = {}


def _largest_div(n: int, cap: int) -> int:
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def mm2d(device, M: int, K: int, N: int, in0_dtype, w_dtype, out_dtype):
    """The 2D-multicast config for one shape, or None (ttnn auto) when disabled or nothing fits."""
    if not ENABLED:
        return None
    key = (M, K, N, in0_dtype, w_dtype, out_dtype)
    if key in _cache:
        return _cache[key]
    grid = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = math.ceil(M / TILE), K // TILE, N // TILE
    pm = math.ceil(Mt / grid.y)
    gy = math.ceil(Mt / pm)
    pn = math.ceil(Nt / grid.x)
    gx = math.ceil(Nt / pn)
    sw = _largest_div(pn, 4)
    sh = _largest_div(pm, max(1, 4 // sw))
    a, b, o = TILE_BYTES[in0_dtype], TILE_BYTES[w_dtype], TILE_BYTES[out_dtype]
    pc = None
    for bw in (16, 8, 4, 2, 1):
        if Kt % bw:
            continue
        l1 = 2 * pm * bw * a + 2 * bw * pn * b + pm * pn * (o + 4096)  # in0 + in1 double-buffered, out + fp32 interm
        if l1 <= L1_BUDGET:
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
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
            break
    _cache[key] = pc
    return pc


def linear_config(x, w, out_dtype):
    """mm2d for ttnn.linear(x, w): M = the product of x's leading dims."""
    M = 1
    for d in tuple(x.shape)[:-1]:
        M *= d
    return mm2d(x.device(), M, x.shape[-1], w.shape[-1], x.dtype, w.dtype, out_dtype)


def minimal_config(device, M_block: int, K_block: int, N_block: int, sub_h: int, sub_w: int):
    if not ENABLED:
        return None
    return ttnn.MinimalMatmulConfig(
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )


def bmm(device, B: int, M: int, K: int, N: int, l1_budget: int = L1_BUDGET):
    """Batched matmul with one weight per batch ([B, M, K] x [B, K, N], MLA per-head absorb), or None (auto).
    MatmulMultiCoreReuseProgramConfig is only correct when every core gets at most ONE output block: with more blocks
    than cores the result is garbage (tests/test_mla_bmm_repro.py::test_mla_bmm_blocks: rel 5..25 vs fp32 for every
    per_core_M with B * Mt / per_core_M > cores, 4.6e-3 at 64 blocks / 110 cores; it only looked right after a
    same-shape auto run left the right data in L1). So: per_core_N = all N tiles, per_core_M the smallest divisor of
    the per-batch M tiles with B * Mt / per_core_M <= cores, the widest in0 block that fits L1; else None.
    o w_uv (64 x 640 x 512 x 256): per_core_M 20, 0.905 -> ~0.25 ms; q w_uk has no fitting config (auto)."""
    if not ENABLED:
        return None
    key = ("bmm", B, M, K, N, l1_budget)
    if key in _cache:
        return _cache[key]
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    sw = _largest_div(Nt, 4)
    pc = None
    pms = [d for d in range(1, Mt + 1) if Mt % d == 0 and B * Mt // d <= cores]
    if pms:
        pm = pms[0]
        sh = _largest_div(pm, max(1, 4 // sw))
        for bw in (4, 2, 1):
            if Kt % bw == 0 and 2 * pm * bw * 2048 + 2 * bw * Nt * 2048 + pm * Nt * (2048 + 4096) <= l1_budget:
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=bw,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=pm,
                    per_core_N=Nt,
                )
                break
    _cache[key] = pc
    return pc
