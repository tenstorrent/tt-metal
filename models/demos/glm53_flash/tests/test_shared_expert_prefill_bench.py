# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0222: the prefill shared expert's two linears (``TtClampedSwiGLU`` through ``DecodeLinear``, i.e. the
default ``ttnn.linear`` dispatch for M > 32) vs explicit 2D-mcast program configs on the full grid, at the 16 K chunk's
slab shapes (one chip of the 8x4 mesh: x [2048, 4096] bf16, w13 [4096, 1024] / w2 [512, 4096] bfp8, the layer's
compute config). Time per call (20 back-to-back calls between synchronizes) and bitwise equality vs the default.

    pytest models/demos/glm53_flash/tests/test_shared_expert_prefill_bench.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.matmul_configs import prefill_linear_config
from models.demos.glm53_flash.tt.mlp import _compute_config

M = int(os.environ.get("GLM53_SE_M", "2048"))
SHAPES = [("w13", 4096, 1024), ("w2", 512, 4096)]
BLOCKS = [int(v) for v in os.environ.get("GLM53_SE_BLOCKS", "2,4,8,16").split(",")]


def _timed(fn, dev, n=20):
    out = fn()
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return out, (time.perf_counter() - t) / n * 1e3


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shared_expert_prefill_bench(mesh_device):
    dev = mesh_device
    grid = dev.compute_with_storage_grid_size()
    cc = _compute_config(dev)
    torch.manual_seed(0)
    for name, K, N in SHAPES:
        x = ttnn.from_torch(
            torch.randn(1, 1, M, K).to(torch.bfloat16) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        w = ttnn.from_torch(
            torch.randn(K, N).to(torch.bfloat16) * 0.02, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev
        )
        ref, t_ref = _timed(lambda: ttnn.linear(x, w, dtype=ttnn.bfloat16, compute_kernel_config=cc), dev)
        ref_h = ttnn.to_torch(ref)
        logger.info(f"[se-bench] {name} [{M}x{K}]x[{K}x{N}] default dispatch: {t_ref:.3f} ms")
        for bw in BLOCKS:
            if (K // 32) % bw:
                continue
            try:
                pc = prefill_linear_config(M, K, N, grid, in0_block_w=bw)
                out, t = _timed(
                    lambda: ttnn.linear(x, w, program_config=pc, dtype=ttnn.bfloat16, compute_kernel_config=cc), dev
                )
            except Exception as ex:  # noqa: BLE001
                logger.info(f"[se-bench] {name} 2D mcast in0_block_w {bw}: failed {ex!r}"[:300])
                continue
            o = ttnn.to_torch(out)
            eq = torch.equal(o, ref_h)
            d = (o.float() - ref_h.float()).abs().max().item()
            g = pc.compute_with_storage_grid_size
            logger.info(
                f"[se-bench] {name} 2D mcast grid {g} in0_block_w {bw} per_core {pc.per_core_M}x{pc.per_core_N}: "
                f"{t:.3f} ms ({t_ref / t:.2f}x) {'BITWISE' if eq else f'DIFF max {d:.3g}'}"
            )
            ttnn.deallocate(out)
        ttnn.deallocate(ref)
        ttnn.deallocate(x)
        ttnn.deallocate(w)
