# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for rms_norm — Refinement 3 (Data-movement co-tune).

NOT a correctness gate (run under --profile; Tracy masks the exit code). Each
test runs the op N times so the profiler CSV carries N device-kernel-duration
rows per shape; take the median of the post-warmup rows.

Target = the latency-bound wide-W / few-tile-row cells the design flags
(W=4096/8192, 1 tile-row, bf16, interleaved). Parametrized over layout + gamma
so the same harness measures every affected kernel path.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

N_ITERS = 12  # discard the first (warm-up); median the rest


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# Wide-W, few-tile-row (most latency-bound on one core).
PERF_SHAPES = [
    (1, 1, 32, 4096),
    (1, 1, 32, 8192),
    (1, 1, 64, 8192),
]


@pytest.mark.parametrize("shape", PERF_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("with_gamma", [False, True], ids=["nog", "wg"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_perf(device, shape, layout, with_gamma, dtype):
    torch.manual_seed(0)
    W = shape[-1]
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    x = torch.randn(shape, dtype=torch_dtype)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tg = None
    if with_gamma:
        g = torch.randn(W, dtype=torch_dtype).reshape(1, 1, 1, W)
        tg = ttnn.from_torch(
            g, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    for _ in range(N_ITERS):
        out = rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=_cfg())
        ttnn.synchronize_device(device)
