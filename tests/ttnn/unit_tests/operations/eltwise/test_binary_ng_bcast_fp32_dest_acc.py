# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue 43196.

The next-gen binary broadcast factory previously enabled fp32 dest accumulation
only when (a) the output dtype was 32-bit, or (b) both inputs were the same
32-bit dtype. The case `bf16 small × fp32 large → bf16` fell through, leaving
fp32 tiles loaded into a DST register configured for bf16. This produced a
timing-sensitive, tile-aligned corruption in broadcast multiply that watcher /
slow-dispatch happened to mask.

The fix enables fp32 dest accumulation whenever any input or output is fp32.
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


N_ITERS = 20


def _run_bcast_mul_iters(device, a_dtype, b_dtype, T, C):
    """Run `relu(beta) * x_up` (broadcast) N_ITERS times and return per-iter
    max absolute deviation from the expected uniform output of 1.0.

    `relu` is used as the producer so the small operand comes from a kernel
    program (not from_torch's DMA path); the bug only fires under that pattern.
    """
    beta = ttnn.from_torch(torch.ones(C) + 1e-6, dtype=a_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x_torch = torch.ones(1, 1, T, C, dtype=torch.float32)

    deviations = []
    for _ in range(N_ITERS):
        x_up = ttnn.from_torch(
            x_torch, dtype=b_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        small = ttnn.relu(beta)
        out = ttnn.multiply(small, x_up)
        flat = ttnn.to_torch(out).flatten().float()
        deviations.append((flat - 1.0).abs().max().item())
    return deviations


@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat16),
    ],
    ids=["a_bf16_b_fp32", "a_fp32_b_bf16", "a_fp32_b_fp32", "a_bf16_b_bf16"],
)
def test_bcast_multiply_dtype_combinations(device, a_dtype, b_dtype):
    """Bug 43196: `bf16 small × fp32 large` raced; other dtype combos were always
    clean. After the fix all four cases must be deterministic and exactly 1.0."""
    deviations = _run_bcast_mul_iters(device, a_dtype, b_dtype, T=8192, C=96)

    max_dev = max(deviations)
    assert max_dev < 1e-3, (
        f"a={a_dtype} b={b_dtype}: max deviation across {N_ITERS} iters = {max_dev}; "
        f"expected uniform output of 1.0. Per-iter deviations: {deviations}"
    )


@pytest.mark.parametrize(
    "T, C",
    [
        (8192, 96),
        (2048, 32),
        (512, 128),
    ],
)
def test_bcast_multiply_bf16_x_fp32_sizes(device, T, C):
    """Bug 43196: explicit coverage of the racing combination across sizes."""
    deviations = _run_bcast_mul_iters(device, ttnn.bfloat16, ttnn.float32, T=T, C=C)

    max_dev = max(deviations)
    assert max_dev < 1e-3, (
        f"T={T} C={C}: max deviation across {N_ITERS} iters = {max_dev}; "
        f"expected uniform output of 1.0. Per-iter deviations: {deviations}"
    )
