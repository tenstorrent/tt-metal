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

from models.common.utility_functions import run_for_blackhole

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


# ─── BH: BF16→FP32 + COL bcast hang ────────────────────────────────────────
# Regression coverage for a BH-specific LLK `unary_bcast` hang: COL bcast with
# any BFLOAT16 inputs and fp32_dest_acc_en hangs the post-op Synchronize.
# Fix lives in `is_llk_bcast`, gated to BH + COL (the only configuration tested).

TILE_W = 32

_OPS = {
    "subtract": (ttnn.subtract, torch.subtract),
    "add": (ttnn.add, torch.add),
    "multiply": (ttnn.multiply, torch.multiply),
}


def _bf16_round(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 precision and back to float32; matches ttnn storage."""
    return x.to(torch.bfloat16).to(torch.float32)


def _run_subtract_fp32_col_b(device, w_tiles, b=5, s=256):
    """Canonical bf16 − bf16 → fp32 + COL_B-bcast on ones inputs; expects zeros."""
    v = w_tiles * TILE_W
    lhs = ttnn.from_torch(torch.ones(b, 1, s, v), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rhs = ttnn.from_torch(torch.ones(b, 1, s, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.subtract(lhs, rhs, dtype=ttnn.float32)
    assert out.dtype == ttnn.float32
    return ttnn.to_torch(out)


@run_for_blackhole("BH-only LLK unary_bcast + fp32_dest_acc regression;")
@pytest.mark.timeout(20)
@pytest.mark.parametrize("w_tiles", [1, 2, 3, 4, 5, 6, 7, 8], ids=lambda w: f"W_{w}")
def test_subtract_bf16_to_fp32_col_b_w_sweep(device, w_tiles):
    """LLK #1338: full W sweep of the canonical hang case."""
    out = _run_subtract_fp32_col_b(device, w_tiles)
    assert torch.equal(out, torch.zeros_like(out)), f"W={w_tiles}: max abs = {out.abs().max()}"


def _run_bcast_subword_in_fp32_out(device, ttnn_op, torch_op, bcast_input, w_tiles, b=5, s=256):
    """Generic BF16→FP32 col-bcast op runner. bcast_input ∈ {'a','b'}."""
    v = w_tiles * TILE_W
    if bcast_input == "a":
        lhs_torch = torch.full((b, 1, s, 1), 0.5, dtype=torch.float32)
        rhs_torch = torch.full((b, 1, s, v), 1.0, dtype=torch.float32)
    else:
        lhs_torch = torch.full((b, 1, s, v), 1.0, dtype=torch.float32)
        rhs_torch = torch.full((b, 1, s, 1), 0.5, dtype=torch.float32)
    ref = torch_op(_bf16_round(lhs_torch), _bf16_round(rhs_torch)).expand(b, 1, s, v).contiguous()

    lhs = ttnn.from_torch(lhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rhs = ttnn.from_torch(rhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn_op(lhs, rhs, dtype=ttnn.float32)
    assert out.dtype == ttnn.float32
    return ttnn.to_torch(out), ref


# Matrix covers subtract+COL_A and add/multiply for both COL_A and COL_B.
# (subtract+COL_B is exercised separately by test_subtract_bf16_to_fp32_col_b_w_sweep.)
_MATRIX_COMBOS = [
    ("subtract", "a"),
    ("add", "a"),
    ("add", "b"),
    ("multiply", "a"),
    ("multiply", "b"),
]


@run_for_blackhole("BH-only LLK unary_bcast + fp32_dest_acc regression")
@pytest.mark.timeout(20)
@pytest.mark.parametrize("w_tiles", [1, 2, 3, 4, 5, 6, 7, 8], ids=lambda w: f"W_{w}")
@pytest.mark.parametrize(
    "op_name, bcast_input", _MATRIX_COMBOS, ids=[f"{op}_COL_{d.upper()}" for op, d in _MATRIX_COMBOS]
)
def test_binary_ng_bcast_bf16_in_fp32_out_matrix(device, op_name, bcast_input, w_tiles):
    """LLK #1338: (op × bcast direction × W) matrix for BF16→FP32 col-bcast."""
    ttnn_op, torch_op = _OPS[op_name]
    out, ref = _run_bcast_subword_in_fp32_out(device, ttnn_op, torch_op, bcast_input, w_tiles)
    assert torch.equal(
        out, ref
    ), f"{op_name} COL_{bcast_input.upper()} W={w_tiles}: max abs diff = {(out - ref).abs().max()}"
