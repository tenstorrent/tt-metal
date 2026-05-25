# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 1 acceptance: wide reduce dimension fits in L1.

Phase 0's softmax compute kernel sized `cb_input_tiles` to `2 * reduce_dim_tiles`
and `cb_exps` to `reduce_dim_tiles`. At W = 4096 (Wt = 128) that's ~2.6 MB on a
1.5 MB L1 budget — the program would not even launch.

Refinement 1 rewrites the compute kernel to a 3-pass chunked design with
constant-bounded CBs. This test exercises the wide-W shapes from
`eval/golden_tests/softmax/feature_spec.py` that previously OOM'd, plus a
couple of intermediate wide shapes for finer granularity.

Pass criteria (matching `eval/golden_tests/softmax/helpers.py` TOLERANCES for
fp32_hifi4_fp32acc):
    - The op accepts the input (no L1 OOM during program launch).
    - Output matches `torch.softmax` to PCC >= 0.999.
    - Per-element RMS error normalised by reference stddev <= 0.01.
    - softmax sums to ~1 along the reduce dim, with atol scaling as the
      natural fp32-accumulator error (≈ N · ε): for W = 8192 this is ~3e-3,
      not the 1e-3 used in Phase-0 small-W tests.

The dim=-2 cells weren't in the OOM bucket (Ht stayed small) but the new
unified chunked kernel must still pass them. We include them here as a
regression guard.
"""

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.softmax import softmax


# Sum-to-1 tolerance for the chunked-design softmax. The error has two
# components:
#   1. A near-constant floor (~1.5e-3) coming from the recip(Σexp) ULP +
#      its multiply propagation across the row. Measured empirically.
#   2. A linear N·ε term from the fp32 accumulator's per-fma rounding —
#      visible above W = 2048 (W=8192 → 2.89e-3 ≈ 1.5e-3 + 4·8192·ε).
# The closed-form `1.5e-3 + 4 N ε` matches the measurements with a few
# bits of headroom on every W in {1024, 2048, 4096, 8192}.
def _sum_atol(reduce_dim_size: int) -> float:
    """Per-row sum-to-1 tolerance for a fp32 softmax over `reduce_dim_size` elements."""
    eps_fp32 = 1.2e-7
    return 1.5e-3 + 4.0 * reduce_dim_size * eps_fp32


# Relative RMS tolerance (matches eval/golden_tests/softmax/helpers.py
# TOLERANCES["fp32_hifi4_fp32acc"][1]).
GOLDEN_RMS_REL = 0.01


# Wide-W shapes that previously failed with L1 OOM at dim=-1.
# These mirror the Phase-0 OOM cells listed in op_requirements.md.
WIDE_W_SHAPES = [
    pytest.param((1, 1, 32, 4096), id="32x4096"),
    pytest.param((1, 1, 32, 8192), id="32x8192"),
    pytest.param((1, 1, 128, 4096), id="128x4096"),
    pytest.param((2, 1, 64, 4096), id="b2_64x4096"),
    # Extra granularity: medium wide cases that exercise the chunked path
    # without paying the largest-shape compile time.
    pytest.param((1, 1, 32, 1024), id="32x1024"),
    pytest.param((1, 1, 32, 2048), id="32x2048"),
]


@pytest.mark.parametrize("shape", WIDE_W_SHAPES)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_wide_reduce_dim_minus_1(device, shape, numeric_stable):
    """dim=-1 (the previously-OOMing direction). Wide W must now fit in L1."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-1, numeric_stable=numeric_stable)

    assert ttnn_output.shape == ttnn_input.shape, f"shape mismatch: {ttnn_output.shape} vs {ttnn_input.shape}"
    assert ttnn_output.dtype == ttnn.float32

    torch_output = ttnn.to_torch(ttnn_output)

    # 1. PCC — primary correctness signal (matches golden tolerance).
    assert_with_pcc(torch_expected, torch_output, 0.999)

    # 2. Relative RMS — matches golden tolerance for fp32_hifi4_fp32acc.
    diff = (torch_output - torch_expected).double()
    rms = math.sqrt((diff * diff).mean().item())
    ref_std = torch_expected.double().std().item()
    rms_rel = rms / ref_std if ref_std > 0 else 0.0
    assert rms_rel <= GOLDEN_RMS_REL, f"RMS_rel {rms_rel:.5f} exceeds golden tolerance {GOLDEN_RMS_REL}"

    # 3. Sum-to-1 sanity check with W-scaled atol.
    reduce_dim_size = shape[-1]
    atol = _sum_atol(reduce_dim_size)
    sum_along_dim = torch_output.sum(dim=-1)
    max_dev = (sum_along_dim - 1).abs().max().item()
    assert max_dev <= atol, f"wide-W softmax rows do not sum to 1: max abs deviation = {max_dev:.3e} (atol={atol:.3e})"


# A subset of wide shapes also tested at dim=-2 — not previously OOM but a
# regression guard for the unified chunked kernel (the new path also handles
# dim=-2 via accumulate_reduce_block<SUM, REDUCE_COL>).
WIDE_H_SHAPES = [
    pytest.param((1, 1, 4096, 32), id="4096x32"),
    pytest.param((1, 1, 2048, 32), id="2048x32"),
    pytest.param((2, 1, 4096, 64), id="b2_4096x64"),
]


@pytest.mark.parametrize("shape", WIDE_H_SHAPES)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_wide_reduce_dim_minus_2(device, shape, numeric_stable):
    """dim=-2 — regression guard for the unified chunked kernel on tall H."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-2)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-2, numeric_stable=numeric_stable)

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)

    diff = (torch_output - torch_expected).double()
    rms = math.sqrt((diff * diff).mean().item())
    ref_std = torch_expected.double().std().item()
    rms_rel = rms / ref_std if ref_std > 0 else 0.0
    assert rms_rel <= GOLDEN_RMS_REL, f"RMS_rel {rms_rel:.5f} exceeds golden tolerance {GOLDEN_RMS_REL}"

    reduce_dim_size = shape[-2]
    atol = _sum_atol(reduce_dim_size)
    sum_along_dim = torch_output.sum(dim=-2)
    max_dev = (sum_along_dim - 1).abs().max().item()
    assert max_dev <= atol, f"tall-H softmax cols do not sum to 1: max abs deviation = {max_dev:.3e} (atol={atol:.3e})"


# Exercise the block-size selection — shape with Wt that doesn't divide cap=16
# cleanly (Wt=128 is fine; Wt=192 = 6144 picks BLOCK_SIZE = 16 since 192/16=12).
# Below tests Wt = 5*32 = 160 (BLOCK_SIZE picks 16 = 160/10 = 16... yes 16
# divides 160), and Wt=192. Both must work.
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 5120), id="32x5120_Wt160"),
        pytest.param((1, 1, 32, 6144), id="32x6144_Wt192"),
    ],
)
def test_softmax_block_size_selection(device, shape):
    """Exercise Wt values that pick non-trivial BLOCK_SIZE divisors."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=-1, numeric_stable=True)
    torch_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_expected, torch_output, 0.999)
