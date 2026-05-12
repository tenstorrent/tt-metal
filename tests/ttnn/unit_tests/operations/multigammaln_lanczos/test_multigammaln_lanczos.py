# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for multigammaln_lanczos — multivariate log-gamma at p=4 via Lanczos.

Math under test:
    multigammaln_lanczos(x) ≈ torch.special.multigammaln(x, 4)
                            = lgamma(x) + lgamma(x − 0.5) + lgamma(x − 1.0)
                              + lgamma(x − 1.5) + 3·log(π)

The operation is a single fused TTNN kernel that translates the Lanczos 6-term
lgamma composite into SFPU tile primitives — see ``op_design.md``. The kernel
zeroes out at the integer poles a=1, a=2 within each of the four lgamma
sub-evaluations.

This test is the immutable spec for the operation. The implementer must NOT
modify this file. If a parametrized case is impossible under the Phase 0
constraints (fp32, TILE_LAYOUT, rank ≥ 2, H/W tile-aligned), the implementer
should fix the kernel rather than relax the test.

Coverage:
- Multiple shapes (single-tile, multi-tile, non-square, multi-batch).
- Numerical correctness against ``torch.special.multigammaln(x, 4)`` over the
  safe domain x ∈ [2.0, 10.0] with random inputs.
- Output dtype / shape / layout preservation.
- The pole values x ∈ {2.0} (exact zero behaviour from the pole-zeroing mask
  inside one of the four lgamma sub-evaluations).
- Negative tests: invalid dtype, layout, rank, non-tile-aligned H/W.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos


# Phase-0 tolerances. The Lanczos 6-term polynomial evaluated at fp32 is
# meaningfully less accurate than torch.special.multigammaln (which uses libm
# at fp64). Tolerances are wide and match the task specification.
RTOL = 0.1
ATOL = 0.5


# Restrict the random domain to the regime where the Lanczos 6-term polynomial
# is well-conditioned at fp32 — same as the task spec's "safe domain".
SAFE_LO = 2.0
SAFE_HI = 10.0


def _torch_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference: torch.special.multigammaln at p=4, computed in fp64 for accuracy."""
    return torch.special.multigammaln(x.double(), 4).float()


def _make_safe_input(shape, seed: int = 42) -> torch.Tensor:
    """
    Uniform sample from [SAFE_LO, SAFE_HI]. Using `torch.rand` (not `torch.randn`)
    keeps every element strictly inside the safe domain — the Lanczos polynomial
    has poles below 1.5 which we don't want to hit for the bulk-correctness tests.
    """
    torch.manual_seed(seed)
    u = torch.rand(shape, dtype=torch.float32)
    return SAFE_LO + (SAFE_HI - SAFE_LO) * u


# -----------------------------------------------------------------------------
# Positive cases — bulk correctness over the safe domain
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # Single tile per (n, c) plane — minimal case.
        pytest.param((1, 1, 32, 32), id="single_tile"),
        # Multi-tile along W and H.
        pytest.param((1, 1, 32, 256), id="multi_tile_W"),
        pytest.param((1, 1, 256, 32), id="multi_tile_H"),
        # Non-square HxW.
        pytest.param((1, 1, 64, 128), id="non_square_64x128"),
        pytest.param((1, 1, 128, 64), id="non_square_128x64"),
        # Multi-batch — exercises the tile-level work distribution across NC*Ht*Wt.
        pytest.param((2, 4, 64, 128), id="multi_batch"),
    ],
)
def test_multigammaln_lanczos_correctness(device, shape):
    """
    multigammaln_lanczos matches torch.special.multigammaln(x, 4) within the
    Phase-0 tolerance band over the safe input domain [2.0, 10.0].
    """
    torch_input = _make_safe_input(shape, seed=42)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = multigammaln_lanczos(ttnn_input)

    # Metadata sanity.
    assert tuple(ttnn_output.shape) == tuple(
        shape
    ), f"Output shape {tuple(ttnn_output.shape)} does not match input shape {shape}"
    assert ttnn_output.dtype == ttnn.float32, f"Output dtype {ttnn_output.dtype} != float32"
    assert ttnn_output.layout == ttnn.TILE_LAYOUT

    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    # Numerical match within tolerance.
    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / (expected.abs() + 1e-6)).max().item()
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
        f"Mismatch for shape={shape}:\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
        f"  actual.flat[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flat[:6] = {expected.flatten()[:6].tolist()}"
    )


def test_multigammaln_lanczos_positional_call(device):
    """multigammaln_lanczos(t) — positional argument call style works."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 32)
    torch_input = _make_safe_input(shape)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = multigammaln_lanczos(ttnn_input)  # positional
    actual = ttnn.to_torch(ttnn_output).float()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL)


def test_multigammaln_lanczos_keyword_call(device):
    """multigammaln_lanczos(input_tensor=t) — keyword argument call style works."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 32)
    torch_input = _make_safe_input(shape)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = multigammaln_lanczos(input_tensor=ttnn_input)  # keyword
    actual = ttnn.to_torch(ttnn_output).float()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------
# Pole behaviour — x = 2.0 hits the a=2 pole inside the lgamma(x) sub-evaluation
# and the a=1 pole inside the lgamma(x-1) sub-evaluation. The kernel must zero
# those contributions; the overall result must still match torch.
# -----------------------------------------------------------------------------


def test_multigammaln_lanczos_pole_x_equals_2(device):
    """
    At x = 2.0, two of the four lgamma sub-evaluations sit exactly on integer
    poles (a=2 and a=1, both of which the kernel zero-masks). torch agrees
    (gamma(2) = gamma(1) = 1 → lgamma = 0 contribution from both), so the
    masked kernel still matches the torch reference within tolerance.
    """
    shape = (1, 1, 32, 32)
    torch_input = torch.full(shape, 2.0, dtype=torch.float32)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = multigammaln_lanczos(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # The expected value is finite (the 4-lgamma sum is finite at x=2.0). The
    # kernel must produce a finite value within tolerance of torch.
    assert torch.isfinite(actual).all(), "Pole-masked output contains NaN/Inf"
    max_abs = (actual - torch_expected).abs().max().item()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL), (
        f"Pole-correctness mismatch at x=2.0:\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  actual[0,0,0,0]   = {actual[0,0,0,0].item():.6f}\n"
        f"  expected[0,0,0,0] = {torch_expected[0,0,0,0].item():.6f}"
    )


# -----------------------------------------------------------------------------
# Negative cases (Python-side validation)
# -----------------------------------------------------------------------------


def _make_tensor(device, shape, *, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    """Helper to build on-device tensors with controlled dtype / layout."""
    torch.manual_seed(42)
    torch_dtype = {
        ttnn.float32: torch.float32,
        ttnn.bfloat16: torch.bfloat16,
    }.get(dtype, torch.float32)

    # Use a safe-domain shift for any value-bearing operations downstream; for
    # validation tests the actual values are irrelevant.
    return ttnn.from_torch(
        SAFE_LO + (SAFE_HI - SAFE_LO) * torch.rand(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_multigammaln_lanczos_rejects_bf16_dtype(device):
    """Phase 0: float32 only. bfloat16 inputs must be rejected."""
    bf16_input = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(bf16_input)


def test_multigammaln_lanczos_rejects_row_major_layout(device):
    """TILE_LAYOUT only. ROW_MAJOR_LAYOUT must be rejected."""
    rm_input = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(rm_input)


def test_multigammaln_lanczos_rejects_rank_lt_2(device):
    """rank < 2 must be rejected. 1-D tensors don't have an inner tile."""
    # rank-1 input — note: ttnn.from_torch on a 1-D tensor with TILE_LAYOUT
    # requires padding; use a rank-1 shape that's tile-friendly to allow the
    # validator to inspect the shape rather than failing during construction.
    torch.manual_seed(42)
    try:
        bad_input = ttnn.from_torch(
            torch.rand((32,), dtype=torch.float32) + SAFE_LO,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except (ValueError, RuntimeError):
        # If ttnn.from_torch itself rejects a rank-1 TILE_LAYOUT tensor, that
        # is also acceptable — the precondition is enforced *somewhere*.
        return
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(bad_input)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 30, 32), id="H_not_tile_aligned"),
        pytest.param((1, 1, 32, 30), id="W_not_tile_aligned"),
    ],
)
def test_multigammaln_lanczos_rejects_non_tile_aligned(device, shape):
    """
    Phase 0 requires H and W divisible by 32. Non-aligned inputs must be
    rejected. Note: ttnn.from_torch with TILE_LAYOUT pads internally; the
    validator must still flag the underlying logical shape.
    """
    bad_input = _make_tensor(device, shape)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(bad_input)
