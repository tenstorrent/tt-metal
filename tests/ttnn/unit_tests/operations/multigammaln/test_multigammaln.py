# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for multigammaln — multivariate log-gamma function at order p = 4.

Math under test:
    multigammaln(a) = lgamma(a) + lgamma(a - 0.5) + lgamma(a - 1.0) + lgamma(a - 1.5)
                    + 3 * log(pi)
                   (the (p*(p-1)/4)*log(pi) constant for p = 4)

Matches `torch.special.multigammaln(x, p=4)` exactly. The order is fixed at 4 —
the operation does NOT accept a `p` argument.

Domain: real-valued only for a > 1.5. Outside the domain, the result is NaN
(matching torch). The kernel must NOT branch on input values; NaN must arise
from the math falling through naturally.

This test is the immutable spec for the operation. The implementer must NOT
modify this file. If a parametrized case is impossible under the Phase 0
constraints (fp32, TILE_LAYOUT, 4D, H/W tile-aligned, no compute_kernel_config
parameter), the implementer should fix the kernel rather than relax the test.

Coverage:
- Multiple shapes (single-tile, multi-tile-W, multi-tile-H, non-square,
  multi-batch).
- Both halves of the in-domain region:
  - `a in (1.5, 2.0)` — exercises the reflection branch of lgamma for the
    `lgamma(a - 1.5)` term (argument in (0, 0.5)), which is non-trivial to
    implement correctly.
  - `a > 2.0` — every lgamma argument >= 0.5, plain Stirling path.
- NaN propagation for out-of-domain `a <= 1.5`.
- Negative tests: invalid dtype, invalid layout, invalid rank, non-tile-aligned
  shape.
"""

import math
import pytest
import torch
import ttnn

from ttnn.operations.multigammaln import multigammaln


# Phase-0 tolerances. The op composes four lgammas plus a sum + constant in
# fp32 with HiFi4 + fp32 dest acc. Each lgamma is the Stirling + reflection
# recipe (multi-step compute band). Use the multi-step tolerance from
# .claude/references/op-design-template.md (multi-step compute: 0.05 / 0.2).
RTOL = 0.05
ATOL = 0.2


# -----------------------------------------------------------------------------
# Reference (PyTorch)
# -----------------------------------------------------------------------------


def _torch_reference(a: torch.Tensor) -> torch.Tensor:
    """
    Direct reference for multigammaln at p = 4. We compute in fp32 to match the
    device-side precision policy. We also cross-check against
    `torch.special.multigammaln(a, 4)` — both forms must agree (and they will,
    since this IS the formula).
    """
    a = a.float()
    return (
        torch.lgamma(a)
        + torch.lgamma(a - 0.5)
        + torch.lgamma(a - 1.0)
        + torch.lgamma(a - 1.5)
        + 3.0 * math.log(math.pi)
    )


def _torch_special_reference(a: torch.Tensor) -> torch.Tensor:
    """Convenience: torch's own multigammaln(., 4) — the user-visible contract."""
    return torch.special.multigammaln(a.float(), 4)


# -----------------------------------------------------------------------------
# Positive cases — in-domain inputs
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # Single tile — minimal case.
        pytest.param((1, 1, 32, 32), id="single_tile"),
        # Multi-tile along W and H (separately).
        pytest.param((1, 1, 32, 256), id="multi_tile_W"),
        pytest.param((1, 1, 256, 32), id="multi_tile_H"),
        # Non-square HxW.
        pytest.param((1, 1, 64, 128), id="non_square_64x128"),
        pytest.param((1, 1, 128, 64), id="non_square_128x64"),
        # Multi-batch — exercises work distribution across (N*C) extra tiles.
        pytest.param((2, 4, 64, 128), id="multi_batch"),
    ],
)
def test_multigammaln_correctness_large_domain(device, shape):
    """
    Standard correctness check over a > 2.0 (the easy half of the domain).
    Every lgamma argument is >= 0.5 here, so the kernel's Stirling path runs
    without invoking the reflection correction.
    """
    torch.manual_seed(42)

    # randn -> shift so values are in [~3.0, ~7.0]. Every lgamma argument is
    # comfortably >= 0.5 (a - 1.5 >= 1.5).
    torch_input = torch.randn(shape, dtype=torch.float32).abs() + 3.0

    torch_expected = _torch_reference(torch_input)

    # Sanity: the formula reference must match torch's own multigammaln(., 4).
    torch_alt = _torch_special_reference(torch_input)
    assert torch.allclose(torch_expected, torch_alt, rtol=1e-5, atol=1e-5)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = multigammaln(ttnn_input)

    # Sanity on metadata.
    assert tuple(ttnn_output.shape) == tuple(
        shape
    ), f"Output shape {tuple(ttnn_output.shape)} does not match input shape {shape}"
    assert ttnn_output.dtype == ttnn.float32, f"Output dtype {ttnn_output.dtype} != float32"
    assert ttnn_output.layout == ttnn.TILE_LAYOUT

    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / (expected.abs() + 1e-6)).max().item()
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
        f"Mismatch for shape={shape}:\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
        f"  actual.flat[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flat[:6] = {expected.flatten()[:6].tolist()}"
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="non_square_64x128"),
    ],
)
def test_multigammaln_reflection_branch(device, shape):
    """
    Correctness in the (1.5, 2.0) sub-region of the domain. Here `a - 1.5`
    lies in (0, 0.5), which forces the kernel's lgamma to use the reflection
    correction. This is the hard half of the domain — a kernel that only
    implements the Stirling path (and skips the reflection) will fail this
    test even when the easy-domain test passes.
    """
    torch.manual_seed(42)

    # Random values in (1.5, 2.0) — strictly in-domain but every term needs
    # `a - 1.5` in (0, 0.5), exercising the reflection branch.
    torch_input = 1.5 + 0.5 * torch.rand(shape, dtype=torch.float32)
    # Nudge away from boundaries to keep numerical noise reasonable.
    torch_input = torch_input.clamp(min=1.55, max=1.95)

    torch_expected = _torch_special_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input)

    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    # Both should be finite for these inputs.
    assert torch.isfinite(actual).all(), "Expected finite outputs for a ∈ (1.55, 1.95)"
    assert torch.isfinite(expected).all()

    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / (expected.abs() + 1e-6)).max().item()
    # Reflection adds more numerical noise than plain Stirling — loosen ATOL
    # slightly but stay within the multi-step tolerance band.
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
        f"Reflection-branch mismatch for shape={shape}:\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
        f"  actual.flat[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flat[:6] = {expected.flatten()[:6].tolist()}"
    )


def test_multigammaln_matches_torch_special(device):
    """
    End-to-end contract test: the operation IS `torch.special.multigammaln(x, 4)`.
    This guards against off-by-half-step bugs (e.g., using the wrong constant or
    the wrong number of lgamma terms) that the formula reference might also
    have if accidentally edited.
    """
    torch.manual_seed(42)
    shape = (1, 1, 64, 64)

    # Mix of "easy" and "hard" sub-regions of the domain.
    torch_input = 1.6 + 5.0 * torch.rand(shape, dtype=torch.float32)

    torch_expected = torch.special.multigammaln(torch_input.float(), 4)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    max_abs = (actual - torch_expected).abs().max().item()
    max_rel = ((actual - torch_expected).abs() / (torch_expected.abs() + 1e-6)).max().item()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL), (
        f"Mismatch vs torch.special.multigammaln(x, 4):\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
    )


def test_multigammaln_call_with_keyword_arg(device):
    """The signature must support the `input_tensor=` keyword form."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 32)
    torch_input = 3.0 + torch.rand(shape, dtype=torch.float32)

    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Keyword form (NOT positional).
    ttnn_output = multigammaln(input_tensor=ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------
# Out-of-domain inputs — NaN propagation
# -----------------------------------------------------------------------------


def test_multigammaln_out_of_domain_produces_nan(device):
    """
    For a <= 1.5, torch.special.multigammaln(., 4) returns NaN. The kernel
    must produce NaN in the same positions, without branching on the input
    value. We use a mix of out-of-domain (a in [0.0, 1.5]) and in-domain
    (a in (1.5, 5.0)) values so the NaN pattern is non-trivial and the test
    catches accidental clamping.
    """
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)

    # Half out-of-domain (a in [0, 1.5]), half in-domain (a in (1.5, 5.0)).
    half = torch.rand(shape, dtype=torch.float32) * 1.5  # [0, 1.5]
    other_half = 1.6 + 3.0 * torch.rand(shape, dtype=torch.float32)  # (1.6, 4.6)
    mask = torch.rand(shape) < 0.5
    torch_input = torch.where(mask, half, other_half)

    torch_expected = torch.special.multigammaln(torch_input.float(), 4)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # NaN positions must match torch's.
    nan_actual = torch.isnan(actual)
    nan_expected = torch.isnan(torch_expected)
    assert torch.equal(
        nan_actual, nan_expected
    ), "NaN positions disagree with torch.special.multigammaln (kernel must NOT branch on input — let math fall through to NaN)."

    # Where both are finite, values should match within tolerance.
    finite_mask = (~nan_actual) & (~nan_expected)
    if finite_mask.any():
        actual_f = actual[finite_mask]
        expected_f = torch_expected[finite_mask]
        assert torch.allclose(actual_f, expected_f, rtol=RTOL, atol=ATOL), (
            f"Finite-region mismatch in mixed-domain test:\n"
            f"  max abs diff = {(actual_f - expected_f).abs().max().item():.6f}"
        )


# -----------------------------------------------------------------------------
# Negative cases (validation)
# -----------------------------------------------------------------------------


def _make_tensor(device, shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    torch.manual_seed(42)
    if dtype == ttnn.float32:
        torch_dtype = torch.float32
    elif dtype == ttnn.bfloat16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    # Use in-domain values to avoid masking validation failures with NaN math.
    base = 3.0 + torch.rand(shape, dtype=torch.float32)
    return ttnn.from_torch(
        base.to(torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_multigammaln_rejects_bf16_dtype(device):
    """Phase 0: float32 only. bfloat16 inputs must be rejected."""
    t = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln(t)


def test_multigammaln_rejects_row_major_layout(device):
    """Phase 0 requires TILE_LAYOUT."""
    t = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln(t)


def test_multigammaln_rejects_rank_lt_2(device):
    """Phase 0 requires rank >= 2 (and the design pins rank == 4)."""
    t = _make_tensor(device, (1, 32))
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln(t)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32), id="rank_3"),
        pytest.param((32, 32), id="rank_2"),
        pytest.param((1, 1, 1, 32, 32), id="rank_5"),
    ],
)
def test_multigammaln_rejects_wrong_rank(device, shape):
    """Phase 0 fixes rank at 4 (N, C, H, W). Other ranks must be rejected."""
    t = _make_tensor(device, shape)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln(t)


def test_multigammaln_rejects_non_tile_aligned(device):
    """
    Phase 0 requires H and W divisible by 32. Non-aligned inputs must be
    rejected. (ttnn.from_torch with TILE_LAYOUT will pad internally, so the
    validator must consult the logical shape, not the padded shape.)
    """
    t = _make_tensor(device, (1, 1, 30, 32))
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln(t)

    t = _make_tensor(device, (1, 1, 32, 30))
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln(t)
