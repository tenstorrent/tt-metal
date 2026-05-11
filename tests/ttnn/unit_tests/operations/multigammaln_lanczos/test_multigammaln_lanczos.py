# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for multigammaln_lanczos — multivariate log-gamma at order p = 4,
implemented via the Lanczos 6-term polynomial approximation as a single fused kernel.

Math under test:
    multigammaln_lanczos(a) = L(a) + L(a - 0.5) + L(a - 1.0) + L(a - 1.5) + 3*log(pi)

where L(.) is the Lanczos approximation of lgamma (see op_design.md "Algorithm").

Matches `torch.special.multigammaln(a, p=4)` within Lanczos-at-fp32 precision —
WIDER tolerances than the Stirling+reflection baseline because Lanczos is
intrinsically less accurate (no reflection correction; series degrades near 0).

The order is fixed at 4 — the operation does NOT accept a `p` argument.

This test is the immutable spec. The implementer must NOT modify this file. If
a parametrized case is impossible under the Phase 0 constraints (fp32,
TILE_LAYOUT, 4D, H/W tile-aligned, no compute_kernel_config parameter), the
implementer fixes the kernel rather than relaxing the test.

Coverage:
- Multiple shapes (single-tile, multi-tile-W, multi-tile-H, non-square, multi-batch).
- Safe sub-domain `a in [2.0, 10.0]` for the main correctness sweep — every
  lgamma argument >= 0.5, which is the comfortable region for Lanczos.
- Hard sub-domain `a in (1.55, 1.95)` — `L(a - 1.5)` argument in (0.05, 0.45)
  where Lanczos is least accurate; documented separately with the same loose
  tolerances.
- Pole zeroing at exact integer / half-integer inputs (where the reference
  zero-clamping kicks in).
- NaN propagation for out-of-domain `a <= 1.5`.
- Negative tests: invalid dtype, invalid layout, invalid rank, non-tile-aligned shape.
- Keyword-argument invocation.
"""

import math
import pytest
import torch
import ttnn

from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos


# Lanczos-at-fp32 tolerances (per problem statement: rtol=0.1, atol=0.5).
# Lanczos is meaningfully less accurate than libm-double lgamma; these
# tolerances are WIDER than the Stirling-based multigammaln baseline.
RTOL = 0.1
ATOL = 0.5


# -----------------------------------------------------------------------------
# Reference (PyTorch)
# -----------------------------------------------------------------------------


def _torch_reference(a: torch.Tensor) -> torch.Tensor:
    """
    The exact formula multigammaln_lanczos is approximating, evaluated with
    torch.lgamma (double precision under the hood). Equivalent to
    torch.special.multigammaln(a, 4).
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
# Positive cases — safe sub-domain a in [2.0, 10.0]
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
def test_multigammaln_lanczos_correctness_safe_domain(device, shape):
    """
    Main correctness sweep over a in [2.0, 10.0] — the safe Lanczos region.
    Every lgamma argument is >= 0.5 here, where the Lanczos polynomial agrees
    well with double-precision lgamma.
    """
    torch.manual_seed(42)

    # randn -> abs + 3 places values in [~3.0, ~7.0]; clamp to the spec's
    # suggested safe range [2.0, 10.0] just in case randn produces a large tail.
    torch_input = (torch.randn(shape, dtype=torch.float32).abs() + 3.0).clamp(min=2.0, max=10.0)

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

    ttnn_output = multigammaln_lanczos(ttnn_input)

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


# -----------------------------------------------------------------------------
# Positive cases — hard sub-domain a in (1.55, 1.95)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="non_square_64x128"),
    ],
)
def test_multigammaln_lanczos_correctness_hard_domain(device, shape):
    """
    Correctness in the (1.5, 2.0) sub-region — `L(a - 1.5)` argument lies in
    (0, 0.5), where the Lanczos polynomial is least accurate. The Phase-0
    tolerances are intentionally loose to accept this. A pass here certifies
    the Lanczos translation is faithful (not the absolute precision).
    """
    torch.manual_seed(42)

    # Random values in (1.55, 1.95) — strictly in-domain.
    torch_input = (1.55 + 0.40 * torch.rand(shape, dtype=torch.float32)).clamp(min=1.55, max=1.95)

    torch_expected = _torch_special_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln_lanczos(ttnn_input)

    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    # Both should be finite for these inputs (a > 1.5 strictly).
    assert torch.isfinite(actual).all(), "Expected finite outputs for a in (1.55, 1.95)"
    assert torch.isfinite(expected).all()

    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / (expected.abs() + 1e-6)).max().item()
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
        f"Hard-domain mismatch for shape={shape}:\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
        f"  actual.flat[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flat[:6] = {expected.flatten()[:6].tolist()}"
    )


# -----------------------------------------------------------------------------
# Contract test: matches torch.special.multigammaln(., 4)
# -----------------------------------------------------------------------------


def test_multigammaln_lanczos_matches_torch_special(device):
    """
    End-to-end contract test: the operation IS torch.special.multigammaln(., 4)
    (within Lanczos precision). Guards against off-by-half-step bugs (wrong
    constant, wrong number of lgamma terms).
    """
    torch.manual_seed(42)
    shape = (1, 1, 64, 64)

    # Comfortable Lanczos domain.
    torch_input = (2.0 + 5.0 * torch.rand(shape, dtype=torch.float32)).clamp(min=2.0, max=10.0)

    torch_expected = torch.special.multigammaln(torch_input.float(), 4)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln_lanczos(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    max_abs = (actual - torch_expected).abs().max().item()
    max_rel = ((actual - torch_expected).abs() / (torch_expected.abs() + 1e-6)).max().item()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL), (
        f"Mismatch vs torch.special.multigammaln(x, 4):\n"
        f"  max abs diff = {max_abs:.6f}\n"
        f"  max rel diff = {max_rel:.6f}\n"
    )


# -----------------------------------------------------------------------------
# Pole zeroing — exact integer / half-integer inputs
# -----------------------------------------------------------------------------


def test_multigammaln_lanczos_pole_zeroing_at_exact_integers(device):
    """
    The Lanczos `_lgamma` zeros out the result at the integer poles x == 1 and
    x == 2 (the Lanczos polynomial diverges there; the true lgamma values are
    0 for both, so zeroing is correct). For multigammaln_lanczos, each of the
    four `L(a - off)` terms is zeroed independently at its own integer poles.

    This test inputs exact half-integer / integer values that hit at least one
    pole and verifies the result matches torch (which uses the same convention:
    lgamma(1) = lgamma(2) = 0).

    Inputs picked so that the entire tile's 1024 elements share the same value,
    making the per-element comparison unambiguous.
    """
    shape = (1, 1, 32, 32)

    # a = 2.0 -> L(2)=0, L(1.5), L(1)=0, L(0.5) — L(0.5) and L(1.5) finite via
    # Lanczos (0.5 and 1.5 are in-domain for the Lanczos formula even though
    # the multigammaln domain a > 1.5 means a = 2.0 is at the boundary plus
    # one half-step).
    # a = 2.5 -> L(2.5), L(2)=0, L(1.5), L(1)=0
    # a = 3.0 -> L(3), L(2.5), L(2)=0, L(1.5)
    for a_value in (2.0, 2.5, 3.0):
        torch_input = torch.full(shape, a_value, dtype=torch.float32)
        torch_expected = _torch_special_reference(torch_input)

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_output = multigammaln_lanczos(ttnn_input)
        actual = ttnn.to_torch(ttnn_output).float()

        max_abs = (actual - torch_expected).abs().max().item()
        max_rel = ((actual - torch_expected).abs() / (torch_expected.abs() + 1e-6)).max().item()
        assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL), (
            f"Pole-zero mismatch at a = {a_value}:\n"
            f"  expected = {torch_expected.flatten()[0].item():.6f}\n"
            f"  actual   = {actual.flatten()[0].item():.6f}\n"
            f"  max abs diff = {max_abs:.6f}\n"
            f"  max rel diff = {max_rel:.6f}"
        )


# -----------------------------------------------------------------------------
# Out-of-domain — NaN / non-finite propagation (no input branching allowed)
# -----------------------------------------------------------------------------


def test_multigammaln_lanczos_out_of_domain_non_finite(device):
    """
    Verifies that the Lanczos kernel does NOT branch on input value — the
    out-of-domain (a ≤ 1.5) NaN/Inf behavior must arise naturally from the
    polynomial math falling through (negative `series` → log(neg) = NaN;
    near-pole recip → ±Inf).

    NOTE on torch reference: `torch.special.multigammaln(a, 4)` for a ∈ (0, 1.5]
    actually returns FINITE values (it sums lgamma terms, and `lgamma` is real
    and finite for negative non-integer args). The Lanczos polynomial
    approximation CANNOT reproduce that — by construction it goes NaN/Inf
    whenever any `input + j ≤ 0` for j ∈ {1..6}. So a strict positional
    NaN-equality check with torch is mathematically impossible to satisfy
    here. We instead verify the structural property that proves no-branching:

      1. In-domain (a > 1.5): kernel is finite AND matches torch within tol.
      2. Out-of-domain (a ≤ 1.5): kernel is non-finite for the vast majority
         of positions (Lanczos series diverges → log of negative → NaN). The
         math falls through to NaN with no input check.
    """
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)

    # Half out-of-domain (a in [0.1, 1.5]), half in-domain (a in (2.0, 5.0)).
    # Use 0.1 lower bound to avoid the exact integer poles in the OOD half,
    # which would zero out via the kernel's pole-zero step and might LOOK
    # finite even when the Lanczos polynomial diverges.
    half = 0.1 + 1.4 * torch.rand(shape, dtype=torch.float32)  # [0.1, 1.5]
    other_half = 2.0 + 3.0 * torch.rand(shape, dtype=torch.float32)  # (2.0, 5.0)
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
    ttnn_output = multigammaln_lanczos(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    in_domain = torch_input > 1.5
    ood = ~in_domain

    # 1. In-domain positions must be finite and match torch within Lanczos tol.
    in_domain_actual = actual[in_domain]
    in_domain_expected = torch_expected[in_domain]
    assert torch.isfinite(in_domain_actual).all(), (
        f"Kernel produced non-finite values for in-domain inputs (a > 1.5):\n"
        f"  non-finite count = {(~torch.isfinite(in_domain_actual)).sum().item()}"
    )
    assert torch.allclose(in_domain_actual, in_domain_expected, rtol=RTOL, atol=ATOL), (
        f"In-domain values disagree with torch.special.multigammaln:\n"
        f"  max abs diff = {(in_domain_actual - in_domain_expected).abs().max().item():.6f}"
    )

    # 2. OOD positions: most must be non-finite (Lanczos series diverges
    # naturally — proves no branching on input value).
    if ood.any():
        non_finite_ood_fraction = (~torch.isfinite(actual[ood])).float().mean().item()
        assert non_finite_ood_fraction > 0.5, (
            f"Expected most OOD positions (a ≤ 1.5) to produce non-finite values "
            f"(proving no input branching). Got {non_finite_ood_fraction:.2%} "
            f"non-finite out of {ood.sum().item()} OOD positions."
        )


# -----------------------------------------------------------------------------
# Keyword-argument invocation
# -----------------------------------------------------------------------------


def test_multigammaln_lanczos_call_with_keyword_arg(device):
    """The signature must support the `input_tensor=` keyword form."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 32)
    torch_input = (3.0 + torch.rand(shape, dtype=torch.float32)).clamp(min=2.0, max=10.0)

    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Keyword form (NOT positional).
    ttnn_output = multigammaln_lanczos(input_tensor=ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL)


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


def test_multigammaln_lanczos_rejects_bf16_dtype(device):
    """Phase 0: float32 only. bfloat16 inputs must be rejected."""
    t = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(t)


def test_multigammaln_lanczos_rejects_row_major_layout(device):
    """Phase 0 requires TILE_LAYOUT."""
    t = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(t)


def test_multigammaln_lanczos_rejects_rank_lt_2(device):
    """Phase 0 requires rank >= 2 (and the design pins rank == 4)."""
    t = _make_tensor(device, (1, 32))
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(t)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32), id="rank_3"),
        pytest.param((32, 32), id="rank_2"),
        pytest.param((1, 1, 1, 32, 32), id="rank_5"),
    ],
)
def test_multigammaln_lanczos_rejects_wrong_rank(device, shape):
    """Phase 0 fixes rank at 4 (N, C, H, W). Other ranks must be rejected."""
    t = _make_tensor(device, shape)
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(t)


def test_multigammaln_lanczos_rejects_non_tile_aligned(device):
    """
    Phase 0 requires H and W divisible by 32. Non-aligned inputs must be
    rejected. (ttnn.from_torch with TILE_LAYOUT pads internally, so the
    validator must consult the logical shape, not the padded shape.)
    """
    t = _make_tensor(device, (1, 1, 30, 32))
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(t)

    t = _make_tensor(device, (1, 1, 32, 30))
    with pytest.raises((ValueError, RuntimeError)):
        multigammaln_lanczos(t)
