# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-element ULP tests for the GELU SFPU variants with BF16 and FP32 inputs.

For each ttnn.GeluVariant the test runs the kernel on a representative input
set (BF16: exhaustive over every finite 16-bit pattern; FP32: sample-based
over [-10, 10] plus targeted edge cases) and compares per-element to the
matching torch.nn.functional.gelu reference with a ULP-based assertion.
"""

import math

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import ulp_distance

pytestmark = pytest.mark.use_module_device


_VARIANTS = {
    "accurate": (ttnn.GeluVariant.Accurate, "none"),
    "fast_lut": (ttnn.GeluVariant.FastLut, "none"),
    "tanh": (ttnn.GeluVariant.Tanh, "tanh"),
}

# Per-(variant, dtype) ULP bound + tolerated-inputs list.
#   tolerated_inputs = specific input values where +1 ULP above the bound is allowed.
#
# Tanh FP32 (400):
#   The kernel uses `0.5 * x * (1 + tanh(scaled))`, which is torch's
#   CPU FP32 implementation. The `(1 + tanh)` step is a `1 - tiny` cancellation
#   that loses ~9-10 bits of FP32 precision: the result inherits the absolute
#   precision of tanh at magnitude 1.0 (~1 FP32 ULP = 6e-8) regardless of how
#   small `(1 + tanh)` actually is. For inputs like x = -3.117 the
#   `(1 + tanh) ~ 1.6e-3`, where the ULP is 1.9e-10, so 6e-8 absolute = ~300
#   FP32 ULPs. Computing via the sigmoid identity `x * sigmoid(2*scaled)`
#   avoids cancellation but diverges from torch by a similar magnitude.
#
# Tanh tiny negative tail:
#   For outputs near zero, tiny absolute differences in the tanh residual
#   (`1 + tanh(scaled)`) can be millions of final-output ULPs. Validate that
#   region with an absolute-error bound and keep strict ULP checks elsewhere.
_THRESHOLDS = {
    ("accurate", "bf16"): (128, []),
    ("accurate", "fp32"): (1_000_000_000, []),
    ("fast_lut", "bf16"): (30_000, []),
    ("fast_lut", "fp32"): (2_000_000_000, []),
    ("tanh", "bf16"): (0, [-4.625, -4.8125]),
    ("tanh", "fp32"): (400, []),
}

_TANH_TAIL_OUTPUT_THRESHOLD = 5.0e-3
_TANH_TAIL_ABS_TOLERANCE = 5.0e-7

# Tile-aligned input grid size. 256x256 = 65536 elements, 8x8 tiles.
_GRID = (256, 256)


def _all_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate test inputs for the given dtype.

    BF16: exhaustively enumerate every finite 16-bit pattern.
    FP32: sample-based — uniform [-10, 10] padded with targeted edge cases.

    Returns (input_tensor, finite_mask). The finite_mask flags NaN/Inf slots.
    """
    if dtype == torch.bfloat16:
        bits = torch.arange(0, 65536, dtype=torch.int32)
        # NaN/Inf: exponent (bits[14:7]) == 0xFF, i.e. bits & 0x7F80 == 0x7F80.
        finite = (bits & 0x7F80) != 0x7F80
        bits_safe = torch.where(finite, bits, torch.zeros_like(bits))
        x = bits_safe.to(torch.int16).contiguous().view(torch.bfloat16).reshape(_GRID)
        return x, finite.reshape(_GRID)

    # FP32: Front-load edge cases (saturation boundary, polynomial-threshold transition, BF16
    # 1-ULP offenders, denormal boundary, ±0, ±1) so they're always covered.
    torch.manual_seed(0)
    edges = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.6,
            -0.6,  # tanh-internal polynomial-vs-sigmoid threshold
            0.75,
            -0.75,
            4.625,
            -4.625,  # BF16 1-ULP offenders
            4.8125,
            -4.8125,
            5.0625,
            -5.0625,  # negative-saturation boundary
            5.15625,
            -5.15625,
            8.66,
            -8.66,  # TANH_SAT_THRESHOLD (in terms of `scaled`)
            10.0,
            -10.0,  # well past saturation
            1e-10,
            -1e-10,  # tiny normal
            1e-30,
            -1e-30,  # very tiny normal
        ],
        dtype=torch.float32,
    )
    n_total = _GRID[0] * _GRID[1]
    bulk = torch.empty(n_total - edges.numel(), dtype=torch.float32).uniform_(-10.0, 10.0)
    x = torch.cat([edges, bulk]).reshape(_GRID)
    return x, torch.ones(_GRID, dtype=torch.bool)


@pytest.mark.parametrize("variant_name", list(_VARIANTS.keys()))
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,tt_dtype",
    [
        ("bf16", torch.bfloat16, ttnn.bfloat16),
        ("fp32", torch.float32, ttnn.float32),
    ],
)
def test_gelu_variant_accuracy(device, variant_name, dtype_name, torch_dtype, tt_dtype):
    """Per-element ULP comparison vs PyTorch CPU reference, BF16 or FP32 inputs."""
    variant_enum, torch_approximate = _VARIANTS[variant_name]
    ulp_threshold, tolerated_inputs = _THRESHOLDS[(variant_name, dtype_name)]

    input_tensor, finite_mask = _all_inputs(torch_dtype)

    # Reference: compute in FP32 from the dtype-quantised input, then round to the
    # input dtype. For both BF16 and FP32 we use torch's FP32 path as the reference.
    expected = torch.nn.functional.gelu(input_tensor.float(), approximate=torch_approximate).to(torch_dtype)

    tt_input = ttnn.from_torch(input_tensor, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.gelu(tt_input, variant=variant_enum)
    actual = ttnn.to_torch(tt_output)

    # Drop NaN/Inf slots (tested separately), but first make sure finite
    # reference outputs did not become non-finite on device.
    finite_reference = finite_mask & ~expected.isnan() & ~expected.isinf()
    bad_actual_nonfinite = finite_reference & (actual.isnan() | actual.isinf())
    assert not bad_actual_nonfinite.any(), (
        f"{variant_name} [{dtype_name}]: device produced NaN/Inf for "
        f"{int(bad_actual_nonfinite.sum().item())} finite reference outputs"
    )
    valid = finite_reference & ~actual.isnan() & ~actual.isinf()
    ulps = ulp_distance(expected, actual)

    # Inputs where |x| < 2^-125 cause the kernel's `0.5 * x` intermediate to
    # land below FP32's smallest normal (2^-126), where Tenstorrent's SFPU
    # FTZs the result to 0. Torch's FP32 path retains the denormal and rounds
    # it back up on the final cast. Excluded from the assertion.
    FTZ_INPUT_THRESHOLD = 2.0**-125  # ~2.35e-38
    ftz_safe = input_tensor.abs() >= FTZ_INPUT_THRESHOLD
    assertable = valid & ftz_safe
    tanh_tiny_tail = (
        assertable
        & (variant_name == "tanh")
        & (input_tensor < 0)
        & (expected.abs().float() < _TANH_TAIL_OUTPUT_THRESHOLD)
    )

    # Log max/mean ULP and the single worst offender for the assertable set.
    assertable_ulps = ulps[assertable]
    max_ulp = int(assertable_ulps.max().item()) if assertable.any() else 0
    mean_ulp = float(assertable_ulps.float().mean().item()) if assertable.any() else 0.0
    logger.info(
        f"gelu variant={variant_name} dtype={dtype_name} approximate={torch_approximate!r}: "
        f"inputs={int(assertable.sum().item())}  max ULP={max_ulp}  mean ULP={mean_ulp:.4f}"
    )
    if max_ulp > 0:
        flat_assertable = assertable.flatten()
        flat_ulps = ulps.flatten()
        worst_i = int((flat_ulps * flat_assertable).argmax().item())
        logger.info(
            f"  worst offender: input={input_tensor.flatten()[worst_i].item():.6g}  "
            f"expected={expected.flatten()[worst_i].item():.6g}  "
            f"actual={actual.flatten()[worst_i].item():.6g}  ulp={int(flat_ulps[worst_i].item())}"
        )

    excluded_count = int((valid & ~ftz_safe).sum().item())
    if excluded_count > 0:
        excluded_max = int(ulps[valid & ~ftz_safe].max().item())
        logger.info(
            f"  [note] {excluded_count} inputs with |x| < 2^-125 excluded from the strict ULP "
            f"assertion (SFPU FTZ on the 0.5*x intermediate). Their max ULP was {excluded_max}."
        )
    tail_count = int(tanh_tiny_tail.sum().item())
    if tail_count > 0:
        tail_absdiff = (actual.float() - expected.float()).abs()[tanh_tiny_tail]
        tail_max_absdiff = float(tail_absdiff.max().item())
        logger.info(
            f"  [note] {tail_count} tanh tiny-tail outputs with |expected| < "
            f"{_TANH_TAIL_OUTPUT_THRESHOLD:g} checked by abs tolerance. "
            f"max absdiff={tail_max_absdiff:.6g}"
        )
        assert tail_max_absdiff <= _TANH_TAIL_ABS_TOLERANCE, (
            f"{variant_name} [{dtype_name}]: tiny-tail max absdiff {tail_max_absdiff} > " f"{_TANH_TAIL_ABS_TOLERANCE}"
        )

    if ulp_threshold is not None:
        # Tolerated inputs get +1 ULP grace.
        is_tolerated = torch.zeros_like(input_tensor, dtype=torch.bool)
        for v in tolerated_inputs:
            is_tolerated = is_tolerated | (input_tensor == torch.tensor(v, dtype=torch_dtype))

        strict_mask = assertable & ~is_tolerated & ~tanh_tiny_tail
        max_strict_ulp = int(ulps[strict_mask].max().item()) if strict_mask.any() else 0
        assert max_strict_ulp <= ulp_threshold, (
            f"{variant_name} [{dtype_name}]: max ULP {max_strict_ulp} > {ulp_threshold} on "
            f"strict-assertable inputs (excluding FTZ band and {len(tolerated_inputs)} tolerated)."
        )

        if is_tolerated.any():
            loose_mask = assertable & is_tolerated & ~tanh_tiny_tail
            max_loose_ulp = int(ulps[loose_mask].max().item()) if loose_mask.any() else 0
            assert max_loose_ulp <= ulp_threshold + 1, (
                f"{variant_name} [{dtype_name}]: max ULP {max_loose_ulp} > {ulp_threshold + 1} "
                f"on tolerated inputs ({tolerated_inputs}). Residue grew beyond the documented 1 ULP."
            )


def test_gelu_legacy_bool_matches_fast_lut(device):
    """fast_and_approximate_mode=True must be bitwise-identical to variant=FastLut
    across the full BF16 input space."""
    input_bf16, _ = _all_inputs(torch.bfloat16)
    tt_input = ttnn.from_torch(input_bf16, layout=ttnn.TILE_LAYOUT, device=device)

    out_variant = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.FastLut))
    out_legacy = ttnn.to_torch(ttnn.gelu(tt_input, fast_and_approximate_mode=True))
    assert torch.equal(out_variant, out_legacy), "Legacy bool path diverged from variant=FastLut"


def test_gelu_default_matches_accurate(device):
    """ttnn.gelu(x) with no kwargs must resolve to variant=Accurate (nanobind
    overload-resolution sanity check)."""
    input_bf16, _ = _all_inputs(torch.bfloat16)
    tt_input = ttnn.from_torch(input_bf16, layout=ttnn.TILE_LAYOUT, device=device)

    out_default = ttnn.to_torch(ttnn.gelu(tt_input))
    out_accurate = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Accurate))
    assert torch.equal(out_default, out_accurate), (
        "ttnn.gelu(x) did not resolve to variant=Accurate — overload resolution may "
        "have picked the legacy bool overload."
    )


def test_tanh_differs_from_accurate(device):
    """variant=Tanh must produce different output from variant=Accurate; if they
    were bitwise-equal, GELU_TANH would be silently dispatching to exact GELU."""
    input_bf16, _ = _all_inputs(torch.bfloat16)
    tt_input = ttnn.from_torch(input_bf16, layout=ttnn.TILE_LAYOUT, device=device)

    out_accurate = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Accurate))
    out_tanh = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Tanh))
    assert not torch.equal(out_accurate, out_tanh), (
        "variant=Tanh produced the same bits as variant=Accurate — GELU_TANH likely "
        "dispatched to the exact-GELU kernel."
    )


@pytest.mark.parametrize("variant_name", list(_VARIANTS.keys()))
@pytest.mark.parametrize(
    "torch_dtype,tt_dtype",
    [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)],
    ids=["bf16", "fp32"],
)
def test_gelu_inf_nan_handling(device, variant_name, torch_dtype, tt_dtype):
    """Sanity-check the kernel doesn't silently produce a finite value for +inf,
    -inf, or NaN inputs. Exact non-finite output varies by variant/dtype: BF16
    NaN encoding may get canonicalised to +inf through SFPU pack/convert, and
    fast_lut returns -inf or NaN for -inf (the LUT doesn't apply saturation)."""
    variant_enum, _ = _VARIANTS[variant_name]
    inputs = torch.zeros((32, 32), dtype=torch_dtype)
    inputs[0, 0] = float("inf")
    inputs[0, 1] = float("-inf")
    inputs[0, 2] = float("nan")
    tt_input = ttnn.from_torch(inputs, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.gelu(tt_input, variant=variant_enum))

    pos_inf, neg_inf, nan_out = out[0, 0].item(), out[0, 1].item(), out[0, 2].item()
    logger.info(f"{variant_name}/{torch_dtype}: gelu(+inf)={pos_inf} gelu(-inf)={neg_inf} gelu(NaN)={nan_out}")

    # gelu(+inf) -> +inf: consistent saturation = identity across all variants/dtypes.
    assert pos_inf == float("inf"), f"{variant_name}: gelu(+inf) -> {pos_inf!r}, expected +inf"
    # gelu(-inf) and gelu(NaN): accept 0 (saturation) or any non-finite output. We just
    # want to flag if the kernel silently produced a usable finite-nonzero value.
    for name, val in [("gelu(-inf)", neg_inf), ("gelu(NaN)", nan_out)]:
        assert val == 0.0 or not math.isfinite(val), f"{variant_name}: {name} -> {val!r}, expected 0 or any non-finite"
