# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict per-element ULP tests for the GELU SFPU variants, BF16 and FP32 inputs.

For each ttnn.GeluVariant the test runs the kernel on a representative input
set (BF16: exhaustive over every finite 16-bit pattern; FP32: sample-based
over [-10, 10] plus targeted edge cases) and compares per-element to the
matching torch.nn.functional.gelu reference. Output is a cumulative ULP
histogram (>= 1 ULP, >= 2 ULP, ...) plus the top-N worst offenders, with a
strict ULP-bound assertion where the kernel is expected to match precisely.
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


_VARIANTS = {
    "accurate": (ttnn.GeluVariant.Accurate, "none"),
    "fast_lut": (ttnn.GeluVariant.FastLut, "none"),
    "tanh": (ttnn.GeluVariant.Tanh, "tanh"),
}

# Per-(variant, dtype) ULP bound + tolerated-inputs list.
#   ULP bound = None means report-only (no assertion). Use this for variants
#   whose error budget is too wide to assert tightly (FastLut) or when you
#   first want to see what the kernel produces before tightening (FP32).
#   tolerated_inputs = specific input values where +1 ULP above the bound is
#   allowed; documents known FP32-precision-floor residues.
_THRESHOLDS = {
    ("accurate", "bf16"): (4, []),
    ("accurate", "fp32"): (None, []),
    ("fast_lut", "bf16"): (None, []),
    ("fast_lut", "fp32"): (None, []),
    # Tanh BF16: x = -4.625 and x = -4.8125 land at the FP32 rounding boundary
    # in tanh's `2*sigmoid(2x) - 1` chain. The SFPU's exp/reciprocal accumulate
    # sub-FP32-ULP error that flips the rounding direction vs PyTorch's tanh,
    # leaking 1 BF16 ULP. Not fixable without bit-exact emulation of torch's tanh.
    ("tanh", "bf16"): (0, [-4.625, -4.8125]),
    ("tanh", "fp32"): (None, []),
}

# Tile-aligned input grid size. 256x256 = 65536 elements, 8x8 tiles.
_GRID = (256, 256)


def _all_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate test inputs for the given dtype.

    BF16: exhaustively enumerate every finite 16-bit pattern.
    FP32: sample-based — uniform [-10, 10] padded with targeted edge cases.

    Returns (input_tensor, finite_mask). The finite_mask flags NaN/Inf slots
    so they can be excluded from ULP comparison.
    """
    if dtype == torch.bfloat16:
        bits = torch.arange(0, 65536, dtype=torch.int32)
        # NaN/Inf: exponent (bits[14:7]) == 0xFF, i.e. bits & 0x7F80 == 0x7F80.
        finite = (bits & 0x7F80) != 0x7F80
        bits_safe = torch.where(finite, bits, torch.zeros_like(bits))
        x = bits_safe.to(torch.int16).contiguous().view(torch.bfloat16).reshape(_GRID)
        return x, finite.reshape(_GRID)

    # FP32: too many values (2^32) to enumerate, so sample. Front-load edge
    # cases (saturation boundary, polynomial-threshold transition, the BF16
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


def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-element ULP distance for BF16 or FP32 tensors.

    Maps the sign-magnitude bit pattern to a monotonic int (negatives ->
    all_ones - bits, positives -> bits + sign_bit) so float ordering is
    preserved by int subtraction; ULP = |mono(a) - mono(b)|.

    Treats +0 and -0 as identical (0 ULP apart): they're numerically equal
    per IEEE 754, and the SFPU's pack/convert pipeline canonicalises -0 to
    +0 anyway, so the 1-ULP gap the monotonic-int mapping would otherwise
    report is an artefact, not a real numerical difference.
    """
    assert a.dtype == b.dtype, f"dtype mismatch: {a.dtype} vs {b.dtype}"
    if a.dtype == torch.bfloat16:
        bits_dtype, wider_dtype, sign_mask, all_mask = torch.int16, torch.int32, 0x8000, 0xFFFF
    elif a.dtype == torch.float32:
        bits_dtype, wider_dtype, sign_mask, all_mask = torch.int32, torch.int64, 0x80000000, 0xFFFFFFFF
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")

    a_bits = a.contiguous().view(bits_dtype).to(wider_dtype) & all_mask
    b_bits = b.contiguous().view(bits_dtype).to(wider_dtype) & all_mask
    a_mono = torch.where((a_bits & sign_mask) != 0, all_mask - a_bits, a_bits + sign_mask)
    b_mono = torch.where((b_bits & sign_mask) != 0, all_mask - b_bits, b_bits + sign_mask)
    diff = (a_mono - b_mono).abs()
    magnitude_mask = all_mask ^ sign_mask
    both_zero = ((a_bits & magnitude_mask) == 0) & ((b_bits & magnitude_mask) == 0)
    return torch.where(both_zero, torch.zeros_like(diff), diff)


def _cumulative_histogram(ulps_flat: torch.Tensor) -> list[tuple[int, int, float]]:
    buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 8192, 65536, 1 << 20]
    total = ulps_flat.numel()
    rows = []
    for b in buckets:
        n = int((ulps_flat >= b).sum().item())
        rows.append((b, n, (100.0 * n / total) if total else 0.0))
    return rows


def _format_report(
    variant_name: str,
    dtype_name: str,
    torch_approximate: str,
    input_tensor: torch.Tensor,
    expected: torch.Tensor,
    actual: torch.Tensor,
    ulps: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[str, int, float]:
    """Format the per-variant report. Histogram and worst-offenders are scoped
    to `mask` (typically `assertable`). FTZ-excluded inputs are surfaced
    separately by the `[note]` line outside this helper."""
    mask_ulps = ulps[mask]
    max_ulp = int(mask_ulps.max().item()) if mask.any() else 0
    mean_ulp = float(mask_ulps.float().mean().item()) if mask.any() else 0.0

    lines = [
        f"\n=== gelu variant={variant_name} dtype={dtype_name} (torch approximate={torch_approximate!r}) ===",
        f"  inputs evaluated: {int(mask.sum().item())}",
        f"  max ULP: {max_ulp}   mean ULP: {mean_ulp:.4f}",
        f"  cumulative histogram ({dtype_name} ULP distance from torch reference):",
    ]
    for thr, count, pct in _cumulative_histogram(mask_ulps):
        lines.append(f"    >= {thr:8d} ULP : {count:6d}  ({pct:7.4f}%)")

    if mask.any() and mask_ulps.max().item() > 0:
        flat_in = input_tensor.flatten()
        flat_exp = expected.flatten()
        flat_act = actual.flatten()
        flat_ulp = ulps.flatten()
        flat_mask = mask.flatten()
        mask_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
        sub_ulp = flat_ulp[mask_idx]
        k = min(10, mask_idx.numel())
        top_vals, top_local = torch.topk(sub_ulp, k)
        lines.append("  top-10 worst offenders:")
        lines.append(f"    {'input':>14}  {'expected':>14}  {'actual':>14}  {'ulp':>10}")
        for ul, li in zip(top_vals.tolist(), top_local.tolist()):
            i = int(mask_idx[li].item())
            lines.append(
                f"    {flat_in[i].item():>14.6g}  {flat_exp[i].item():>14.6g}  "
                f"{flat_act[i].item():>14.6g}  {int(ul):>10d}"
            )

    return "\n".join(lines), max_ulp, mean_ulp


@pytest.mark.parametrize("variant_name", list(_VARIANTS.keys()))
@pytest.mark.parametrize(
    "dtype_name,torch_dtype,tt_dtype",
    [
        ("bf16", torch.bfloat16, ttnn.bfloat16),
        ("fp32", torch.float32, ttnn.float32),
    ],
)
def test_gelu_variant_accuracy(device, variant_name, dtype_name, torch_dtype, tt_dtype, capsys):
    """Per-element ULP comparison vs PyTorch CPU reference, BF16 or FP32 inputs."""
    variant_enum, torch_approximate = _VARIANTS[variant_name]
    ulp_threshold, tolerated_inputs = _THRESHOLDS[(variant_name, dtype_name)]

    input_tensor, finite_mask = _all_inputs(torch_dtype)

    # Reference: compute in FP64 from the dtype-quantised input, then round
    # to the input dtype. FP64 dodges the 2*sigmoid(2x) - 1 cancellation error
    # that limits torch's CPU FP32 path to ~5 ULPs at magnitude 1 (which
    # magnifies to ~hundreds of ULPs at small (1 + tanh) magnitudes). Our
    # kernel uses the sigmoid-identity formulation to avoid the same
    # cancellation, so an FP64-rounded reference is the right yardstick for
    # how close it gets to true math. For BF16 inputs the difference between
    # FP32-ref and FP64-ref is hidden by BF16 rounding in 99.9%+ of cases.
    expected = torch.nn.functional.gelu(input_tensor.double(), approximate=torch_approximate).to(torch_dtype)

    tt_input = ttnn.from_torch(input_tensor, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.gelu(tt_input, variant=variant_enum)
    actual = ttnn.to_torch(tt_output)

    # Drop NaN/Inf slots: ULP distance between two NaNs is meaningless.
    valid = finite_mask & ~expected.isnan() & ~expected.isinf() & ~actual.isnan() & ~actual.isinf()
    ulps = _ulp_diff(expected, actual)

    # Inputs where |x| < 2^-125 cause the kernel's `0.5 * x` intermediate to
    # land below FP32's smallest normal (2^-126), where Tenstorrent's SFPU
    # FTZs the result to 0. Torch's FP32 path retains the denormal and rounds
    # it back up on the final cast. Excluded from the strict assertion; still
    # visible in the histogram so any regression beyond the known band shows up.
    FTZ_INPUT_THRESHOLD = 2.0**-125  # ~2.35e-38
    ftz_safe = input_tensor.abs() >= FTZ_INPUT_THRESHOLD
    assertable = valid & ftz_safe

    report, max_ulp, _mean_ulp = _format_report(
        variant_name,
        dtype_name,
        torch_approximate,
        input_tensor,
        expected,
        actual,
        ulps,
        mask=assertable,
    )

    with capsys.disabled():
        print(report)
        excluded_count = int((valid & ~ftz_safe).sum().item())
        if excluded_count > 0:
            excluded_max = int(ulps[valid & ~ftz_safe].max().item())
            print(
                f"  [note] {excluded_count} inputs with |x| < 2^-125 excluded from the strict ULP "
                f"assertion (SFPU FTZ on the 0.5*x intermediate). Their max ULP was {excluded_max}."
            )

    if ulp_threshold is not None:
        # Strict assertion: every assertable input that isn't tolerated must
        # be within `ulp_threshold`. Tolerated inputs get +1 ULP grace.
        is_tolerated = torch.zeros_like(input_tensor, dtype=torch.bool)
        for v in tolerated_inputs:
            is_tolerated = is_tolerated | (input_tensor == torch.tensor(v, dtype=torch_dtype))

        strict_mask = assertable & ~is_tolerated
        max_strict_ulp = int(ulps[strict_mask].max().item()) if strict_mask.any() else 0
        assert max_strict_ulp <= ulp_threshold, (
            f"{variant_name} [{dtype_name}]: max ULP {max_strict_ulp} > {ulp_threshold} on "
            f"strict-assertable inputs (excluding FTZ band and {len(tolerated_inputs)} tolerated)."
        )

        if is_tolerated.any():
            loose_mask = assertable & is_tolerated
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
