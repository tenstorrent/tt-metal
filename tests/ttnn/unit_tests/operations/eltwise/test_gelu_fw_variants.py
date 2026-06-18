# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict per-element BF16 ULP tests for the GELU SFPU variants.

For each ttnn.GeluVariant we exhaustively enumerate every finite BF16 input
(2^16 bit patterns minus NaN/Inf), run the kernel, and compare to the
matching torch.nn.functional.gelu reference. Output is a cumulative ULP
histogram (>=1 ULP, >=2 ULP, ...) plus the top-N worst offenders, with a
strict ULP-bound assertion for variants that should match BF16 precision
(Accurate, Tanh). FastLut is informational-only — its 6-segment piecewise
LUT inherently diverges from exact GELU by ~1% (128+ ULPs near transition).
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


# variant_name: (enum, torch `approximate=` flag, ULP bound — None means report-only).
_VARIANTS = {
    "accurate": (ttnn.GeluVariant.Accurate, "none", 4),
    "fast_lut": (ttnn.GeluVariant.FastLut, "none", None),
    "tanh": (ttnn.GeluVariant.Tanh, "tanh", 4),
}


def _all_bf16_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Every finite BF16 bit pattern arranged as (256, 256). Non-finite slots
    (NaN/Inf) are filled with +0 to keep the shape tile-aligned and masked
    out via the returned finite_mask."""
    bits = torch.arange(0, 65536, dtype=torch.int32)
    # NaN/Inf: exponent (bits[14:7]) == 0xFF, i.e. bits & 0x7F80 == 0x7F80.
    finite = (bits & 0x7F80) != 0x7F80
    bits_safe = torch.where(finite, bits, torch.zeros_like(bits))
    bf16 = bits_safe.to(torch.int16).contiguous().view(torch.bfloat16).reshape(256, 256)
    return bf16, finite.reshape(256, 256)


def _bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-element BF16 ULP distance.

    Maps the 16-bit sign-magnitude bit pattern to a monotonic int (negatives
    -> 0xFFFF - bits, positives -> bits + 0x8000) so that float ordering is
    preserved by int subtraction. ULP = |mono(a) - mono(b)|.
    """
    a_bits = a.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    b_bits = b.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    a_mono = torch.where((a_bits & 0x8000) != 0, 0xFFFF - a_bits, a_bits + 0x8000)
    b_mono = torch.where((b_bits & 0x8000) != 0, 0xFFFF - b_bits, b_bits + 0x8000)
    return (a_mono - b_mono).abs()


def _cumulative_histogram(ulps_flat: torch.Tensor) -> list[tuple[int, int, float]]:
    buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 8192]
    total = ulps_flat.numel()
    rows = []
    for b in buckets:
        n = int((ulps_flat >= b).sum().item())
        rows.append((b, n, (100.0 * n / total) if total else 0.0))
    return rows


def _format_report(
    variant_name: str,
    torch_approximate: str,
    input_bf16: torch.Tensor,
    expected_bf16: torch.Tensor,
    actual_bf16: torch.Tensor,
    ulps: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[str, int, float]:
    valid_ulps = ulps[valid]
    max_ulp = int(valid_ulps.max().item()) if valid.any() else 0
    mean_ulp = float(valid_ulps.float().mean().item()) if valid.any() else 0.0

    lines = [
        f"\n=== gelu variant={variant_name} (torch approximate={torch_approximate!r}) ===",
        f"  inputs evaluated: {int(valid.sum().item())}",
        f"  max ULP: {max_ulp}   mean ULP: {mean_ulp:.4f}",
        "  cumulative histogram (BF16 ULP distance from torch reference):",
    ]
    for thr, count, pct in _cumulative_histogram(valid_ulps):
        lines.append(f"    >= {thr:5d} ULP : {count:6d}  ({pct:7.4f}%)")

    if valid.any() and max_ulp > 0:
        # Top-10 worst offenders by ULP.
        flat_in = input_bf16.flatten()
        flat_exp = expected_bf16.flatten()
        flat_act = actual_bf16.flatten()
        flat_ulp = ulps.flatten()
        flat_valid = valid.flatten()
        valid_idx = torch.nonzero(flat_valid, as_tuple=False).flatten()
        sub_ulp = flat_ulp[valid_idx]
        k = min(10, valid_idx.numel())
        top_vals, top_local = torch.topk(sub_ulp, k)
        lines.append("  top-10 worst offenders:")
        lines.append(f"    {'input':>14}  {'expected':>14}  {'actual':>14}  {'ulp':>6}")
        for ul, li in zip(top_vals.tolist(), top_local.tolist()):
            i = int(valid_idx[li].item())
            lines.append(
                f"    {flat_in[i].item():>14.6g}  {flat_exp[i].item():>14.6g}  "
                f"{flat_act[i].item():>14.6g}  {int(ul):>6d}"
            )

    return "\n".join(lines), max_ulp, mean_ulp


@pytest.mark.parametrize("variant_name", list(_VARIANTS.keys()))
def test_gelu_variant_accuracy(device, variant_name, capsys):
    """All finite BF16 inputs, per-element ULP vs PyTorch CPU reference."""
    variant_enum, torch_approximate, ulp_threshold = _VARIANTS[variant_name]

    input_bf16, finite_mask = _all_bf16_inputs()

    # Reference computed at FP32 from the BF16-quantized input, then rounded
    # back to BF16 — matches what the SFPU does when fp32_dest_acc is on, and
    # what the BF16 dest-acc path does too (FP32 SFPU math then final convert).
    expected_bf16 = torch.nn.functional.gelu(input_bf16.float(), approximate=torch_approximate).to(torch.bfloat16)

    tt_input = ttnn.from_torch(input_bf16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.gelu(tt_input, variant=variant_enum)
    actual_bf16 = ttnn.to_torch(tt_output)

    # Drop NaN/Inf slots: not a fair ULP comparison there. ULP distance between
    # two NaNs is meaningless and varies by NaN payload; SFPU may flush to zero
    # while torch propagates NaN.
    valid = finite_mask & ~expected_bf16.isnan() & ~expected_bf16.isinf() & ~actual_bf16.isnan() & ~actual_bf16.isinf()
    ulps = _bf16_ulp_diff(expected_bf16, actual_bf16)

    # Inputs where |x| < 2^-125 cause the kernel's `0.5 * x` intermediate to
    # land below FP32's smallest normal (2^-126), where Tenstorrent's SFPU
    # FTZ's the result to 0. Torch's FP32 path retains the denormal and rounds
    # it back up to the smallest BF16 normal (2^-126) on the final cast — a
    # discrepancy we can't reproduce in software given the hardware FTZ.
    # These inputs are physically rare in real workloads (the model would have
    # to operate at 2^-126 magnitudes) and are excluded from the strict ULP
    # assertion below. They remain visible in the histogram and worst-offender
    # list so any regression beyond this known band is still surfaced.
    FTZ_INPUT_THRESHOLD = 2.0**-125  # ~2.35e-38
    ftz_safe = input_bf16.abs() >= FTZ_INPUT_THRESHOLD
    assertable = valid & ftz_safe

    report, max_ulp, _mean_ulp = _format_report(
        variant_name, torch_approximate, input_bf16, expected_bf16, actual_bf16, ulps, valid
    )

    # Always show the report — even on pass — so the user can see how tight
    # the kernel actually is. `capsys.disabled()` makes the report visible
    # without requiring `pytest -s`.
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
        max_assert_ulp = int(ulps[assertable].max().item()) if assertable.any() else 0
        assert (
            max_assert_ulp <= ulp_threshold
        ), f"{variant_name}: max ULP {max_assert_ulp} > {ulp_threshold} on FTZ-safe inputs."


def test_gelu_legacy_bool_matches_fast_lut(device):
    """fast_and_approximate_mode=True must be bitwise-identical to variant=FastLut
    across the full BF16 input space."""
    input_bf16, _ = _all_bf16_inputs()
    tt_input = ttnn.from_torch(input_bf16, layout=ttnn.TILE_LAYOUT, device=device)

    out_variant = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.FastLut))
    out_legacy = ttnn.to_torch(ttnn.gelu(tt_input, fast_and_approximate_mode=True))
    assert torch.equal(out_variant, out_legacy), "Legacy bool path diverged from variant=FastLut"


def test_gelu_default_matches_accurate(device):
    """ttnn.gelu(x) with no kwargs must resolve to variant=Accurate (nanobind
    overload-resolution sanity check)."""
    input_bf16, _ = _all_bf16_inputs()
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
    input_bf16, _ = _all_bf16_inputs()
    tt_input = ttnn.from_torch(input_bf16, layout=ttnn.TILE_LAYOUT, device=device)

    out_accurate = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Accurate))
    out_tanh = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Tanh))
    assert not torch.equal(out_accurate, out_tanh), (
        "variant=Tanh produced the same bits as variant=Accurate — GELU_TANH likely "
        "dispatched to the exact-GELU kernel."
    )
