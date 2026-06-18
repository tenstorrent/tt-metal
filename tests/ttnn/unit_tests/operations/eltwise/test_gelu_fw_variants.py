# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict per-element accuracy tests for the GELU SFPU variants.

For each ttnn.GeluVariant we compare against torch.nn.functional.gelu with the
matching `approximate=` flag, asserting both max-abs and mean-abs diff fit
under per-variant thresholds. Stricter than PCC because PCC can mask a kernel
that's systematically off by ~1% absolute (PCC ~0.997 still passes).

Thresholds reflect the SFPU error budget for each variant:
  - Accurate: piecewise CDF (BF16) or FP32 erf, MaxULP ≤ 1 vs exact GELU.
  - FastLut:  6-segment piecewise-linear LUT, peak ~1-2% absolute error.
  - Tanh:     FP32 tanh formula, MaxULP ≤ 1 vs F.gelu(approximate="tanh").
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


# (variant enum, torch `approximate=` flag, {max_abs, mean_abs} bounds for bf16 input)
_VARIANTS = {
    "accurate": (ttnn.GeluVariant.Accurate, "none", {"max_abs": 0.02, "mean_abs": 5e-4}),
    "fast_lut": (ttnn.GeluVariant.FastLut, "none", {"max_abs": 0.05, "mean_abs": 1e-2}),
    "tanh": (ttnn.GeluVariant.Tanh, "tanh", {"max_abs": 0.02, "mean_abs": 5e-4}),
}


def _diff_stats(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    diff = (expected.float() - actual.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def _make_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Sample uniformly over GELU's interesting transition zone plus some saturation."""
    torch.manual_seed(0)
    return torch.empty(shape, dtype=dtype).uniform_(-6.0, 6.0)


@pytest.mark.parametrize("variant_name", list(_VARIANTS.keys()))
@pytest.mark.parametrize("shape", [(64, 128), (1, 8192)])
def test_gelu_variant_accuracy(device, variant_name, shape):
    """ttnn.gelu output vs torch.nn.functional.gelu, strict mean-abs and max-abs bounds."""
    variant_enum, torch_approximate, bounds = _VARIANTS[variant_name]

    torch_input = _make_input(shape, torch.bfloat16)

    # Reference: compute GELU in FP32 from the bf16-quantized input, then quantize
    # the result back to bf16. This isolates the kernel's accuracy from input
    # quantization noise, matching what the SFPU does when fp32_dest_acc is on.
    torch_output_bf16 = torch.nn.functional.gelu(torch_input.float(), approximate=torch_approximate).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.gelu(tt_input, variant=variant_enum)
    tt_output_torch = ttnn.to_torch(tt_output)

    stats = _diff_stats(torch_output_bf16, tt_output_torch)
    assert stats["max_abs"] <= bounds["max_abs"], (
        f"{variant_name}: max-abs-diff {stats['max_abs']:.4g} > {bounds['max_abs']:.4g}. "
        f"Mean-abs-diff {stats['mean_abs']:.4g}."
    )
    assert stats["mean_abs"] <= bounds["mean_abs"], (
        f"{variant_name}: mean-abs-diff {stats['mean_abs']:.4g} > {bounds['mean_abs']:.4g}. "
        f"Max-abs-diff {stats['max_abs']:.4g}."
    )


def test_gelu_legacy_bool_matches_fast_lut(device):
    """fast_and_approximate_mode=True must be bitwise-identical to variant=FastLut."""
    torch_input = _make_input((64, 128), torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    out_variant = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.FastLut))
    out_legacy = ttnn.to_torch(ttnn.gelu(tt_input, fast_and_approximate_mode=True))
    assert torch.equal(out_variant, out_legacy), "Legacy bool path diverged from variant=FastLut"


def test_gelu_default_matches_accurate(device):
    """ttnn.gelu(x) with no kwargs must resolve to variant=Accurate.

    This guards against nanobind overload resolution picking the bool overload
    when both overloads have all-default kwargs.
    """
    torch_input = _make_input((64, 128), torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    out_default = ttnn.to_torch(ttnn.gelu(tt_input))
    out_accurate = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Accurate))
    assert torch.equal(out_default, out_accurate), (
        "ttnn.gelu(x) did not resolve to variant=Accurate — overload resolution may have picked "
        "the legacy bool overload, which would still produce the same result for "
        "fast_and_approximate_mode=False but indicates incorrect nanobind dispatch."
    )


def test_tanh_differs_from_accurate(device):
    """variant=Tanh must produce different output from variant=Accurate.

    Their max-abs gap is ~3e-5 mathematically but in BF16 it manifests as a
    handful of differing tiles. If they were bitwise-equal, GELU_TANH would
    be silently dispatching to the exact-GELU kernel.
    """
    torch_input = _make_input((64, 128), torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    out_accurate = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Accurate))
    out_tanh = ttnn.to_torch(ttnn.gelu(tt_input, variant=ttnn.GeluVariant.Tanh))
    assert not torch.equal(out_accurate, out_tanh), (
        "variant=Tanh produced the same bits as variant=Accurate — GELU_TANH likely dispatched "
        "to the exact-GELU kernel, not the new tanh-formula kernel."
    )
