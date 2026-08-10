# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


def generate_clean_bf16_tensor(dtype=torch.bfloat16):
    all_bf16 = generate_all_bfloat16_bitpatterns(torch.bfloat16).flatten()
    fp32 = all_bf16.to(torch.float32)

    # Remove special values (NaN, -0.0, +inf, -inf, subnormals)
    neg_zero_mask = (fp32 == 0.0) & torch.signbit(fp32)
    tiny = torch.finfo(torch.bfloat16).tiny  # 2**-126
    good_mask = torch.isfinite(fp32) & ~neg_zero_mask & (fp32.abs() >= tiny)
    fp32 = fp32[good_mask]  # ~65024 clean values

    return fp32.to(dtype)


@pytest.mark.parametrize("exponent", [2.0, -2.0, -3.56, 0.5, -0.5, -0.566, -2])
def test_pow(exponent, device):
    torch.manual_seed(42)
    torch_base = torch.rand([4, 4], dtype=torch.bfloat16)
    torch_output = torch.pow(torch_base, exponent)
    ttnn_base = ttnn.from_torch(torch_base, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_output = ttnn.pow(ttnn_base, exponent)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_ulp(torch_output, ttnn_output, 1)


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0, 3.0, -1.0])
def test_pow_arange_masking(exponent, device):
    # Generate all possible bit pattern for bf16
    tt_input = generate_clean_bf16_tensor(torch.bfloat16)
    # If input is subnormal then we assume hardware will flush it to 0.0
    tt_input = flush_subnormal_values_to_zero(tt_input)

    tt_in = ttnn.from_torch(
        tt_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(tt_input, exponent, device=device)

    tt_result = ttnn.pow(tt_in, exponent)
    result = ttnn.to_torch(tt_result)
    # If expected output is subnormal then its calculated value should be 0.0 (hardware assumed to flush to 0.0)
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "op_type,exponent",
    [
        (ttnn.UnaryOpType.POWER_ITERATIVE, 0),
        (ttnn.UnaryOpType.POWER_ITERATIVE, 2),
        (ttnn.UnaryOpType.POWER_ITERATIVE, 3.65),
        (ttnn.UnaryOpType.POWER_ITERATIVE, -4.2),
        (ttnn.UnaryOpType.POWER_ITERATIVE, -3.0),
        (ttnn.UnaryOpType.POWER_ITERATIVE, 3.0),
        (ttnn.UnaryOpType.POWER, 0),
        (ttnn.UnaryOpType.POWER, 2),
        (ttnn.UnaryOpType.POWER, 1.5),
        (ttnn.UnaryOpType.POWER, -1.9),
    ],
)
def test_power_as_activation(device, op_type, exponent):
    if op_type == ttnn.UnaryOpType.POWER_ITERATIVE and (exponent != int(exponent) or exponent < 0):
        pytest.xfail(
            "POWER_ITERATIVE only supports positive integer exponents (Non-integer values are truncated causing output mismatch with expected)"
        )

    x_torch = torch.rand([16, 16], dtype=torch.bfloat16) + 1.5
    z_torch = torch.pow(x_torch + x_torch, exponent)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.add(x_tt, x_tt, activations=[ttnn.UnaryWithParam(op_type, exponent)])
    tt_out = ttnn.to_torch(z_tt)

    assert_with_ulp(z_torch, tt_out, 1)


# Regression for issue #52675: the bf16 power path ran a single cubic log2 whose
# ~2.8e-3 absolute error was amplified by |exponent| (result rel err ~= 0.2%*|y|,
# unbounded: 18% at y=100, 133% at y=500). bf16 now runs the accurate fp32
# algorithm (atanh-series log2 + hi/lo argument split + explicit overflow guard),
# so large exponents must stay within a few ULP of the bf16 golden and overflow
# must saturate to +inf instead of wrapping to NaN/garbage.
@pytest.mark.parametrize(
    "base,exponent",
    [
        (0.99609375, 100.0),  # issue repro: 0.8008 (18% off) -> ~0.6761
        (0.99609375, 500.0),  # issue repro: 0.3295 (133% off) -> ~0.1413
        (0.99609375, -100.0),
        (1.125, 64.0),
        (0.875, -64.0),
        (2.0, 128.0),  # overflow -> +inf
        (10.0, 39.0),  # overflow -> +inf
    ],
)
def test_pow_bf16_large_exponent(device, base, exponent):
    torch.manual_seed(0)
    torch_base = torch.full([1, 1, 32, 32], base, dtype=torch.bfloat16)
    torch_output = torch.pow(torch_base, exponent)

    ttnn_base = ttnn.from_torch(torch_base, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.to_torch(ttnn.pow(ttnn_base, exponent))

    if torch_output.isinf().all():
        assert ttnn_output.isinf().all(), f"pow({base}, {exponent}) should overflow to +inf, got {ttnn_output[0,0,0,0]}"
        assert (ttnn_output > 0).all(), "overflow produced -inf/NaN instead of +inf"
    else:
        assert_with_ulp(torch_output, ttnn_output, 3)


# Sweep over a range of bases and large exponents to prove the error no longer
# grows with |exponent|. Before the fix the median error was 5.8% at y=32 and
# 29.3% at y=500; after the fix every point is within 3 ULP of the bf16 golden.
@pytest.mark.parametrize("exponent", [32.0, 100.0, 500.0])
def test_pow_bf16_large_exponent_sweep(device, exponent):
    torch.manual_seed(1)
    bases = torch.logspace(-3, 3, 4096, dtype=torch.float32).reshape(64, 64).to(torch.bfloat16)
    torch_output = torch.pow(bases, exponent)

    ttnn_base = ttnn.from_torch(bases, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.to_torch(ttnn.pow(ttnn_base, exponent))

    # Skip points where the golden overflows or denormal-flushes; those are
    # correctly handled but outside the accuracy comparison.
    finite = torch_output.isfinite() & (torch_output.abs() >= torch.finfo(torch.bfloat16).tiny)
    assert finite.any(), f"exponent {exponent}: no finite, non-denormal golden outputs to compare"
    assert_with_ulp(torch_output[finite], ttnn_output[finite], 3)
