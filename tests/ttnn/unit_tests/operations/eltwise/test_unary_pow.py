# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp
from tests.ttnn.unit_tests.operations.eltwise.test_expm1 import flush_subnormal_values


def generate_clean_bf16_tensor(dtype=torch.bfloat16):
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)  # 65536 values
    fp32 = input_tensor.to(torch.float32)

    # Remove special values (NaN, -0.0, +inf, -inf, subnormals)
    neg_zero_mask = (fp32 == 0.0) & torch.signbit(fp32)
    tiny = torch.finfo(torch.bfloat16).tiny  # 2**-126
    good_mask = torch.isfinite(fp32) & ~neg_zero_mask & (fp32.abs() >= tiny)
    fp32 = fp32[good_mask]  # 65024 values

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
    tt_input = flush_subnormal_values(tt_input)

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
    result = flush_subnormal_values(result)
    golden = flush_subnormal_values(golden)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


@pytest.mark.parametrize("exponent", [12.0, -0.6484])
def test_pow_arange_masking_fp32(exponent, device):
    tt_input = generate_clean_bf16_tensor(torch.float32)

    tt_in = ttnn.from_torch(
        tt_input,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(tt_input, exponent, device=device)

    tt_result = ttnn.pow(tt_in, exponent)
    # tt_result = ttnn.multiply(tt_in, exponent, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)], activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)], use_legacy=False )
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)
