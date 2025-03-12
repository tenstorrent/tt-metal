# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import skip_for_grayskull, torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_broken_remainder(input_shapes, device):
    torch_lhs = torch.ones(32, 32, dtype=torch.bfloat16)
    torch_rhs = torch.zeros(32, 32, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_lhs, torch_rhs, device=device)

    tt_lhs = ttnn.from_torch(torch_lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_rhs = ttnn.from_torch(torch_rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_result = ttnn.remainder(tt_lhs, tt_rhs)
    result = ttnn.to_torch(tt_result)
    assert torch.allclose(result, golden, atol=0.01, rtol=0)


@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_broken_remainder1(input_shapes, device):
    torch_lhs = torch.ones(32, 32, dtype=torch.bfloat16) * 95
    torch_rhs = torch.ones(32, 32, dtype=torch.bfloat16) * (-94.5)

    golden_function = ttnn.get_golden_function(ttnn.remainder)  # all -94.0
    golden = golden_function(torch_lhs, torch_rhs, device=device)

    tt_lhs = ttnn.from_torch(torch_lhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_rhs = ttnn.from_torch(torch_rhs, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    tt_result = ttnn.remainder(tt_lhs, tt_rhs)
    result = ttnn.to_torch(tt_result)  # all 0.5
    assert torch.allclose(result, golden, atol=0.01, rtol=0)


@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 10, 10])),),
)
def test_remainder_scalar(input_shapes, device):
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16
    )(input_shapes)
    scalar = 2
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar, device=device)

    output_tensor = ttnn.remainder(input_tensor_a, scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    print("tt input a: ", input_tensor_a)
    print("Torch out: ", torch_output_tensor)
    print("TT out: ", output_tensor)
    print("Diff: ", torch_output_tensor - output_tensor)
    diff = torch.abs(torch_output_tensor - output_tensor)
    max_atol = torch.max(diff)
    print(f"Max absolute tolerance (atol): {max_atol.item()}")

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999


def test_remainder_small_matrix(device):
    torch_input_tensor_a = torch.tensor([[5, 2], [3, 4]], dtype=torch.bfloat16)
    print("torch input: ", torch_input_tensor_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    scalar = 0.003
    # golden_function = ttnn.get_golden_function(ttnn.remainder)
    # torch_output_tensor = golden_function(torch_input_tensor_a, scalar, device=device)
    torch_output_tensor = torch.remainder(torch_input_tensor_a.float(), scalar).bfloat16()
    output_tensor = ttnn.remainder(input_tensor_a, scalar)
    print("tt input: ", input_tensor_a)
    print("Torch input: ", torch_input_tensor_a)
    output_tensor = ttnn.to_torch(output_tensor)

    print("Torch out: ", torch_output_tensor)
    print("TT out: ", output_tensor)
    print("Diff: ", torch_output_tensor - output_tensor)
    diff = torch.abs(torch_output_tensor - output_tensor)
    max_atol = torch.max(diff)
    print(f"Max absolute tolerance (atol): {max_atol.item()}")

    # assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
    assert torch.allclose(output_tensor, torch_output_tensor, atol=0.01, rtol=0)
