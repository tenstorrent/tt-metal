# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


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


# This test was added for #17361
# If input is a multiple of the scalar, the result should be 0, but both Torch and TT output either 0 or the scalar value itself depending on the operands.
# This inconsistency is persistent due to some fp precision loss in both Torch and TT.
# Eg: torch.remainder of (3, 1.5) = 0.0 and of (3, 0.003) = 0.003
# Eg: ttnn.remainder of (4, 0.004) = 0.004 and of (3, 0.003) = 0.0
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([6, 5, 320, 320])),
        (torch.Size([2, 1, 384, 320])),
        (torch.Size([3, 123, 115])),
        (torch.Size([69, 178])),
        (torch.Size([1024])),
        (torch.Size([])),
    ),
)
@pytest.mark.parametrize("scalar", [-0.002, -0.001, -0.0006, -0.0003, 0.0, 0.0005, 0.0007, 0.001, 0.002])
def test_remainder_scalar(input_shapes, scalar, device):
    torch.manual_seed(0)
    if len(input_shapes) == 0:
        torch_input_tensor = torch.tensor(5.0, dtype=torch.bfloat16)
    else:
        torch_input_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16
        )(input_shapes)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    output_tensor = ttnn.remainder(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    if scalar == 0.0:
        assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
    else:
        assert torch.allclose(output_tensor, torch_output_tensor, atol=0.001, rtol=0)
