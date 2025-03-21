# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.utility_functions import skip_for_grayskull, torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


# This test was added for #17361
# If input is a multiple of the scalar, the result should be 0, but both Torch and TT output either 0 or the scalar value itself depending on the operands.
# This inconsistency is persistent due to some fp precision loss in both Torch and TT.
# Eg: torch.remainder of (3, 1.5) = 0.0 and of (3, 0.003) = 0.003
# Eg: ttnn.remainder of (4, 0.004) = 0.004 and of (3, 0.003) = 0.0
@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
@pytest.mark.parametrize(
    "input_shapes",
    (
        # (torch.Size([6, 5, 90, 112])),
        # (torch.Size([2, 1, 120, 11])),
        # (torch.Size([3, 123, 115])),
        (torch.Size([69, 178])),
    ),
)
@pytest.mark.parametrize("scalar", [0.002])
def test_fmod_scalar(input_shapes, scalar, device):
    torch.manual_seed(0)
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16
    )(input_shapes)
    # torch_input_tensor_a = torch.tensor([[2, 3], [4, 5]], dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar, device=device)

    output_tensor = ttnn.fmod(input_tensor_a, scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    print("Input: ", input_tensor_a)
    print("Torch Out: ", torch_output_tensor)
    print("TT Out: ", output_tensor)
    diff = torch_output_tensor - output_tensor
    print("Diff: ", diff)
    max_atol = torch.max(diff)
    print(f"Max absolute tolerance (atol): {max_atol.item()}")

    if scalar == 0.0:
        assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999
    else:
        assert torch.allclose(output_tensor, torch_output_tensor, atol=0.001, rtol=0)
