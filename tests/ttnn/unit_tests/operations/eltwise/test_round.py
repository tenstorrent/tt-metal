# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from functools import partial
from models.utility_functions import skip_for_grayskull, torch_random
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@pytest.mark.parametrize(
    "shape",
    (
        torch.Size([1, 1, 10, 10]),
        torch.Size([4, 3, 32, 32]),
        torch.Size([2, 2, 32, 32]),
        torch.Size([6, 4, 32, 32]),
        torch.Size([1, 1, 320, 320]),
        torch.Size([1, 3, 320, 64]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
    ),
)
@pytest.mark.parametrize("decimal", range(-6, 7))
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@skip_for_grayskull("Requires wormhole_b0 to run")
def test_round_new(shape, dtypes, decimal, device):
    torch.manual_seed(0)
    torch_dtype, tt_dtype = dtypes
    torch_input_tensor = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch_dtype), tt_dtype)(
        shape
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=tt_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_input_tensor = ttnn.to_torch(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.round)
    torch_output_tensor = golden_function(torch_input_tensor, decimal)

    output_tensor = ttnn.round(input_tensor, decimal)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.997
