# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from models.utility_functions import torch_random
from functools import partial
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [1.0, 5.0, 10.0])
def test_device_subalpha_no_bcast(input_shapes, alpha, device):
    torch.manual_seed(0)
    in_data1 = (torch.rand(input_shapes, dtype=torch.bfloat16) * 200) - 100
    in_data2 = (torch.rand(input_shapes, dtype=torch.bfloat16) * 300) - 150


    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.subalpha(input_tensor_a, input_tensor_b, alpha)

    golden_function = ttnn.get_golden_function(ttnn.subalpha)
    golden_tensor = golden_function(in_data1, in_data2, alpha)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998
