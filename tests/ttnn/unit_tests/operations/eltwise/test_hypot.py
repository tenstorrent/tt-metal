# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
from tests.ttnn.utils_for_testing import assert_with_pcc

@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)

def test_binary_hypot_ttnn(input_shapes, device):
    in_data1 = torch.rand(input_shapes, dtype=torch.bfloat16) * (200 - 100)
    in_data2  = torch.rand(input_shapes, dtype=torch.bfloat16) * (200 - 100)

    input_tensor_a = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(in_data2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.experimental.hypot(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.hypot)
    golden_tensor = golden_function(in_data1, in_data2)

    comp_pass = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert comp_pass >= 0.9998

