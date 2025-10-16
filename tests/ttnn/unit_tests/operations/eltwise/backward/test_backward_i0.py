# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_i0(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 199, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, required_grad=True)
    in_data = in_data.float()

    tt_output_tensor_on_device = ttnn.i0_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.i0_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
        [4, 7, 21, 133],
        [4, 6, 105, 245],
    ],
)
def test_i0_bw_range(device, shapes):
    torch.manual_seed(3624344)  # 16305027

    high = -10
    low = 10
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32, requires_grad=True) * (high - low) + low

    high = 5
    low = -5
    grad_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low

    golden_fn = ttnn.get_golden_function(ttnn.i0_bw)
    torch_output_tensor = golden_fn(grad_tensor_a, torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grad_tensor = ttnn.from_torch(
        grad_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0_bw(grad_tensor, input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor[0])

    torch_output_tensor = torch_output_tensor[0]

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert pcc >= 0.9998
