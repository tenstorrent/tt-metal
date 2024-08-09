# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes, input_shapes_2, dimension",
    (
        ((torch.Size([12, 1, 30, 32])), (torch.Size([2, 1, 30, 32])), 0),
        ((torch.Size([1, 2, 45, 64])), (torch.Size([1, 1, 45, 64])), 1),
        ((torch.Size([1, 1, 125, 32])), (torch.Size([1, 1, 32, 32])), 2),
        (
            (torch.Size([1, 1, 64, 80])),
            (torch.Size([1, 1, 64, 16])),
            3,
        ),  # size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values
        # Tile shape
        ((torch.Size([4, 1, 32, 32])), (torch.Size([5, 1, 32, 32])), 0),
        ((torch.Size([1, 2, 64, 64])), (torch.Size([1, 1, 64, 64])), 1),
        ((torch.Size([1, 1, 64, 32])), (torch.Size([1, 1, 32, 32])), 2),
        ((torch.Size([1, 1, 64, 64])), (torch.Size([1, 1, 64, 32])), 3),
    ),
)
def test_bw_concat(input_shapes, input_shapes_2, dimension, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, True)

    other_data, other_tensor = data_gen_with_range(input_shapes_2, -100, 100, device, True, True)

    pyt_y = torch.concat((in_data, other_data), dim=dimension)

    grad_data, grad_tensor = data_gen_with_range(pyt_y.shape, -100, 100, device, True, True)

    tt_output_tensor_on_device = ttnn.concat_bw(grad_tensor, input_tensor, other_tensor, dimension)

    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, dimension)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes, input_shapes_2",
    (
        ((torch.Size([12, 1, 30, 32])), (torch.Size([2, 1, 30, 32]))),
        ((torch.Size([4, 1, 32, 32])), (torch.Size([5, 1, 32, 32]))),
    ),
)
def test_bw_concat_Default(input_shapes, input_shapes_2, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, True)

    other_data, other_tensor = data_gen_with_range(input_shapes_2, -100, 100, device, True, True)

    pyt_y = torch.concat((in_data, other_data))

    grad_data, grad_tensor = data_gen_with_range(pyt_y.shape, -100, 100, device, True, True)

    tt_output_tensor_on_device = ttnn.concat_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
