# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range
from tests.ttnn.backward_shape_matrix import broadcast_grad_shape, matrix_ids, matrix_shapes


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_squared_difference(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    tt_output_tensor_on_device = ttnn.squared_difference_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.squared_difference_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("input_shape,other_shape", matrix_shapes(), ids=matrix_ids())
def test_bw_squared_difference_broadcast(input_shape, other_shape, device):
    grad_shape = broadcast_grad_shape(input_shape, other_shape)
    grad_data, grad_tensor = data_gen_with_range(grad_shape, -5, 5, device)
    in_data, input_tensor = data_gen_with_range(torch.Size(input_shape), -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(torch.Size(other_shape), -10, 10, device, True)

    tt_output_tensor_on_device = ttnn.squared_difference_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.squared_difference_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    assert [list(t.shape) for t in tt_output_tensor_on_device] == [list(input_shape), list(other_shape)]
    assert compare_pcc(tt_output_tensor_on_device, golden_tensor)
