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
@pytest.mark.parametrize(
    "scalar",
    (
        1.5,
        15.9,
        7.1,
        0.0,
    ),
)
def test_bw_unary_remainder(input_shapes, scalar, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, seed=0)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True, seed=1)

    tt_output_tensor_on_device = ttnn.remainder_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.remainder_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_binary_remainder(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 30, device, True, seed=0)
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=1)

    high = 50
    low = -50
    other_data = torch.rand(input_shapes, requires_grad=True).bfloat16() * (high - low) + low
    other_data[other_data == 0] = 1.0
    other_tensor = ttnn.from_torch(other_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output_tensor_on_device = ttnn.remainder_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.remainder_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize("input_shape,other_shape", matrix_shapes(), ids=matrix_ids())
def test_bw_remainder_broadcast(input_shape, other_shape, device):
    grad_shape = broadcast_grad_shape(input_shape, other_shape)
    grad_data, grad_tensor = data_gen_with_range(grad_shape, -5, 5, device, seed=0)
    in_data, input_tensor = data_gen_with_range(torch.Size(input_shape), -10, 10, device, True, seed=1)
    other_data, other_tensor = data_gen_with_range(torch.Size(other_shape), 1, 5, device, True, seed=2)

    tt_output_tensor_on_device = ttnn.remainder_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.remainder_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    assert [list(t.shape) for t in tt_output_tensor_on_device] == [list(input_shape), list(other_shape)]
    assert compare_pcc(tt_output_tensor_on_device, golden_tensor)
