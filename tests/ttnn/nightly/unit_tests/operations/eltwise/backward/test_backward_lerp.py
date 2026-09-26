# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_lerp(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, True)
    end_data, end_tensor = data_gen_with_range(input_shapes, -199, 199, device, True)
    weight_data, weight_tensor = data_gen_with_range(input_shapes, -201, 201, device, True)

    tt_output_tensor_on_device = ttnn.lerp_bw(grad_tensor, input_tensor, end_tensor, weight_tensor)

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight_data)
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("weight", [-0.25, -25.0, 0.05, 1.0, 25.0])
def test_bw_lerp_weight_scalar(input_shapes, weight, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 101, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, True)
    end_data, end_tensor = data_gen_with_range(input_shapes, -199, 199, device, True)

    tt_output_tensor_on_device = ttnn.lerp_bw(grad_tensor, input_tensor, end_tensor, weight)

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "are_required_outputs",
    [[True, True, True], [True, False, False], [False, True, False], [False, False, True]],
)
def test_bw_lerp_tensor_weight_required_outputs(input_shapes, are_required_outputs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    end_data, end_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    weight_data, weight_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.lerp_bw(
        grad_tensor, input_tensor, end_tensor, weight_tensor, are_required_outputs=are_required_outputs
    )

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight_data)

    assert len(tt_output_tensor_on_device) == 3
    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
        else:
            assert tt_output_tensor_on_device[i] is None
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_lerp_scalar_weight_required_outputs(input_shapes, weight, are_required_outputs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    end_data, end_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.lerp_bw(
        grad_tensor, input_tensor, end_tensor, weight, are_required_outputs=are_required_outputs
    )

    golden_function = ttnn.get_golden_function(ttnn.lerp_bw)
    golden_tensor = golden_function(grad_data, in_data, end_data, weight)

    assert len(tt_output_tensor_on_device) == 2
    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
        else:
            assert tt_output_tensor_on_device[i] is None
    assert status
