# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_lerp(input_shapes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    end_data, end_tensor = data_gen_pt_tt(input_shapes, device, True)
    weight_data, weight_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    tt_output_tensor_on_device = tt_lib.tensor.lerp_bw(grad_tensor, input_tensor, end_tensor, weight_tensor)

    in_data.retain_grad()
    end_data.retain_grad()

    pyt_y = torch.lerp(in_data, end_data, weight_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, end_data.grad]

    status = compare_results(tt_output_tensor_on_device, golden_tensor)
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
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    end_data, end_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    tt_output_tensor_on_device = tt_lib.tensor.lerp_bw(grad_tensor, input_tensor, end_tensor, weight)

    in_data.retain_grad()
    end_data.retain_grad()

    pyt_y = torch.lerp(in_data, end_data, weight)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, end_data.grad]

    status = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert status
