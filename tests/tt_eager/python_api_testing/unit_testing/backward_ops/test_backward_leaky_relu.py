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
@pytest.mark.parametrize("negative_slope", [-0.25, -0.5, 0.01, 0.5, 5.0])
def test_bw_leaky_relu(input_shapes, negative_slope, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    pyt_y = torch.nn.functional.leaky_relu(in_data, negative_slope=negative_slope, inplace=False)

    tt_output_tensor_on_device = tt_lib.tensor.leaky_relu_bw(grad_tensor, input_tensor, negative_slope)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
