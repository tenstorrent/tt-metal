# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize("sizes", [[12, 1, 1, 1], [6, 1, 1, 1], [1, 24, 1, 1], [1, 3, 1, 1]])
def test_bw_repeat(input_shapes, sizes, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)

    pyt_y = in_data.repeat(sizes)

    grad_data, grad_tensor = data_gen_pt_tt(pyt_y.shape, device, True)

    tt_output_tensor_on_device = tt_lib.tensor.repeat_bw(grad_tensor, input_tensor, sizes)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize("sizes", [[12, 1, 1, 1], [6, 1, 1, 1], [1, 24, 1, 1], [1, 3, 1, 1]])
@pytest.mark.parametrize(
    "are_required_outputs",
    [
        [True],
        [False],
    ],
)
def test_bw_repeat_optional_output_cq_id(input_shapes, sizes, are_required_outputs, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    if are_required_outputs[0]:
        _, optional_output_tensor = data_gen_pt_tt(input_shapes, device, True)
    else:
        optional_output_tensor = None
    pyt_y = in_data.repeat(sizes)

    grad_data, grad_tensor = data_gen_pt_tt(pyt_y.shape, device, True)

    cq_id = 0
    if are_required_outputs[0]:
        tt_output_tensor_on_device = tt_lib.tensor.repeat_bw(
            grad_tensor,
            input_tensor,
            sizes,
            are_required_outputs=are_required_outputs,
            input_grad=optional_output_tensor,
            queue_id=cq_id,
        )
    else:
        tt_output_tensor_on_device = tt_lib.tensor.repeat_bw(
            grad_tensor, input_tensor, sizes, are_required_outputs=are_required_outputs
        )

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = True

    if are_required_outputs[0]:
        status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
        assert status
