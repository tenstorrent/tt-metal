# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        -0.01,
        -1.0,
    ],
)
def test_negative_exponent(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    with pytest.raises(RuntimeError) as _e:
        tt_output_tensor_on_device = tt_lib.tensor.unary_pow_bw(grad_tensor, input_tensor, exponent=exponent)
    assert "exponent >= 0.0" in str(_e)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        0,
    ],
)
def test_fw_exponent(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    golden_tensor = [
        torch.pow(grad_data, 0.0),
    ]
    tt_output_tensor_on_device = tt_lib.tensor.pow(grad_tensor, 0.0)
    status = compare_results([tt_output_tensor_on_device], golden_tensor)
    assert status

    # assert "exponent >= 0.0" in str(_e)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    [
        0.0,
        1.0,
        2.0,
        5.0,
    ],
)
def test_bw_unary_pow(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    tt_output_tensor_on_device = tt_lib.tensor.unary_pow_bw(grad_tensor, input_tensor, exponent=exponent)

    in_data.retain_grad()

    pyt_y = torch.pow(in_data, exponent)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert status
