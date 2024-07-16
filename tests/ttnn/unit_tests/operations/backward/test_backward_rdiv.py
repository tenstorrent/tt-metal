# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_pt_tt, compare_results


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "round_mode",
    (
        "None",
        "trunc",
        "floor",
    ),
)
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12])
def test_bw_rdiv(input_shapes, scalar, round_mode, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    tt_output_tensor_on_device = ttnn.rdiv_bw(grad_tensor, input_tensor, scalar, round_mode=round_mode)

    in_data.retain_grad()

    if round_mode == "None":
        round_mode = None
    pyt_y = torch.div(scalar, in_data, rounding_mode=round_mode)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

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
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12])
def test_bw_rdiv_default(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    tt_output_tensor_on_device = ttnn.rdiv_bw(grad_tensor, input_tensor, scalar)

    in_data.retain_grad()

    pyt_y = torch.div(scalar, in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert status
