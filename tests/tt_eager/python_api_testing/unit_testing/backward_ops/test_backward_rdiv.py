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

    tt_output_tensor_on_device = tt_lib.tensor.rdiv_bw(grad_tensor, input_tensor, scalar=scalar, round_mode=round_mode)

    in_data.retain_grad()

    if round_mode == "None":
        round_mode = None
    pyt_y = torch.div(scalar, in_data, rounding_mode=round_mode)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert status
