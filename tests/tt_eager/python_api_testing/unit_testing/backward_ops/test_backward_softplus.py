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
    "beta",
    [0.5, -3, 1, 4],
)
@pytest.mark.parametrize(
    "threshold",
    [-20, -10, 10, 20, 5],
)
def test_bw_softplus(input_shapes, beta, threshold, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device, True)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    in_data.retain_grad()

    pyt_y = torch.nn.functional.softplus(in_data, beta=beta, threshold=threshold)

    tt_output_tensor_on_device = tt_lib.tensor.softplus_bw(grad_tensor, input_tensor, beta=beta, threshold=threshold)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
