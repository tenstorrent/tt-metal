# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_multigammaln(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 2.5, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    pyt_y = torch.mvlgamma(in_data, 4)

    tt_output_tensor_on_device = ttnn.multigammaln_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass
