# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
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
@pytest.mark.parametrize("lambd", [2.2, 0.5, 1.0, 5.5, 9.9])
def test_bw_hardshrink(input_shapes, lambd, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    pyt_y = torch.nn.functional.hardshrink(in_data, lambd=lambd)

    tt_output_tensor_on_device = ttnn.hardshrink_bw(grad_tensor, input_tensor, lambd)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
