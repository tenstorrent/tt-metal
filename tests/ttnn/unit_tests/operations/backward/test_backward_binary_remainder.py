# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import compare_pcc, data_gen_with_range
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@skip_for_grayskull("#ToDo: GS implementation needs to be done for binary remainder backward")
def test_bw_remainder(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 30, device, True)
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -50, 50, device, True)
    pyt_y = torch.remainder(in_data, other_data)

    tt_output_tensor_on_device = ttnn.binary_remainder_bw(grad_tensor, input_tensor, other_tensor)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
