# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_val,
)
import ttnn


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_sqrt(input_shapes, device):
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, val=0)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, val=-1)

    pyt_y = torch.sqrt(in_data)

    tt_output_tensor_on_device = ttnn.sqrt_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    print("PT out ", golden_tensor)
    print("TT out ", tt_output_tensor_on_device)
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
