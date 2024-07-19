# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_i0(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 199, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, required_grad=True)
    in_data = in_data.float()

    tt_output_tensor_on_device = ttnn.i0_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.i0_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
