# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 5.0])
def test_bw_addcdiv(input_shapes, value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, False)

    tt_output_tensor_on_device = ttnn.addcdiv_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcdiv_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    comp_pcc = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pcc
