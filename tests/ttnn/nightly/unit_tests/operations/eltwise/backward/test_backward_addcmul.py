# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 0.12])
def test_bw_addcmul(input_shapes, value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.addcmul_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcmul_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0])
@pytest.mark.parametrize(
    "are_required_outputs",
    [[True, True, True], [True, False, False], [False, True, False], [False, False, True], [False, True, True]],
)
def test_bw_addcmul_required_outputs(input_shapes, value, are_required_outputs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.addcmul_bw(
        grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value, are_required_outputs=are_required_outputs
    )

    golden_function = ttnn.get_golden_function(ttnn.addcmul_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    assert len(tt_output_tensor_on_device) == 3
    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
        else:
            assert tt_output_tensor_on_device[i] is None
    assert status
