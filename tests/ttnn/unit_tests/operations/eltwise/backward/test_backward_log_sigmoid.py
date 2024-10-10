# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_val,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -120, 120, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 20, device)

    tt_output_tensor_on_device = ttnn.log_sigmoid_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_neg_inp(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -120, -1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 50, device)

    tt_output_tensor_on_device = ttnn.log_sigmoid_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_pos_inp(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 50, device)

    tt_output_tensor_on_device = ttnn.log_sigmoid_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_zero_inp(input_shapes, device):
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, 0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    tt_output_tensor_on_device = ttnn.log_sigmoid_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_log_sigmoid_zero_grad(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -50, 50, device, True)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, True, 0)

    tt_output_tensor_on_device = ttnn.log_sigmoid_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.log_sigmoid_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
