# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        -0.01,
        -1.0,
    ],
)
def test_negative_exponent(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    with pytest.raises(RuntimeError) as _e:
        tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)
    assert "exponent >= 0.0" in str(_e.value)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        0,
    ],
)
def test_fw_exponent(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    golden_tensor = [
        torch.pow(grad_data, exponent),
    ]
    tt_output_tensor_on_device = ttnn.pow(grad_tensor, exponent)
    status = compare_pcc([tt_output_tensor_on_device], golden_tensor)
    assert status

    # assert "exponent >= 0.0" in str(_e)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent_and_pcc",
    [
        (0.0, 0.99),
        (1.0, 0.99),
        (2.0, 0.99),
        (5.0, 0.99),
        (0.5, 0.92),
        (1.5, 0.84),
        (2.5, 0.57),
    ],
)
def test_bw_unary_pow(input_shapes, exponent_and_pcc, device):
    exponent, pcc = exponent_and_pcc
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device)

    tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=pcc)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_unary_pow_test_inf(input_shapes, device):
    exponent = 2
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 9, device)

    tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)
    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_unary_pow_test_neg_inf(input_shapes, device):
    exponent = 2
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, -1, device)

    tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent_and_pcc",
    [
        (0.0, 0.99),
        (1.0, 0.99),
        (2.0, 0.99),
        (5.0, 0.99),
        (0.5, 0.92),
        (1.5, 0.84),
        (2.5, 0.57),
    ],
)
def test_bw_unary_pow_output(input_shapes, exponent_and_pcc, device):
    exponent, pcc = exponent_and_pcc
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    input_grad = None

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.pow_bw(
        grad_tensor,
        input_tensor,
        exponent=exponent,
        input_grad=input_grad,
        queue_id=cq_id,
    )

    in_data.retain_grad()

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=pcc)
    assert status
