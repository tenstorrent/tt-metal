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
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device, seed=1)

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
    in_data, input_tensor = data_gen_with_range(input_shapes, -90, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device, seed=1)

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
    "exponent",
    [(0.5), (0.0), (1.5), (4.5), (7.25), (10.0), (12.75), (16.0)],
)
def test_bw_unary_pow(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device, seed=1)

    tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=0.99)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_unary_pow_test_inf(input_shapes, device):
    exponent = 2
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 9, device, seed=1)

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
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, -1, device, seed=1)

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
    "exponent",
    [(0.5), (0.0), (1.5), (4.5), (7.25), (10.0), (12.75), (16.0)],
)
def test_bw_unary_pow_output(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device, seed=1)
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

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=0.99)
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
    "exponent",
    [(0.5), (0.0), (1.5), (4.5), (7.25), (10.0), (12.75), (16.0)],
)
def test_bw_unary_pow_negative_inputs(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device, seed=1)
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

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=0.99)
    assert status
