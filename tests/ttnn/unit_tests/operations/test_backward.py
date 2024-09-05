# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from models.utility_functions import is_wormhole_b0, is_blackhole
from tests.ttnn.unit_tests.operations.backward.utility_funcs import (
    data_gen_with_val,
    compare_all_close,
)


def run_backward_unary_test(device, h, w, in_val, grad_val, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor, input_tensor = data_gen_with_val([1, 1, h, w], device, True, val=in_val)
    torch_grad_tensor, grad_tensor = data_gen_with_val([1, 1, h, w], device, val=grad_val)

    torch_output_tensor = torch_function(torch_input_tensor)

    output_tensor = ttnn_function(grad_tensor, input_tensor)

    torch_input_tensor.retain_grad()

    torch_output_tensor.backward(gradient=torch_grad_tensor)

    golden_tensor = [torch_input_tensor.grad]

    comp_pass = compare_all_close(output_tensor, golden_tensor, atol=0.01)

    assert comp_pass


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 0, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
def test_atan(device, h, w, in_val, grad_val):
    run_backward_unary_test(device, h, w, in_val, grad_val, ttnn.atan_bw, torch.atan)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 0, 1])
@pytest.mark.parametrize("grad_val", [1])
def test_atanh(device, h, w, in_val, grad_val):
    run_backward_unary_test(device, h, w, in_val, grad_val, ttnn.atanh_bw, torch.atanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 0, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Skipped due to hardware restriction in storing nan")
def test_atanh_nan(device, h, w, in_val, grad_val):
    run_backward_unary_test(device, h, w, in_val, grad_val, ttnn.atanh_bw, torch.atanh)


def run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor, input_tensor = data_gen_with_val([1, 1, h, w], device, True, val=in_val)
    torch_other_tensor, other_tensor = data_gen_with_val([1, 1, h, w], device, True, val=other_val)
    torch_grad_tensor, grad_tensor = data_gen_with_val([1, 1, h, w], device, val=grad_val)

    torch_output_tensor = torch_function(torch_input_tensor, torch_other_tensor)

    output_tensor = ttnn_function(grad_tensor, input_tensor, other_tensor)

    torch_input_tensor.retain_grad()
    torch_other_tensor.retain_grad()

    torch_output_tensor.backward(gradient=torch_grad_tensor)

    golden_tensor = [torch_input_tensor.grad, torch_other_tensor.grad]

    comp_pass = compare_all_close(output_tensor, golden_tensor, atol=0.01)

    assert comp_pass


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_atan2(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.atan2_bw, torch.atan2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [0])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [0])
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Skipped due to hardware restriction in storing nan")
def test_atan2_zero(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.atan2_bw, torch.atan2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Skipped due to hardware restriction in storing nan")
def test_xlogy(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.xlogy_bw, torch.xlogy)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_hypot(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.hypot_bw, torch.hypot)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_ldexp(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.ldexp_bw, torch.ldexp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_logaddexp(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.logaddexp_bw, torch.logaddexp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_logaddexp2(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.logaddexp2_bw, torch.logaddexp2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_squared_difference(device, h, w, in_val, grad_val, other_val):
    torch_squared_diff = lambda x, y: torch.square(torch.sub(x, y))
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.squared_difference_bw, torch_squared_diff)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_min(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.min_bw, torch.min)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("in_val", [-1, 1])
@pytest.mark.parametrize("grad_val", [-1, 0, 1])
@pytest.mark.parametrize("other_val", [-1, 1])
def test_max(device, h, w, in_val, grad_val, other_val):
    run_backward_binary_test(device, h, w, in_val, grad_val, other_val, ttnn.max_bw, torch.max)
