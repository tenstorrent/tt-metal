# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
    data_gen_with_range,
    data_gen_with_val,
)
from models.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "order",
    [1, 2, 3, 6, 7, 10],
)
def test_bw_polygamma(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)
    n = order

    tt_output_tensor_on_device = ttnn.polygamma_bw(grad_tensor, input_tensor, n)

    golden_function = ttnn.get_golden_function(ttnn.polygamma_bw)
    golden_tensor = golden_function(grad_data, in_data, n)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [1, 4, 7, 10],
)
def test_bw_polygamma_range_pos(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 30, device)
    n = order

    tt_output_tensor_on_device = ttnn.polygamma_bw(grad_tensor, input_tensor, n)

    golden_function = ttnn.get_golden_function(ttnn.polygamma_bw)
    golden_tensor = golden_function(grad_data, in_data, n)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# grad and input zero
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [2, 5],
)
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_bw_polygamma_zero(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, 0)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, True, 0)
    n = order

    tt_output_tensor_on_device = ttnn.polygamma_bw(grad_tensor, input_tensor, n)

    golden_function = ttnn.get_golden_function(ttnn.polygamma_bw)
    golden_tensor = golden_function(grad_data, in_data, n)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# grad zero
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [2, 5],
)
def test_bw_polygamma_grad_zero(input_shapes, order, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, True, 0)
    n = order

    tt_output_tensor_on_device = ttnn.polygamma_bw(grad_tensor, input_tensor, n)

    golden_function = ttnn.get_golden_function(ttnn.polygamma_bw)
    golden_tensor = golden_function(grad_data, in_data, n)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# input zero
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "order",
    [1, 2, 5],
)
def test_bw_polygamma_input_zero(input_shapes, order, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, 0)
    n = order

    tt_output_tensor_on_device = ttnn.polygamma_bw(grad_tensor, input_tensor, n)

    golden_function = ttnn.get_golden_function(ttnn.polygamma_bw)
    golden_tensor = golden_function(grad_data, in_data, n)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
