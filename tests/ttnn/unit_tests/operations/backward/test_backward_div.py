# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, data_gen_with_val, compare_pcc
from models.utility_functions import (
    skip_for_wormhole_b0,
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
    "round_mode",
    (
        None,
        "trunc",
        "floor",
    ),
)
def test_bw_div_binary(input_shapes, round_mode, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    golden_function = ttnn.get_golden_function(ttnn.div_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, round_mode)

    if round_mode == None:
        round_mode = "None"
    tt_output_tensor_on_device = ttnn.div_bw(grad_tensor, input_tensor, other_tensor, round_mode=round_mode)

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
def test_bw_div_binary_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    golden_function = ttnn.get_golden_function(ttnn.div_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    tt_output_tensor_on_device = ttnn.div_bw(grad_tensor, input_tensor, other_tensor)
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
    "round_mode",
    (
        "None",
        "trunc",
        "floor",
    ),
)
@pytest.mark.parametrize("scalar", [0.0])
@skip_for_wormhole_b0()
def test_bw_unary_div_0(input_shapes, scalar, round_mode, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, False, val=0)

    tt_output_tensor_on_device = ttnn.div_bw(grad_tensor, input_tensor, scalar, round_mode=round_mode)

    if round_mode == "None":
        round_mode = None
    golden_function = ttnn.get_golden_function(ttnn.div_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar, round_mode)

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
    "round_mode",
    (
        "None",
        "trunc",
        "floor",
    ),
)
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12, 0.0, -0.05, -1.0, -0.5, -0.12])
def test_bw_unary_div(input_shapes, scalar, round_mode, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    tt_output_tensor_on_device = ttnn.div_bw(grad_tensor, input_tensor, scalar, round_mode=round_mode)

    in_data.retain_grad()

    if round_mode == "None":
        round_mode = None
    golden_function = ttnn.get_golden_function(ttnn.div_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar, round_mode)

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
@pytest.mark.parametrize("scalar", [0.0])
@skip_for_wormhole_b0()
def test_bw_unary_div_0_default(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, False, val=0)

    tt_output_tensor_on_device = ttnn.div_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.div_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

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
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12, 0.0, -0.05, -1.0, -0.5, -0.12])
def test_bw_unary_div_default(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    tt_output_tensor_on_device = ttnn.div_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.div_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
