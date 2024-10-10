# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
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
def test_bw_cosh(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -9, 9, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device)

    tt_output_tensor_on_device = ttnn.cosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.cosh_bw)
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
def test_bw_cosh_inf(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 90, 95, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -7, 7, device)

    tt_output_tensor_on_device = ttnn.cosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.cosh_bw)
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
def test_bw_cosh_neg_inf(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -95, -89, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -7, 7, device)

    tt_output_tensor_on_device = ttnn.cosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.cosh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_bw_cosh_nan_test1(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 86, 89, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 35, 50, device)

    tt_output_tensor_on_device = ttnn.cosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.cosh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_bw_cosh_nan_test2(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 86, 89, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -50, -35, device)

    tt_output_tensor_on_device = ttnn.cosh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.cosh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
