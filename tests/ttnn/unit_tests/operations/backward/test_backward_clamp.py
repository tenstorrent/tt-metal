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
def test_bw_clamp(input_shapes, device):
    min = -10.0
    max = 10.0
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, -1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, -1, device, True)

    tt_output_tensor_on_device = ttnn.clamp_bw(grad_tensor, input_tensor, min=min, max=max)

    golden_function = ttnn.get_golden_function(ttnn.clamp_bw)
    golden_tensor = golden_function(grad_data, in_data, min, max)
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
@pytest.mark.parametrize("min", (-1.0, 1.0, 0.0, -0.5, -10.0, 10.0))
def test_bw_clamp_min(input_shapes, min, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, -1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, -1, device, True)

    tt_output_tensor_on_device = ttnn.clamp_bw(grad_tensor, input_tensor, min=min)

    golden_function = ttnn.get_golden_function(ttnn.clamp_bw)
    golden_tensor = golden_function(grad_data, in_data, min, None)

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
@pytest.mark.parametrize("max", (1.0, 0.5, 0.0, -1.0, 10.0))
def test_bw_clamp_max(input_shapes, max, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, -1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, -1, device, True)

    tt_output_tensor_on_device = ttnn.clamp_bw(grad_tensor, input_tensor, max=max)

    golden_function = ttnn.get_golden_function(ttnn.clamp_bw)
    golden_tensor = golden_function(grad_data, in_data, None, max)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
