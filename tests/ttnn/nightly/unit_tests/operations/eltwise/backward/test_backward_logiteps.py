# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
    compare_pcc,
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
    "eps",
    (2, -0.001, 0.4, 0.5, 1.0),
)
def test_bw_logiteps(input_shapes, eps, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -2, 2, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    tt_output_tensor_on_device = ttnn.logiteps_bw(grad_tensor, input_tensor, eps=eps)
    golden_function = ttnn.get_golden_function(ttnn.logiteps_bw)
    golden_tensor = golden_function(grad_data, in_data, eps)
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
def test_bw_logiteps_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -2, 2, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    tt_output_tensor_on_device = ttnn.logiteps_bw(grad_tensor, input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.logiteps_bw)
    golden_tensor = golden_function(grad_data, in_data)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
