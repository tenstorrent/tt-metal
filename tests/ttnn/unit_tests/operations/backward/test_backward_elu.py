# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "alpha",
    (
        1.0,
        -1.5,
        0.0,
        0.5,
        2.5,
    ),
)
def test_bw_elu(input_shapes, alpha, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device, True)

    tt_output_tensor_on_device = ttnn.elu_bw(grad_tensor, input_tensor, alpha=alpha)

    golden_function = ttnn.get_golden_function(ttnn.elu_bw)
    golden_tensor = golden_function(grad_data, in_data, alpha)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
