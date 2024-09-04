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
    "threshold",
    [-2.0, -0.5, 0.0, 0.5, 2.0],
)
@pytest.mark.parametrize("value", [1.0])
def test_bw_threshold(input_shapes, threshold, value, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device, True)

    tt_output_tensor_on_device = ttnn.threshold_bw(grad_tensor, input_tensor, threshold, value)

    golden_function = ttnn.get_golden_function(ttnn.threshold_bw)
    golden_tensor = golden_function(grad_data, in_data, threshold, value)
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
