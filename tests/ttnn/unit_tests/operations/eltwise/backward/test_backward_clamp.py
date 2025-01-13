# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (-10.0, 10.0),
        (10.0, -10.0),
        (1, -1),
        (0, 0),
        (-1.0, None),
        (None, 1.0),
        (None, None),
        (-0.5, None),
        (None, -0.5),
        (1.0, 0.0),
        (0.0, 1.0),
        ("tensor", None),
        (None, "tensor"),
        ("tensor", "tensor"),
    ],
)
def test_unary_bw_clamp_ttnn(input_shapes, min_val, max_val, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, -1, device)
    if min_val == "tensor":
        min, min_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    elif min_val is None:
        min, min_tensor = None, None
    else:
        min, min_tensor = min_val, min_val

    if max_val == "tensor":
        max, max_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    elif max_val is None:
        max, max_tensor = None, None
    else:
        max, max_tensor = max_val, max_val

    if min is None and max is None:
        pytest.xfail("Only one of 'min' or 'max' can be None. Please provide one value")
    else:
        tt_output_tensor_on_device = ttnn.clamp_bw(grad_tensor, input_tensor, min_tensor, max_tensor)
        golden_function = ttnn.get_golden_function(ttnn.clamp_bw)
        golden_tensor = golden_function(grad_data, in_data, min, max)
        comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
        assert comp_pass
