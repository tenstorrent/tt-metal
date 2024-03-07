# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_unary_le(input_shapes, device):
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)
    tt_output_tensor_on_device = tt_lib.tensor.le_bw(grad_tensor)

    pyt_y = torch.zeros_like(grad_data)

    golden_tensor = [pyt_y]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
