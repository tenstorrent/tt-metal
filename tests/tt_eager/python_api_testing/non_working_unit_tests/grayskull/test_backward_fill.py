# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([8, 17, 160, 32])),
    ),
)
# Pytorch Reference
# - name: fill.Tensor(Tensor self, Tensor value) -> Tensor
#   self: zeros_like(grad)
#   value: grad.sum()
#   result: at::fill(self_t, value_t)
def test_bw_fill(input_shapes, device):
    # torch.manual_seed(12386)
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)
    pyt_y = torch.zeros_like(grad_data)
    grad_sum = grad_data.sum()
    pyt_y.fill_(grad_sum)

    tt_output_tensor_on_device = ttnn.fill_bw(grad_tensor)

    golden_tensor = [pyt_y]
    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
