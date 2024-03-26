# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_all_close,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
# Pytorch Reference
# - name: fill.Tensor(Tensor self, Tensor value) -> Tensor
#   self: zeros_like(grad)
#   value: grad.sum()
#   result: at::fill(self_t, value_t)
def test_bw_fill(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    pyt_y = torch.zeros_like(grad_data)
    grad_sum = grad_data.sum()
    pyt_y.fill_(grad_sum)

    tt_output_tensor_on_device = tt_lib.tensor.fill_bw(grad_tensor)

    golden_tensor = [pyt_y]
    comp_pass = compare_all_close(tt_output_tensor_on_device, golden_tensor, atol=100, rtol=1e-6)
    assert comp_pass
