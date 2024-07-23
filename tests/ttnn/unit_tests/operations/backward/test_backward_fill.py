# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.backward.utility_funcs import (
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
@skip_for_wormhole_b0()
def test_bw_fill(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    tt_output_tensor_on_device = ttnn.fill_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.fill_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_all_close(tt_output_tensor_on_device, golden_tensor, atol=150, rtol=1e-6)
    assert comp_pass
