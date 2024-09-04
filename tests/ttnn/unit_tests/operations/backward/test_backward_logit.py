# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import (
    compare_pcc,
    data_gen_with_range,
    data_gen_with_val,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_logit(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -2, 2, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    tt_output_tensor_on_device = ttnn.logit_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.logit_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
