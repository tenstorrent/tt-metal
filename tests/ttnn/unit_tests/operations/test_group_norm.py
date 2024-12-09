# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_batch_norm,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        # (torch.Size([1, 3, 32, 32])),
        # (torch.Size([1, 1, 32, 32])),
        (torch.Size([2, 3, 32, 32])),
    ),
)
@pytest.mark.parametrize("training", [True])
@pytest.mark.parametrize("weight", [True])
@pytest.mark.parametrize("bias", [True])
def test_batch_norm(input_shapes, training, weight, bias, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 10, device, False)
    if not training:
        mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 1, 10, device, False)
        var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 10, 20, device, False)
    else:
        mean_data = None
        mean_tensor = None
        var_data = None
        var_tensor = None
    if weight:
        weight_data, weight_tensor = data_gen_with_range_batch_norm(input_shapes, 1, 10, device, False)
    else:
        weight_data = None
        weight_tensor = None
    if bias:
        bias_data, bias_tensor = data_gen_with_range_batch_norm(input_shapes, 1, 10, device, False)
    else:
        bias_data = None
        bias_tensor = None

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        3,
        # running_mean=mean_tensor,
        # running_var=var_tensor,
        gamma=weight_tensor,
        beta=bias_tensor,
    )
    print(tt_output_tensor_on_device)
    print("Mean in pytorch : ", in_data.mean(dim=(0, 2, 3)))
    return True
