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
    in_data, input_tensor = data_gen_with_range(input_shapes, 5, 10, device, False)
    # in_data = torch.ones(torch.Size([1, 3, 1, 1])).bfloat16() * 4
    # print("Input tensor : ",in_data)
    # padding = (0, 31, 0, 31 )
    # in_data = torch.nn.functional.pad(in_data, padding, mode='constant', value=0)
    # print("\n\nInput tensor on padding to [1,3,32,32]: ",in_data)
    # input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if not training:
        mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False)
        var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device, False)
    else:
        mean_data = None
        mean_tensor = None
        var_data = None
        var_tensor = None
    if weight:
        weight_data, weight_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False)
    else:
        weight_data = None
        weight_tensor = None
    if bias:
        bias_data, bias_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False)
    else:
        bias_data = None
        bias_tensor = None

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        1.0,
        # running_mean=mean_tensor,
        # running_var=var_tensor,
        gamma=weight_tensor,
        beta=bias_tensor,
    )
    print("TT GROUP NORM MEAN OUTPUT : ", tt_output_tensor_on_device[1])
    tt_mean_to_torch = ttnn.to_torch(tt_output_tensor_on_device[1]).to(torch.bfloat16)
    sliced_tensor = tt_mean_to_torch[:, :, 0, 0].unsqueeze(2).unsqueeze(3)
    print("\n\nSlicing the positions we need --> [1,3,1,1]", sliced_tensor)
    print("\n\nMean in pytorch - expected: ", in_data.mean(dim=(0, 2, 3)))
    return True
