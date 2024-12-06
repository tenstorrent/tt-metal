# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range_batch_norm,
    compare_results_batch_norm,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 32, 32])),
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([2, 1, 32, 32])),
        (torch.Size([2, 2, 32, 32])),
        (torch.Size([2, 3, 32, 32])),
        (torch.Size([3, 1, 32, 32])),
        (torch.Size([3, 2, 32, 32])),
        (torch.Size([3, 3, 32, 32])),
        (torch.Size([4, 1, 32, 32])),
        (torch.Size([4, 2, 32, 32])),
        (torch.Size([4, 3, 32, 32])),
        (torch.Size([4, 4, 32, 32])),
        (torch.Size([1, 1, 23, 23])),
        (torch.Size([1, 2, 23, 23])),
        (torch.Size([1, 3, 23, 23])),
        (torch.Size([2, 1, 23, 23])),
        (torch.Size([2, 2, 23, 23])),
        (torch.Size([2, 3, 23, 23])),
        (torch.Size([3, 1, 23, 23])),
        (torch.Size([3, 2, 23, 23])),
        (torch.Size([3, 3, 23, 23])),
        (torch.Size([4, 1, 23, 23])),
        (torch.Size([4, 2, 23, 23])),
        (torch.Size([4, 3, 23, 23])),
        (torch.Size([4, 4, 23, 23])),
        (torch.Size([1, 1, 64, 120])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 64, 120])),
        (torch.Size([2, 1, 64, 120])),
        (torch.Size([2, 2, 64, 120])),
        (torch.Size([2, 3, 64, 120])),
        (torch.Size([3, 1, 64, 120])),
        (torch.Size([3, 2, 64, 120])),
    ),
)
@pytest.mark.parametrize("training", [False])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("eps", [1.0, 0.0, 2.34, 1e-05])
def test_batch_norm(input_shapes, training, weight, bias, eps, device):
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, True, False)
    if not training:
        mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False, False)
        var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device, False, False)
    else:
        mean_data = None
        mean_tensor = None
        var_data = None
        var_tensor = None
    if weight:
        weight_data, weight_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False, False)
    else:
        weight_data = None
        weight_tensor = None
    if bias:
        bias_data, bias_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device, False, False)
    else:
        bias_data = None
        bias_tensor = None

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        weight=weight_tensor,
        bias=bias_tensor,
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    # ttnn.set_printoptions(profile="full")
    # print("TT result : ", tt_output, tt_output.shape)
    # torch.set_printoptions(precision=5, sci_mode=False)
    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
    )
    # print("Torch result : ",torch_result)
    comp_pass = compare_results_batch_norm([tt_output], [torch_result])
    assert comp_pass
