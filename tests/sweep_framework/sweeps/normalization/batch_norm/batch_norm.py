# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import itertools
import ttnn
import pytest
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_batch_norm

TIMEOUT = 30

random.seed(0)

parameters = {
    "BN_Testing": {
        "input_shape": gen_shapes([1, 1, 32, 32], [6, 12, 256, 256], [2, 4, 64, 64], 4),
        "input_dtype": [ttnn.bfloat16, ttnn.float32],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "training": [True, False],
        "check_mean": [True, False],
        "check_var": [True, False],
        "weight": [True, False],
        "bias": [True, False],
        "eps": [1.0, 0.0, 2.34, 1e-05],
        "momentum": [0.0, 0.1, 0.5],
    },
}


def run_batch_norm(
    input_shape,
    input_dtype,
    input_layout,
    input_memory_config,
    training,
    check_mean,
    check_var,
    weight,
    bias,
    eps,
    momentum,
    device,
):
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    dtype_names = "float32"
    if input_dtype == ttnn.bfloat16:
        dtype_names = "bfloat16"

    in_data, input_tensor = data_gen_with_range_batch_norm(
        input_shape, 5, 10, device, is_input=True, testing_dtype=dtype_names, memory_config=input_memory_config
    )

    if input_dtype == ttnn.float32 and ttnn.device.is_grayskull(device):
        return [(False, "Dest Fp32 mode is not supported for arch grayskull"), 0]

    mean_data, mean_tensor = (
        data_gen_with_range_batch_norm(
            input_shape, 4, 10, device, testing_dtype=dtype_names, memory_config=input_memory_config
        )
        if (check_mean or (not training))
        else (None, None)
    )
    var_data, var_tensor = (
        data_gen_with_range_batch_norm(
            input_shape, 4, 20, device, testing_dtype=dtype_names, memory_config=input_memory_config
        )
        if (check_var or (not training))
        else (None, None)
    )
    weight_data, weight_tensor = (
        data_gen_with_range_batch_norm(
            input_shape, 4, 10, device, testing_dtype=dtype_names, memory_config=input_memory_config
        )
        if weight
        else (None, None)
    )
    bias_data, bias_tensor = (
        data_gen_with_range_batch_norm(
            input_shape, 4, 10, device, testing_dtype=dtype_names, memory_config=input_memory_config
        )
        if bias
        else (None, None)
    )

    start_time = start_measuring_time()
    result = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        weight=weight_tensor,
        bias=bias_tensor,
        momentum=momentum,
        memory_config=input_memory_config,
    )
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
        momentum=momentum,
    )

    expected_pcc = 0.99
    if training:
        channels = input_shape[1]
        if check_mean:
            tt_updated_mean = ttnn.to_torch(mean_tensor)
            passed_, output_string_ = check_with_pcc(tt_updated_mean, mean_data.view(1, channels, 1, 1), expected_pcc)
            if not passed_:
                return [(passed_, output_string_), e2e_perf]
        if check_var:
            tt_updated_var = ttnn.to_torch(var_tensor)
            passed_, output_string_ = check_with_pcc(tt_updated_var, var_data.view(1, channels, 1, 1), expected_pcc)
            if not passed_:
                return [(passed_, output_string_), e2e_perf]

    tensors = [input_tensor, result]
    if mean_tensor is not None:
        tensors.append(mean_tensor)
    if var_tensor is not None:
        tensors.append(var_tensor)
    if weight_tensor is not None:
        tensors.append(weight_tensor)
    if bias_tensor is not None:
        tensors.append(bias_tensor)
    return get_run_return(torch_result, output_tensor, expected_pcc, tensors, e2e_perf)


def run(
    input_shape,
    input_dtype,
    input_layout,
    input_memory_config,
    training,
    check_mean,
    check_var,
    weight,
    bias,
    eps,
    momentum,
    *,
    device,
) -> list:
    return run_batch_norm(
        input_shape,
        input_dtype,
        input_layout,
        input_memory_config,
        training,
        check_mean,
        check_var,
        weight,
        bias,
        eps,
        momentum,
        device,
    )


param_keys = parameters["BN_Testing"].keys()
param_values = itertools.product(*parameters["BN_Testing"].values())


@pytest.mark.parametrize(",".join(param_keys), list(param_values))
def test_batch_norm(
    input_shape,
    input_dtype,
    input_layout,
    input_memory_config,
    training,
    check_mean,
    check_var,
    weight,
    bias,
    eps,
    momentum,
    device,
):
    run_batch_norm(
        input_shape,
        input_dtype,
        input_layout,
        input_memory_config,
        training,
        check_mean,
        check_var,
        weight,
        bias,
        eps,
        momentum,
        device,
    )
