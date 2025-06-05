# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30


def run_sum(
    input_shape,
    dim,
    keepdim,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if input_a_dtype == ttnn.float32 and ttnn.device.is_grayskull(device):
        return [(False, "Dest Fp32 mode is not supported for arch grayskull"), 0]

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    dim = dim % len(input_shape)

    torch_output_tensor = torch.sum(torch_input_tensor_a, dim=dim, keepdim=keepdim)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    op_output_tensor = ttnn.sum(input_tensor_a, dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor_a, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


def run_prod(
    input_shape,
    dim,
    keepdim,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if input_a_dtype == ttnn.float32 and ttnn.device.is_grayskull(device):
        return [(False, "Dest Fp32 mode is not supported for arch grayskull"), 0]

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    dim = dim % len(input_shape)

    torch_output_tensor = torch.prod(torch_input_tensor_a, dim=dim, keepdim=keepdim)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    op_output_tensor = ttnn.prod(input_tensor_a, dim=dim, keepdim=keepdim, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor_a, op_output_tensor]
    assert len(output_tensor.shape) == len(torch_output_tensor.shape)
    assert output_tensor.shape == torch_output_tensor.shape
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)
