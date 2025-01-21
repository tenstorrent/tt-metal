# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
import json
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 360

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 4)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 4)
        + gen_shapes([1, 1], [256, 256], [1, 1], 4),
        "reduction": ["__none", "__mean", "__sum"],
        "input_reference_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_reference_layout": [ttnn.TILE_LAYOUT],
        "input_reference_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_prediction_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_prediction_layout": [ttnn.TILE_LAYOUT],
        "input_prediction_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    reduction,
    input_reference_dtype,
    input_reference_layout,
    input_reference_memory_config,
    input_prediction_dtype,
    input_prediction_layout,
    input_prediction_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_reference_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_reference_dtype
    )(input_shape)

    torch_input_prediction_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_prediction_dtype
    )(input_shape)

    reduction_0 = "none"
    reduction_1 = ttnn.LossReductionMode.NONE

    if reduction == "__mean":
        reduction_0 = "mean"
        reduction_1 = ttnn.LossReductionMode.MEAN

    if reduction == "__sum":
        reduction_0 = "sum"
        reduction_1 = ttnn.LossReductionMode.SUM

    golden_function = ttnn.get_golden_function(ttnn.mse_loss)

    torch_output_tensor = golden_function(
        torch_input_reference_tensor.to(torch.float32),
        torch_input_prediction_tensor.to(torch.float32),
        reduction=reduction_0,
    )

    input_reference_tensor = ttnn.from_torch(
        torch_input_reference_tensor,
        dtype=input_reference_dtype,
        layout=input_reference_layout,
        device=device,
        memory_config=input_reference_memory_config,
    )

    input_prediction_tensor = ttnn.from_torch(
        torch_input_prediction_tensor,
        dtype=input_prediction_dtype,
        layout=input_prediction_layout,
        device=device,
        memory_config=input_prediction_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.mse_loss(
        input_reference_tensor,
        input_prediction_tensor,
        reduction=reduction_1,
        output_tensor=None,
        memory_config=output_memory_config,
    )

    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
