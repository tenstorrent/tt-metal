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

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "xfail": {
        "input_shape": [[4, 256, 40, 40]],  # gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 16),
        "kH": [2],
        "kW": [2],
        "stride": [1],
        "padding": [0],
        "dilation": [1],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],  # ttnn.ROW_MAJOR_LAYOUT
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],  # ttnn.L1_MEMORY_CONFIG
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # input_shape = test_vector["input_shape"]

    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_a_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b/bfloat4_b requires TILE_LAYOUT!"

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    kH,
    kW,
    stride,
    padding,
    dilation,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if input_a_layout == ttnn.ROW_MAJOR_LAYOUT and input_shape[-3] % 2 == 1:
        input_shape[-3] += 1

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    # print(f"input_shape {input_shape} input_a_dtype {input_a_dtype} input_a_layout {input_a_layout}")

    torch_output_tensor = torch.nn.functional.max_pool2d(
        torch_input_tensor_a, (kH, kW), stride=(stride, stride), padding=padding, dilation=dilation
    )

    # The input tensor is expected to be in [NHW, C]
    [N, C, H, W] = input_shape
    torch_input_tensor_a = torch.permute(torch_input_tensor_a, (0, 2, 3, 1))
    torch_input_tensor_a = torch.reshape(torch_input_tensor_a, [1, 1, N * H * W, C])

    # print(f"bla {torch_input_tensor_a.shape}")

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.max_pool2d(
        input_tensor=input_tensor_a,
        batch_size=N,
        input_h=H,
        input_w=W,
        channels=C,
        kernel_size=[kH, kW],
        stride=[stride, stride],
        padding=[padding, padding],
        dilation=[dilation, dilation],
        memory_config=output_memory_config,
        applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )

    result = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # ttnn operates on channels-last tensors
    output_tensor = torch.permute(result, (0, 3, 1, 2))

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    print(f"pcc {pcc}")
    return [pcc, e2e_perf]


# Run sweeps locally
# from tests.sweep_framework.framework.permutations import *

# start_time = start_measuring_time()
# for suite in parameters.keys():
#     device_id = 0
#     device = ttnn.open_device(device_id=device_id)
#     suite_vectors = list(permutations(parameters[suite]))
#     print(len(suite_vectors))
#     for vector in suite_vectors:
#         invalidate_res = invalidate_vector(vector)
#         if invalidate_res[0]:
#             print(f"Invalidated: {invalidate_res[1]}")
#             continue
#         try:
#             passed, _ = run(**vector, device=device)
#             if passed[0] != True:
#                 print(passed)
#         except Exception as e:
#             print(e)

#     ttnn.close_device(device)

# e2e_perf = stop_measuring_time(start_time)
# print(f"time {e2e_perf / 1000000000}s")
