# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial, product
import random
from typing import Optional, Tuple

import torch

from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
import ttnn

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30
random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

TILE_WIDTH = 32  # ???


def _gen_spec(input_shapes, cache_sizes, decode_mode=(False, True)) -> dict:
    for shape, csz, decode_mode in product(input_shapes, cache_sizes, decode_mode):
        yield {"input_shape": shape, "cache_size": csz, "decode_mode": decode_mode}


parameters = {
    "nightly": {
        "input_spec": _gen_spec(
            [(1, 1, 32, 128)],
            [32],
            [True],
            # gen_shapes([1, 1, 32, 64], [1, 1, 32, 128], [1, 1, 32, 64]),
            # [random.randint(1, 2048) for i in range(8)],
        ),
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_spec,
    input_dtype,
    input_layout,
    input_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    input_shape, cache_size, decode = input_spec.values()
    seq_length, batch_size, num_heads, head_dim = input_shape

    sin_cos_cache_shape = (1, 1, cache_size, head_dim)
    transform_shape = (1, 1, cache_size, TILE_WIDTH)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape)
    torch_cos_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(sin_cos_cache_shape)
    torch_sin_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(sin_cos_cache_shape)
    torch_transform_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(transform_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )
    cos_cache_tensor = ttnn.from_torch(
        torch_cos_cache_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )
    sin_cache_tensor = ttnn.from_torch(
        torch_sin_cache_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )
    transform_tensor = ttnn.from_torch(
        torch_sin_cache_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.rotary_embedding(
        input_tensor,
        cos_cache_tensor,
        sin_cache_tensor,
        transform_tensor,
        is_decode_mode=decode,
        memory_config=output_memory_config,
    )
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [(True, ""), e2e_perf]
