# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial, product
import random
from typing import Generator, Optional, Tuple

import torch

from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
import ttnn

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


def _generate_spec(
    shape_list: list[Tuple[int, int, int, int]],
    num_q_heads: Iterable[int],
    num_kv_heads_range: Iterable[Optional[int]],
    overlap_qk_coregrid: list[bool],
) -> Generator[dict, None, None]:
    for shape, nh, nkv, ovlp in product(shape_list, num_q_heads, num_kv_heads_range, overlap_qk_coregrid):
        yield {"input_tensor": shape, "num_heads": nh, "num_kv_heads": nkv, "overlap_qk_coregrid": ovlp}


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
# parameters = {
#     "nightly": {
#         "input_spec": _generate_spec()
#         "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
#         "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
#         "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
#         "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
#     },
# }

parameters = {
    "nightly": {
        "input_spec": {
            "input_tensor_shape": (1, 1, 32, 128),
            "num_heads": 32,
            "num_kv_heads": None,
            "overlap_qk_coregrid": True,
        },
        "input_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_MEMORY_CONFIG],
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

    input_shape, num_heads, num_kv_heads, overlap_qk_coregrid = input_spec.values()
    seq_length, batch_size, num_heads, head_dim = input_shape

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.nlp_create_qkv_heads_decode(
        input_tensor,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        overlap_qk_coregrid=overlap_qk_coregrid,
        memory_config=output_memory_config,
    )
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [(True, ""), e2e_perf]
