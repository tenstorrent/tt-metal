# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, gen_rotary_embedding_spec
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_spec": gen_rotary_embedding_spec(
            input_shape_list=gen_shapes([1, 1, 32, 64], [6, 12, 256, 512], [1, 1, 32, 64], 16),
            cache_size_list=[random.randint(1, 2048) for i in range(8)],
        ),
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b/bfloat4_b requires TILE_LAYOUT!"
    if test_vector["input_spec"]["input_shape"][-1] % 64 != 0:
        return True, "Input X dimension (133) must be divisible by 64 for tiling"
    if test_vector["input_spec"]["token_idx"] and test_vector["input_spec"]["input_shape"][0] != 1:
        return True, "When passing token_idx, sequence length must be 1"
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

    input_shape, cache_size, token_idx = input_spec.values()
    seq_length, batch_size, num_heads, head_dim = input_shape

    sin_cos_cache_shape = [1, 1, cache_size, head_dim]

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape)
    torch_cos_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(sin_cos_cache_shape)
    torch_sin_cache_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(sin_cos_cache_shape)

    if token_idx:
        golden_function = partial(ttnn.get_golden_function(ttnn.experimental.rotary_embedding), token_idx=token_idx)
    else:

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=None):
            seq_len = x.shape[-2]
            if token_idx is None:
                cos = cos_cached[:, :, :seq_len, ...]
                sin = sin_cached[:, :, :seq_len, ...]
            else:
                cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
                sin = sin_cached[:, :, token_idx : token_idx + 1, ...]

            x_embed = (x * cos) + (rotate_half(x) * sin)
            return x_embed

        golden_function = apply_rotary_pos_emb

    torch_output_tensor = golden_function(torch_input_tensor, torch_cos_cache_tensor, torch_sin_cache_tensor)

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

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.rotary_embedding(
        input_tensor, cos_cache_tensor, sin_cache_tensor, token_idx, memory_config=output_memory_config
    )
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
