# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_topk_similarity
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 32, 64], [2, 6, 128, 128], [1, 1, 32, 64], 64)
        + gen_shapes([1, 1, 33, 65], [2, 6, 127, 129], [1, 1, 33, 63], 128)
        + gen_shapes([1, 1, 31, 63], [2, 6, 128, 128], [1, 1, 32, 64], 7)
        + gen_shapes([1, 32, 64], [12, 200, 1025], [1, 32, 64], 8)
        + gen_shapes([1, 32, 64], [12, 256, 1023], [1, 32, 164], 9)
        + gen_shapes([1, 7, 20], [12, 300, 1024], [1, 32, 64], 10)
        + gen_shapes([32, 64], [256, 1024], [32, 64], 8)
        + gen_shapes([32, 6], [256, 404], [32, 264], 18)
        + gen_shapes([32, 17], [256, 124], [32, 624], 28),
        "dim": [
            0,
            1,
            2,
            3,
            None,
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
        "largest": [True, False],
        "k": [32],  # small-k suite; large k is swept by the "large_k" suite below
        "input_a_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "large_k": {
        # Large-k coverage (64 < k <= 2048). On Blackhole + bfloat16 + last-dim +
        # largest these route through the ttnn.experimental.topk_large_indices
        # composite (indices from the op, values gathered back from the
        # original tensor — exact); everywhere else stock single-core would
        # take seconds-to-minutes per vector, so those combinations are
        # invalidated below rather than swept.
        "input_shape": gen_shapes([1, 1, 1, 8192], [1, 1, 4, 131072], [1, 1, 1, 8192], 20)
        + gen_shapes([1, 1, 1, 100000], [1, 1, 2, 100000], [1, 1, 1, 1], 2)
        + gen_shapes([1, 1, 32, 8192], [1, 1, 64, 65536], [1, 1, 32, 8192], 6)
        # Routed-envelope ceiling is 2^19: the value-recovery gather parks a
        # full W * 2 B input row-stick in L1, so wider rows fall back to the
        # stock single-core factory (seconds-to-minutes per vector here).
        + gen_shapes([1, 1, 4, 262144], [1, 1, 8, 524288], [1, 1, 4, 262144], 4),
        "dim": [-1],
        "largest": [True],
        "k": [96, 128, 256, 512, 1024, 2048],
        "input_a_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "xfail": {
        "input_shape": gen_shapes([1, 1, 32, 64], [6, 12, 256, 1024], [1, 1, 32, 64], 64)
        + gen_shapes([1, 32, 64], [12, 256, 1024], [1, 32, 64], 8)
        + gen_shapes([32, 64], [256, 1024], [32, 64], 8),
        "dim": [
            0,
            1,
            2,
            3,
            None,
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
        "largest": [True, False],
        "k": [32],
        "input_a_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if len(test_vector["input_shape"]) != 4:
        return True, "Input shape must be 4D"
    if test_vector["k"] > 64:
        # The large_k suite covers the Blackhole routed path
        # (ttnn.experimental.topk_large_indices). The standard sweep matrix
        # schedules every regular module on the wormhole-n150 lane
        # (compute_sweep_matrix.py: compute_standard_matrix) and has no
        # Blackhole lane, so these vectors are pre-staged as INVALID: on WH
        # the stock single-core factory takes seconds-to-minutes per vector
        # (recorded as hangs, triggering device resets). Routed-path CI
        # coverage lives in tests/ttnn/unit_tests/operations/reduce/
        # test_topk.py (Blackhole-gated). Flip this to arch-conditional
        # sub-checks when a Blackhole sweeps lane exists.
        return True, "large_k suite is Blackhole-only (routed path); no Blackhole sweep lane yet"
    if test_vector["dim"] != -1:
        return True, "Only the last dim is supported right now"
    if test_vector["dim"] * (-1) > (len(test_vector["input_shape"])):
        return True, "Absolute value of dim must be less or equal than the rank of input tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor"
    if test_vector["input_layout"] == ttnn.ROW_MAJOR_LAYOUT and not (
        test_vector["input_a_dtype"] == ttnn.float32 or test_vector["input_a_dtype"] == ttnn.bfloat16
    ):
        return True, "Row major is only supported for fp32 & fp16"

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    dim,
    largest,
    k,
    input_a_dtype,
    input_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    if k > 64:
        # The large_k suite exists to cover the Blackhole routed path
        # (ttnn.experimental.topk_large_indices). Elsewhere these vectors land
        # on the stock linear single-core factory (seconds-to-minutes per
        # vector against the 30s timeout, recorded as hangs). Generation-time
        # invalidation cannot see the target arch (the generator is offline),
        # so fail fast here instead.
        assert "blackhole" in ttnn.get_arch_name().lower(), "large_k suite is Blackhole-only (routed path)"
        # sanitize_shape rounds the last dim up to the next power of two,
        # which would silently retarget the non-pow2 vectors (e.g. W=100000)
        # onto already-covered pow2 widths. Non-pow2 width is one of the two
        # structural reasons the routed path exists — run the declared shape.
    else:
        input_shape = sanitize_shape(input_shape, "topk")

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function(ttnn.topk)
    torch_output_values, torch_output_indices = golden_function(
        torch_input_tensor_a, k, dim=dim, largest=largest, sorted=True
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_values, output_indices = ttnn.topk(input_tensor_a, k=k, dim=dim, largest=largest, sorted=True)
    e2e_perf = stop_measuring_time(start_time)

    output_values, output_indices = ttnn.to_torch(output_values), ttnn.to_torch(output_indices).to(torch.int64)
    output_gathered_values = torch.gather(torch_input_tensor_a, dim, output_indices)

    passing, output_str = comp_topk_similarity(
        [torch_output_values, torch_output_indices], [output_values, output_gathered_values]
    )

    return [(passing, output_str), e2e_perf]
