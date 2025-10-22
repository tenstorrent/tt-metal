# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import pytest
import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import sanitize_shape_rm
from tests.sweep_framework.sweep_utils.sharding_utils import (
    gen_sharded_spec_unary,
    parse_sharding_spec,
    invalidate_vector_sharding,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 120

random.seed(0)

# Parameters provided to the test vector generator,
# defined as dict-type suites that contain the arguments to the run function as keys,
# and lists of possible inputs as values.
parameters = {
    "xfail": {
        "input_spec": gen_sharded_spec_unary(16, max_tensor_size_per_core=20 * 1024, layouts=["TILE_LAYOUT"]),
        "input_a_dtype": [ttnn.bfloat16],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_layout = test_vector["input_spec"]["input_layout"]
    sharding_invalidated, output_str = invalidate_vector_sharding(test_vector["input_spec"])

    if input_layout == "ROW_MAJOR_LAYOUT":
        return True, "Inputs to eltwise binary must be tilized"
    if sharding_invalidated:
        return sharding_invalidated, output_str
    return False, None


# The actual test function.
def run_softmax_sharded(
    input_spec,
    input_a_dtype,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    (
        input_shape,
        core_grid,
        sharding_strategy,
        shard_orientation,
        tensor_hw_as_shard_shape,
        input_layout,
        shard_height_mul_of_32,
    ) = parse_sharding_spec(input_spec)

    if input_layout == ttnn.ROW_MAJOR_LAYOUT:
        input_shape = sanitize_shape_rm(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(input_shape)
    golden_function = ttnn.get_golden_function(ttnn.softmax)

    torch_output_tensor = golden_function(torch_input_tensor_a, dim=-1)

    sharded_config = ttnn.create_sharded_memory_config_(
        shape=input_shape,
        core_grid=core_grid,
        strategy=sharding_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
        tile_layout=shard_height_mul_of_32,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=sharded_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.softmax(input_tensor_a, dim=-1, memory_config=sharded_config)
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(output_tensor)

    expected_pcc = 0.999
    tensors = [input_tensor_a, output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


# Entry point for the sweep framework.
# Takes one test vector (as defined above) as the input.
def run(
    input_spec,
    input_a_dtype,
    *,
    device,
) -> list:
    return run_softmax_sharded(input_spec, input_a_dtype, device)


# Entry point for pytest.
@pytest.mark.xfail
@pytest.mark.parametrize("input_spec", parameters["xfail"]["input_spec"])
@pytest.mark.parametrize("input_a_dtype", parameters["xfail"]["input_a_dtype"])
def test_softmax_sharded(device, input_spec, input_a_dtype):
    run_softmax_sharded(input_spec, input_a_dtype, device)
