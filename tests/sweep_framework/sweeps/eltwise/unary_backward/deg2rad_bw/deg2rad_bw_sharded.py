# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import json
import torch
import random
import ttnn
import math
from tests.sweep_framework.sweep_utils.utils import gen_shapes, sanitize_shape_rm
from tests.sweep_framework.sweep_utils.sharding_utils import (
    gen_sharded_spec_unary,
    parse_sharding_spec,
    invalidate_vector_sharding,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import assert_with_ulp, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 120

random.seed(0)


parameters = {
    "nightly": {
        "input_spec": gen_sharded_spec_unary(16, layouts=["TILE_LAYOUT"]),
        "input_a_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_layout = test_vector["input_spec"]["input_layout"]
    sharding_invalidated, output_str = invalidate_vector_sharding(test_vector["input_spec"])

    if input_layout == "ROW_MAJOR_LAYOUT":
        return True, "Input to eltwise binary must be tilized"
    if sharding_invalidated:
        return sharding_invalidated, output_str

    return False, None


def run(
    input_spec,
    input_a_dtype,
    *,
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

    sharded_config = ttnn.create_sharded_memory_config_(
        shape=input_shape,
        core_grid=core_grid,
        strategy=sharding_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
        tile_layout=shard_height_mul_of_32,
    )

    torch_grad_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    torch_input_tensor.requires_grad = True
    golden_function = ttnn.get_golden_function(ttnn.deg2rad_bw)
    torch_output_tensor = golden_function(torch_grad_tensor, torch_input_tensor)[0]

    grad_tensor = ttnn.from_torch(
        torch_grad_tensor,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=sharded_config,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_a_dtype,
        layout=input_layout,
        device=device,
        memory_config=sharded_config,
    )

    scale = 0.017453292519943295
    reference_tensor = ttnn.multiply(grad_tensor, scale, memory_config=sharded_config)

    start_time = start_measuring_time()
    output_tensor = ttnn.deg2rad_bw(grad_tensor, input_tensor, memory_config=sharded_config)[0]
    output_tensor_device = output_tensor
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(reference_tensor, output_tensor_device, ulp_threshold=4)
    assert_with_ulp(torch_output_tensor, output_tensor_device, ulp_threshold=4)
    return [True, e2e_perf]
