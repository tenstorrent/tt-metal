# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import json
import random
import torch
import ttnn

from tests.sweep_framework.sweep_utils.sharding_utils import (
    gen_sharded_spec_unary,
    parse_sharding_spec,
    invalidate_vector_sharding,
)
from tests.ttnn.utils_for_testing import (
    check_with_pcc,
    start_measuring_time,
    stop_measuring_time,
)
from models.common.utility_functions import torch_random


# Override default timeout (in seconds) for hang detection
TIMEOUT = 30

random.seed(42)

sharded_specs = gen_sharded_spec_unary(2, layouts=["TILE_LAYOUT"])
sharded_specs = random.sample(sharded_specs, 2)

# Parameter suite
parameters = {
    "nightly": {
        "input_spec": [None, *(json.dumps(sharded_spec) for sharded_spec in sharded_specs)],
        "input_dtype": [ttnn.bfloat16],
        "input_memory_config": [None, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "op_name": ["reciprocal", "log", "exp", "gelu", "tanh"],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_spec = test_vector["input_spec"]
    input_memory_config = test_vector["input_memory_config"]

    if input_spec is None:
        if input_memory_config is None:
            return True, "Must select either input_spec or input_memory_config"
        else:
            return False, None
    elif input_memory_config is not None:
        return True, "Cannot select both input_spec and input_memory_config"

    return invalidate_vector_sharding(json.loads(input_spec))


# Main run function
def run(
    input_spec,
    input_dtype,
    input_memory_config,
    op_name,
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
    ) = parse_sharding_spec(sharded_specs[0] if input_spec is None else json.loads(input_spec))

    # Define value ranges depending on op (for numerical stability)
    if op_name in ["reciprocal", "log"]:
        low, high = 0.1, 10.0  # avoid zeros/negatives
    else:
        low, high = -3.0, 3.0

    # Generate random input tensor
    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.float32)

    # Resolve TTNN and PyTorch ops dynamically
    ttnn_op = getattr(ttnn, op_name)
    torch_op = ttnn.get_golden_function(ttnn_op)

    # Compute golden reference
    torch_output_tensor = torch_op(torch_input_tensor)

    memory_config = (
        input_memory_config
        if input_spec is None
        else ttnn.create_sharded_memory_config_(
            shape=input_shape,
            core_grid=core_grid,
            strategy=sharding_strategy,
            orientation=shard_orientation,
            use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
            tile_layout=shard_height_mul_of_32,
        )
    )

    # Convert input to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=memory_config,
    )

    # Measure execution time
    start_time = start_measuring_time()
    output_tensor = ttnn_op(input_tensor, memory_config=memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Return correctness + perf
    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
