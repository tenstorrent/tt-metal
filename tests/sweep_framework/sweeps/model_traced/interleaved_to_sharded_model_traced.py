# Model traced sweep for interleaved_to_sharded
# Generated automatically - DO NOT EDIT MANUALLY

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("interleaved_to_sharded", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # For interleaved_to_sharded, the output is the same tensor but in sharded memory layout
    torch_output_tensor = torch_input_tensor_a.clone()

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # The traced operation expects the output to be sharded
    # Use a working sharding configuration like the unit tests (DRAM, reasonable core range)

    # Create core range (use 4 cores like working unit tests)
    num_cores = 4
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))
    core_range_set = ttnn.CoreRangeSet({core_range})

    # Calculate shard shape - must be divisible by tile size (32 for TILE_LAYOUT)
    tile_size = 32
    total_height = shape[-2]
    total_width = shape[-1]

    # Shard height: divide total height by number of cores, ensure tile alignment
    shard_height = (total_height // num_cores) // tile_size * tile_size
    if shard_height == 0:
        shard_height = tile_size  # Minimum tile size

    # Shard width: full width, ensure tile alignment
    shard_width = (total_width // tile_size) * tile_size
    if shard_width == 0:
        shard_width = tile_size

    shard_shape = [shard_height, shard_width]

    shard_spec = ttnn.ShardSpec(core_range_set, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

    # Use DRAM for sharding (like working unit tests)
    sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,  # Use DRAM instead of L1
        shard_spec=shard_spec,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.interleaved_to_sharded(input_tensor_a, sharded_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - should be identical since it's just a memory layout change
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
