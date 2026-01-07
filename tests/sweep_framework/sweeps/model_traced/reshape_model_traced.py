# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
model_traced_params = loader.get_suite_parameters("reshape", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "target_shape": [(1, 32, 1, 32)],
        "storage_type": [
            "StorageType::DEVICE"
        ],  # NOTE: HOST storage does not work properly for reshape - always use DEVICE
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
    target_shape,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        input_shape_tuple = tuple(input_shape)
    else:
        input_shape_tuple = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape_tuple)

    torch_output_tensor = torch.reshape(torch_input_tensor_a, target_shape)

    # Check if shard shape is tile-aligned
    # If shard shape height or width is not divisible by tile size (32), use ROW_MAJOR layout
    actual_layout = input_a_layout
    shard_spec = None
    shard_shape = None

    # Handle both dict (from JSON) and MemoryConfig object
    if isinstance(input_a_memory_config, dict):
        # Memory config can be a dict with 'data' key containing the actual config
        data = input_a_memory_config.get("data", input_a_memory_config)
        shard_spec = data.get("shard_spec") if isinstance(data, dict) else None
        if shard_spec is not None and isinstance(shard_spec, dict):
            shard_shape = shard_spec.get("shape")
    elif hasattr(input_a_memory_config, "shard_spec"):
        # MemoryConfig object
        shard_spec = input_a_memory_config.shard_spec
        if shard_spec is not None:
            # ShardSpec object has shape attribute
            if hasattr(shard_spec, "shape"):
                shard_shape = shard_spec.shape

    if shard_shape is not None and isinstance(shard_shape, (list, tuple)) and len(shard_shape) >= 2:
        shard_height, shard_width = shard_shape[0], shard_shape[1]
        # Check if shard dimensions are tile-aligned (must be divisible by 32)
        if shard_height % 32 != 0 or shard_width % 32 != 0:
            # Shard shape is not tile-aligned, use ROW_MAJOR layout
            actual_layout = ttnn.ROW_MAJOR_LAYOUT

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    # NOTE: HOST storage does not work properly for reshape operation - always use DEVICE
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=actual_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.reshape(input_tensor_a, target_shape, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
