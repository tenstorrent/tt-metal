# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from typing import Tuple, Optional
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 240

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 12, 32, 64)],  # Batch, heads, seq, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture for nlp_concat_heads operation.
    Using explicit DispatchCoreConfig to handle sharded memory configs.
    """
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_device(device)
    del device


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Check if test vector is valid.
    Returns (True, "reason") if invalid (should skip), (False, None) if valid.
    """
    input_a_memory_config = test_vector.get("input_a_memory_config")
    input_shape = test_vector.get("input_shape")

    # Skip HEIGHT_SHARDED config with large grid that causes TT_FATAL validation error
    # Error: "physical_width == physical_shard_width" check fails
    # Specific case: shape (8, 12, 384, 64) with HEIGHT_SHARDED on grid [0,0] to [5,7]
    # GitHub Issue: #35358

    # Handle both dict (from JSON) and ttnn.MemoryConfig object (during generation)
    if isinstance(input_a_memory_config, dict):
        # JSON format
        mem_layout = input_a_memory_config.get("data", {}).get("memory_layout")
        shard_spec = input_a_memory_config.get("data", {}).get("shard_spec", {})

        if mem_layout == "HEIGHT_SHARDED" and shard_spec:
            grid = shard_spec.get("grid", [])
            if grid and len(grid) > 0:
                end = grid[0].get("end", {})
                if end.get("x", 0) >= 5 and end.get("y", 0) >= 7:
                    return True, "HEIGHT_SHARDED with large grid: physical_width != shard_width (TT_FATAL)"

    elif hasattr(input_a_memory_config, "memory_layout") and hasattr(input_a_memory_config, "shard_spec"):
        # ttnn.MemoryConfig object format (during parameter generation)
        mem_layout_str = str(input_a_memory_config.memory_layout)

        if "HEIGHT_SHARDED" in mem_layout_str and input_a_memory_config.shard_spec:
            shard_spec = input_a_memory_config.shard_spec

            if hasattr(shard_spec, "grid"):
                grid = shard_spec.grid
                # grid is a CoreRangeSet, check its ranges
                if hasattr(grid, "ranges") and grid.ranges:
                    for core_range in grid.ranges():
                        # Check if end coordinates indicate large grid
                        if hasattr(core_range, "end"):
                            end = core_range.end
                            if hasattr(end, "x") and hasattr(end, "y"):
                                if end.x >= 5 and end.y >= 7:
                                    return (
                                        True,
                                        "HEIGHT_SHARDED with large grid: physical_width != shard_width (TT_FATAL)",
                                    )

    return False, None


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_concat_heads concatenates heads: [B, H, S, D] -> [B, 1, S, H*D]
    # So we need to compute the expected output shape
    if len(shape) == 4:
        batch, num_heads, seq_len, head_dim = shape
        expected_output_shape = (batch, 1, seq_len, num_heads * head_dim)
        # Reshape input to match expected output for comparison
        torch_output_tensor = (
            torch_input_tensor_a.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, seq_len, num_heads * head_dim)
            .unsqueeze(1)
        )
    else:
        # Fallback: just clone the input
        torch_output_tensor = torch_input_tensor_a.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.nlp_concat_heads(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
