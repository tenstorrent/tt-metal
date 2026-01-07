# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from math import prod
from typing import Optional, Tuple

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import (
    device_context,
    get_mem_configs,
    get_serializable_shard_specs,
    mesh_shape_iterator,
    validate_serializable_shard_spec,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 45

NUM_DEVICES = ttnn.get_num_devices()

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("experimental::all_gather_async", all_cases=False)

FABRIC_CONFIGS_1D = [
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricConfig.FABRIC_1D_RING,
]

FABRIC_CONFIGS_2D = [
    ttnn.FabricConfig.FABRIC_2D,
    ttnn.FabricConfig.FABRIC_2D_DYNAMIC,
]

FABRIC_CONFIGS = FABRIC_CONFIGS_1D + FABRIC_CONFIGS_2D

LEAD_MODEL_SHARD_SPECS = [
    get_serializable_shard_specs(
        input_shape=(32, 32),
        input_cores=(1, 1),
        input_strategy="w",
        output_shape=None,  # (32, 128) in production on Galaxy
        output_cores=(1, 1),
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 32]],
    ),
    get_serializable_shard_specs(
        input_shape=(32, 128),
        input_cores=(2, 4),
        input_strategy="h",
        output_shape=None,  # (32, 128) in production on Galaxy
        output_cores=(2, 4),
        output_strategy="h",
        valid_tensor_shapes=[[1, 8, 8, 128]],
    ),
]


GENERALITY_PARAMETERS = {
    "mesh_shape": list(mesh_shape_iterator(NUM_DEVICES)),
    "fabric_config": FABRIC_CONFIGS,
    "num_links": [1],
    "input_shape": [
        [1, 1, 32, 32],
        [1, 1, 32, 31],
        [1, 1, 1, 32, 32],
        [2, 32, 32],
        [1, 1, 32, 16384],
        [1, 1, 1, 2048],
    ],
    "dim": [0, 1, 2, 3, 4],
    "cluster_axis": [0, 1, None],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "input_dtype": [ttnn.bfloat16],
    "buffer_type": [ttnn.BufferType.DRAM],
    "shard_specs": [None],
    "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
    "num_iters": [1],
}

parameters = {
    "generality_suite": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS},
    "generality_suite_fabric_1d": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS_1D},
    "generality_suite_fabric_2d": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS_2D},
    "lead_model_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 1440],  # GPT-OSS 20B. Dim: 3, cluster_axis 1
            [1, 1, 32, 32],  # Qwen3, Llama on Glx, DeepSeek dim:3 cluster_axis: 1
            [1, 8, 8, 128],  # Qwen3, Llama on Glx dim:3 cluster_axis: 1
            [3, 1, 4096, 192],  # Gemma3 Dim: 3
            [3, 1, 4096, 144],  # Gemma3 Dim: 3
            [1, 1, 32, 896],  # DeepSeek dim:3 cluster_axis 1
            [1, 1, 32, 192],  # DeepSeek dim:3 cluster_axis 1
            [1, 1, 32, 576],  # DeepSeek dim: 1 cluster_axis 1
            [1, 1, 32, 224],  # DeepSeek dim:3 cluster_axis 0
            [1, 4, 128, 512],  # DeepSeek dim: 1 cluster_axis 1
        ],
        "dim": [1, 3],
        "cluster_axis": [0, 1],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
        "shard_specs": [None] + LEAD_MODEL_SHARD_SPECS,
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params and any(len(v) > 0 for v in model_traced_params.values() if isinstance(v, list)):
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # Check if this is a model_traced vector (has input_a_memory_config instead of buffer_type)
    is_model_traced = "input_a_memory_config" in test_vector

    if is_model_traced:
        # For model_traced vectors, we don't have mesh_shape, dim, cluster_axis, etc.
        # These are hardcoded in the run function, so we can skip validation
        # Just check basic shape validity
        input_shape = test_vector.get("input_shape")
        if input_shape and isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 0:
                return True, "Empty input shape"
        return False, None

    # Original validation for generality/lead_model suites
    # L1 sharding only
    if test_vector["shard_specs"] is not None and test_vector["buffer_type"] == ttnn.BufferType.DRAM:
        return True, "L1 Sharding only"

    cluster_axis = test_vector["cluster_axis"]
    mesh_shape = test_vector["mesh_shape"]
    input_shape = test_vector["input_shape"]
    dim = test_vector["dim"]
    cluster_size = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)

    if not validate_serializable_shard_spec(input_shape, test_vector["shard_specs"], dim, cluster_size, "gather"):
        return True, "Invalid shard spec"

    # hardcode for 6U
    if mesh_shape in [(16, 2), (2, 16)]:
        return True, "Invalid mesh shape for 6U"

    if cluster_axis is not None and test_vector["mesh_shape"][cluster_axis] == 1:
        return True, "Only one device along axis"

    if dim >= len(input_shape):
        return True, "Dim greater than rank"
    if (
        test_vector["topology"] == ttnn.Topology.Ring
        and test_vector["fabric_config"] != ttnn.FabricConfig.FABRIC_1D_RING
    ):
        return True, "Ring fabric config required for ring topology"

    return False, None


# dummy device fixture so we can sweep over device parameters as part of the test body
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, dtype, buffer_type, shard_specs, layout, device):
    torch_input = torch.rand(input_shape).bfloat16()

    replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
    torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

    input_memory_config, output_memory_config = get_mem_configs(buffer_type, shard_specs, layout, torch_reference.shape)

    assert input_memory_config.memory_layout == output_memory_config.memory_layout

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
    )

    return tt_input, torch_reference, output_memory_config


def run(
    mesh_shape=None,
    fabric_config=None,
    input_shape=None,
    dim=None,
    cluster_axis=None,
    num_links=None,
    input_dtype=None,
    layout=None,
    buffer_type=None,
    shard_specs=None,
    num_iters=None,
    topology=None,
    # Model traced parameters
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    output_memory_config=None,
    *,
    device,  # unused
    **kwargs,
) -> list:
    # Check if this is a model_traced run (has input_a_memory_config)
    is_model_traced = input_a_memory_config is not None

    if is_model_traced:
        # Model traced format - use defaults for multi-device setup
        if NUM_DEVICES < 2:
            logger.warning("Skipping all_gather_async test: requires multi-device setup (2+ devices)")
            return [1.0, 0.0]

        # Use defaults for model_traced
        mesh_shape = (2, 1)
        fabric_config = ttnn.FabricConfig.FABRIC_1D
        dim = 3  # Default dim
        cluster_axis = 0  # Default cluster_axis
        num_links = 1
        num_iters = 1
        topology = ttnn.Topology.Linear

        # Convert model_traced parameters to generality format
        input_dtype = input_a_dtype
        layout = input_a_layout

        # Create reference output
        replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
        torch_input = torch.rand(input_shape).bfloat16()
        torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

        # Use provided memory configs directly
        input_memory_config = input_a_memory_config

        # Ensure output_memory_config is a MemoryConfig object
        # It might come as a string from JSON serialization, so parse it if needed
        if output_memory_config is None:
            raise ValueError("output_memory_config is None - required parameter missing")
        elif isinstance(output_memory_config, str):
            # Parse the string representation back to a MemoryConfig
            import json
            import ast

            # Try to parse as dict (might be a string representation of dict)
            mem_config_dict = ast.literal_eval(output_memory_config)
            if not isinstance(mem_config_dict, dict):
                raise ValueError(
                    f"Failed to parse output_memory_config string: expected dict, got {type(mem_config_dict)}"
                )
            # Use the loader's parse_memory_config to convert dict to MemoryConfig
            from tests.sweep_framework.master_config_loader import MasterConfigLoader

            loader = MasterConfigLoader()
            # Output shape is input shape with width doubled
            output_shape = input_shape.copy() if input_shape else []
            if len(output_shape) >= 4:
                output_shape[3] = output_shape[3] * 2
            output_memory_config = loader.parse_memory_config(mem_config_dict, output_shape)
        elif isinstance(output_memory_config, dict):
            # It's a dict, convert to MemoryConfig
            from tests.sweep_framework.master_config_loader import MasterConfigLoader

            loader = MasterConfigLoader()
            output_shape = input_shape.copy() if input_shape else []
            if len(output_shape) >= 4:
                output_shape[3] = output_shape[3] * 2
            output_memory_config = loader.parse_memory_config(output_memory_config, output_shape)
        elif not isinstance(output_memory_config, ttnn.MemoryConfig):
            raise ValueError(f"output_memory_config is not a MemoryConfig (type: {type(output_memory_config)})")
    else:
        # Original generality/lead_model format
        # Create reference output
        replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
        torch_input = torch.rand(input_shape).bfloat16()
        torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

        # Get memory configs from buffer_type and shard_specs
        input_memory_config, output_memory_config = get_mem_configs(
            buffer_type, shard_specs, layout, torch_reference.shape
        )

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None

        if is_model_traced:
            # Create input tensor directly with provided memory config
            tt_input = ttnn.from_torch(
                torch_input,
                layout=layout,
                memory_config=input_memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                device=device,
            )
        else:
            # Use _get_tensors helper for generality format
            tt_input, torch_reference, output_memory_config = _get_tensors(
                input_shape,
                mesh_shape,
                dim,
                cluster_axis,
                input_dtype,
                buffer_type,
                shard_specs,
                layout,
                device,
            )

        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        semaphores = [ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(2)]

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()
                # Use exact same signature as test_all_gather_config.py which works correctly
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    tt_input,
                    dim=dim,
                    cluster_axis=cluster_axis,
                    mesh_device=device,
                    topology=topology,
                    multi_device_global_semaphore=semaphores,  # List of semaphores
                    num_links=num_links,
                    memory_config=output_memory_config,
                )
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                raise RuntimeError(f"Execution failed: {e}")

        for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
            tt_output_tensor = ttnn.to_torch(t)

            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_output_tensor, torch_reference)
            else:
                eq, output = comp_pcc(tt_output_tensor, torch_reference)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
            return [(eq, output), e2e_perf]
