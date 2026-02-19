# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import random
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("embedding")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 32)],  # (batch_size, seq_length)
        "input_a_dtype": [ttnn.uint32],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(128, 32)],  # (num_embeddings, embeddings_dim)
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "dtype": [ttnn.bfloat16],  # output dtype
        "memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # For embedding, check the input_a_layout (indices) and input_b_layout (weights)
    if test_vector.get("input_a_layout") == ttnn.TILE_LAYOUT:
        return True, "Input indices must be in row major layout"
    if test_vector.get("input_b_layout") == ttnn.TILE_LAYOUT:
        return True, "Weights must be in row major layout"
    if test_vector.get("dtype") == ttnn.bfloat8_b:
        return True, "bfloat8_b is not supported for output tensor"
    if (
        test_vector.get("input_b_layout") == ttnn.ROW_MAJOR_LAYOUT
        and test_vector.get("input_b_dtype") == ttnn.bfloat8_b
    ):
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if mesh_shape:
        # Create mesh device based on env var
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"⚠️ Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # Single device (default)
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,  # indices shape: (batch_size, seq_length)
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape,  # weights shape: (num_embeddings, embeddings_dim)
    input_b_dtype,
    input_b_layout,
    input_b_memory_config,
    dtype=None,  # output dtype
    memory_config=None,  # output memory_config
    storage_type="StorageType::DEVICE",
    layout=None,  # Additional layout parameter from JSON
    weight_shape=None,  # Alternative weight shape parameter
    weight_dtype=None,  # Alternative weight dtype parameter
    weight_layout=None,  # Alternative weight layout parameter
    weight_memory_config=None,  # Alternative weight memory_config parameter
    padding_idx=None,  # Padding index for embeddings
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method

    # V2 format provides separate shapes
    input_shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    weight_shape = tuple(input_b_shape) if isinstance(input_b_shape, (list, tuple)) else input_b_shape

    num_embeddings = weight_shape[0]

    # Generate input indices tensor (random integers in range [0, num_embeddings))
    torch_input_tensor = torch_random(input_shape, 0, num_embeddings, torch.int64)

    # Generate weight tensor
    torch_weight_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(weight_shape)

    golden_function = ttnn.get_golden_function(ttnn.embedding)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight_tensor).squeeze()

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Create input tensor (indices)
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor = create_tensor_on_mesh(
                torch_input_tensor,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            input_tensor = ttnn.from_torch(
                torch_input_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_a_dtype, layout=input_a_layout)

    # Create weight tensor
    if not is_host:
        if is_mesh_device and input_b_tensor_placement:
            # Use mesh with placement
            weight_tensor = create_tensor_on_mesh(
                torch_weight_tensor,
                device,
                input_b_dtype,
                input_b_layout,
                input_b_memory_config,
                input_b_tensor_placement,
            )
        else:
            # Regular single-device tensor
            weight_tensor = ttnn.from_torch(
                torch_weight_tensor,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=input_b_memory_config,
            )
    else:
        # Host storage
        weight_tensor = ttnn.from_torch(torch_weight_tensor, dtype=input_b_dtype, layout=input_b_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.embedding(input_tensor, weight_tensor, dtype=dtype, memory_config=memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None).squeeze()

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
