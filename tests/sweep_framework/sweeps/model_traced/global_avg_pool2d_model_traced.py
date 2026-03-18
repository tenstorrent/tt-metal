# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    infer_mesh_shape_from_params,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("global_avg_pool2d")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if not mesh_shape:
        mesh_shape = infer_mesh_shape_from_params(model_traced_params)

    if mesh_shape:
        # Create mesh device based on env var
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
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
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # V2 format provides input_a_shape
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Global average pooling: average over spatial dimensions (H, W)
    # TTNN global_avg_pool2d preserves the last dimension (width/channels) from input
    # For input shape [B, C, H, W], output is [B, C, 1, W] but rounded up to tile-aligned size (multiple of 32)
    # We compute the expected output using the original input width, then slice TTNN output to match
    input_width = shape[-1]
    # Compute expected output with original width (true global avg pool)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor_a, (1, input_width))

    # Check if shard shape is tile-aligned (same fix as reshape)
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
    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                actual_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=actual_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=actual_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.global_avg_pool2d(input_tensor_a, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # TTNN output may be padded to tile-aligned width, slice to match expected shape
    if output_tensor.shape != torch_output_tensor.shape:
        # Slice the last dimension to match expected width
        output_tensor = output_tensor[..., : torch_output_tensor.shape[-1]]

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
