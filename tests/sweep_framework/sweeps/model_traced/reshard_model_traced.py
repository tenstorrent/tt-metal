# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("reshard")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Invalidate test vectors with non-tile-aligned shard shapes.
    ttnn.reshard cannot work with tensors that have non-tile-aligned shard dimensions
    in either input or output.
    """
    # Check both input and output memory configs for non-tile-aligned shards
    for config_key in ["input_a_memory_config", "output_memory_config"]:
        mem_config = test_vector.get(config_key)

        if mem_config:
            # Check if it's a dict (during generation) or object (during execution)
            if isinstance(mem_config, dict):
                shard_spec = mem_config.get("data", {}).get("shard_spec")
                if shard_spec and "shape" in shard_spec:
                    shard_shape = shard_spec["shape"]
                    if len(shard_shape) >= 2:
                        height, width = shard_shape[-2], shard_shape[-1]
                        if height % 32 != 0 or width % 32 != 0:
                            return (
                                True,
                                f"{config_key} shard shape {shard_shape} not tile-aligned (must be divisible by 32)",
                            )
            elif hasattr(mem_config, "shard_spec") and mem_config.shard_spec:
                shard_spec = mem_config.shard_spec
                if hasattr(shard_spec, "shape"):
                    shard_shape = shard_spec.shape
                    if len(shard_shape) >= 2:
                        height, width = shard_shape[-2], shard_shape[-1]
                        if height % 32 != 0 or width % 32 != 0:
                            return True, f"{config_key} shard shape ({height}, {width}) not tile-aligned"

    return False, None


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
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # Single device (default)
        device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
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

    shape = input_a_shape if isinstance(input_a_shape, (tuple, list)) else (1, 1, 32, 32)

    # Tensor creation
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )
    torch_output = torch_input.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor = create_tensor_on_mesh(
                torch_input,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        input_tensor = ttnn.from_torch(torch_input, dtype=input_a_dtype, layout=input_a_layout)

    # Op call — reshard's second arg is the output memory config
    # V2 loader stores it as arg1 (raw dict) or input_b_memory_config (parsed)
    start_time = start_measuring_time()
    reshard_mem_config = output_memory_config or op_kwargs.pop("memory_config", None)
    if reshard_mem_config is None:
        # Try input_b_memory_config (V2 loader treats 2nd positional as tensor)
        reshard_mem_config = kwargs.get("input_b_memory_config")
    if reshard_mem_config is None:
        # Try arg1 as a raw memory config dict
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        arg1 = kwargs.get("arg1")
        if arg1 is not None and isinstance(arg1, dict):
            reshard_mem_config = parse_dict_value("memory_config", arg1)
    if reshard_mem_config is None:
        return [(False, "Missing output_memory_config for reshard"), 0.0]
    output_tensor = ttnn.reshard(input_tensor, reshard_mem_config, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
