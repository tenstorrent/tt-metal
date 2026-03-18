# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("sharded_to_interleaved_partial")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if mesh_shape:
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
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    arg2=None,
    arg3=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    """
    sharded_to_interleaved_partial: write a sharded slice into an interleaved output buffer.

    Positional args from JSON:
        arg0: input sharded slice tensor
        arg1: cache/output buffer tensor (interleaved)
        arg2: num_slices
        arg3: slice_index
        memory_config: output memory config
    """
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    num_slices = int(arg2) if arg2 is not None else 2
    slice_index = int(arg3) if arg3 is not None else 0

    # Generate input slice tensor
    torch_input_slice = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Generate output buffer tensor (cache)
    if input_b_shape is not None:
        shape_b = tuple(input_b_shape) if isinstance(input_b_shape, (tuple, list)) else input_b_shape
    else:
        # Default: output is num_slices * input height
        shape_b = list(shape_a)
        shape_b[-2] = shape_b[-2] * num_slices
        shape_b = tuple(shape_b)

    cache_dtype = input_b_dtype if input_b_dtype else input_a_dtype
    cache_layout = input_b_layout if input_b_layout else input_a_layout
    cache_mem = input_b_memory_config if input_b_memory_config else ttnn.DRAM_MEMORY_CONFIG

    torch_cache = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), cache_dtype)(
        shape_b
    )

    # Golden: write slice into cache at the correct position
    golden_cache = torch_cache.clone()
    flat_cache = golden_cache.reshape(1, 1, -1, golden_cache.shape[-1])
    slice_size = flat_cache.shape[-2] // num_slices
    start = slice_index * slice_size
    stop = start + slice_size
    flat_slice = torch_input_slice.reshape(1, 1, -1, torch_input_slice.shape[-1])
    flat_cache[:, :, start:stop, :] = flat_slice
    torch_output_tensor = flat_cache.reshape(golden_cache.shape)

    is_host = storage_type and "HOST" in str(storage_type)

    # Create slice tensor on device
    input_is_sharded = hasattr(input_a_memory_config, "is_sharded") and input_a_memory_config.is_sharded()
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_slice,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        elif input_is_sharded:
            input_tensor_a = ttnn.from_torch(
                torch_input_slice,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_a, input_a_memory_config)
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_slice,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_slice, dtype=input_a_dtype, layout=input_a_layout)

    # Create cache tensor on device (always interleaved)
    if not is_host:
        if is_mesh_device and input_b_tensor_placement:
            cache_tensor = create_tensor_on_mesh(
                torch_cache,
                device,
                cache_dtype,
                cache_layout,
                cache_mem,
                input_b_tensor_placement,
            )
        else:
            cache_tensor = ttnn.from_torch(
                torch_cache,
                dtype=cache_dtype,
                layout=cache_layout,
                device=device,
                memory_config=cache_mem,
            )
    else:
        cache_tensor = ttnn.from_torch(torch_cache, dtype=cache_dtype, layout=cache_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.sharded_to_interleaved_partial(
        input_tensor_a,
        cache_tensor,
        num_slices,
        slice_index,
        **op_kwargs,
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
