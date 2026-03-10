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
model_traced_params = loader.get_suite_parameters("interleaved_to_sharded_partial")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 64, 64)],
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
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
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
    arg1=None,
    arg2=None,
    arg3=None,
    arg4=None,
    arg5=None,
    arg6=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    """
    interleaved_to_sharded_partial: extract a slice from an interleaved tensor into sharded memory.

    Positional args from JSON:
        arg0: input tensor (interleaved)
        arg1: grid size [x, y]
        arg2: shard shape [h, w]
        arg3: num_slices
        arg4: slice_index
        arg5: TensorMemoryLayout (e.g., HEIGHT_SHARDED)
        arg6: ShardOrientation (e.g., ROW_MAJOR)
    """
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    # Parse positional args with defaults for sample suite
    grid_size = arg1 if arg1 is not None else [8, 8]
    shard_shape = arg2 if arg2 is not None else [32, 32]
    num_slices = int(arg3) if arg3 is not None else 2
    slice_index = int(arg4) if arg4 is not None else 0

    # Parse TensorMemoryLayout
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if arg5 is not None:
        if isinstance(arg5, dict):
            repr_str = arg5.get("repr", "")
            if "BLOCK_SHARDED" in repr_str:
                shard_scheme = ttnn.TensorMemoryLayout.BLOCK_SHARDED
            elif "WIDTH_SHARDED" in repr_str:
                shard_scheme = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif hasattr(arg5, "name"):
            shard_scheme = arg5

    # Parse ShardOrientation
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    if arg6 is not None:
        if isinstance(arg6, dict):
            repr_str = arg6.get("repr", "")
            if "COL_MAJOR" in repr_str:
                shard_orientation = ttnn.ShardOrientation.COL_MAJOR
        elif hasattr(arg6, "name"):
            shard_orientation = arg6

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Golden: extract the slice from the input tensor
    flat_tensor = torch_input_tensor_a.reshape(1, 1, -1, torch_input_tensor_a.shape[-1])
    slice_size = flat_tensor.shape[-2] // num_slices
    start = slice_index * slice_size
    stop = start + slice_size
    torch_output_tensor = flat_tensor[:, :, start:stop, :].clone()

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    grid_coord = ttnn.CoreCoord(grid_size[0], grid_size[1]) if isinstance(grid_size, (list, tuple)) else grid_size

    start_time = start_measuring_time()
    output_tensor = ttnn.interleaved_to_sharded_partial(
        input_tensor_a,
        grid_coord,
        shard_shape,
        num_slices,
        slice_index,
        shard_scheme,
        shard_orientation,
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
