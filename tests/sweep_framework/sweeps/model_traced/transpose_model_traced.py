# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args, parse_dict_value

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transpose")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 32, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "dim0": [0],
        "dim1": [1],
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
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def _is_sharded(memory_config):
    """Check if a memory config uses sharded memory layout."""
    if memory_config is None:
        return False
    if hasattr(memory_config, "is_sharded"):
        return memory_config.is_sharded()
    return False


def _shard_grid_fits_device(memory_config, device):
    """Check if a sharded memory config's shard grid fits within the device's compute grid."""
    if not _is_sharded(memory_config):
        return True
    shard_spec = getattr(memory_config, "shard_spec", None)
    if shard_spec is None:
        return True
    num_shards = shard_spec.grid.num_cores()
    compute_grid = device.compute_with_storage_grid_size()
    max_cores = compute_grid.x * compute_grid.y
    return num_shards <= max_cores


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    dim0=None,
    dim1=None,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2"}, output_memory_config=output_memory_config)

    pos_args = extract_positional_args(kwargs)
    dim0 = dim0 or pos_args.get(1, 0)
    dim1 = dim1 or pos_args.get(2, 1)
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    # Pass output memory_config to ttnn.transpose — without it, transpose inherits
    # the input's sharded memory_config which may become invalid after transposing
    # dimensions (e.g., shard grid may exceed available cores for the output shape).
    # Always provide an explicit output memory_config for sharded inputs to avoid this.
    if output_memory_config is not None and "memory_config" not in op_kwargs:
        parsed_mc = parse_dict_value("memory_config", output_memory_config)
        if parsed_mc is not None:
            # If the parsed output memory config is sharded but its shard grid exceeds
            # the device's compute grid (e.g., traced on a 32-core device but running
            # on a 4-core Galaxy node), fall back to DRAM interleaved.
            if _is_sharded(parsed_mc) and not _shard_grid_fits_device(parsed_mc, device):
                op_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
            else:
                op_kwargs["memory_config"] = parsed_mc
    elif _is_sharded(input_a_memory_config) and "memory_config" not in op_kwargs:
        # Sharded input but no explicit output memory_config — use DRAM interleaved
        # to avoid the transposed output inheriting an incompatible shard spec.
        op_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    torch_output_tensor = torch.transpose(torch_input_tensor_a, dim0, dim1)

    is_host = storage_type and "HOST" in str(storage_type)

    # Create tensor using interleaved→sharded for sharded memory configs.
    # Direct from_torch with sharded config triggers TilizeDeviceOperation which
    # can clash with L1 circular buffers.
    # If input shard grid exceeds device cores, fall back to DRAM interleaved
    if _is_sharded(input_a_memory_config) and not _shard_grid_fits_device(input_a_memory_config, device):
        input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG

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
        elif _is_sharded(input_a_memory_config):
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_a, input_a_memory_config)
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

    start_time = start_measuring_time()
    output_tensor = ttnn.transpose(input_tensor_a, dim0, dim1, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output_tensor.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
