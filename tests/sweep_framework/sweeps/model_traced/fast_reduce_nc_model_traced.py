# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial

import torch
import ttnn
from ttnn import ShardTensor2dMesh

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader, dict_to_memory_config
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    infer_mesh_shape_from_params,
    detect_mesh_shape_from_hardware,
)

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("fast_reduce_nc")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 4, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
        "dims": [[0, 1]],  # Default dimensions for sample
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if not mesh_shape:
        mesh_shape = infer_mesh_shape_from_params(model_traced_params)
    if not mesh_shape:
        mesh_shape = detect_mesh_shape_from_hardware()
    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0)
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)


def _parse_mesh_shape(mesh_device_shape):
    if isinstance(mesh_device_shape, (list, tuple)):
        return tuple(int(x) for x in mesh_device_shape)
    if isinstance(mesh_device_shape, str):
        nums = re.findall(r"\d+", mesh_device_shape)
        if len(nums) >= 2:
            return tuple(int(x) for x in nums[:2])
    return None


def _parse_shard_dims_from_placement(tensor_placement):
    if not tensor_placement:
        return None
    placement = tensor_placement.get("placement", "")
    if isinstance(placement, list):
        placement = " ".join(str(p) for p in placement)
    dims = []
    for m in re.finditer(r"PlacementShard\((-?\d+)\)|PlacementReplicate", placement):
        if m.group(1) is not None:
            dims.append(int(m.group(1)))
        else:
            dims.append(None)
    return tuple(dims) if len(dims) == 2 else None


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
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

    dims = kwargs.get("dims", [0, 1])
    op_kwargs = build_op_kwargs(kwargs, exclude={"dims"}, output_memory_config=output_memory_config)

    traced_memory_config = memory_config
    if traced_memory_config == "__ABSENT__":
        traced_memory_config = None
    if isinstance(traced_memory_config, dict):
        traced_memory_config = dict_to_memory_config(traced_memory_config)
    if traced_memory_config is not None:
        op_kwargs["memory_config"] = traced_memory_config

    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    is_host = storage_type and "HOST" in str(storage_type)

    mesh_shape = None
    shard_dims = None
    if is_mesh_device and input_a_tensor_placement:
        mesh_shape = _parse_mesh_shape(input_a_tensor_placement.get("mesh_device_shape"))
        shard_dims = _parse_shard_dims_from_placement(input_a_tensor_placement)

    if is_mesh_device and mesh_shape:
        try:
            dev_shape = device.shape
            if callable(dev_shape):
                dev_shape = dev_shape()
            dev_rows, dev_cols = dev_shape[0], dev_shape[1]
        except Exception:
            dev_rows, dev_cols = 1, 1
        if dev_rows < mesh_shape[0] or dev_cols < mesh_shape[1]:
            return [
                (
                    False,
                    f"Device mesh ({dev_rows}x{dev_cols}) too small for traced mesh shape "
                    f"({mesh_shape[0]}x{mesh_shape[1]}). Set MESH_DEVICE_SHAPE or run on a larger device.",
                ),
                None,
            ]

    if shard_dims is not None and mesh_shape is not None:
        global_shape = list(shape)
        for axis_idx, sd in enumerate(shard_dims):
            if sd is not None:
                esd = sd if sd >= 0 else len(shape) + sd
                global_shape[esd] *= mesh_shape[axis_idx]

        torch_global = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(tuple(global_shape))

        dev0_slices = [slice(None)] * len(global_shape)
        for axis_idx, sd in enumerate(shard_dims):
            if sd is not None:
                esd = sd if sd >= 0 else len(shape) + sd
                dev0_slices[esd] = slice(0, shape[esd])
        torch_input_dev0 = torch_global[tuple(dev0_slices)]
        torch_output_tensor = torch.sum(torch_input_dev0, dim=tuple(dims), keepdim=True)

        input_tensor_a = ttnn.from_torch(
            torch_global,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=input_a_memory_config,
            mesh_mapper=ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
        )
    else:
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(shape)
        torch_output_tensor = torch.sum(torch_input_tensor_a, dim=tuple(dims), keepdim=True)

        if not is_host:
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
    output_tensor = ttnn.experimental.fast_reduce_nc(input_tensor_a, dims=dims, output=None, **op_kwargs)

    output_shape = list(shape)
    for dim in dims:
        output_shape[dim] = 1

    if is_mesh_device:
        device_tensors = ttnn.get_device_tensors(output_tensor)
        output_tensor = device_tensors[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(output_shape).to_torch()
    else:
        output_tensor = output_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).unpad_from_tile(output_shape).to_torch()
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
