# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re

import torch
import ttnn
from ttnn import ShardTensor2dMesh

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, parse_dict_value

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("add")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(1, 1, 32, 32)],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def _parse_mesh_shape(mesh_device_shape):
    """Parse mesh_device_shape which may be a list, tuple, or string like '[4, 8]'."""
    if isinstance(mesh_device_shape, (list, tuple)):
        return tuple(int(x) for x in mesh_device_shape)
    if isinstance(mesh_device_shape, str):
        nums = re.findall(r"\d+", mesh_device_shape)
        if len(nums) >= 2:
            return tuple(int(x) for x in nums[:2])
    return None


def _parse_shard_dims_from_placement(tensor_placement):
    """Extract (dim0, dim1) for ShardTensor2dMesh from a traced tensor_placement dict.

    Returns a tuple like (2, 1) for PlacementShard dims, None for PlacementReplicate entries,
    or None if the field cannot be parsed.
    """
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


def _build_mesh_tensor(per_device_shape, device, dtype, layout, memory_config, tensor_placement, mesh_shape):
    """Create a TTNN tensor on mesh matching the model trace's placement.

    For sharded placements, builds a global tensor scaled by mesh dims and shards it
    with ShardTensor2dMesh.  For replicated placements, replicates to all devices.

    Returns (ttnn_tensor, torch_device0_data) where torch_device0_data is the data
    that device 0 sees (for golden reference computation).
    """
    shard_dims = _parse_shard_dims_from_placement(tensor_placement)

    if shard_dims is not None and any(d is not None for d in shard_dims):
        global_shape = list(per_device_shape)
        for axis_idx, sd in enumerate(shard_dims):
            if sd is not None:
                esd = sd if sd >= 0 else len(per_device_shape) + sd
                global_shape[esd] *= mesh_shape[axis_idx]

        torch_global = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(
            tuple(global_shape)
        )

        target_sharded = None
        create_mem = memory_config
        if memory_config is not None and hasattr(memory_config, "memory_layout"):
            if "SHARDED" in str(memory_config.memory_layout):
                target_sharded = memory_config
                create_mem = ttnn.DRAM_MEMORY_CONFIG

        tt_tensor = ttnn.from_torch(
            torch_global,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=create_mem,
            mesh_mapper=ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
        )

        if target_sharded is not None:
            tt_tensor = ttnn.to_memory_config(tt_tensor, target_sharded)

        ref_slices = [slice(None)] * len(global_shape)
        for axis_idx, sd in enumerate(shard_dims):
            if sd is not None:
                esd = sd if sd >= 0 else len(per_device_shape) + sd
                ref_slices[esd] = slice(0, per_device_shape[esd])
        torch_dev0 = torch_global[tuple(ref_slices)]

        return tt_tensor, torch_dev0

    torch_data = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(
        tuple(per_device_shape)
    )

    # On a 2D mesh with full-replicate placement, use ShardTensor2dMesh with
    # dims=(None, None) so the tracer records 2D-style placement metadata
    # (['PlacementReplicate', 'PlacementReplicate'] with distribution_shape=[R,C])
    # instead of the 1D ['PlacementReplicate'] from ReplicateTensorToMesh.
    is_2d_replicate = shard_dims is not None and len(shard_dims) == 2 and all(d is None for d in shard_dims)
    if is_2d_replicate:
        mesh_mapper = ShardTensor2dMesh(device, dims=(None, None), mesh_shape=mesh_shape)
    else:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)

    tt_tensor = ttnn.from_torch(
        torch_data,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )
    return tt_tensor, torch_data


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
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    scalar = kwargs.get("scalar", kwargs.get("arg1", None))
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    is_mesh_device = hasattr(device, "get_num_devices")

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else input_b_shape

    is_scalar_add = shape_b is None or scalar is not None
    is_host = storage_type and "HOST" in str(storage_type)

    mesh_shape = None
    if is_mesh_device and input_a_tensor_placement:
        mesh_shape = _parse_mesh_shape(input_a_tensor_placement.get("mesh_device_shape"))

    if not is_host and is_mesh_device and mesh_shape is not None:
        # ── Model-traced mesh path ──
        # Use ShardTensor2dMesh with shard dims from the model trace so the
        # per-tensor placement matches exactly.
        input_tensor_a, torch_ref_a = _build_mesh_tensor(
            shape_a,
            device,
            input_a_dtype,
            input_a_layout,
            input_a_memory_config,
            input_a_tensor_placement,
            mesh_shape,
        )

        if is_scalar_add:
            scalar_value = scalar if scalar is not None else 1.0
            torch_output_tensor = torch.add(torch_ref_a, scalar_value)
        else:
            input_tensor_b, torch_ref_b = _build_mesh_tensor(
                shape_b,
                device,
                input_b_dtype,
                input_b_layout,
                input_b_memory_config,
                input_b_tensor_placement,
                mesh_shape,
            )
            torch_output_tensor = torch.add(torch_ref_a, torch_ref_b)

    elif not is_host:
        # ── Single device path ──
        torch_input_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(shape_a)
        input_tensor_a = ttnn.from_torch(
            torch_input_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=input_a_memory_config,
        )

        if is_scalar_add:
            scalar_value = scalar if scalar is not None else 1.0
            torch_output_tensor = torch.add(torch_input_a, scalar_value)
        else:
            torch_input_b = gen_func_with_cast_tt(
                partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
            )(shape_b)
            input_tensor_b = ttnn.from_torch(
                torch_input_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=input_b_memory_config,
            )
            torch_output_tensor = torch.add(torch_input_a, torch_input_b)

    else:
        # ── Host path ──
        torch_input_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(shape_a)
        input_tensor_a = ttnn.from_torch(torch_input_a, dtype=input_a_dtype, layout=input_a_layout)

        if is_scalar_add:
            scalar_value = scalar if scalar is not None else 1.0
            torch_output_tensor = torch.add(torch_input_a, scalar_value)
        else:
            torch_input_b = gen_func_with_cast_tt(
                partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
            )(shape_b)
            input_tensor_b = ttnn.from_torch(torch_input_b, dtype=input_b_dtype, layout=input_b_layout)
            torch_output_tensor = torch.add(torch_input_a, torch_input_b)

    # Build op kwargs — use build_op_kwargs for most things, then re-add
    # memory_config which build_op_kwargs filters by default.
    op_kwargs = build_op_kwargs(kwargs, exclude={"scalar", "arg1"})
    memory_config_raw = kwargs.get("memory_config", None)
    if memory_config_raw is not None:
        op_kwargs["memory_config"] = parse_dict_value("memory_config", memory_config_raw)

    start_time = start_measuring_time()

    if is_scalar_add:
        scalar_value = scalar if scalar is not None else 1.0
        output_tensor = ttnn.add(input_tensor_a, scalar_value, **op_kwargs)
    else:
        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, **op_kwargs)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Trim tile padding to match expected shape
    output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
