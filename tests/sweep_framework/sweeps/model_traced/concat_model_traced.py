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
    infer_mesh_shape_from_params,
    detect_mesh_shape_from_hardware,
)

# Import V2 master config loader and standalone helpers for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import (
    MasterConfigLoader,
    dict_to_memory_config,
    parse_dtype,
    parse_layout,
)


def _parse_shard_dims_from_placement(placements):
    """Extract (dim0, dim1) for ShardTensor2dMesh from a placements list.

    ``placements`` is a list like ['PlacementShard(2)', 'PlacementShard(1)']
    or ['PlacementReplicate', 'PlacementShard(3)'].

    Returns a tuple of (dim_or_None, dim_or_None) for a 2-entry list, else None.
    """
    if not placements:
        return None
    if isinstance(placements, list):
        placement_str = " ".join(str(p) for p in placements)
    else:
        placement_str = str(placements)
    dims = []
    for m in re.finditer(r"PlacementShard\((-?\d+)\)|PlacementReplicate", placement_str):
        dims.append(int(m.group(1)) if m.group(1) is not None else None)
    return tuple(dims) if len(dims) == 2 else None


def _get_placement_from_tensor_spec(tensor_spec):
    """Extract placement info from a raw tensor spec's mesh_device field.

    Returns (shard_dims, mesh_shape) or (None, None) if not available.
    shard_dims is a tuple like (2, 1) or (None, 3).
    mesh_shape is a tuple like (4, 8).
    """
    mesh_device = tensor_spec.get("mesh_device")
    if not mesh_device:
        return None, None
    placements = mesh_device.get("placements", [])
    mesh_shape = mesh_device.get("shape", mesh_device.get("distribution_shape"))
    if not mesh_shape or not placements:
        return None, None
    mesh_shape = tuple(mesh_shape)
    shard_dims = _parse_shard_dims_from_placement(placements)
    return shard_dims, mesh_shape


# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("concat")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "arg0": [
            [
                {"shape": (1, 1, 32, 16), "dtype": "ttnn.bfloat16", "layout": "ttnn.TILE_LAYOUT"},
                {"shape": (1, 1, 32, 8), "dtype": "ttnn.bfloat16", "layout": "ttnn.TILE_LAYOUT"},
            ]
        ],
        "dim": [3],
        "memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    if not mesh_shape:
        mesh_shape = detect_mesh_shape_from_hardware()

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
    arg0,  # List of tensor specs: [{"shape": ..., "dtype": ..., "layout": ..., ...}, ...]
    arg1=None,  # dim value (positional in JSON)
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    dim=None,  # dim as kwarg (fallback)
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Handle dim parameter - can be arg1 (positional) or dim (kwarg)
    dim_value = arg1 if arg1 is not None else dim
    if dim_value is None:
        dim_value = -1

    # Handle memory_config - prefer output_memory_config, fallback to memory_config
    mem_config = output_memory_config if output_memory_config is not None else memory_config
    mem_config = dict_to_memory_config(mem_config)

    is_mesh_device = hasattr(device, "get_num_devices")
    is_host = storage_type and "HOST" in str(storage_type)

    if not isinstance(arg0, list):
        raise ValueError(f"arg0 must be a list of tensor specs, got {type(arg0)}")

    # Determine actual device mesh shape for compatibility checks
    if is_mesh_device:
        try:
            dev_shape = device.shape
            actual_mesh = (dev_shape[0], dev_shape[1])
        except Exception:
            actual_mesh = (1, 1)
    else:
        actual_mesh = (1, 1)

    torch_tensors = []
    ttnn_tensors = []

    for i, tensor_spec in enumerate(arg0):
        shape = tensor_spec.get("original_shape", tensor_spec.get("shape"))
        dtype_str = tensor_spec.get("original_dtype", tensor_spec.get("dtype", "ttnn.bfloat16"))
        layout_str = tensor_spec.get("layout", "ttnn.TILE_LAYOUT")
        tensor_mem_config = tensor_spec.get("memory_config", mem_config)
        tensor_mem_config = dict_to_memory_config(tensor_mem_config)

        dtype = parse_dtype(dtype_str)
        layout = parse_layout(layout_str)

        per_device_shape = tuple(shape) if isinstance(shape, list) else shape

        # Generate torch tensor at per-device shape (golden reference uses this)
        torch_tensor = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(
            per_device_shape
        )
        torch_tensors.append(torch_tensor)

        if is_host:
            ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)
        elif is_mesh_device:
            shard_dims, traced_mesh = _get_placement_from_tensor_spec(tensor_spec)
            mesh_compatible = (
                traced_mesh is not None
                and len(traced_mesh) == 2
                and actual_mesh[0] >= traced_mesh[0]
                and actual_mesh[1] >= traced_mesh[1]
            )

            if shard_dims is not None and mesh_compatible:
                # Build global tensor: scale per-device shape by mesh size on each sharded axis.
                # When shard_dims=(None, None) (full 2D replicate) the shape is unchanged
                # but we still use ShardTensor2dMesh so the tracer records a 2D distribution.
                global_shape = list(per_device_shape)
                for axis_idx, sd in enumerate(shard_dims):
                    if sd is not None:
                        esd = sd if sd >= 0 else len(per_device_shape) + sd
                        global_shape[esd] *= traced_mesh[axis_idx]

                torch_global = torch_tensor.repeat(
                    *[global_shape[d] // per_device_shape[d] for d in range(len(per_device_shape))]
                )
                ttnn_tensor = ttnn.from_torch(
                    torch_global,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    memory_config=tensor_mem_config,
                    mesh_mapper=ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=traced_mesh),
                )
            else:
                ttnn_tensor = ttnn.from_torch(
                    torch_tensor,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    memory_config=tensor_mem_config,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
        else:
            ttnn_tensor = ttnn.from_torch(
                torch_tensor,
                dtype=dtype,
                layout=layout,
                device=device,
                memory_config=tensor_mem_config,
            )

        ttnn_tensors.append(ttnn_tensor)

    # Golden reference: per-device concat (each device runs independently)
    torch_output_tensor = torch.cat(torch_tensors, dim=dim_value)

    start_time = start_measuring_time()

    # Build op_kwargs incrementally — only include memory_config when non-None
    # to avoid tracing "memory_config": null when the model didn't pass it.
    op_kwargs = {}
    if mem_config is not None:
        op_kwargs["memory_config"] = mem_config

    output_tensor = ttnn.concat(ttnn_tensors, dim=dim_value, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
