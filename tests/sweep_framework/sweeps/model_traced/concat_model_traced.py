# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    get_model_traced_mesh_shape,
    create_mesh_device,
    mesh_tensor_to_torch,
    get_mesh_composer,
    reconcile_golden_to_actual,
)

# Import V2 master config loader and standalone helpers for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import (
    MasterConfigLoader,
    dict_to_memory_config,
    parse_dtype,
    parse_layout,
)


def _parse_shard_dims_from_placement(placements):
    """Extract (dim0, dim1) for ShardTensor2dMesh from a placements value.

    ``placements`` may be a Python list like ['PlacementShard(2)', 'PlacementShard(1)']
    or its JSON-serialized string form "['PlacementShard(2)', 'PlacementShard(1)']".

    Returns a tuple of (dim_or_None, dim_or_None) for a 2-entry list, else None.
    """
    if not placements:
        return None
    if isinstance(placements, list):
        placement_str = " ".join(str(p) for p in placements)
    else:
        placement_str = str(placements)
    dims = []
    for m in re.finditer(r"PlacementShard\((?:dim=)?(-?\d+)\)|PlacementReplicate", placement_str):
        dims.append(int(m.group(1)) if m.group(1) is not None else None)
    return tuple(dims) if len(dims) == 2 else None


def _parse_mesh_shape(mesh_device_shape):
    """Parse a mesh_device_shape value (list, tuple, or string like '[4, 8]') -> tuple."""
    if isinstance(mesh_device_shape, (list, tuple)):
        return tuple(int(x) for x in mesh_device_shape)
    if isinstance(mesh_device_shape, str):
        nums = re.findall(r"-?\d+", mesh_device_shape)
        if len(nums) >= 2:
            return tuple(int(x) for x in nums[:2])
    return None


def _get_placement_from_tensor_spec(tensor_spec):
    """Extract (shard_dims, mesh_shape) from a tensor_spec's tensor_placement field.

    The vector format stores tensor_placement = {
        'distribution_shape': '[4, 8]',
        'mesh_device_shape':  '[4, 8]',
        'placement':          "['PlacementShard(2)', 'PlacementShard(3)']",
    }
    """
    tensor_placement = tensor_spec.get("tensor_placement") if isinstance(tensor_spec, dict) else None
    if not tensor_placement:
        return None, None
    mesh_shape = _parse_mesh_shape(
        tensor_placement.get("mesh_device_shape") or tensor_placement.get("distribution_shape")
    )
    shard_dims = _parse_shard_dims_from_placement(tensor_placement.get("placement"))
    if mesh_shape is None or shard_dims is None:
        return None, None
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
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)


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
    # Use arg0[0]'s tensor_placement to drive the mesh composer (all inputs share
    # the same placement for concat in the traced configs we've seen).
    primary_placement = arg0[0].get("tensor_placement") if isinstance(arg0[0], dict) else None
    mesh_composer = get_mesh_composer(device, primary_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, primary_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
