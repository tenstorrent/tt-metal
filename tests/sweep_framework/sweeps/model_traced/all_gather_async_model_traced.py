# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import re
from math import prod
from typing import Optional, Tuple

import torch
import ttnn
from ttnn import ShardTensor2dMesh

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.sweep_framework.sweep_utils.ccl_common import (
    device_context,
    get_mem_configs,
    get_serializable_shard_specs,
    mesh_shape_iterator,
    validate_serializable_shard_spec,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

NUM_DEVICES = ttnn.get_num_devices()

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("ttnn.experimental.all_gather_async")

FABRIC_CONFIGS_1D = [
    ttnn.FabricConfig.FABRIC_1D,
    ttnn.FabricConfig.FABRIC_1D_RING,
]

FABRIC_CONFIGS_2D = [
    ttnn.FabricConfig.FABRIC_2D,
]

FABRIC_CONFIGS = FABRIC_CONFIGS_1D + FABRIC_CONFIGS_2D

LEAD_MODEL_SHARD_SPECS = [
    get_serializable_shard_specs(
        input_shape=(32, 32),
        input_cores=(1, 1),
        input_strategy="w",
        output_shape=None,  # (32, 128) in production on Galaxy
        output_cores=(1, 1),
        output_strategy="w",
        valid_tensor_shapes=[[1, 1, 32, 32]],
    ),
    get_serializable_shard_specs(
        input_shape=(32, 128),
        input_cores=(2, 4),
        input_strategy="h",
        output_shape=None,  # (32, 128) in production on Galaxy
        output_cores=(2, 4),
        output_strategy="h",
        valid_tensor_shapes=[[1, 8, 8, 128]],
    ),
]


def _parse_mesh_shape(mesh_device_shape):
    """Parse mesh_device_shape which may be a list, tuple, or string like '[4, 8]'.

    Returns a tuple of ints, or None if unparseable.
    """
    if isinstance(mesh_device_shape, (list, tuple)):
        return tuple(int(x) for x in mesh_device_shape)
    if isinstance(mesh_device_shape, str):
        nums = re.findall(r"\d+", mesh_device_shape)
        if len(nums) >= 2:
            return tuple(int(x) for x in nums[:2])
    return None


def _parse_shard_dims_from_placement(tensor_placement):
    """Extract (dim0, dim1) for ShardTensor2dMesh from a traced tensor_placement dict.

    ``placement`` may be a Python list of strings (runtime) or a single
    string representation of that list (from JSON).  Examples::

        ['PlacementShard(2)', 'PlacementShard(1)']   -> (2, 1)
        ['PlacementReplicate', 'PlacementShard(3)']  -> (None, 3)

    Returns None when the field cannot be parsed.
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


GENERALITY_PARAMETERS = {
    "mesh_shape": list(mesh_shape_iterator(NUM_DEVICES)),
    "fabric_config": FABRIC_CONFIGS,
    "num_links": [1],
    "input_shape": [
        [1, 1, 32, 32],
        [1, 1, 32, 31],
        [1, 1, 1, 32, 32],
        [2, 32, 32],
        [1, 1, 32, 16384],
        [1, 1, 1, 2048],
    ],
    "dim": [0, 1, 2, 3, 4],
    "cluster_axis": [0, 1, None],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "input_dtype": [ttnn.bfloat16],
    "buffer_type": [ttnn.BufferType.DRAM],
    "shard_specs": [None],
    "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
    "num_iters": [1],
}

parameters = {
    "generality_suite": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS},
    "generality_suite_fabric_1d": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS_1D},
    "generality_suite_fabric_2d": GENERALITY_PARAMETERS | {"fabric_config": FABRIC_CONFIGS_2D},
    "lead_model_suite": {
        "mesh_shape": mesh_shape_iterator(NUM_DEVICES),
        "fabric_config": FABRIC_CONFIGS,
        "num_links": [1],
        "input_shape": [
            [1, 1, 32, 1440],  # GPT-OSS 20B. Dim: 3, cluster_axis 1
            [1, 1, 32, 32],  # Qwen3, Llama on Glx, DeepSeek dim:3 cluster_axis: 1
            [1, 8, 8, 128],  # Qwen3, Llama on Glx dim:3 cluster_axis: 1
            [3, 1, 4096, 192],  # Gemma3 Dim: 3
            [3, 1, 4096, 144],  # Gemma3 Dim: 3
            [1, 1, 32, 896],  # DeepSeek dim:3 cluster_axis 1
            [1, 1, 32, 192],  # DeepSeek dim:3 cluster_axis 1
            [1, 1, 32, 576],  # DeepSeek dim: 1 cluster_axis 1
            [1, 1, 32, 224],  # DeepSeek dim:3 cluster_axis 0
            [1, 4, 128, 512],  # DeepSeek dim: 1 cluster_axis 1
        ],
        "dim": [1, 3],
        "cluster_axis": [0, 1],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
        "shard_specs": [None] + LEAD_MODEL_SHARD_SPECS,
        "topology": [ttnn.Topology.Linear, ttnn.Topology.Ring],
        "num_iters": [1],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params and any(len(v) > 0 for v in model_traced_params.values() if isinstance(v, list)):
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # Check if this is a model_traced vector (has input_a_memory_config instead of buffer_type)
    is_model_traced = "input_a_memory_config" in test_vector

    if is_model_traced:
        # Model traced vectors are pre-validated by the tracer.
        # Do NOT check device count here — vector generation may run on a
        # smaller machine (e.g., N150) than the actual test runner (Galaxy).
        input_shape = test_vector.get("input_shape")
        if input_shape and isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 0:
                return True, "Empty input shape"
        return False, None

    # Original validation for generality/lead_model suites
    # L1 sharding only
    shard_specs = test_vector.get("shard_specs")
    buffer_type = test_vector.get("buffer_type")
    if shard_specs is not None and buffer_type == ttnn.BufferType.DRAM:
        return True, "L1 Sharding only"

    cluster_axis = test_vector.get("cluster_axis")
    mesh_shape = test_vector.get("mesh_shape")
    input_shape = test_vector.get("input_shape")
    dim = test_vector.get("dim")

    # If any required field is missing, skip validation (shouldn't happen for generality/lead suites)
    if None in [cluster_axis, mesh_shape, input_shape, dim]:
        return False, None

    cluster_size = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)

    if not validate_serializable_shard_spec(input_shape, shard_specs, dim, cluster_size, "gather"):
        return True, "Invalid shard spec"

    # hardcode for 6U
    if mesh_shape in [(16, 2), (2, 16)]:
        return True, "Invalid mesh shape for 6U"

    if cluster_axis is not None and mesh_shape[cluster_axis] == 1:
        return True, "Only one device along axis"

    if dim >= len(input_shape):
        return True, "Dim greater than rank"

    topology = test_vector.get("topology")
    fabric_config = test_vector.get("fabric_config")
    if topology == ttnn.Topology.Ring and fabric_config != ttnn.FabricConfig.FABRIC_1D_RING:
        return True, "Ring fabric config required for ring topology"

    return False, None


# dummy device fixture so we can sweep over device parameters as part of the test body
def mesh_device_fixture():
    yield None, "Device creation in sweep body"


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, dtype, buffer_type, shard_specs, layout, device):
    torch_input = torch.rand(input_shape).bfloat16()

    replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
    torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

    input_memory_config, output_memory_config = get_mem_configs(buffer_type, shard_specs, layout, torch_reference.shape)

    assert input_memory_config.memory_layout == output_memory_config.memory_layout

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        device=device,
    )

    return tt_input, torch_reference, output_memory_config


def run(
    mesh_shape=None,
    fabric_config=None,
    input_shape=None,
    dim=None,
    cluster_axis=None,
    num_links=None,
    input_dtype=None,
    layout=None,
    buffer_type=None,
    shard_specs=None,
    num_iters=None,
    topology=None,
    # Model traced parameters (V2 format)
    input_a_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    input_a_tensor_placement=None,
    memory_config=None,  # output memory_config
    persistent_output_buffer=None,
    multi_device_global_semaphore=None,  # From traced config (ignored, we create fresh)
    barrier_semaphore=None,  # From traced config (ignored, we create fresh)
    mesh_device=None,  # From traced config (ignored, we use device param)
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    subdevice_id=None,
    use_broadcast=None,
    *,
    device,  # unused
    **kwargs,
) -> list:
    is_mesh_device = hasattr(device, "get_num_devices")

    # Check if this is a model_traced run (V2 format has input_a_shape)
    is_model_traced = input_a_shape is not None

    if is_model_traced:
        if NUM_DEVICES < 2:
            logger.warning("Skipping all_gather_async test: requires multi-device setup (2+ devices)")
            return [(True, "Skipped: requires 2+ devices"), 0.0]

        input_shape = input_a_shape
        input_dtype = input_a_dtype
        layout = input_a_layout
        input_memory_config = input_a_memory_config

        if isinstance(memory_config, dict):
            mem_layout_str = memory_config.get("memory_layout", "")
            buffer_type_str = memory_config.get("buffer_type", "")

            if buffer_type_str == "BufferType.DRAM" or buffer_type_str == "DRAM":
                buffer_type_enum = ttnn.BufferType.DRAM
            elif buffer_type_str == "BufferType.L1" or buffer_type_str == "L1":
                buffer_type_enum = ttnn.BufferType.L1
            else:
                buffer_type_enum = ttnn.BufferType.DRAM

            if "SHARDED" in str(mem_layout_str):
                output_memory_config = ttnn.DRAM_MEMORY_CONFIG
            elif "INTERLEAVED" in str(mem_layout_str):
                output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type_enum)
            else:
                output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            output_memory_config = memory_config

        # All op kwargs (memory_config, topology, etc.) are passed explicitly to
        # all_gather_async below. Don't use build_op_kwargs to avoid leaking extra
        # traced keys (use_broadcast, subdevice_id) that conflict or are invalid.

        if num_links is None:
            num_links = 1
        if num_iters is None:
            num_iters = 1
        if topology is None:
            topology = ttnn.Topology.Linear

        if input_a_tensor_placement:
            mesh_shape = _parse_mesh_shape(input_a_tensor_placement.get("mesh_device_shape"))
            if mesh_shape is None:
                mesh_shape = (1, 2)
        else:
            mesh_shape = (1, 2)

        if dim is None:
            raise ValueError("dim is required for all_gather_async")

        # Resolve negative dim for internal calculations only; keep original
        # dim value so the op call matches the model trace exactly.
        effective_dim = dim if dim >= 0 else len(input_shape) + dim

        if cluster_axis is None:
            if mesh_shape[0] == 1 or mesh_shape[1] == 1:
                cluster_axis = 0 if mesh_shape[0] > 1 else 1
            else:
                cluster_axis = 1 if effective_dim > 1 else 0

        if mesh_shape[0] == 1 or mesh_shape[1] == 1:
            fabric_config = ttnn.FabricConfig.FABRIC_1D
        else:
            fabric_config = ttnn.FabricConfig.FABRIC_2D

        replicate_dim = mesh_shape[cluster_axis]
        is_2d_mesh = mesh_shape[0] > 1 and mesh_shape[1] > 1

        # Parse the model's actual tensor placement for 2D meshes so the
        # sweep input matches the model's distribution exactly.
        shard_dims = _parse_shard_dims_from_placement(input_a_tensor_placement) if is_2d_mesh else None

        if shard_dims is not None:
            # Build a global tensor whose shape accounts for sharding on
            # ALL mesh axes.  input_shape is the per-device shape, so each
            # sharded dim must be scaled by the number of devices on that axis.
            global_shape = list(input_shape)
            for axis_idx, sd in enumerate(shard_dims):
                if sd is not None:
                    esd = sd if sd >= 0 else len(input_shape) + sd
                    global_shape[esd] *= mesh_shape[axis_idx]

            torch_global = torch.rand(global_shape).bfloat16()

            # Golden reference for device 0 after all_gather: the gather dim
            # is fully present but non-gathered sharded dims remain split.
            ref_slices = [slice(None)] * len(global_shape)
            for axis_idx, sd in enumerate(shard_dims):
                if sd is not None and axis_idx != cluster_axis:
                    esd = sd if sd >= 0 else len(input_shape) + sd
                    ref_slices[esd] = slice(0, input_shape[esd])

            torch_reference = torch_global[tuple(ref_slices)]
            torch_input = torch_global
        else:
            # 1D mesh or unparseable placement: shard only along gather dim.
            output_shape = list(input_shape)
            output_shape[effective_dim] = input_shape[effective_dim] * replicate_dim
            torch_reference = torch.rand(output_shape).bfloat16()
            torch_input = torch_reference
    else:
        # Original generality/lead_model format
        # Create reference output
        replicate_dim = mesh_shape[cluster_axis] if cluster_axis is not None else prod(mesh_shape)
        torch_input = torch.rand(input_shape).bfloat16()
        torch_reference = torch_input.repeat(tuple((1 if i != dim else replicate_dim) for i in range(len(input_shape))))

        # Get memory configs from buffer_type and shard_specs
        input_memory_config, output_memory_config = get_mem_configs(
            buffer_type, shard_specs, layout, torch_reference.shape
        )

    with device_context(mesh_shape, fabric_config) as (device, device_err):
        assert tuple(device.shape) == mesh_shape

        if device_err is not None:
            return False, device_err, None, None

        if is_model_traced:
            if is_2d_mesh:
                if shard_dims is not None:
                    mapper_dims = shard_dims
                elif cluster_axis == 1:
                    mapper_dims = (None, effective_dim)
                else:
                    mapper_dims = (effective_dim, None)
                tt_input = ttnn.from_torch(
                    torch_input,
                    layout=layout,
                    dtype=input_dtype,
                    memory_config=input_memory_config,
                    mesh_mapper=ShardTensor2dMesh(device, dims=mapper_dims, mesh_shape=mesh_shape),
                    device=device,
                )
            else:
                tt_input = ttnn.from_torch(
                    torch_input,
                    layout=layout,
                    dtype=input_dtype,
                    memory_config=input_memory_config,
                    mesh_mapper=ttnn.ShardTensorToMesh(device, dim=effective_dim),
                    device=device,
                )
        else:
            # Use _get_tensors helper for generality format
            tt_input, torch_reference, output_memory_config = _get_tensors(
                input_shape,
                mesh_shape,
                dim,
                cluster_axis,
                input_dtype,
                buffer_type,
                shard_specs,
                layout,
                device,
            )

        # Setup SubDevice and semaphores (match test_minimal_all_gather_async.py pattern)
        compute_grid_size = device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]

        # Set sub-device stall group
        device.set_sub_device_stall_group(sub_device_stall_group)

        # Create semaphores for CCL operations - one set per iteration
        ccl_semaphore_handles = [
            [ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(2)] for _ in range(num_iters)
        ]

        # Create barrier semaphore if needed
        barrier_semaphore_handles = []
        if barrier_semaphore is not None:
            barrier_semaphore_handles = [
                ttnn.create_global_semaphore(device, ccl_sub_device_crs, 0) for _ in range(num_iters)
            ]

        for i in range(num_iters):
            try:
                start_time = start_measuring_time()

                if is_model_traced:
                    # Build kwargs to match the model trace exactly: include
                    # mesh_device, and only pass optional args that were
                    # present (non-None) in the original model call.
                    op_kwargs = {
                        "dim": dim,
                        "multi_device_global_semaphore": ccl_semaphore_handles[i],
                        "num_links": num_links,
                        "topology": topology,
                        "cluster_axis": cluster_axis,
                        "mesh_device": device,
                    }
                    if output_memory_config is not None:
                        op_kwargs["memory_config"] = output_memory_config
                    if barrier_semaphore_handles:
                        op_kwargs["barrier_semaphore"] = barrier_semaphore_handles[i]
                    if persistent_output_buffer is not None:
                        op_kwargs["persistent_output_buffer"] = persistent_output_buffer
                    if subdevice_id is not None:
                        op_kwargs["subdevice_id"] = worker_sub_device_id
                    if chunks_per_sync is not None:
                        op_kwargs["chunks_per_sync"] = chunks_per_sync
                    if num_workers_per_link is not None:
                        op_kwargs["num_workers_per_link"] = num_workers_per_link
                    if num_buffers_per_channel is not None:
                        op_kwargs["num_buffers_per_channel"] = num_buffers_per_channel
                    if use_broadcast is not None:
                        op_kwargs["use_broadcast"] = use_broadcast
                    tt_out_tensor = ttnn.experimental.all_gather_async(tt_input, **op_kwargs)
                else:
                    tt_out_tensor = ttnn.experimental.all_gather_async(
                        tt_input,
                        persistent_output_buffer=persistent_output_buffer if persistent_output_buffer else None,
                        dim=dim,
                        multi_device_global_semaphore=ccl_semaphore_handles[i],
                        num_links=num_links,
                        memory_config=output_memory_config,
                        topology=topology,
                        subdevice_id=worker_sub_device_id,
                        barrier_semaphore=barrier_semaphore_handles[i] if barrier_semaphore_handles else None,
                        cluster_axis=cluster_axis,
                        chunks_per_sync=chunks_per_sync,
                        num_workers_per_link=num_workers_per_link,
                        num_buffers_per_channel=num_buffers_per_channel,
                    )

                ttnn.synchronize_device(device, sub_device_ids=sub_device_stall_group)
                e2e_perf = stop_measuring_time(start_time)
            except Exception as e:
                raise RuntimeError(f"Execution failed: {e}")

        device.reset_sub_device_stall_group()

        # After all_gather, every device in the gather group has the full tensor.
        # Read a single device's output for comparison.
        device_tensors = ttnn.get_device_tensors(tt_out_tensor)
        tt_output_tensor = ttnn.to_torch(device_tensors[0])

        # Trim tile padding to match expected shape
        tt_output_tensor = tt_output_tensor[tuple(slice(0, s) for s in torch_reference.shape)]

        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, torch_reference)
        else:
            eq, output = comp_pcc(tt_output_tensor, torch_reference)

        return [(eq, output), e2e_perf]
