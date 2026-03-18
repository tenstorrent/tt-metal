# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import random
import re
import torch
import ttnn
from ttnn import ShardTensor2dMesh
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    mesh_tensor_to_torch,
    infer_mesh_shape_from_params,
    detect_mesh_shape_from_hardware,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("embedding")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 32)],  # (batch_size, seq_length)
        "input_a_dtype": [ttnn.uint32],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(128, 32)],  # (num_embeddings, embeddings_dim)
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "dtype": [ttnn.bfloat16],  # output dtype
        "memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    """Extract (dim0, dim1) for ShardTensor2dMesh from tensor_placement dict.

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


def _create_mesh_tensor(torch_tensor, device, dtype, layout, memory_config, tensor_placement, mesh_shape):
    """Create a TTNN tensor on a mesh device with placement matching the model trace.

    Builds a global tensor by repeating the per-device tensor along shard dims,
    then distributes with ShardTensor2dMesh so each device gets the original shape.
    Falls back to ReplicateTensorToMesh when no shard dims are found.
    """
    shard_dims = _parse_shard_dims_from_placement(tensor_placement)
    if shard_dims is not None and any(d is not None for d in shard_dims):
        repeat_factors = [1] * torch_tensor.ndim
        for axis_idx, sd in enumerate(shard_dims):
            if sd is not None:
                esd = sd if sd >= 0 else torch_tensor.ndim + sd
                repeat_factors[esd] *= mesh_shape[axis_idx]
        global_tensor = torch_tensor.repeat(*repeat_factors)
        return ttnn.from_torch(
            global_tensor,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=memory_config,
            mesh_mapper=ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
        )
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


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
    input_a_shape,  # indices shape: (batch_size, seq_length)
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape,  # weights shape: (num_embeddings, embeddings_dim)
    input_b_dtype,
    input_b_layout,
    input_b_memory_config,
    dtype=None,  # output dtype
    memory_config=None,  # output memory_config
    storage_type="StorageType::DEVICE",
    layout=None,  # Additional layout parameter from JSON
    weight_shape=None,  # Alternative weight shape parameter
    weight_dtype=None,  # Alternative weight dtype parameter
    weight_layout=None,  # Alternative weight layout parameter
    weight_memory_config=None,  # Alternative weight memory_config parameter
    padding_idx=None,  # Padding index for embeddings
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    weight_tensor_placement = kwargs.get("weight_tensor_placement", input_b_tensor_placement)

    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"weight_tensor_placement", "padding_idx"})

    # The model trace captures the weight tensor under the key "weight" when
    # passed as a keyword argument, and under "arg1" when passed positionally.
    # The loader maps "weight" → weight_shape and "arg1" → input_b_shape.
    # Mirror the original calling convention so the tracer records the same key.
    use_weight_kwarg = input_b_shape is None and weight_shape is not None

    input_shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    if input_b_shape is not None:
        weight_shape_actual = tuple(input_b_shape) if isinstance(input_b_shape, (list, tuple)) else input_b_shape
    elif weight_shape is not None:
        weight_shape_actual = tuple(weight_shape) if isinstance(weight_shape, (list, tuple)) else weight_shape
    else:
        raise ValueError("Either input_b_shape or weight_shape must be provided")

    if isinstance(weight_shape_actual, (list, tuple)) and len(weight_shape_actual) > 2:
        num_embeddings = weight_shape_actual[-2]
    else:
        num_embeddings = weight_shape_actual[0]

    torch_input_tensor = torch_random(input_shape, 0, num_embeddings, torch.int64)

    weight_dtype_actual = weight_dtype if weight_dtype is not None else input_b_dtype
    weight_layout_actual = weight_layout if weight_layout is not None else input_b_layout
    weight_memory_config_actual = weight_memory_config if weight_memory_config is not None else input_b_memory_config

    torch_weight_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), weight_dtype_actual
    )(weight_shape_actual)

    golden_function = ttnn.get_golden_function(ttnn.embedding)
    torch_weight_2d = torch_weight_tensor.reshape(-1, torch_weight_tensor.shape[-1])
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight_2d).squeeze()

    is_host = storage_type and "HOST" in str(storage_type)

    # Derive mesh shape from tensor placements for correct distribution
    mesh_shape = None
    if is_mesh_device:
        for tp in [input_a_tensor_placement, weight_tensor_placement]:
            if tp:
                mesh_shape = _parse_mesh_shape(tp.get("mesh_device_shape"))
                if mesh_shape:
                    break

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

    # Create input tensor (indices)
    if not is_host:
        if is_mesh_device and mesh_shape and input_a_tensor_placement:
            input_tensor = _create_mesh_tensor(
                torch_input_tensor,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
                mesh_shape,
            )
        elif is_mesh_device:
            input_tensor = ttnn.from_torch(
                torch_input_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_a_dtype, layout=input_a_layout)

    # Create weight tensor
    if not is_host:
        if is_mesh_device and mesh_shape and weight_tensor_placement:
            weight_tensor = _create_mesh_tensor(
                torch_weight_tensor,
                device,
                weight_dtype_actual,
                weight_layout_actual,
                weight_memory_config_actual,
                weight_tensor_placement,
                mesh_shape,
            )
        elif is_mesh_device:
            weight_tensor = ttnn.from_torch(
                torch_weight_tensor,
                dtype=weight_dtype_actual,
                layout=weight_layout_actual,
                device=device,
                memory_config=weight_memory_config_actual,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            weight_tensor = ttnn.from_torch(
                torch_weight_tensor,
                dtype=weight_dtype_actual,
                layout=weight_layout_actual,
                device=device,
                memory_config=weight_memory_config_actual,
            )
    else:
        weight_tensor = ttnn.from_torch(torch_weight_tensor, dtype=weight_dtype_actual, layout=weight_layout_actual)

    start_time = start_measuring_time()
    emb_kwargs = {}
    if dtype is not None:
        emb_kwargs["dtype"] = dtype
    if memory_config is not None:
        emb_kwargs["memory_config"] = memory_config
    if layout is not None:
        emb_kwargs["layout"] = layout
    emb_kwargs.update(op_kwargs)

    if use_weight_kwarg:
        output_tensor = ttnn.embedding(input_tensor, weight=weight_tensor, **emb_kwargs)
    else:
        output_tensor = ttnn.embedding(input_tensor, weight_tensor, **emb_kwargs)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None).squeeze()

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
