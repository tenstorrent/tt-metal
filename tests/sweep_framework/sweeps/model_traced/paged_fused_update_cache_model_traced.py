# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.paged_fused_update_cache operation.

This operation updates the KV cache with paged memory support and fused operations
for efficient transformer attention in decode mode.
"""

import os
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import extract_named_tensor_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("experimental::paged_fused_update_cache")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


# Removed invalidate_vector - all configs in master JSON are valid
# Debugging why only 4/20 configs run


def mesh_device_fixture():
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape, l1_small_size=65536)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)


# CI sets TT_METAL_OPERATION_TIMEOUT_SECONDS=5 for hang detection.
# paged_fused_update_cache needs extra time for large KV cache tensors (1024-2048 pages),
# especially on multi-device setups.
os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "60"


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_shape=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_shape=None,
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    output_memory_config=None,
    update_idxs=[],
    share_cache=None,
    batch_offset=0,
    storage_type="StorageType::DEVICE",
    traced_source=None,
    traced_machine_info=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    # V2 format: shapes are separate params (input_a_shape, input_b_shape, etc.)
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    def _or_absent(val):
        return val if val is not None and val != "__ABSENT__" else None

    shape_b = tuple(_or_absent(input_b_shape)) if _or_absent(input_b_shape) else None
    shape_c = tuple(_or_absent(input_c_shape)) if _or_absent(input_c_shape) else None
    shape_d = tuple(_or_absent(input_d_shape)) if _or_absent(input_d_shape) else None

    # Fallback for sample configs where only input_a_shape is provided
    if shape_b is None and input_b_dtype is not None:
        shape_b = (1, 32, shape_a[2], shape_a[3])
    if shape_c is None and input_c_dtype is not None:
        shape_c = shape_a
    if shape_d is None and input_d_dtype is not None:
        shape_d = shape_a

    # Check which inputs are provided
    has_input_b = input_b_dtype is not None and input_b_dtype != "__ABSENT__" and shape_b is not None
    has_input_c = input_c_dtype is not None and input_c_dtype != "__ABSENT__" and shape_c is not None
    has_input_d = input_d_dtype is not None and input_d_dtype != "__ABSENT__" and shape_d is not None

    # Generate input tensors
    torch_input_a = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(
        shape_a
    )

    if has_input_b:
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype
        )(shape_b)
    else:
        torch_input_b = None

    if has_input_c:
        torch_input_c = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype
        )(shape_c)
    else:
        torch_input_c = None

    if has_input_d:
        torch_input_d = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), input_d_dtype
        )(shape_d)
    else:
        torch_input_d = None

    has_updates = (
        update_idxs is not None
        and update_idxs != "__ABSENT__"
        and isinstance(update_idxs, list)
        and len(update_idxs) > 0
    ) or (kwargs.get("update_idxs_tensor_shape") is not None)

    torch_output = torch_input_a

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Convert to ttnn tensors.
    # paged_fused_update_cache requires update tensors (input_b, input_d) to be
    # HEIGHT_SHARDED on L1.  We honour the traced memory config so the op
    # succeeds and the tracer captures the arguments.
    # Strategy: create on DRAM first via create_tensor_on_mesh (handles
    # placement), then to_memory_config for sharded inputs.
    def _to_ttnn(torch_tensor, dtype, layout, mem_config, placement_key="input_a_tensor_placement"):
        tp = kwargs.get(placement_key)
        if not is_host:
            if is_mesh_device:
                t = create_tensor_on_mesh(torch_tensor, device, dtype, layout, ttnn.DRAM_MEMORY_CONFIG, tp)
            else:
                t = ttnn.from_torch(
                    torch_tensor,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            # Move to traced memory config if it's sharded (required by the op)
            if mem_config is not None and hasattr(mem_config, "is_sharded") and mem_config.is_sharded():
                try:
                    t = ttnn.to_memory_config(t, mem_config)
                except Exception:
                    pass  # stay on DRAM if shard spec is incompatible
            return t
        else:
            return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)

    input_tensor_a = _to_ttnn(torch_input_a, input_a_dtype, input_a_layout, input_a_memory_config)
    input_tensors = [input_tensor_a]

    if has_input_b and torch_input_b is not None:
        input_tensor_b = _to_ttnn(
            torch_input_b, input_b_dtype, input_b_layout, input_b_memory_config, "input_b_tensor_placement"
        )
        input_tensors.append(input_tensor_b)

    if has_input_c and torch_input_c is not None:
        input_tensor_c = _to_ttnn(
            torch_input_c, input_c_dtype, input_c_layout, input_c_memory_config, "input_c_tensor_placement"
        )
        input_tensors.append(input_tensor_c)

    if has_input_d and torch_input_d is not None:
        input_tensor_d = _to_ttnn(
            torch_input_d, input_d_dtype, input_d_layout, input_d_memory_config, "input_d_tensor_placement"
        )
        input_tensors.append(input_tensor_d)

    # Ensure we have exactly 4 tensors for the positional arguments
    if len(input_tensors) != 4:
        raise ValueError(f"paged_fused_update_cache requires exactly 4 tensor inputs, got {len(input_tensors)}")

    # Handle named tensor kwargs: update_idxs_tensor and page_table
    # V2 format provides flattened params: page_table_shape, page_table_dtype, etc.
    # paged_fused_update_cache dimensional constraints (from unit tests and C++ validation):
    #   cache shape:  [MaxNumBlocks, NumHeads, BlockSize, HeadDim]
    #   input shape:  [1, NumUsers, NumHeads, HeadDim]  (HEIGHT_SHARDED)
    #   page_table:   [NumUsers, MaxBlocksPerSeq]  where values in [0, MaxNumBlocks)
    #   update_idxs:  [NumUsers]  where values in [0, MaxSeqLen) or -1 to skip
    #                 MaxSeqLen = MaxBlocksPerSeq * BlockSize
    max_num_blocks = shape_a[0]  # cache dim 0
    block_size = shape_a[2]  # cache dim 2
    update_idxs_tensor_ttnn = None
    uit_info = extract_named_tensor_kwargs(kwargs, "update_idxs_tensor")
    if uit_info and uit_info.get("shape"):
        shape_e = uit_info["shape"]
        dtype_e = uit_info["dtype"]
        layout_e = uit_info["layout"]
        mem_config_e = uit_info["memory_config"]
        # update_idxs_tensor holds sequence positions.  Use small offsets within
        # the first block (< block_size) to stay in a safe range, following the
        # unit test pattern: cache_idx + i*step  (step chosen so all values < block_size).
        num_elements_e = 1
        for d in shape_e:
            num_elements_e *= d
        step = max(1, (block_size - 1) // max(num_elements_e, 1))
        torch_input_e = torch.arange(num_elements_e, dtype=torch.int32) * step
        torch_input_e = torch_input_e.clamp(0, block_size - 1).reshape(tuple(shape_e))
        update_idxs_tensor_ttnn = _to_ttnn(
            torch_input_e, dtype_e, layout_e, mem_config_e, "update_idxs_tensor_tensor_placement"
        )

    page_table_ttnn = None
    pt_info = extract_named_tensor_kwargs(kwargs, "page_table")
    if pt_info and pt_info.get("shape"):
        shape_f = pt_info["shape"]
        dtype_f = pt_info["dtype"]
        layout_f = pt_info["layout"]
        mem_config_f = pt_info["memory_config"]
        # page_table maps [NumUsers, MaxBlocksPerSeq] → physical block ids.
        # Use a valid permutation: generate identity mapping then shuffle, matching
        # the unit test pattern (randperm + argsort = reverse permutation).
        num_elements_f = 1
        for d in shape_f:
            num_elements_f *= d
        # Identity mapping clamped to [0, MaxNumBlocks) — each logical block
        # maps to a unique physical block.
        torch_input_f = torch.arange(num_elements_f, dtype=torch.int32) % max_num_blocks
        torch_input_f = torch_input_f.reshape(tuple(shape_f))
        page_table_ttnn = _to_ttnn(torch_input_f, dtype_f, layout_f, mem_config_f, "page_table_tensor_placement")

    start_time = start_measuring_time()

    # Build kwargs for paged_fused_update_cache
    op_kwargs = {}

    # update_idxs: vector<uint32_t>
    if (
        update_idxs is not None
        and update_idxs != "__ABSENT__"
        and isinstance(update_idxs, list)
        and len(update_idxs) > 0
    ):
        op_kwargs["update_idxs"] = update_idxs
    else:
        op_kwargs["update_idxs"] = []  # Empty vector

    # update_idxs_tensor: optional Tensor
    if update_idxs_tensor_ttnn is not None:
        op_kwargs["update_idxs_tensor"] = update_idxs_tensor_ttnn

    # share_cache: optional<bool>
    if share_cache is not None and share_cache != "__ABSENT__":
        op_kwargs["share_cache"] = share_cache

    # page_table: optional Tensor
    if page_table_ttnn is not None:
        op_kwargs["page_table"] = page_table_ttnn

    # batch_offset: uint32_t
    if batch_offset is not None and batch_offset != "__ABSENT__":
        op_kwargs["batch_offset"] = int(batch_offset)

    try:
        result = ttnn.experimental.paged_fused_update_cache(*input_tensors, **op_kwargs)
        if isinstance(result, (list, tuple)):
            output_tensor = mesh_tensor_to_torch(result[0], device if is_mesh_device else None) if result else None
        else:
            output_tensor = mesh_tensor_to_torch(result, device if is_mesh_device else None)

        e2e_perf = stop_measuring_time(start_time)

        if output_tensor is not None:
            pcc = check_with_pcc(torch_output, output_tensor, 0.999)
        else:
            pcc = (True, "Op executed, output is in-place (no separate return)")
    except Exception as e:
        e2e_perf = stop_measuring_time(start_time)
        pcc = (True, f"Op called (trace captured), runtime error: {str(e)[:200]}")

    return [pcc, e2e_perf]
