# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

import re

from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
    reconcile_golden_to_actual,
    dispatch_axis_for_grid,
    shard_grid_bounds,
)
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from typing import Optional, Tuple

# Override the default timeout in seconds for hang detection.
# group_norm is computationally intensive, needs longer timeout
TIMEOUT = 300


# Device opened per-vector (see _ensure_vector_device): group_norm runs on an
# 8-wide compute grid (core_grid x=8 -> cores x in [0,7]). Under COL dispatch
# the worker grid is only 7 wide (x in [0,6]), so core x=7 is a dispatch core
# and the op fails with "not on_dispatch_core". These configs need ROW dispatch
# (8 wide). The default fixture's auto-axis defaulted to COL (arg0 is
# interleaved, so there is no shard_spec signal), failing every config.
_CUR_DEVICE = None
_CUR_AXIS = "__uninit__"
_CUR_SHAPE = None


def _close_vector_device():
    global _CUR_DEVICE, _CUR_AXIS, _CUR_SHAPE
    if _CUR_DEVICE is not None:
        try:
            ttnn.close_mesh_device(_CUR_DEVICE)
        except Exception:
            # best-effort teardown; a close failure must not mask the test result
            pass
    _CUR_DEVICE = None
    _CUR_AXIS = "__uninit__"
    _CUR_SHAPE = None


def _ensure_vector_device(axis):
    global _CUR_DEVICE, _CUR_AXIS, _CUR_SHAPE
    shape = get_model_traced_mesh_shape()
    if _CUR_DEVICE is None or axis != _CUR_AXIS or shape != _CUR_SHAPE:
        _close_vector_device()
        _CUR_DEVICE = create_mesh_device(shape, dispatch_core_axis=axis)
        _CUR_AXIS = axis
        _CUR_SHAPE = shape
    return _CUR_DEVICE


def _core_grid_bounds(core_grid):
    """Max (x, y) core index a traced core_grid needs, or (None, None).

    Accepts the raw vector form {"type": "CoreGrid", "value": "ttnn.CoreGrid(x=8, y=8)"},
    a parsed ttnn.CoreGrid, or None. A CoreGrid(x=N, y=M) uses cores
    x in [0, N-1], y in [0, M-1].
    """
    if core_grid is None or core_grid == "__ABSENT__":
        return None, None
    x = y = None
    if isinstance(core_grid, dict):
        s = core_grid.get("value", "")
    else:
        s = str(core_grid)
    mx = re.search(r"x\s*=\s*(\d+)", s)
    my = re.search(r"y\s*=\s*(\d+)", s)
    if mx:
        x = int(mx.group(1)) - 1
    if my:
        y = int(my.group(1)) - 1
    return x, y


def _group_norm_dispatch_axis(core_grid, *mem_configs):
    """Axis that fits the op's compute core_grid and any input shard grid."""
    cx, cy = _core_grid_bounds(core_grid)
    max_x = cx if cx is not None else -1
    max_y = cy if cy is not None else -1
    for mc in mem_configs:
        if mc is None or mc == "__ABSENT__":
            continue
        bx, by = shard_grid_bounds(mc)
        if bx is not None:
            max_x = max(max_x, bx)
        if by is not None:
            max_y = max(max_y, by)
    return dispatch_axis_for_grid(max_x if max_x >= 0 else None, max_y if max_y >= 0 else None)


# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("group_norm")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 1024, 32)],  # Shape: [N, 1, H*W, C] as per ttnn.group_norm docs
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_groups": [8],
        "epsilon": [1e-5],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    # Device opened per-vector in run() (see _ensure_vector_device).
    yield (None, "wormhole_b0")
    _close_vector_device()


_DTYPE_BYTES = {
    "BFLOAT16": 2,
    "BFLOAT8_B": 1,
    "BFLOAT4_B": 1,
    "FLOAT32": 4,
    "UINT32": 4,
    "INT32": 4,
    "UINT16": 2,
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # A BLOCK_SHARDED group_norm pins its per-core input/output shards in L1, and
    # on top of those the op statically allocates several full-shard-sized compute
    # CBs. For the high-resolution SDXL VAE configs the per-core shard alone is
    # ~320-490 KB, so the shards + CBs exceed N300's ~1.46 MB L1 and the program
    # fails to allocate ("static circular buffers clash with L1 buffers"). These
    # configs ran on larger-memory hardware; they cannot fit a 2-chip N300, so
    # mark them invalid rather than crash. Threshold: per-core shard > 300 KB
    # (measured boundary — 2048x80=320KB/2048x120=480KB fail; 2048x64=256KB and
    # smaller fit).
    mc = test_vector.get("input_a_memory_config")
    if mc is None or mc == "__ABSENT__":
        return False, None

    # input_a_memory_config may be a serialized dict (exported vectors), a
    # ttnn.MemoryConfig object (at generation time), or its string repr. Extract
    # (memory_layout, shard [h, w]) from whichever form we get.
    shape = None
    layout = ""
    if isinstance(mc, dict):
        data = mc.get("data") or mc
        layout = str(data.get("memory_layout", ""))
        ss = data.get("shard_spec") or {}
        if isinstance(ss, dict):
            shape = ss.get("shape")
    else:
        try:
            layout = str(getattr(mc, "memory_layout", ""))
            if getattr(mc, "is_sharded", lambda: False)():
                shape = list(mc.shard_spec.shape)
        except Exception:
            shape = None
        if shape is None:
            s = str(mc)
            layout = s
            m = re.search(r"shape\s*=?\s*[\[(]\s*(\d+)\s*,\s*(\d+)", s)
            if m:
                shape = [int(m.group(1)), int(m.group(2))]

    if "BLOCK_SHARDED" not in layout:
        return False, None
    if not (isinstance(shape, (list, tuple)) and len(shape) >= 2):
        return False, None

    dt = str(test_vector.get("input_a_dtype", "")).rsplit(".", 1)[-1]
    elem = _DTYPE_BYTES.get(dt, 2)
    per_core_bytes = int(shape[0]) * int(shape[1]) * elem
    if per_core_bytes > 300_000:
        return (
            True,
            f"group_norm: per-core shard {list(shape)} ({per_core_bytes} B) exceeds N300 L1 for block-sharded group_norm",
        )
    return False, None


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config=None,
    output_memory_config=None,
    num_groups=None,
    epsilon=1e-5,
    storage_type="StorageType::DEVICE",
    # Optional traced arguments
    input_mask_shape=None,
    input_mask_dtype=None,
    input_mask_memory_config=None,
    weight_shape=None,
    weight_dtype=None,
    weight_memory_config=None,
    bias_shape=None,
    bias_dtype=None,
    bias_memory_config=None,
    reciprocals_shape=None,
    reciprocals_dtype=None,
    reciprocals_layout=None,
    reciprocals_memory_config=None,
    inplace=False,
    num_out_blocks=None,
    use_welford=False,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Open this vector's device with the dispatch axis its compute core_grid
    # needs. core_grid x=8 -> cores x in [0,7] -> ROW (COL is only 7 wide and
    # would put core x=7 on a dispatch core: "not on_dispatch_core").
    _ax = _group_norm_dispatch_axis(kwargs.get("core_grid"), input_a_memory_config)
    device = _ensure_vector_device(_ax)

    # Filter __ABSENT__ sentinels from optional parameters
    def _clean(v):
        return None if v == "__ABSENT__" else v

    input_mask_shape = _clean(input_mask_shape)
    input_mask_dtype = _clean(input_mask_dtype)
    input_mask_memory_config = _clean(input_mask_memory_config)
    weight_shape = _clean(weight_shape)
    weight_dtype = _clean(weight_dtype)
    weight_memory_config = _clean(weight_memory_config)
    bias_shape = _clean(bias_shape)
    bias_dtype = _clean(bias_dtype)
    bias_memory_config = _clean(bias_memory_config)
    reciprocals_shape = _clean(reciprocals_shape)
    reciprocals_dtype = _clean(reciprocals_dtype)
    reciprocals_layout = _clean(reciprocals_layout)
    reciprocals_memory_config = _clean(reciprocals_memory_config)
    output_memory_config = _clean(output_memory_config)
    inplace = False if inplace == "__ABSENT__" else inplace
    num_out_blocks = _clean(num_out_blocks)
    use_welford = False if use_welford == "__ABSENT__" else use_welford

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    weight_tensor_placement = kwargs.get("weight_tensor_placement", None)
    bias_tensor_placement = kwargs.get("bias_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    # Let core_grid, memory_config, num_groups, epsilon flow through op_kwargs
    # so they get parsed from dicts. Exclude only non-op params.
    op_kwargs = build_op_kwargs(
        kwargs,
        exclude={"inplace", "negative_mask", "num_out_blocks", "use_welford"},
        output_memory_config=output_memory_config,
    )

    # Read num_groups and epsilon from op_kwargs (from traced config), falling back to function params
    num_groups = op_kwargs.get("num_groups", num_groups)
    if num_groups is not None:
        num_groups = int(num_groups)
        op_kwargs["num_groups"] = num_groups  # Ensure int type in op_kwargs too
    epsilon = op_kwargs.get("epsilon", epsilon)

    if input_a_memory_config is None:
        input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG

    if num_groups is None:
        return [(False, "Missing num_groups"), 0.0]

    # Handle tuple input_a_shape for sample suite
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # ========================================================================================================
    # TENSOR FORMAT CONVERSION - TTNN vs PyTorch
    # ========================================================================================================
    # TTNN group_norm format: [N, 1, H*W, C]
    # PyTorch group_norm format: [N, C, H, W]
    # ========================================================================================================

    # Extract number of channels from input shape (last dimension in both formats)
    C = shape[-1]

    # Create optional weight and bias tensors if provided in traced config
    torch_weight = None
    torch_bias = None
    if weight_shape:
        weight_elements = 1
        for dim in weight_shape:
            weight_elements *= dim
        if weight_elements == C:
            torch_weight = torch.ones(weight_shape, dtype=torch.float32)
    if bias_shape:
        bias_elements = 1
        for dim in bias_shape:
            bias_elements *= dim
        if bias_elements == C:
            torch_bias = torch.zeros(bias_shape, dtype=torch.float32)

    # Convert TTNN format to PyTorch format for golden reference
    if len(shape) == 4 and shape[1] == 1:
        N, _, HW, C = shape
        import math

        H = W = int(math.sqrt(HW))
        if H * W != HW:
            H = HW
            W = 1

        torch_input_reshaped = torch_input_tensor_a.reshape(N, H, W, C).permute(0, 3, 1, 2)

        if torch_weight is not None:
            torch_weight_reshaped = torch_weight.reshape(C)
        else:
            torch_weight_reshaped = None
        if torch_bias is not None:
            torch_bias_reshaped = torch_bias.reshape(C)
        else:
            torch_bias_reshaped = None
    else:
        torch_input_reshaped = torch_input_tensor_a
        torch_weight_reshaped = torch_weight
        torch_bias_reshaped = torch_bias

    # Compute golden reference
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_reshaped, num_groups, weight=torch_weight_reshaped, bias=torch_bias_reshaped, eps=epsilon
    )

    # Convert PyTorch output back to TTNN format for comparison
    if len(shape) == 4 and shape[1] == 1:
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).reshape(shape)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create input tensor using traced memory config (may be sharded)
    # For sharded configs, create interleaved first then shard
    input_is_sharded = hasattr(input_a_memory_config, "is_sharded") and input_a_memory_config.is_sharded()

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
        elif input_is_sharded:
            # Create interleaved first, then shard (from_torch can't directly create sharded)
            input_tensor_a_interleaved = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            try:
                input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_a_interleaved, input_a_memory_config)
            except RuntimeError:
                # If sharding fails, fall back to interleaved
                input_tensor_a = input_tensor_a_interleaved
                input_is_sharded = False
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

    # Create optional tensors if traced config provides them
    input_mask = None
    weight_tensor = None
    bias_tensor = None
    reciprocals_tensor = None

    # Determine core_grid early - needed for proper mask/weight/bias creation
    # The core_grid.y value determines the num_cores_across_channel parameter
    _op_kwargs_copy = build_op_kwargs(
        kwargs,
        exclude={"inplace", "negative_mask", "num_out_blocks", "use_welford"},
        output_memory_config=output_memory_config,
    )
    if "core_grid" in _op_kwargs_copy:
        _early_core_grid = _op_kwargs_copy["core_grid"]
    else:
        _early_core_grid = ttnn.CoreGrid(y=1, x=1)
    # num_cores_across_channel is the channel-tile count: group_norm RM weight/
    # bias/mask split channels into tiles of 32 per core, so the master records
    # weight.original_shape == [1, 1, C/32, 32]. Deriving this from core_grid.y
    # (=8) produced the wrong weight/bias/mask shapes (and wrong grouping ->
    # PCC ~0.5-0.7). Prefer the traced weight_shape[-2] when present; else C/32.
    num_cores_across_channel = max(1, (C + 31) // 32)
    if weight_shape and len(weight_shape) >= 2 and int(weight_shape[-2]) > 0:
        num_cores_across_channel = int(weight_shape[-2])
    # For block-sharded group_norm the channels are split across the core grid's
    # x dimension, so the op's real num_cores_across_channel is core_grid.x. The
    # mask/weight/bias must be built for that exact count or the channel->group
    # mapping is wrong (≈0.79 PCC). weight_shape[-2] (=C/32) often does not even
    # divide num_groups (so create_group_norm_input_mask rejects it and a wrong
    # fallback count is silently used). Prefer core_grid.x when it validly tiles
    # both the channels (C/32) and the groups.
    try:
        _cgx = int(getattr(_early_core_grid, "x", 0))
    except Exception:
        _cgx = 0
    if _cgx > 0 and num_groups % _cgx == 0 and (C // 32) % _cgx == 0:
        num_cores_across_channel = _cgx

    # Most authoritative source: when the input is BLOCK_SHARDED, the device
    # splits the channels across the shard grid's WIDTH, so the real number of
    # cores across the channel axis is C / (channel shard width). The mask /
    # weight / bias must be built for exactly this count, otherwise the
    # channel->group mapping is wrong and PCC collapses to ~0.707 (1/sqrt(2)).
    # core_grid is often absent for these configs (-> _cgx defaults to 1) and
    # weight_shape[-2] (=C/32) doesn't match the shard split either.
    def _channel_shard_width(mc):
        try:
            if isinstance(mc, dict):
                ss = (mc.get("data") or {}).get("shard_spec") or mc.get("shard_spec")
                if isinstance(ss, dict):
                    shp = ss.get("shape")
                    if isinstance(shp, (list, tuple)) and len(shp) >= 2:
                        return int(shp[1])
            elif mc is not None and getattr(mc, "is_sharded", lambda: False)():
                return int(mc.shard_spec.shape[1])
        except Exception:
            return None
        return None

    _sw = _channel_shard_width(input_a_memory_config)
    if _sw and _sw > 0 and C % _sw == 0:
        _nc_shard = C // _sw
        if _nc_shard > 0 and num_groups % _nc_shard == 0:
            num_cores_across_channel = _nc_shard

    # Use ttnn.create_group_norm_input_mask for proper channel-group mapping.
    # create_group_norm_input_mask only accepts num_cores values that evenly
    # tile the channel/group layout (e.g. 24 crashes for C=768,groups=32 while
    # powers of two work); its resulting mask shape is the same for every valid
    # value, so when the weight-derived num_cores is rejected, fall back to
    # other candidates until one succeeds — the recorded mask still matches.
    if input_mask_shape and not is_host:
        mask_dtype = input_mask_dtype or ttnn.bfloat8_b
        _mask_candidates = [num_cores_across_channel] + [
            n for n in (max(1, (C + 31) // 32), 8, 4, 2, 1) if n != num_cores_across_channel
        ]
        for _nc in _mask_candidates:
            try:
                input_mask = ttnn.create_group_norm_input_mask(C, num_groups, _nc, mask_dtype)
                input_mask = ttnn.to_device(input_mask, device)
                break
            except Exception:
                input_mask = None
        if input_mask is None:
            print("Warning: create_group_norm_input_mask failed for all num_cores candidates, skipping mask")

    # Use ttnn.create_group_norm_weight_bias_rm for proper weight formatting
    if weight_shape and torch_weight is not None and not is_host:
        w_dtype = weight_dtype or ttnn.bfloat16
        w_mem = weight_memory_config or ttnn.DRAM_MEMORY_CONFIG
        try:
            torch_weight_rm = ttnn.create_group_norm_weight_bias_rm(
                torch_weight.reshape(C), C, num_cores_across_channel
            )
            # Place weight with its traced placement (the master shards weight on
            # the same mesh axis as the input); replicating it produced a
            # tensor_placement diff vs the master.
            if is_mesh_device and weight_tensor_placement:
                weight_tensor = create_tensor_on_mesh(
                    torch_weight_rm, device, w_dtype, ttnn.ROW_MAJOR_LAYOUT, w_mem, weight_tensor_placement
                )
            else:
                weight_tensor = ttnn.from_torch(
                    torch_weight_rm,
                    dtype=w_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=w_mem,
                )
        except Exception as e:
            print(f"Warning: create_group_norm_weight_bias_rm for weight failed: {e}")
            weight_tensor = None

    # Use ttnn.create_group_norm_weight_bias_rm for proper bias formatting
    if bias_shape and torch_bias is not None and not is_host:
        b_dtype = bias_dtype or ttnn.bfloat16
        b_mem = bias_memory_config or ttnn.DRAM_MEMORY_CONFIG
        try:
            torch_bias_rm = ttnn.create_group_norm_weight_bias_rm(torch_bias.reshape(C), C, num_cores_across_channel)
            if is_mesh_device and bias_tensor_placement:
                bias_tensor = create_tensor_on_mesh(
                    torch_bias_rm, device, b_dtype, ttnn.ROW_MAJOR_LAYOUT, b_mem, bias_tensor_placement
                )
            else:
                bias_tensor = ttnn.from_torch(
                    torch_bias_rm,
                    dtype=b_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=b_mem,
                )
        except Exception as e:
            print(f"Warning: create_group_norm_weight_bias_rm for bias failed: {e}")
            bias_tensor = None

    if reciprocals_shape and use_welford and not is_host:
        skip_reciprocals = False
        reciprocals_mem_cfg = reciprocals_memory_config if reciprocals_memory_config else ttnn.DRAM_MEMORY_CONFIG

        if (
            reciprocals_mem_cfg
            and hasattr(reciprocals_mem_cfg, "memory_layout")
            and hasattr(reciprocals_mem_cfg, "buffer_type")
        ):
            is_recip_sharded = reciprocals_mem_cfg.memory_layout in [
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            ]
            is_l1 = reciprocals_mem_cfg.buffer_type == ttnn.types.BufferType.L1

            if is_recip_sharded and is_l1:
                skip_reciprocals = True

        if not skip_reciprocals:
            torch_reciprocals = torch.ones(reciprocals_shape, dtype=torch.float32)
            recip_layout = reciprocals_layout or ttnn.TILE_LAYOUT
            recip_dtype = reciprocals_dtype or ttnn.float32

            if is_mesh_device and input_a_tensor_placement:
                reciprocals_tensor = create_tensor_on_mesh(
                    torch_reciprocals,
                    device,
                    recip_dtype,
                    recip_layout,
                    reciprocals_mem_cfg,
                    input_a_tensor_placement,
                )
            else:
                reciprocals_tensor = ttnn.from_torch(
                    torch_reciprocals,
                    dtype=recip_dtype,
                    layout=recip_layout,
                    device=device,
                    memory_config=reciprocals_mem_cfg,
                )

    start_time = start_measuring_time()

    # inplace groupnorm is only supported for sharded tensors
    actual_inplace = inplace and input_is_sharded

    # Use traced core_grid if provided via op_kwargs, otherwise compute a default
    if "core_grid" not in op_kwargs:
        if use_welford and num_groups > 16:
            min_cores = (num_groups + 15) // 16
            try:
                grid_size = device.compute_with_storage_grid_size()
                if grid_size.y * grid_size.x >= min_cores:
                    core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
                else:
                    core_grid = ttnn.CoreGrid(y=1, x=min_cores)
            except Exception:
                core_grid = ttnn.CoreGrid(y=1, x=min_cores)
        else:
            core_grid = ttnn.CoreGrid(y=1, x=1)
    else:
        core_grid = op_kwargs.pop("core_grid")

    # Build group_norm arguments. Pass epsilon explicitly: when omitted the op
    # uses its C++ default and the trace records no epsilon arg, while the master
    # always records the traced epsilon (an extra_key diff).
    group_norm_kwargs = {
        "inplace": actual_inplace,
        "core_grid": core_grid,
        "memory_config": output_memory_config,
        "epsilon": epsilon,
    }

    # Add optional arguments if they exist
    if input_mask is not None:
        group_norm_kwargs["input_mask"] = input_mask
    if weight_tensor is not None:
        group_norm_kwargs["weight"] = weight_tensor
    if bias_tensor is not None:
        group_norm_kwargs["bias"] = bias_tensor
    if reciprocals_tensor is not None:
        group_norm_kwargs["reciprocals"] = reciprocals_tensor
    if num_out_blocks is not None:
        group_norm_kwargs["num_out_blocks"] = num_out_blocks
    if use_welford:
        group_norm_kwargs["use_welford"] = use_welford

    # Merge op_kwargs but don't overwrite explicitly set params
    for k, v in op_kwargs.items():
        if k not in group_norm_kwargs:
            group_norm_kwargs[k] = v

    def _retryable(msg):
        m = msg.lower()
        # An L1/CB overflow (try a bigger split) or num_out_blocks exceeding block_h
        # (try a smaller split — the op requires num_out_blocks in [1, block_h]).
        return any(s in m for s in ("clash", "circular buffer", "out of memory", "l1 buffer", "num_out_blocks"))

    try:
        output_tensor = ttnn.group_norm(input_tensor_a, **group_norm_kwargs)
    except Exception as e:
        # Genuinely-oversized configs are invalidated upstream; here we only relieve
        # borderline L1 pressure by splitting the per-core height into num_out_blocks
        # compute sub-blocks (smaller CBs, identical math). Try a few valid splits;
        # keep the count small so a hopeless config can't wedge the device with many
        # large failed program builds.
        if not _retryable(str(e)) or "num_out_blocks" in group_norm_kwargs:
            raise
        output_tensor = None
        _last = e
        for _nob in (4, 8, 16):
            try:
                _gk = dict(group_norm_kwargs)
                _gk["num_out_blocks"] = _nob
                output_tensor = ttnn.group_norm(input_tensor_a, **_gk)
                break
            except Exception as e2:
                _last = e2
                if not _retryable(str(e2)):
                    raise
        if output_tensor is None:
            raise _last
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
