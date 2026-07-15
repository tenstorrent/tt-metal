# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader, dict_to_memory_config
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args, parse_dict_value
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
    reconcile_golden_to_actual,
)

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("interleaved_to_sharded")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
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
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1"}, output_memory_config=output_memory_config)

    # Extract positional non-tensor args dropped by build_op_kwargs
    # interleaved_to_sharded(input, output_mem_cfg, tensor_memory_layout, shard_orientation, output_dtype)
    pos_args = extract_positional_args(kwargs)
    arg1_raw = pos_args.get(1, None)
    arg1 = arg1_raw
    if isinstance(arg1, dict):
        arg1 = dict_to_memory_config(arg1)  # None for a non-MemoryConfig dict (e.g. a CoreCoord grid)

    # arg3 = TensorMemoryLayout, arg4 = ShardOrientation, arg5 = output DataType
    arg3 = pos_args.get(3, None)
    if isinstance(arg3, dict):
        arg3 = parse_dict_value("arg3", arg3)
    arg4 = pos_args.get(4, None)
    if isinstance(arg4, dict):
        arg4 = parse_dict_value("arg4", arg4)
    arg5 = pos_args.get(5, None)
    if isinstance(arg5, dict):
        arg5 = parse_dict_value("arg5", arg5)

    # arg1 IS the target sharded memory config — always prioritize it.
    # The loader's fallback sets output_memory_config from the input tensor's
    # memory_config (DRAM/interleaved), which is wrong for this op.
    if arg1 is not None:
        output_memory_config = arg1
    elif output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    # Grid call form: interleaved_to_sharded(input, grid, shard_shape, scheme,
    # orientation, dtype). Here arg1 is a CoreCoord grid (not a MemoryConfig) and
    # the traced output_memory_config may be interleaved -> "Output memory config
    # must be sharded". Reconstruct the sharded config from grid + arg2 shard_shape
    # + scheme(arg3) + orientation(arg4) ONLY when we don't already have a sharded
    # output config (don't override the configs that already carry one).
    _have_sharded_out = (
        output_memory_config is not None
        and getattr(output_memory_config, "is_sharded", None) is not None
        and output_memory_config.is_sharded()
    )
    # Use the RAW kwargs dict for arg1: parse_dict_value returns None for a
    # CoreCoord dict, so pos_args[1] can't carry the grid form.
    _arg1_dict = kwargs.get("arg1")
    if not _have_sharded_out and isinstance(_arg1_dict, dict) and _arg1_dict.get("type") == "CoreCoord":
        import re as _re_i2s

        def _enum_from_repr(d, enum_cls):
            # The traced scheme/orientation are {'type':..,'repr':'TensorMemoryLayout.BLOCK_SHARDED'}
            # dicts that parse_dict_value can't resolve; map the repr's last token to the enum.
            if not isinstance(d, dict):
                return None
            return getattr(enum_cls, str(d.get("repr", "")).rsplit(".", 1)[-1], None)

        _scheme = _enum_from_repr(kwargs.get("arg3"), ttnn.TensorMemoryLayout)
        _orient = _enum_from_repr(kwargs.get("arg4"), ttnn.ShardOrientation)
        _gm = _re_i2s.match(r"(\d+)\s*-\s*(\d+)", str(_arg1_dict.get("value", "")))
        _arg2 = pos_args.get(2, None)
        if _gm and isinstance(_arg2, (list, tuple)) and len(_arg2) == 2 and _scheme is not None and _orient is not None:
            gx, gy = int(_gm.group(1)), int(_gm.group(2))
            grid_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
            shard_spec = ttnn.ShardSpec(grid_crs, [int(_arg2[0]), int(_arg2[1])], _orient)
            output_memory_config = ttnn.MemoryConfig(_scheme, ttnn.BufferType.L1, shard_spec)
            arg3 = None  # scheme/orientation now baked into the MemoryConfig
            arg4 = None
            if arg5 is not None and "output_dtype" not in op_kwargs:
                op_kwargs["output_dtype"] = arg5
                arg5 = None

    # Handle tuple input_a_shape
    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # For interleaved_to_sharded, the output is the same tensor but in sharded memory layout
    torch_output_tensor = torch_input_tensor_a.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
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

    # Use the traced output_memory_config directly - no hardcoding
    # The traced config contains the exact memory layout and shard spec from real model runs
    # If output_memory_config is not available, fall back to DRAM interleaved
    if output_memory_config is None:
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    start_time = start_measuring_time()
    i2s_extra_args = [a for a in [arg3, arg4, arg5] if a is not None]
    output_tensor = ttnn.interleaved_to_sharded(input_tensor_a, output_memory_config, *i2s_extra_args, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - should be identical since it's just a memory layout change
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
