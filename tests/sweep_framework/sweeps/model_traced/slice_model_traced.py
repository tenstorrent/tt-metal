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
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("slice")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

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
    memory_config="__ABSENT__",  # __ABSENT__ sentinel: distinguishes "not in trace" from "trace had None"
    storage_type="StorageType::DEVICE",
    arg1=None,  # May contain starts from V2 traced configs (positional)
    arg2=None,  # May contain ends from V2 traced configs (positional)
    arg3=None,  # May contain steps from V2 traced configs (positional)
    dtype="__ABSENT__",  # __ABSENT__ sentinel: distinguishes "not in trace" from "trace had None"
    use_legacy=None,  # Legacy mode flag from V2 traced configs
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    # Detect tensor-arg slice: the config has valid num_devices + slice_dim
    # (arg1/arg2 are __ABSENT__ when they were tensor dicts expanded by loader)
    _has_slice_meta = kwargs.get("num_devices") not in (None, "__ABSENT__") and kwargs.get("slice_dim") not in (
        None,
        "__ABSENT__",
    )
    _tensor_arg_slice = (isinstance(arg1, dict) and isinstance(arg2, dict)) or _has_slice_meta

    if _tensor_arg_slice:
        # For tensor-arg slice, include slice_dim and num_devices (required by C++ op)
        # and keep None values (e.g., output_tensor=None) for trace matching
        op_kwargs = build_op_kwargs(
            kwargs,
            exclude={"starts", "ends", "steps"},
            output_memory_config=output_memory_config,
            keep_none=True,
            extra_kwargs={"memory_config": memory_config, "dtype": dtype},
        )
    else:
        # For coordinate-based slice, exclude slice_dim/num_devices (not valid kwargs)
        op_kwargs = build_op_kwargs(
            kwargs,
            exclude={"starts", "ends", "steps", "slice_dim", "num_devices"},
            output_memory_config=output_memory_config,
            extra_kwargs={"memory_config": memory_config, "dtype": dtype},
        )

    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    if _tensor_arg_slice:
        slice_dim = kwargs.get("slice_dim", len(shape) - 1)
        num_devices = kwargs.get("num_devices", 2)
        chunk = shape[slice_dim] // num_devices
        slice_start = [0] * len(shape)
        slice_end = list(shape)
        slice_end[slice_dim] = chunk
        slices = [slice(s, e) for s, e in zip(slice_start, slice_end)]
        torch_output_tensor = torch_input_tensor_a[tuple(slices)]
    else:
        _arg1 = arg1 if (arg1 is not None and arg1 != "__ABSENT__" and not isinstance(arg1, dict)) else None
        _arg2 = arg2 if (arg2 is not None and arg2 != "__ABSENT__" and not isinstance(arg2, dict)) else None
        slice_start = kwargs.get("starts", None) or _arg1 or [0] * len(shape)
        slice_end = kwargs.get("ends", None) or _arg2
        slice_step = kwargs.get("steps", None) or arg3 or [1] * len(shape)

        if not slice_end:
            slice_end = list(shape)
            slice_end[-1] = shape[-1] // 2

        slices = []
        for start, end, step in zip(slice_start, slice_end, slice_step):
            if step == 1:
                slices.append(slice(start, end))
            else:
                slices.append(slice(start, end, step))
        torch_output_tensor = torch_input_tensor_a[tuple(slices)]

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

    start_time = start_measuring_time()
    if _tensor_arg_slice:
        start_tensor = ttnn.from_torch(
            torch.tensor(slice_start, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        end_tensor = ttnn.from_torch(
            torch.tensor(slice_end, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        extra = {}
        ot_info = extract_named_tensor_kwargs(kwargs, "output_tensor")
        if ot_info and ot_info.get("shape"):
            op_kwargs.pop("output_tensor", None)
            out_shape = tuple(ot_info["shape"])
            ot_dtype = ot_info.get("dtype", input_a_dtype)
            ot_layout = ot_info.get("layout", input_a_layout)
            ot_mem = ot_info.get("memory_config", output_memory_config or ttnn.DRAM_MEMORY_CONFIG)
            ot_torch = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), ot_dtype)(
                out_shape
            )
            if is_mesh_device and input_a_tensor_placement:
                extra["output_tensor"] = create_tensor_on_mesh(
                    ot_torch,
                    device,
                    ot_dtype,
                    ot_layout,
                    ot_mem,
                    input_a_tensor_placement,
                )
            else:
                extra["output_tensor"] = ttnn.from_torch(
                    ot_torch,
                    dtype=ot_dtype,
                    layout=ot_layout,
                    device=device,
                    memory_config=ot_mem,
                )
        output_tensor = ttnn.slice(input_tensor_a, start_tensor, end_tensor, **extra, **op_kwargs)
    elif arg3 is not None:
        output_tensor = ttnn.slice(input_tensor_a, slice_start, slice_end, slice_step, **op_kwargs)
    else:
        output_tensor = ttnn.slice(input_tensor_a, slice_start, slice_end, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
