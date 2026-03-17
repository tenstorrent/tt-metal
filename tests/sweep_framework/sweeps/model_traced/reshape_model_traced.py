# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("reshape")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "target_shape": [(1, 32, 1, 32)],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if mesh_shape:
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
    output_memory_config=None,
    target_shape=None,
    shape=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2"}, output_memory_config=output_memory_config)

    # The model trace records the target shape under two different argument
    # names depending on how the model called ttnn.reshape:
    #   positional  ->  "arg1"   (MasterConfigLoader delivers via kwargs)
    #   keyword     ->  "shape"  (MasterConfigLoader delivers via the run() param)
    # We must replicate the exact calling convention so the tracer records the
    # same argument name and the config_hash matches.
    arg1_raw = kwargs.get("arg1", None)
    use_shape_kwarg = shape is not None and arg1_raw is None

    if use_shape_kwarg:
        tgt_shape = shape
    elif arg1_raw is not None:
        tgt_shape = arg1_raw
    elif target_shape is not None:
        tgt_shape = target_shape
    else:
        tgt_shape = (1, 32, 1, 32)

    if isinstance(tgt_shape, dict) and "value" in tgt_shape:
        import re

        m = re.search(r"\[([0-9, -]+)\]", str(tgt_shape["value"]))
        if m:
            tgt_shape = tuple(int(x.strip()) for x in m.group(1).split(","))
    if isinstance(tgt_shape, list):
        tgt_shape = tuple(tgt_shape)

    arg2 = kwargs.get("arg2", None)
    if arg2 is not None and isinstance(arg2, dict) and "value" in arg2:
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(arg2["value"]))
        if m:
            arg2 = tuple(int(x) for x in m.group(1).split(","))
    if isinstance(arg2, list):
        arg2 = tuple(arg2)

    in_shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        in_shape
    )

    import math

    input_numel = math.prod(in_shape)

    resolved_shape = list(tgt_shape)
    if -1 in resolved_shape:
        neg_idx = resolved_shape.index(-1)
        known_prod = abs(math.prod(s for s in resolved_shape if s != -1))
        if known_prod > 0:
            resolved_shape[neg_idx] = input_numel // known_prod

    tgt_numel = math.prod(resolved_shape)
    has_padded_shape = tgt_numel != input_numel and arg2 is not None and math.prod(arg2) == input_numel
    if has_padded_shape:
        torch_output = torch.reshape(torch_input, arg2)
        slices = tuple(slice(0, s) for s in resolved_shape)
        torch_output = torch_output[slices]
    else:
        torch_output = torch.reshape(torch_input, resolved_shape)

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    if use_shape_kwarg:
        output_tensor = ttnn.reshape(input_tensor, shape=tgt_shape, **op_kwargs)
    elif has_padded_shape and arg2 is not None:
        output_tensor = ttnn.reshape(input_tensor, tgt_shape, arg2, **op_kwargs)
    else:
        output_tensor = ttnn.reshape(input_tensor, tgt_shape, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
