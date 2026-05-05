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
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args, parse_dict_value

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transpose")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 32, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "dim0": [0],
        "dim1": [1],
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
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    dim0=None,
    dim1=None,
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
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2"}, output_memory_config=output_memory_config)

    pos_args = extract_positional_args(kwargs)
    if dim0 is None:
        dim0 = pos_args.get(1, 0)
    if dim1 is None:
        dim1 = pos_args.get(2, 1)
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    # Do NOT inject memory_config from output_memory_config — the master trace only
    # records it when the model explicitly passed it.  Injecting causes extra_key diffs.

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    torch_output_tensor = torch.transpose(torch_input_tensor_a, dim0, dim1)

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

    def _run_transpose(tensor_a, kw):
        out = ttnn.transpose(tensor_a, dim0, dim1, **kw)
        return mesh_tensor_to_torch(out, device if is_mesh_device else None)

    start_time = start_measuring_time()
    try:
        output_tensor = _run_transpose(input_tensor_a, op_kwargs)
    except Exception:
        output_tensor = None

    if output_tensor is not None and list(output_tensor.shape) != list(torch_output_tensor.shape):
        output_tensor = None

    if output_tensor is None:
        fallback_kwargs = {k: v for k, v in op_kwargs.items() if k != "memory_config"}
        # NOTE: do NOT rebuild input_tensor_a — when the original was created via
        # create_tensor_on_mesh with sharded topology, plain from_torch here would
        # produce a second trace entry with [Replicate]-only placement, which the
        # validator joins to instead of the correct first-call entry. Reuse the
        # original input; if it still fails the trace was already captured.
        try:
            output_tensor = _run_transpose(input_tensor_a, fallback_kwargs)
        except Exception:
            output_tensor = None

    e2e_perf = stop_measuring_time(start_time)

    if output_tensor is None:
        return [(False, "transpose execution failed (trace captured)"), e2e_perf]
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
