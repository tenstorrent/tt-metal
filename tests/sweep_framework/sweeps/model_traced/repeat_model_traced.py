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
    get_mesh_composer,
    reconcile_golden_to_actual,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("repeat")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "repeat_shape": [(1, 1, 2, 1)],
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
    input_a_memory_config=None,
    output_memory_config=None,
    repeat_shape=None,
    repeat_dims=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "repeat_dims"}, output_memory_config=output_memory_config)

    # v2 tracer puts repeat vector in arg1 or repeat_dims
    pos_args = extract_positional_args(kwargs)
    repetition_vector = repeat_shape or repeat_dims or pos_args.get(1, None)
    if repetition_vector is None:
        repetition_vector = (1, 1, 2, 1)  # fallback for sample

    if isinstance(repetition_vector, dict) and "value" in repetition_vector:
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(repetition_vector["value"]))
        if m:
            repetition_vector = tuple(int(x) for x in m.group(1).split(","))

    if isinstance(repetition_vector, list):
        repetition_vector = tuple(repetition_vector)

    # Use named memory_config if output_memory_config not set
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )

    torch_output = torch_input.repeat(repetition_vector)

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config or ttnn.DRAM_MEMORY_CONFIG,
                input_a_tensor_placement,
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config or ttnn.DRAM_MEMORY_CONFIG,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    # Reproduce master's exact call form (positional vs `repeat_dims=` kwarg,
    # ttnn.Shape vs plain list) so the trace records the same arg key /
    # value type master saw — the config_hash is sensitive to both.
    _absent = kwargs.get("__absent_keys__", set()) or set()
    _master_arg1 = kwargs.get("arg1")
    # repeat_dims is in the function signature (so it's bound to the local
    # parameter, not kwargs). Use the local variable directly to detect the
    # Shape-dict form master recorded.
    _master_repeat_dims = repeat_dims

    def _wrap_if_shape_meta(meta, plain):
        if isinstance(meta, dict) and meta.get("type") == "Shape":
            try:
                return ttnn.Shape(list(plain))
            except Exception:
                return plain
        return plain

    if "arg1" in _absent and "repeat_dims" not in _absent:
        # Master called ttnn.repeat(input, repeat_dims=<Shape>) — named form.
        rep_arg = _wrap_if_shape_meta(_master_repeat_dims, repetition_vector)
        output_tensor = ttnn.repeat(input_tensor, repeat_dims=rep_arg, **op_kwargs)
    else:
        # Master called positionally: ttnn.repeat(input, <Shape>) or (input, <list>).
        rep_arg = _wrap_if_shape_meta(_master_arg1, repetition_vector)
        output_tensor = ttnn.repeat(input_tensor, rep_arg, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output = reconcile_golden_to_actual(torch_output, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
