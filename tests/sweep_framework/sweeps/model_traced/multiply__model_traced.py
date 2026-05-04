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
    reconcile_golden_to_actual,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("multiply_")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(1, 1, 32, 32)],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "scalar_value": [0.5],
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
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    output_memory_config=None,
    scalar_value=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(
        kwargs,
        exclude={"scalar_value", "fast_and_approximate_mode", "use_legacy"},
        output_memory_config=output_memory_config,
    )

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    # arg1 may be a traced tensor (unpacked to input_b_*) or a scalar. The Flux
    # master records arg1 as a tensor; using a scalar produces a config_hash diff.
    has_input_b = input_b_shape not in (None, "__ABSENT__")
    if not has_input_b and scalar_value is None:
        scalar_value = kwargs.get("scalar_value", 0.5)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    if has_input_b:
        shape_b = tuple(input_b_shape) if isinstance(input_b_shape, (list, tuple)) else input_b_shape
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32),
            input_b_dtype or input_a_dtype,
        )(shape_b)
        torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)
    else:
        torch_output_tensor = torch.mul(torch_input_tensor_a, scalar_value)

    is_host = storage_type and "HOST" in str(storage_type)

    def _make_b(mem_config):
        if is_host:
            return ttnn.from_torch(
                torch_input_tensor_b, dtype=input_b_dtype or input_a_dtype, layout=input_b_layout or input_a_layout
            )
        if is_mesh_device and input_b_tensor_placement:
            return create_tensor_on_mesh(
                torch_input_tensor_b,
                device,
                input_b_dtype or input_a_dtype,
                input_b_layout or input_a_layout,
                mem_config,
                input_b_tensor_placement,
            )
        return ttnn.from_torch(
            torch_input_tensor_b,
            dtype=input_b_dtype or input_a_dtype,
            layout=input_b_layout or input_a_layout,
            device=device,
            memory_config=mem_config,
        )

    # Create tensor A and run multiply_ with L1 clash fallback.
    # Wrap both tensor creation and op call so we can deallocate and retry on DRAM.
    def _create_and_run(mem_config_a, mem_config_b, extra_kwargs):
        if not is_host:
            if is_mesh_device and input_a_tensor_placement:
                t = create_tensor_on_mesh(
                    torch_input_tensor_a,
                    device,
                    input_a_dtype,
                    input_a_layout,
                    mem_config_a,
                    input_a_tensor_placement,
                )
            else:
                t = ttnn.from_torch(
                    torch_input_tensor_a,
                    dtype=input_a_dtype,
                    layout=input_a_layout,
                    device=device,
                    memory_config=mem_config_a,
                )
        else:
            t = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)
        st = start_measuring_time()
        if has_input_b:
            tb = _make_b(mem_config_b)
            ttnn.multiply_(t, tb, **extra_kwargs)
        else:
            ttnn.multiply_(t, scalar_value, **extra_kwargs)
        out = mesh_tensor_to_torch(t, device if is_mesh_device else None)
        perf = stop_measuring_time(st)
        return out, perf

    try:
        output_tensor, e2e_perf = _create_and_run(
            input_a_memory_config, input_b_memory_config or input_a_memory_config, op_kwargs
        )
    except Exception as e:
        if "circular buffers" in str(e) and "clash with L1 buffers" in str(e):
            output_tensor, e2e_perf = _create_and_run(ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, {})
        else:
            raise

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output_tensor.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    # Check with PCC
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
