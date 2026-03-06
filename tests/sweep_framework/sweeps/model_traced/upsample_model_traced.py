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

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("upsample")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Note: upsample requires scale_factor and mode from JSON
    # Sample test skipped - use model_traced suite only
}

# Only add model_traced suite if it has valid configurations
if model_traced_params and any(len(v) > 0 for v in model_traced_params.values() if isinstance(v, list)):
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
    scale_factor=None,
    mode=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    """
    Run upsample test with parameters extracted from traced JSON.
    All parameters are now extracted from JSON including scale_factor.
    """
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Handle tuple input_a_shape for sample suite
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # scale_factor must be extracted from JSON - no fallbacks
    if scale_factor is None:
        raise ValueError(f"Missing scale_factor from JSON")

    # Handle scale_factor - can be int or list [H, W]
    if isinstance(scale_factor, list):
        # If array format [H, W], use first element if both are same, otherwise use tuple
        if len(scale_factor) == 2:
            if scale_factor[0] == scale_factor[1]:
                scale_factor = scale_factor[0]
            else:
                scale_factor = tuple(scale_factor)
        else:
            raise ValueError(f"Invalid scale_factor format from JSON: {scale_factor}")
    elif not isinstance(scale_factor, (int, tuple)):
        raise ValueError(f"Invalid scale_factor type from JSON: {type(scale_factor)}, value: {scale_factor}")

    # mode must be extracted from JSON - no fallbacks
    if mode is None:
        raise ValueError(f"Missing mode from JSON")
    # mode is validated but not used directly (passed via ttnn.upsample kwargs)

    torch_output_tensor = ttnn.get_golden_function(ttnn.upsample)(torch_input_tensor_a, scale_factor=scale_factor)

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

    start_time = start_measuring_time()
    # Handle scale_factor - can be int or tuple
    if isinstance(scale_factor, (tuple, list)) and len(scale_factor) == 2:
        output_tensor = ttnn.upsample(input_tensor_a, scale_factor=tuple(scale_factor), **op_kwargs)
    else:
        output_tensor = ttnn.upsample(input_tensor_a, scale_factor=scale_factor, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
