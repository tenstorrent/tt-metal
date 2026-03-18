# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("remainder")

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
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


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
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    raw_placement_a = kwargs.get("input_a_tensor_placement", None)
    input_a_tensor_placement = raw_placement_a
    raw_placement_b = kwargs.get("input_b_tensor_placement", None)
    input_b_tensor_placement = raw_placement_b
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config or ttnn.DRAM_MEMORY_CONFIG)

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    # Check if arg1 is a scalar (traced configs often have remainder(tensor, scalar))
    # The V2 loader stores non-tensor positional args as "arg1" in kwargs.
    arg1_scalar = kwargs.get("arg1", None)
    if arg1_scalar is not None and arg1_scalar != "__ABSENT__":
        # arg1 might be a scalar value or a dict with {"value": ...}
        if isinstance(arg1_scalar, dict) and "value" in arg1_scalar:
            arg1_scalar = arg1_scalar["value"]
        if isinstance(arg1_scalar, (int, float)):
            scalar_b = arg1_scalar
        else:
            scalar_b = None
    else:
        scalar_b = None

    # Determine if second operand is a tensor or scalar.
    # Filter out __ABSENT__ sentinel values from V2 loader.
    has_tensor_b = (
        input_b_shape is not None
        and input_b_shape != "__ABSENT__"
        and input_b_dtype is not None
        and input_b_dtype != "__ABSENT__"
    )
    use_scalar_b = scalar_b is not None and not has_tensor_b

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    if use_scalar_b:
        # Scalar remainder: ttnn.remainder(tensor, scalar)
        # Ensure scalar is positive to avoid division by zero
        if scalar_b == 0:
            scalar_b = 1
        torch_output_tensor = torch.remainder(torch_input_tensor_a, scalar_b)
    else:
        # Tensor remainder: ttnn.remainder(tensor, tensor)
        shape_b = (
            tuple(input_b_shape)
            if input_b_shape is not None and isinstance(input_b_shape, (list, tuple))
            else input_b_shape
        )
        if shape_b is None:
            shape_b = shape_a

        # Avoid division by zero for remainder
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=1, high=100, dtype=torch.float32), input_b_dtype or input_a_dtype
        )(shape_b)

        torch_output_tensor = torch.remainder(torch_input_tensor_a, torch_input_tensor_b)

    is_host = storage_type and "HOST" in str(storage_type)

    # Create tensor A
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

    if use_scalar_b:
        # Pass scalar directly to ttnn.remainder
        input_b_operand = scalar_b
    else:
        # Create tensor B
        if not is_host:
            if is_mesh_device and input_b_tensor_placement:
                input_b_operand = create_tensor_on_mesh(
                    torch_input_tensor_b,
                    device,
                    input_b_dtype or input_a_dtype,
                    input_b_layout or input_a_layout,
                    input_b_memory_config or input_a_memory_config,
                    input_b_tensor_placement,
                )
            else:
                input_b_operand = ttnn.from_torch(
                    torch_input_tensor_b,
                    dtype=input_b_dtype or input_a_dtype,
                    layout=input_b_layout or input_a_layout,
                    device=device,
                    memory_config=input_b_memory_config or input_a_memory_config,
                )
        else:
            input_b_operand = ttnn.from_torch(
                torch_input_tensor_b,
                dtype=input_b_dtype or input_a_dtype,
                layout=input_b_layout or input_a_layout,
            )

    start_time = start_measuring_time()
    output_tensor = ttnn.remainder(input_tensor_a, input_b_operand, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
