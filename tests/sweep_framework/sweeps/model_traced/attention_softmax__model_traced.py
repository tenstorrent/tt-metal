# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::attention_softmax_")

# Parameters provided to the test vector generator are defined here.
parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
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
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    scalar=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    """
    attention_softmax_: in-place attention softmax with mask

    Based on working transformer sweep test implementation.
    Key difference from unit tests: uses binary mask (0/1) instead of causal_mask parameter.
    """
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"head_size"})

    # Parse input_a_shape - can be tuple/list or dict (from binary operation extraction)
    if isinstance(input_a_shape, dict):
        # Binary operation format: {"input_a": shape_a, "input_b": shape_b}
        shape_a = tuple(input_a_shape.get("input_a", input_a_shape.get("self", [1, 32, 32])))
        shape_b = tuple(input_a_shape.get("input_b", input_a_shape.get("other", shape_a)))
    elif isinstance(input_a_shape, (tuple, list)):
        shape_a = tuple(input_a_shape)
        shape_b = shape_a  # Mask has same shape as input
    else:
        shape_a = input_a_shape
        shape_b = shape_a

    # Get head_size from scalar if provided (as traced configs do)
    # Ensure it's an int, not float
    head_size = int(scalar) if scalar is not None else None

    # Generate input tensor
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # attention_softmax_ requires an attention_mask
    # Use binary mask (0 or 1) as in the working transformer sweep test
    # NOT -inf masks as in some unit tests
    torch_mask_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32),
        input_b_dtype if input_b_dtype else input_a_dtype,
    )(shape_b)
    # Convert to binary mask: values > 0 become 1, else 0
    torch_mask_tensor = (torch_mask_tensor > 0).to(torch.float32)

    # Get golden output using the ttnn golden function
    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax_)
    # Clone input as operation is in-place
    tmp_input = torch.clone(torch_input_tensor)
    torch_output_tensor = golden_function(tmp_input, head_size=head_size, attention_mask=torch_mask_tensor)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Convert to TTNN tensors with mesh support and interleaved→sharded fallback
    input_is_sharded = hasattr(input_a_memory_config, "is_sharded") and input_a_memory_config.is_sharded()

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input_tensor,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        elif input_is_sharded:
            input_tensor = ttnn.from_torch(
                torch_input_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            input_tensor = ttnn.interleaved_to_sharded(input_tensor, input_a_memory_config)
        else:
            input_tensor = ttnn.from_torch(
                torch_input_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_a_dtype, layout=input_a_layout)

    # Convert attention mask to TTNN tensor
    mask_dtype = input_b_dtype if input_b_dtype else input_a_dtype
    mask_layout = input_b_layout if input_b_layout else input_a_layout
    mask_memory_config = input_b_memory_config if input_b_memory_config else input_a_memory_config
    mask_is_sharded = hasattr(mask_memory_config, "is_sharded") and mask_memory_config.is_sharded()

    if not is_host:
        if is_mesh_device and input_b_tensor_placement:
            mask_tensor = create_tensor_on_mesh(
                torch_mask_tensor,
                device,
                mask_dtype,
                mask_layout,
                mask_memory_config,
                input_b_tensor_placement,
            )
        elif mask_is_sharded:
            mask_tensor = ttnn.from_torch(
                torch_mask_tensor,
                dtype=mask_dtype,
                layout=mask_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mask_tensor = ttnn.interleaved_to_sharded(mask_tensor, mask_memory_config)
        else:
            mask_tensor = ttnn.from_torch(
                torch_mask_tensor,
                dtype=mask_dtype,
                layout=mask_layout,
                device=device,
                memory_config=mask_memory_config,
            )
    else:
        mask_tensor = ttnn.from_torch(torch_mask_tensor, dtype=mask_dtype, layout=mask_layout)

    # Run operation (in-place operation modifies input)
    # Note: attention_softmax_ does NOT support numeric_stable parameter
    # Do NOT use causal_mask parameter - use the binary mask instead
    start_time = start_measuring_time()
    result = ttnn.transformer.attention_softmax_(
        input_tensor, head_size=head_size, attention_mask=mask_tensor, **op_kwargs
    )
    output_tensor = mesh_tensor_to_torch(result, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check PCC - using 0.999 as in transformer sweep test
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    # Return result in the format expected by sweeps_runner
    return [pcc, e2e_perf]
