# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader and helpers for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import (
    MasterConfigLoader,
    dict_to_memory_config,
    dict_to_core_grid,
    dict_to_compute_kernel_config,
    dict_to_program_config,
    parse_dtype,
)

# Override the default timeout in seconds for hang detection.
# Linear operations with large shapes can take longer, increase timeout
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("linear")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(32, 32)],  # Input shape (m, k)
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(32, 32)],  # Weight shape (k, n)
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "bias_shape": [(32,)],  # Bias shape (n,) - optional
        "bias_dtype": [ttnn.bfloat16],
        "bias_layout": [ttnn.TILE_LAYOUT],
        "transpose_a": [False],
        "transpose_b": [False],
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
        # Create mesh device based on env var
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"⚠️ Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # Single device (default)
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,  # Input shape (m, k)
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape,  # Weight shape (k, n)
    input_b_dtype,
    input_b_layout,
    input_b_memory_config,
    bias_shape=None,  # Optional bias shape (n,)
    bias_dtype=None,
    bias_layout=None,
    bias_memory_config=None,
    transpose_a=False,
    transpose_b=False,
    storage_type="StorageType::DEVICE",
    memory_config=None,  # Alternative memory_config parameter
    dtype=None,  # Output dtype
    core_grid=None,  # Core grid configuration
    program_config=None,  # Program configuration
    compute_kernel_config=None,  # Compute kernel configuration
    activation=None,  # Activation function
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Convert traced dict params to proper ttnn objects
    memory_config = dict_to_memory_config(memory_config)
    core_grid = dict_to_core_grid(core_grid)
    compute_kernel_config = dict_to_compute_kernel_config(compute_kernel_config)
    if isinstance(dtype, (dict, str)):
        dtype = parse_dtype(dtype.get("repr", "") if isinstance(dtype, dict) else dtype)
    if isinstance(program_config, dict):
        program_config = dict_to_program_config(program_config, input_b_memory_config, input_a_memory_config)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    bias_tensor_placement = kwargs.get("bias_tensor_placement", None)
    output_memory_config = dict_to_memory_config(kwargs.get("output_memory_config", None))

    # When program_config can't be reconstructed (incomplete traced data), the
    # shard_spec in memory_config/output_memory_config was computed by the
    # original program_config and is invalid without it. Also, DRAM-sharded
    # input B requires a matching DRAMSharded program_config. Clear all
    # sharded configs so ttnn.linear auto-determines compatible settings.
    if program_config is None:
        if memory_config is not None and "SHARDED" in str(memory_config):
            memory_config = None
        if output_memory_config is not None and "SHARDED" in str(output_memory_config):
            output_memory_config = None
        if input_b_memory_config is not None and "SHARDED" in str(input_b_memory_config):
            input_b_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method

    # V2 format provides separate shapes
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if isinstance(input_b_shape, (list, tuple)) else input_b_shape

    # Create random tensors
    torch_a = torch.randn(*shape_a, dtype=torch.float32)
    torch_b = torch.randn(*shape_b, dtype=torch.float32)

    # For linear operations, use the weight as-is (TTNN handles the format)
    torch_weight = torch_b

    # Create bias tensor if needed
    torch_bias = None
    ttnn_bias = None
    has_bias = bias_shape is not None and bias_shape != tuple()

    if has_bias:
        shape_bias = tuple(bias_shape) if isinstance(bias_shape, (list, tuple)) else bias_shape
        torch_bias = torch.randn(*shape_bias, dtype=torch.float32) if shape_bias != tuple() else torch.randn(())

        # Check if storage_type is HOST
        is_host = storage_type and "HOST" in str(storage_type)

        # Create bias tensor with mesh support if needed
        if not is_host:
            if is_mesh_device and bias_tensor_placement:
                ttnn_bias = create_tensor_on_mesh(
                    torch_bias,
                    device,
                    bias_dtype if bias_dtype else input_a_dtype,
                    bias_layout if bias_layout else input_a_layout,
                    bias_memory_config if bias_memory_config else ttnn.DRAM_MEMORY_CONFIG,
                    bias_tensor_placement,
                )
            else:
                ttnn_bias = ttnn.from_torch(
                    torch_bias,
                    dtype=bias_dtype if bias_dtype else input_a_dtype,
                    layout=bias_layout if bias_layout else input_a_layout,
                    device=device,
                    memory_config=bias_memory_config if bias_memory_config else ttnn.DRAM_MEMORY_CONFIG,
                )
        else:
            ttnn_bias = ttnn.from_torch(
                torch_bias,
                dtype=bias_dtype if bias_dtype else input_a_dtype,
                layout=bias_layout if bias_layout else input_a_layout,
            )

    # Golden output using PyTorch
    # Use matmul for multi-dimensional tensors (like traced 4D configs)
    # Use linear for 2D tensors (like sample tests)
    if len(torch_a.shape) > 2:
        torch_output_tensor = torch.matmul(torch_a, torch_weight)
        if torch_bias is not None:
            torch_output_tensor = torch_output_tensor + torch_bias
    else:
        torch_weight_for_linear = torch_weight
        if len(torch_weight.shape) >= 2:
            torch_weight_for_linear = torch_weight.transpose(-1, -2)
        torch_output_tensor = torch.nn.functional.linear(torch_a, torch_weight_for_linear, torch_bias)

    # Apply activation to golden reference to match ttnn.linear behavior
    if activation is not None:
        act = str(activation).lower()
        if "silu" in act or "swish" in act:
            torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)
        elif "gelu" in act:
            approx = "tanh" if "approx" in act else "none"
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor, approximate=approx)
        elif "relu" in act:
            torch_output_tensor = torch.nn.functional.relu(torch_output_tensor)

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Create input tensor A
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            ttnn_a = create_tensor_on_mesh(
                torch_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            ttnn_a = ttnn.from_torch(
                torch_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        ttnn_a = ttnn.from_torch(torch_a, dtype=input_a_dtype, layout=input_a_layout)

    # Create weight tensor B
    # Use the traced memory config as-is - with program_config, sharded weights may be supported
    weight_memory_config = input_b_memory_config

    if not is_host:
        if is_mesh_device and input_b_tensor_placement:
            # Use mesh with placement
            ttnn_b = create_tensor_on_mesh(
                torch_b,
                device,
                input_b_dtype,
                input_b_layout,
                weight_memory_config,
                input_b_tensor_placement,
            )
        else:
            # Regular single-device tensor
            ttnn_b = ttnn.from_torch(
                torch_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=weight_memory_config,
            )
    else:
        # Host storage
        ttnn_b = ttnn.from_torch(torch_b, dtype=input_b_dtype, layout=input_b_layout)

    # Run TTNN linear
    start_time = start_measuring_time()

    # Prepare linear kwargs
    linear_kwargs = {
        "bias": ttnn_bias,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
    }

    # Add optional parameters from traced config
    if memory_config is not None:
        linear_kwargs["memory_config"] = memory_config
    elif output_memory_config is not None:
        linear_kwargs["memory_config"] = output_memory_config

    if dtype is not None:
        linear_kwargs["dtype"] = dtype

    if program_config is not None:
        linear_kwargs["program_config"] = program_config

    if compute_kernel_config is not None:
        linear_kwargs["compute_kernel_config"] = compute_kernel_config

    if core_grid is not None:
        linear_kwargs["core_grid"] = core_grid

    if activation is not None:
        linear_kwargs["activation"] = activation

    output_tensor = ttnn.linear(ttnn_a, ttnn_b, **linear_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
