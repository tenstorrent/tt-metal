# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
# group_norm is computationally intensive, needs longer timeout
TIMEOUT = 300

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("group_norm", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 1024, 32)],  # Shape: [N, 1, H*W, C] as per ttnn.group_norm docs
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_groups": [8],
        "epsilon": [1e-5],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture.
    Using explicit DispatchCoreConfig to handle sharded memory configs.
    """
    import ttnn

    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_device(device)
    del device


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    num_groups,
    epsilon=1e-5,
    storage_type="StorageType::DEVICE",
    # Optional traced arguments
    input_mask_shape=None,
    input_mask_dtype=None,
    input_mask_layout=None,
    input_mask_memory_config=None,
    weight_shape=None,
    weight_dtype=None,
    weight_layout=None,
    weight_memory_config=None,
    bias_shape=None,
    bias_dtype=None,
    bias_layout=None,
    bias_memory_config=None,
    reciprocals_shape=None,
    reciprocals_dtype=None,
    reciprocals_layout=None,
    reciprocals_memory_config=None,
    inplace=False,
    num_out_blocks=None,
    use_welford=False,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Create optional weight and bias tensors if provided in traced config
    # IMPORTANT: Validate that weight/bias shapes are compatible with input channels
    # Weight and bias last dimension must match input's last dimension
    C = shape[-1]
    torch_weight = None
    torch_bias = None
    if weight_shape:
        # Check if weight shape is compatible
        weight_elements = 1
        for dim in weight_shape:
            weight_elements *= dim
        # Weight must have total elements == C (will be reshaped to [C])
        if weight_elements == C:
            torch_weight = torch.ones(weight_shape, dtype=torch.float32)
    if bias_shape:
        # Check if bias shape is compatible
        bias_elements = 1
        for dim in bias_shape:
            bias_elements *= dim
        # Bias must have total elements == C (will be reshaped to [C])
        if bias_elements == C:
            torch_bias = torch.zeros(bias_shape, dtype=torch.float32)

    # ttnn group_norm expects shape [N, 1, H*W, C] where C is divisible by num_groups
    # torch group_norm expects shape (N, C, *)
    if len(shape) == 4 and shape[1] == 1:
        # ttnn format: [N, 1, H*W, C] -> torch format: [N, C, H, W]
        N, _, HW, C = shape
        # Calculate H and W
        import math

        H = W = int(math.sqrt(HW))
        if H * W != HW:
            # If not a perfect square, use H*W and W=1
            H = HW
            W = 1
        torch_input_reshaped = torch_input_tensor_a.reshape(N, H, W, C).permute(0, 3, 1, 2)

        # Reshape weight and bias for torch
        if torch_weight is not None:
            torch_weight_reshaped = torch_weight.reshape(C)
        else:
            torch_weight_reshaped = None
        if torch_bias is not None:
            torch_bias_reshaped = torch_bias.reshape(C)
        else:
            torch_bias_reshaped = None
    else:
        torch_input_reshaped = torch_input_tensor_a
        torch_weight_reshaped = torch_weight
        torch_bias_reshaped = torch_bias

    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_reshaped, num_groups, weight=torch_weight_reshaped, bias=torch_bias_reshaped, eps=epsilon
    )

    # Reshape back to ttnn format
    if len(shape) == 4 and shape[1] == 1:
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).reshape(shape)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    # Create optional tensors if traced config provides them
    input_mask = None
    weight_tensor = None
    bias_tensor = None
    reciprocals_tensor = None

    if input_mask_shape and not is_host:
        torch_input_mask = torch.ones(input_mask_shape, dtype=torch.float32)
        input_mask = ttnn.from_torch(
            torch_input_mask,
            dtype=input_mask_dtype or ttnn.bfloat16,
            layout=input_mask_layout or ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_mask_memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    if weight_shape and torch_weight is not None and not is_host:
        weight_tensor = ttnn.from_torch(
            torch_weight,
            dtype=weight_dtype or ttnn.bfloat16,
            layout=weight_layout or ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=weight_memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    if bias_shape and torch_bias is not None and not is_host:
        bias_tensor = ttnn.from_torch(
            torch_bias,
            dtype=bias_dtype or ttnn.bfloat16,
            layout=bias_layout or ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=bias_memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    if reciprocals_shape and use_welford and not is_host:
        # Create reciprocals tensor (needed for Welford's algorithm)
        torch_reciprocals = torch.ones(reciprocals_shape, dtype=torch.float32)

        # Use the reciprocals memory config if provided, otherwise use DRAM
        reciprocals_mem_cfg = reciprocals_memory_config if reciprocals_memory_config else ttnn.DRAM_MEMORY_CONFIG

        # For sharded memory configs with ROW_MAJOR layout, use TILE layout instead
        # ROW_MAJOR + HEIGHT_SHARDED requires tile-aligned shard shapes
        recip_layout = reciprocals_layout or ttnn.ROW_MAJOR_LAYOUT
        if reciprocals_mem_cfg and hasattr(reciprocals_mem_cfg, "memory_layout"):
            if reciprocals_mem_cfg.memory_layout in [
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            ]:
                # Use TILE layout for sharded configs (handles non-tile-aligned shard shapes)
                recip_layout = ttnn.TILE_LAYOUT

        reciprocals_tensor = ttnn.from_torch(
            torch_reciprocals,
            dtype=reciprocals_dtype or ttnn.float32,
            layout=recip_layout,
            device=device,
            memory_config=reciprocals_mem_cfg,
        )

    start_time = start_measuring_time()

    # Determine core_grid (try to infer from device or use default)
    # Use a smaller grid for sample tests that don't have reciprocals/welford
    if reciprocals_tensor is not None and use_welford:
        try:
            grid_size = device.compute_with_storage_grid_size()
            core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
        except:
            core_grid = ttnn.CoreGrid(y=8, x=8)
    else:
        # For simple tests without Welford, use smaller grid
        core_grid = ttnn.CoreGrid(y=1, x=1)

    # Build group_norm arguments
    group_norm_kwargs = {
        "num_groups": num_groups,
        "epsilon": epsilon,
        "inplace": inplace,
        "core_grid": core_grid,
        "memory_config": output_memory_config,
    }

    # Add optional arguments if they exist
    if input_mask is not None:
        group_norm_kwargs["input_mask"] = input_mask
    if weight_tensor is not None:
        group_norm_kwargs["weight"] = weight_tensor
    if bias_tensor is not None:
        group_norm_kwargs["bias"] = bias_tensor
    if reciprocals_tensor is not None:
        group_norm_kwargs["reciprocals"] = reciprocals_tensor
    if num_out_blocks is not None:
        group_norm_kwargs["num_out_blocks"] = num_out_blocks
    if use_welford:
        group_norm_kwargs["use_welford"] = use_welford

    output_tensor = ttnn.group_norm(input_tensor_a, **group_norm_kwargs)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
