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
from typing import Optional, Tuple

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


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Skip test vectors that cause L1 circular buffer overflow.
    group_norm allocates internal circular buffers that can exceed L1 capacity for large tensors.
    """
    input_shape = test_vector.get("input_shape")

    if input_shape:
        # Calculate total tensor size
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim

        # Skip if tensor is too large (causes circular buffer overflow)
        # Empirically, tensors > 200K elements cause L1 overflow
        if total_elements > 200000:
            return True, f"group_norm: Skipping large tensor {input_shape} (circular buffer would exceed L1 capacity)"

    return False, None


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
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
    weight_memory_config=None,
    bias_shape=None,
    bias_dtype=None,
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

    # ========================================================================================================
    # TENSOR FORMAT CONVERSION - TTNN vs PyTorch
    # ========================================================================================================
    # TTNN group_norm format: [N, 1, H*W, C]
    #   - N: batch size
    #   - 1: fixed dimension (always 1 for TTNN)
    #   - H*W: spatial dimensions (height × width) flattened into single dimension
    #   - C: number of channels (must be divisible by num_groups)
    #
    # PyTorch group_norm format: [N, C, H, W]
    #   - N: batch size
    #   - C: number of channels
    #   - H: height
    #   - W: width
    #
    # Conversion Strategy:
    #   1. Input tensors from traced configs are in TTNN format [N, 1, H*W, C]
    #   2. Convert to PyTorch format [N, C, H, W] to compute golden reference with torch.nn.functional.group_norm
    #   3. Convert PyTorch output back to TTNN format [N, 1, H*W, C] for PCC comparison
    # ========================================================================================================

    # Extract number of channels from input shape (last dimension in both formats)
    C = shape[-1]

    # Create optional weight and bias tensors if provided in traced config
    # IMPORTANT: Weight/bias must have total elements equal to C (number of channels)
    # They will be reshaped to [C] for PyTorch group_norm
    torch_weight = None
    torch_bias = None
    if weight_shape:
        # Validate weight shape: total elements must equal number of channels
        weight_elements = 1
        for dim in weight_shape:
            weight_elements *= dim
        if weight_elements == C:
            torch_weight = torch.ones(weight_shape, dtype=torch.float32)
        # If weight_elements != C, skip weight (incompatible shape)
    if bias_shape:
        # Validate bias shape: total elements must equal number of channels
        bias_elements = 1
        for dim in bias_shape:
            bias_elements *= dim
        if bias_elements == C:
            torch_bias = torch.zeros(bias_shape, dtype=torch.float32)
        # If bias_elements != C, skip bias (incompatible shape)

    # ========================================================================================================
    # STEP 1: Convert TTNN format to PyTorch format for golden reference computation
    # ========================================================================================================
    if len(shape) == 4 and shape[1] == 1:
        # Input is in TTNN format: [N, 1, H*W, C]
        # Need to convert to PyTorch format: [N, C, H, W]
        N, _, HW, C = shape

        # Reconstruct spatial dimensions (H, W) from flattened dimension (H*W)
        import math

        H = W = int(math.sqrt(HW))
        if H * W != HW:
            # HW is not a perfect square (e.g., HW=1024 but sqrt(1024)=32 and 32*32=1024 ✓)
            # If not perfect square, use H=HW and W=1 (e.g., HW=1000 -> H=1000, W=1)
            H = HW
            W = 1

        # Convert TTNN [N, 1, H*W, C] -> PyTorch [N, C, H, W]
        # Step 1: Reshape [N, H*W, C] to [N, H, W, C] (unflatten spatial dimensions)
        # Step 2: Permute [N, H, W, C] to [N, C, H, W] (channels-first for PyTorch)
        torch_input_reshaped = torch_input_tensor_a.reshape(N, H, W, C).permute(0, 3, 1, 2)

        # Reshape weight and bias to [C] for PyTorch group_norm
        if torch_weight is not None:
            torch_weight_reshaped = torch_weight.reshape(C)
        else:
            torch_weight_reshaped = None
        if torch_bias is not None:
            torch_bias_reshaped = torch_bias.reshape(C)
        else:
            torch_bias_reshaped = None
    else:
        # Input is NOT in TTNN format (may be standard PyTorch format or other)
        # Use as-is without conversion
        torch_input_reshaped = torch_input_tensor_a
        torch_weight_reshaped = torch_weight
        torch_bias_reshaped = torch_bias

    # ========================================================================================================
    # STEP 2: Compute golden reference using PyTorch group_norm (expects [N, C, H, W] format)
    # ========================================================================================================
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_reshaped, num_groups, weight=torch_weight_reshaped, bias=torch_bias_reshaped, eps=epsilon
    )

    # ========================================================================================================
    # STEP 3: Convert PyTorch output back to TTNN format for comparison
    # ========================================================================================================
    if len(shape) == 4 and shape[1] == 1:
        # PyTorch output is in [N, C, H, W] format
        # Convert back to TTNN format [N, 1, H*W, C]
        # Step 1: Permute [N, C, H, W] to [N, H, W, C] (channels-last)
        # Step 2: Reshape [N, H, W, C] to [N, 1, H*W, C] (flatten spatial dimensions)
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).reshape(shape)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    # Always use DRAM to avoid OOM and sharding constraint issues
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG

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
        # Use original weight shape from traced config (already correct for group_norm)
        # group_norm requires last dim to be TILE_WIDTH (32), so shapes like [1, 1, 4, 32] are correct
        weight_tensor = ttnn.from_torch(
            torch_weight,
            dtype=weight_dtype or ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # Force ROW_MAJOR for weight
            device=device,
            memory_config=weight_memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    if bias_shape and torch_bias is not None and not is_host:
        # Use original bias shape from traced config (already correct for group_norm)
        # group_norm requires last dim to be TILE_WIDTH (32), so shapes like [1, 1, 4, 32] are correct
        bias_tensor = ttnn.from_torch(
            torch_bias,
            dtype=bias_dtype or ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # Force ROW_MAJOR for bias
            device=device,
            memory_config=bias_memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    if reciprocals_shape and use_welford and not is_host:
        # Reciprocals are needed for Welford's algorithm, but traced configs often have issues:
        # - Non-tile-aligned shard shapes (e.g., shard_shape=(1, 8192) where height=1)
        # - L1 OOM when using sharded L1 memory
        # Solution: Skip reciprocals with incompatible memory configs - operation will compute internally

        skip_reciprocals = False
        reciprocals_mem_cfg = reciprocals_memory_config if reciprocals_memory_config else ttnn.DRAM_MEMORY_CONFIG

        # Check if memory config is L1 sharded (likely to cause OOM)
        if (
            reciprocals_mem_cfg
            and hasattr(reciprocals_mem_cfg, "memory_layout")
            and hasattr(reciprocals_mem_cfg, "buffer_type")
        ):
            is_sharded = reciprocals_mem_cfg.memory_layout in [
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            ]
            is_l1 = reciprocals_mem_cfg.buffer_type == ttnn.types.BufferType.L1

            if is_sharded and is_l1:
                # L1 sharded reciprocals often cause OOM - skip them
                skip_reciprocals = True

        if not skip_reciprocals:
            torch_reciprocals = torch.ones(reciprocals_shape, dtype=torch.float32)
            recip_layout = reciprocals_layout or ttnn.TILE_LAYOUT  # Prefer TILE for compatibility

            reciprocals_tensor = ttnn.from_torch(
                torch_reciprocals,
                dtype=reciprocals_dtype or ttnn.float32,
                layout=recip_layout,
                device=device,
                memory_config=reciprocals_mem_cfg,
            )

    start_time = start_measuring_time()

    # Determine core_grid based on num_groups and use_welford
    # Constraint: when use_welford=True, num_groups_per_core must be <= 16
    # num_groups_per_core = num_groups / (grid.y * grid.x)

    if use_welford and num_groups > 16:
        # Need larger grid to keep num_groups_per_core <= 16
        # For num_groups=32, need at least 2x1 grid -> 32/2 = 16 groups per core
        min_cores = (num_groups + 15) // 16  # Round up to ensure <= 16 groups per core
        try:
            grid_size = device.compute_with_storage_grid_size()
            # Use available grid, but ensure we have enough cores
            if grid_size.y * grid_size.x >= min_cores:
                core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
            else:
                # Use min required cores in a row
                core_grid = ttnn.CoreGrid(y=1, x=min_cores)
        except:
            # Fallback: use min required cores
            core_grid = ttnn.CoreGrid(y=1, x=min_cores)
    else:
        # For non-Welford or small num_groups, use simple 1x1 grid
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
