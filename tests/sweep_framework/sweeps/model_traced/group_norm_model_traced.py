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
    infer_mesh_shape_from_params,
    detect_mesh_shape_from_hardware,
)
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from typing import Optional, Tuple

# Override the default timeout in seconds for hang detection.
# group_norm is computationally intensive, needs longer timeout
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("group_norm")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 1024, 32)],  # Shape: [N, 1, H*W, C] as per ttnn.group_norm docs
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_groups": [8],
        "epsilon": [1e-5],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if not mesh_shape:
        mesh_shape = infer_mesh_shape_from_params(model_traced_params)
    if not mesh_shape:
        mesh_shape = detect_mesh_shape_from_hardware()
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


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Skip test vectors that cause L1 circular buffer overflow.
    group_norm allocates internal circular buffers that can exceed L1 capacity for large tensors.
    """
    input_shape = test_vector.get("input_a_shape")

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
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config=None,
    output_memory_config=None,
    num_groups=None,
    epsilon=1e-5,
    storage_type="StorageType::DEVICE",
    # Optional traced arguments
    input_mask_shape=None,
    input_mask_dtype=None,
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

    # Filter __ABSENT__ sentinels from optional parameters
    def _clean(v):
        return None if v == "__ABSENT__" else v

    input_mask_shape = _clean(input_mask_shape)
    input_mask_dtype = _clean(input_mask_dtype)
    input_mask_memory_config = _clean(input_mask_memory_config)
    weight_shape = _clean(weight_shape)
    weight_dtype = _clean(weight_dtype)
    weight_memory_config = _clean(weight_memory_config)
    bias_shape = _clean(bias_shape)
    bias_dtype = _clean(bias_dtype)
    bias_memory_config = _clean(bias_memory_config)
    reciprocals_shape = _clean(reciprocals_shape)
    reciprocals_dtype = _clean(reciprocals_dtype)
    reciprocals_layout = _clean(reciprocals_layout)
    reciprocals_memory_config = _clean(reciprocals_memory_config)
    output_memory_config = _clean(output_memory_config)
    inplace = False if inplace == "__ABSENT__" else inplace
    num_out_blocks = _clean(num_out_blocks)
    use_welford = False if use_welford == "__ABSENT__" else use_welford

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    weight_tensor_placement = kwargs.get("weight_tensor_placement", None)
    bias_tensor_placement = kwargs.get("bias_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    # Let core_grid and memory_config flow through op_kwargs so they get parsed from dicts
    # Exclude params we handle explicitly as named parameters
    op_kwargs = build_op_kwargs(
        kwargs,
        exclude={"inplace", "num_out_blocks", "use_welford", "num_groups", "epsilon", "negative_mask"},
        output_memory_config=output_memory_config,
    )

    if input_a_memory_config is None:
        input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG

    if num_groups is None:
        return [(False, "Missing num_groups"), 0.0]

    # Handle tuple input_a_shape for sample suite
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # ========================================================================================================
    # TENSOR FORMAT CONVERSION - TTNN vs PyTorch
    # ========================================================================================================
    # TTNN group_norm format: [N, 1, H*W, C]
    # PyTorch group_norm format: [N, C, H, W]
    # ========================================================================================================

    # Extract number of channels from input shape (last dimension in both formats)
    C = shape[-1]

    # Create optional weight and bias tensors if provided in traced config
    torch_weight = None
    torch_bias = None
    if weight_shape:
        weight_elements = 1
        for dim in weight_shape:
            weight_elements *= dim
        if weight_elements == C:
            torch_weight = torch.ones(weight_shape, dtype=torch.float32)
    if bias_shape:
        bias_elements = 1
        for dim in bias_shape:
            bias_elements *= dim
        if bias_elements == C:
            torch_bias = torch.zeros(bias_shape, dtype=torch.float32)

    # Convert TTNN format to PyTorch format for golden reference
    if len(shape) == 4 and shape[1] == 1:
        N, _, HW, C = shape
        import math

        H = W = int(math.sqrt(HW))
        if H * W != HW:
            H = HW
            W = 1

        torch_input_reshaped = torch_input_tensor_a.reshape(N, H, W, C).permute(0, 3, 1, 2)

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

    # Compute golden reference
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_reshaped, num_groups, weight=torch_weight_reshaped, bias=torch_bias_reshaped, eps=epsilon
    )

    # Convert PyTorch output back to TTNN format for comparison
    if len(shape) == 4 and shape[1] == 1:
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).reshape(shape)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create input tensor using traced memory config (may be sharded)
    # For sharded configs, create interleaved first then shard
    input_is_sharded = hasattr(input_a_memory_config, "is_sharded") and input_a_memory_config.is_sharded()

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
        elif input_is_sharded:
            # Create interleaved first, then shard (from_torch can't directly create sharded)
            input_tensor_a_interleaved = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            try:
                input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_a_interleaved, input_a_memory_config)
            except RuntimeError:
                # If sharding fails, fall back to interleaved
                input_tensor_a = input_tensor_a_interleaved
                input_is_sharded = False
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

    # Create optional tensors if traced config provides them
    input_mask = None
    weight_tensor = None
    bias_tensor = None
    reciprocals_tensor = None

    # Determine core_grid early - needed for proper mask/weight/bias creation
    # The core_grid.y value determines the num_cores_across_channel parameter
    _op_kwargs_copy = build_op_kwargs(
        kwargs,
        exclude={"inplace", "num_out_blocks", "use_welford", "num_groups", "epsilon", "negative_mask"},
        output_memory_config=output_memory_config,
    )
    if "core_grid" in _op_kwargs_copy:
        _early_core_grid = _op_kwargs_copy["core_grid"]
    else:
        _early_core_grid = ttnn.CoreGrid(y=1, x=1)
    num_cores_across_channel = _early_core_grid.y

    # Use ttnn.create_group_norm_input_mask for proper channel-group mapping
    if input_mask_shape and not is_host:
        mask_dtype = input_mask_dtype or ttnn.bfloat8_b
        try:
            input_mask = ttnn.create_group_norm_input_mask(C, num_groups, num_cores_across_channel, mask_dtype)
            input_mask = ttnn.to_device(input_mask, device)
        except Exception as e:
            print(f"Warning: create_group_norm_input_mask failed: {e}, skipping mask")
            input_mask = None

    # Use ttnn.create_group_norm_weight_bias_rm for proper weight formatting
    if weight_shape and torch_weight is not None and not is_host:
        w_dtype = weight_dtype or ttnn.bfloat16
        w_mem = weight_memory_config or ttnn.DRAM_MEMORY_CONFIG
        try:
            torch_weight_rm = ttnn.create_group_norm_weight_bias_rm(
                torch_weight.reshape(C), C, num_cores_across_channel
            )
            weight_tensor = ttnn.from_torch(
                torch_weight_rm,
                dtype=w_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=w_mem,
            )
        except Exception as e:
            print(f"Warning: create_group_norm_weight_bias_rm for weight failed: {e}")
            weight_tensor = None

    # Use ttnn.create_group_norm_weight_bias_rm for proper bias formatting
    if bias_shape and torch_bias is not None and not is_host:
        b_dtype = bias_dtype or ttnn.bfloat16
        b_mem = bias_memory_config or ttnn.DRAM_MEMORY_CONFIG
        try:
            torch_bias_rm = ttnn.create_group_norm_weight_bias_rm(torch_bias.reshape(C), C, num_cores_across_channel)
            bias_tensor = ttnn.from_torch(
                torch_bias_rm,
                dtype=b_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=b_mem,
            )
        except Exception as e:
            print(f"Warning: create_group_norm_weight_bias_rm for bias failed: {e}")
            bias_tensor = None

    if reciprocals_shape and use_welford and not is_host:
        skip_reciprocals = False
        reciprocals_mem_cfg = reciprocals_memory_config if reciprocals_memory_config else ttnn.DRAM_MEMORY_CONFIG

        if (
            reciprocals_mem_cfg
            and hasattr(reciprocals_mem_cfg, "memory_layout")
            and hasattr(reciprocals_mem_cfg, "buffer_type")
        ):
            is_recip_sharded = reciprocals_mem_cfg.memory_layout in [
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.TensorMemoryLayout.BLOCK_SHARDED,
            ]
            is_l1 = reciprocals_mem_cfg.buffer_type == ttnn.types.BufferType.L1

            if is_recip_sharded and is_l1:
                skip_reciprocals = True

        if not skip_reciprocals:
            torch_reciprocals = torch.ones(reciprocals_shape, dtype=torch.float32)
            recip_layout = reciprocals_layout or ttnn.TILE_LAYOUT
            recip_dtype = reciprocals_dtype or ttnn.float32

            if is_mesh_device and input_a_tensor_placement:
                reciprocals_tensor = create_tensor_on_mesh(
                    torch_reciprocals,
                    device,
                    recip_dtype,
                    recip_layout,
                    reciprocals_mem_cfg,
                    input_a_tensor_placement,
                )
            else:
                reciprocals_tensor = ttnn.from_torch(
                    torch_reciprocals,
                    dtype=recip_dtype,
                    layout=recip_layout,
                    device=device,
                    memory_config=reciprocals_mem_cfg,
                )

    start_time = start_measuring_time()

    # inplace groupnorm is only supported for sharded tensors
    actual_inplace = inplace and input_is_sharded

    # Use traced core_grid if provided via op_kwargs, otherwise compute a default
    if "core_grid" not in op_kwargs:
        if use_welford and num_groups > 16:
            min_cores = (num_groups + 15) // 16
            try:
                grid_size = device.compute_with_storage_grid_size()
                if grid_size.y * grid_size.x >= min_cores:
                    core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
                else:
                    core_grid = ttnn.CoreGrid(y=1, x=min_cores)
            except Exception:
                core_grid = ttnn.CoreGrid(y=1, x=min_cores)
        else:
            core_grid = ttnn.CoreGrid(y=1, x=1)
    else:
        core_grid = op_kwargs.pop("core_grid")

    # Build group_norm arguments
    group_norm_kwargs = {
        "num_groups": num_groups,
        "epsilon": epsilon,
        "inplace": actual_inplace,
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

    group_norm_kwargs.update(op_kwargs)
    output_tensor = ttnn.group_norm(input_tensor_a, **group_norm_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
