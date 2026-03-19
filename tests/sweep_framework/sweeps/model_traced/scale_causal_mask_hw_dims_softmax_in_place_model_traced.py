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
model_traced_params = loader.get_suite_parameters("scale_causal_mask_hw_dims_softmax_in_place")

parameters = {}

# This op requires HEIGHT_SHARDED input + SoftmaxShardedMultiCoreProgramConfig;
# no simple sample suite is possible. Only traced configs are used.
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
    scale_causal_mask_hw_dims_softmax_in_place: softmax(scale * input + causal_mask)

    Positional args from JSON:
        arg0: input tensor (HEIGHT_SHARDED L1)
        arg1: scale (float, e.g. 0.125)
        arg2: causal mask tensor (BFLOAT4_B, INTERLEAVED DRAM)
    Named kwargs: program_config, compute_kernel_config
    """
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"head_size", "program_config"})
    if (
        "memory_config" in op_kwargs
        and hasattr(op_kwargs["memory_config"], "is_sharded")
        and op_kwargs["memory_config"].is_sharded()
    ):
        del op_kwargs["memory_config"]

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    # Scale factor (arg1 in JSON, passed as scalar by loader)
    scale = float(scalar) if scalar is not None else 1.0

    # Generate input tensor
    torch_input_a = gen_func_with_cast_tt(partial(torch_random, low=-10, high=10, dtype=torch.float32), input_a_dtype)(
        shape_a
    )

    # Generate causal mask if mask params are provided (arg2 in JSON)
    mask_shape = kwargs.get("input_b_shape", None)
    if mask_shape is not None:
        mask_shape = tuple(mask_shape) if isinstance(mask_shape, (list, tuple)) else mask_shape
        torch_mask = gen_func_with_cast_tt(
            partial(torch_random, low=-10, high=10, dtype=torch.float32),
            input_b_dtype if input_b_dtype else input_a_dtype,
        )(mask_shape)
    else:
        torch_mask = None

    # Golden: softmax(scale * input + mask)
    golden_input = scale * torch_input_a.float()
    if torch_mask is not None:
        mask_float = torch_mask.float()
        # Pad mask height to match input if needed (causal mask may be smaller)
        input_h = golden_input.shape[-2]
        mask_h = mask_float.shape[-2]
        if mask_h < input_h:
            pad_h = input_h - mask_h
            mask_float = torch.nn.functional.pad(mask_float, (0, 0, pad_h, 0), value=0.0)
        elif mask_h > input_h:
            mask_float = mask_float[..., -input_h:, :]
        # Pad mask width if needed
        input_w = golden_input.shape[-1]
        mask_w = mask_float.shape[-1]
        if mask_w < input_w:
            pad_w = input_w - mask_w
            mask_float = torch.nn.functional.pad(mask_float, (pad_w, 0, 0, 0), value=0.0)
        elif mask_w > input_h:
            mask_float = mask_float[..., -input_w:]
        golden_input = golden_input + mask_float
    x_max = torch.max(golden_input, dim=-1, keepdim=True)[0]
    x_exp = torch.exp(golden_input - x_max)
    torch_output = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

    is_host = storage_type and "HOST" in str(storage_type)

    # Create input tensor with interleaved→sharded fallback
    input_is_sharded = hasattr(input_a_memory_config, "is_sharded") and input_a_memory_config.is_sharded()

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        elif input_is_sharded:
            input_tensor_a = ttnn.from_torch(
                torch_input_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            try:
                input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_a, input_a_memory_config)
            except Exception:
                pass  # Stay on DRAM if shard spec doesn't fit device
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_a, dtype=input_a_dtype, layout=input_a_layout)

    # Create mask tensor if provided
    mask_tensor = None
    if torch_mask is not None and input_b_dtype is not None:
        mask_layout = input_b_layout if input_b_layout else ttnn.TILE_LAYOUT
        mask_mem = input_b_memory_config if input_b_memory_config else ttnn.DRAM_MEMORY_CONFIG
        if not is_host:
            if is_mesh_device and input_b_tensor_placement:
                mask_tensor = create_tensor_on_mesh(
                    torch_mask,
                    device,
                    input_b_dtype,
                    mask_layout,
                    mask_mem,
                    input_b_tensor_placement,
                )
            else:
                mask_tensor = ttnn.from_torch(
                    torch_mask,
                    dtype=input_b_dtype,
                    layout=mask_layout,
                    device=device,
                    memory_config=mask_mem,
                )
        else:
            mask_tensor = ttnn.from_torch(torch_mask, dtype=input_b_dtype, layout=mask_layout)

    start_time = start_measuring_time()
    if mask_tensor is not None:
        output_tensor = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
            input_tensor_a,
            scale,
            mask_tensor,
            **op_kwargs,
        )
    else:
        output_tensor = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
            input_tensor_a,
            scale,
            **op_kwargs,
        )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
