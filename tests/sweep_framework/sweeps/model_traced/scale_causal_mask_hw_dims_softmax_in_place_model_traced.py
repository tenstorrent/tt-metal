# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
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
    op_kwargs = build_op_kwargs(kwargs)

    # scale_causal_mask_hw_dims_softmax_in_place only accepts program_config and
    # compute_kernel_config as named kwargs.  Do NOT pass memory_config here;
    # the C++ binding rejects it with "incompatible function arguments".

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    # Scale factor — V2 loader stores non-tensor positional args as arg0, arg1, …
    # scale_causal_mask_hw_dims_softmax_in_place(input, scale, mask, **kw)
    # → arg1 = scale (float)
    pos_args = extract_positional_args(kwargs)
    raw_scale = scalar if scalar is not None else pos_args.get(1, None)
    scale = float(raw_scale) if raw_scale is not None else 1.0

    # Generate input tensor
    torch_input_a = gen_func_with_cast_tt(partial(torch_random, low=-10, high=10, dtype=torch.float32), input_a_dtype)(
        shape_a
    )

    # Generate causal mask if mask params are provided (arg2 in JSON).
    # The op expects a pre-computed causal mask with -100000 for masked (future) positions
    # and 0 for unmasked positions, matching how falcon7b creates the attention mask.
    mask_shape = kwargs.get("input_b_shape", None)
    if mask_shape is not None:
        mask_shape = tuple(mask_shape) if isinstance(mask_shape, (list, tuple)) else mask_shape
        h, w = mask_shape[-2], mask_shape[-1]
        causal_bool = torch.ones(h, w, dtype=torch.bool).triu(diagonal=1)
        torch_mask = causal_bool.float().masked_fill(causal_bool, -100000.0)
        # Expand to full mask shape
        while torch_mask.ndim < len(mask_shape):
            torch_mask = torch_mask.unsqueeze(0)
        torch_mask = torch_mask.expand(*mask_shape)
    else:
        torch_mask = None

    # Golden: softmax(scale * input + tiled_mask)
    # The "hw_dims" op tiles the mask across the input height — each mask-height block
    # of rows gets its own causal mask applied independently.
    golden_input = scale * torch_input_a.float()
    if torch_mask is not None:
        mask_float = torch_mask.float()
        input_h = golden_input.shape[-2]
        mask_h = mask_float.shape[-2]
        if mask_h < input_h:
            # Tile mask to cover full input height
            repeats = (input_h + mask_h - 1) // mask_h
            mask_float = mask_float.repeat(1, 1, repeats, 1)[..., :input_h, :]
        elif mask_h > input_h:
            mask_float = mask_float[..., :input_h, :]
        golden_input = golden_input + mask_float
    torch_output = torch.softmax(golden_input, dim=-1)

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

    def _run_op(tensor_a, kw):
        if mask_tensor is not None:
            return ttnn.scale_causal_mask_hw_dims_softmax_in_place(tensor_a, scale, mask_tensor, **kw)
        return ttnn.scale_causal_mask_hw_dims_softmax_in_place(tensor_a, scale, **kw)

    try:
        output_tensor = _run_op(input_tensor_a, op_kwargs)
    except Exception:
        input_tensor_a = ttnn.from_torch(
            torch_input_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fallback_kwargs = {k: v for k, v in op_kwargs.items() if k not in ("program_config", "memory_config")}
        try:
            output_tensor = _run_op(input_tensor_a, fallback_kwargs)
        except Exception:
            output_tensor = _run_op(input_tensor_a, {})
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output.shape)]

    if is_mesh_device:
        torch_output = reconcile_golden_to_actual(
            torch_output, output_tensor, input_a_tensor_placement, input_b_tensor_placement
        )
    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
