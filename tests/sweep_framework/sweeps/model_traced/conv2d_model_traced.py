# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re
import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    check_with_pcc_without_tensor_printout,
    start_measuring_time,
    stop_measuring_time,
)
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("conv2d")

parameters = {
    "model_traced_sample": {
        "input_specs": [
            (1, 16, 8, 4, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1, False),
        ],
        "is_conv1d": [False],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape, l1_small_size=65536)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=65536, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=65536, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


_DTYPE_MAP = {
    "DataType.BFLOAT16": ttnn.bfloat16,
    "DataType.BFLOAT8_B": ttnn.bfloat8_b,
    "DataType.BFLOAT4_B": ttnn.bfloat4_b,
    "DataType.FLOAT32": ttnn.float32,
    "DataType.UINT16": ttnn.uint16,
    "DataType.UINT32": ttnn.uint32,
    "DataType.INT32": ttnn.int32,
    ttnn.bfloat16: ttnn.bfloat16,
    ttnn.bfloat8_b: ttnn.bfloat8_b,
    ttnn.bfloat4_b: ttnn.bfloat4_b,
    ttnn.float32: ttnn.float32,
    ttnn.uint16: ttnn.uint16,
    ttnn.uint32: ttnn.uint32,
    ttnn.int32: ttnn.int32,
}

_LAYOUT_MAP = {
    "Layout.ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT,
    "Layout.TILE": ttnn.TILE_LAYOUT,
    ttnn.ROW_MAJOR_LAYOUT: ttnn.ROW_MAJOR_LAYOUT,
    ttnn.TILE_LAYOUT: ttnn.TILE_LAYOUT,
}

_SHARD_LAYOUT_MAP = {
    "HEIGHT_SHARDED": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "BLOCK_SHARDED": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    "WIDTH_SHARDED": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
}

_WEIGHTS_DTYPE_MAP = {
    "BFLOAT8_B": ttnn.bfloat8_b,
    "BFLOAT16": ttnn.bfloat16,
    "BFLOAT4_B": ttnn.bfloat4_b,
    "FLOAT32": ttnn.float32,
}

_OUTPUT_LAYOUT_MAP = {
    "TILE": ttnn.TILE_LAYOUT,
    "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT,
}

_ACTIVATION_MAP = {
    "RELU": ttnn.UnaryOpType.RELU,
    "GELU": ttnn.UnaryOpType.GELU,
    "SILU": ttnn.UnaryOpType.SILU,
    "SIGMOID": ttnn.UnaryOpType.SIGMOID,
}


def _parse_conv_config(traced_conv_config):
    """Parse serialized Conv2dConfig dict into ttnn.Conv2dConfig."""
    if not traced_conv_config or not isinstance(traced_conv_config, dict):
        return None
    if traced_conv_config.get("type") != "Conv2dConfig":
        return None

    value_str = traced_conv_config.get("value", "")
    conv_config = ttnn.Conv2dConfig()

    sl_m = re.search(r"shard_layout=TensorMemoryLayout::(\w+)", value_str)
    if sl_m:
        sl_val = _SHARD_LAYOUT_MAP.get(sl_m.group(1))
        if sl_val:
            conv_config.shard_layout = sl_val

    wdt_m = re.search(r"weights_dtype=DataType::(\w+)", value_str)
    if wdt_m:
        wdt_val = _WEIGHTS_DTYPE_MAP.get(wdt_m.group(1))
        if wdt_val:
            conv_config.weights_dtype = wdt_val

    ol_m = re.search(r"output_layout=Layout::(\w+)", value_str)
    if ol_m:
        ol_val = _OUTPUT_LAYOUT_MAP.get(ol_m.group(1))
        if ol_val:
            conv_config.output_layout = ol_val

    act_m = re.search(r"activation=UnaryWithParam\(op_type=UnaryOpType::(\w+)", value_str)
    if act_m:
        act_op = _ACTIVATION_MAP.get(act_m.group(1))
        if act_op:
            conv_config.activation = ttnn.UnaryWithParam(act_op)

    bool_attrs = {
        "deallocate_activation",
        "reallocate_halo_output",
        "reshard_if_not_optimal",
        "override_sharding_config",
        "override_output_sharding_config",
        "transpose_shards",
        "enable_act_double_buffer",
        "enable_weights_double_buffer",
        "enable_kernel_stride_folding",
        "enable_activation_reuse",
        "full_inner_dim",
        "config_tensors_in_dram",
    }
    for attr in bool_attrs:
        m = re.search(rf"{attr}=(\w+)", value_str)
        if m:
            setattr(conv_config, attr, m.group(1).lower() in ("true", "1"))

    int_attrs = {"act_block_h_override", "act_block_w_div"}
    for attr in int_attrs:
        m = re.search(rf"{attr}=(\d+)", value_str)
        if m:
            setattr(conv_config, attr, int(m.group(1)))

    return conv_config


def _parse_compute_config(device, compute_config_dict):
    """Parse compute_config dict into ttnn ComputeKernelConfig."""
    if not compute_config_dict or not isinstance(compute_config_dict, dict):
        return None

    fidelity_map = {
        "MathFidelity.LoFi": ttnn.MathFidelity.LoFi,
        "MathFidelity.HiFi2": ttnn.MathFidelity.HiFi2,
        "MathFidelity.HiFi3": ttnn.MathFidelity.HiFi3,
        "MathFidelity.HiFi4": ttnn.MathFidelity.HiFi4,
    }
    fidelity_str = compute_config_dict.get("math_fidelity", "MathFidelity.HiFi4")
    math_fidelity = fidelity_map.get(fidelity_str, ttnn.MathFidelity.HiFi4)

    math_approx = str(compute_config_dict.get("math_approx_mode", "False")).lower() in ("true", "1")
    fp32_acc = bool(compute_config_dict.get("fp32_dest_acc_en", False))
    packer_l1 = bool(compute_config_dict.get("packer_l1_acc", False))

    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1,
    )


def _parse_list_param(val, default=(1, 1)):
    """Parse kernel_size/stride/dilation from list/tuple/int."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return int(val[0]), int(val[1])
    elif isinstance(val, int):
        return val, val
    elif isinstance(val, (list, tuple)) and len(val) == 1:
        return int(val[0]), int(val[0])
    return default


def _parse_padding(val):
    """Parse padding - could be 2-element (h,w) or 4-element (top,bottom,left,right)."""
    if isinstance(val, (list, tuple)):
        if len(val) == 4:
            return (0, 0), tuple(int(x) for x in val)
        elif len(val) >= 2:
            return (int(val[0]), int(val[1])), None
        elif len(val) == 1:
            return (int(val[0]), int(val[0])), None
    elif isinstance(val, int):
        return (val, val), None
    return (0, 0), None


def _parse_memory_config(mem_config):
    """Parse memory_config dict to ttnn.MemoryConfig."""
    if mem_config is None or mem_config == "__ABSENT__":
        return None
    if isinstance(mem_config, dict):
        data = mem_config.get("data", mem_config)
        buf_type = data.get("buffer_type", "DRAM")
        if "L1" in str(buf_type):
            return ttnn.L1_MEMORY_CONFIG
        return ttnn.DRAM_MEMORY_CONFIG
    return mem_config


def run(
    input_specs=None,
    is_conv1d=False,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # --- Legacy path for model_traced_sample suite ---
    if input_specs is not None:
        from tests.sweep_framework.sweep_utils.conv2d_common import run_conv2d_short_sweep, run_conv1d_short_sweep

        if is_conv1d:
            return run_conv1d_short_sweep(input_specs, device)
        result = run_conv2d_short_sweep(input_specs, device)
        pcc_passed = bool(result[0])
        pcc_value = float(result[1])
        return [(pcc_passed, f"PCC: {pcc_value:.6f}"), result[2]]

    # --- Model traced path: call ttnn.conv2d directly with traced args ---

    batch_size = int(kwargs.get("batch_size", 1))
    out_channels = int(kwargs.get("out_channels", 1))
    in_channels = int(kwargs.get("in_channels", 1))
    input_height = int(kwargs.get("input_height") or kwargs.get("input_h") or 4)
    input_width = int(kwargs.get("input_width") or kwargs.get("input_w") or 4)
    groups = int(kwargs.get("groups") or 1)

    kernel_h, kernel_w = _parse_list_param(kwargs.get("kernel_size"), (1, 1))
    stride_h, stride_w = _parse_list_param(kwargs.get("stride"), (1, 1))
    dilation_h, dilation_w = _parse_list_param(kwargs.get("dilation"), (1, 1))
    (pad_h, pad_w), full_padding = _parse_padding(kwargs.get("padding"))

    has_bias = bool(kwargs.get("bias_tensor_shape") and kwargs.get("bias_tensor_shape") not in (None, "None", ""))

    # Parse dtypes from traced args
    input_dtype = _DTYPE_MAP.get(kwargs.get("input_tensor_dtype"), ttnn.bfloat16)
    weight_dtype = _DTYPE_MAP.get(kwargs.get("weight_tensor_dtype"), ttnn.bfloat16)
    bias_dtype = _DTYPE_MAP.get(kwargs.get("bias_tensor_dtype"), ttnn.bfloat16)
    output_dtype = _DTYPE_MAP.get(kwargs.get("dtype"), ttnn.bfloat16)

    # Parse layout
    input_layout = _LAYOUT_MAP.get(kwargs.get("input_tensor_layout"), ttnn.ROW_MAJOR_LAYOUT)

    # Parse memory configs from traced args
    input_memory_config = _parse_memory_config(kwargs.get("input_tensor_memory_config"))

    # Parse conv_config
    conv_config = _parse_conv_config(kwargs.get("conv_config"))
    if conv_config is None:
        conv_config = ttnn.Conv2dConfig()

    # Parse compute_config
    compute_config = _parse_compute_config(device, kwargs.get("compute_config"))

    # --- Determine input NHWC shape from traced shape ---
    # The trace records the exact NHWC tensor shape (e.g. (1,1,49,320) for a
    # flattened spatial dim). Use it directly so the physical dimensions match
    # the traced shard_spec.
    traced_input_shape_raw = kwargs.get("input_tensor_shape")
    if traced_input_shape_raw:
        if isinstance(traced_input_shape_raw, (list, tuple)):
            nhwc_shape = list(traced_input_shape_raw)
        else:
            nhwc_shape = [int(x) for x in re.findall(r"\d+", str(traced_input_shape_raw))]
    else:
        nhwc_shape = [batch_size, input_height, input_width, in_channels]

    # --- Create torch tensors ---
    conv_weight_shape = [out_channels, in_channels // groups, kernel_h, kernel_w]
    conv_bias_shape = [1, 1, 1, out_channels]

    # The traced NHWC shape may have a padded channel dim (e.g. 3→16 for alignment).
    # Check if reshaping back to (N, H, W, C) is possible; if not, the channel dim
    # was padded by the pipeline. In that case, create the true (N, C, H, W) NCHW
    # tensor first and derive the NHWC tensor from it, zero-padding the channel dim
    # to match the traced shape.
    traced_channels = nhwc_shape[-1]
    nchw_elements = batch_size * in_channels * input_height * input_width
    nhwc_elements = 1
    for d in nhwc_shape:
        nhwc_elements *= d

    if nhwc_elements == nchw_elements:
        torch_input_nhwc = torch.randn(nhwc_shape, dtype=torch.bfloat16).float()
        torch_input_nchw = (
            torch_input_nhwc.reshape(batch_size, input_height, input_width, in_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
    else:
        torch_input_nchw = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.bfloat16).float()
        nhwc_from_nchw = torch_input_nchw.permute(0, 2, 3, 1).reshape(
            batch_size, 1, input_height * input_width, in_channels
        )
        if traced_channels > in_channels:
            pad_width = traced_channels - in_channels
            nhwc_from_nchw = torch.nn.functional.pad(nhwc_from_nchw, (0, pad_width))
        torch_input_nhwc = nhwc_from_nchw.reshape(nhwc_shape)

    torch_weight = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None

    # --- Golden reference (uses standard NCHW conv2d) ---
    if full_padding is not None:
        pt, pb, pl, pr = full_padding
        golden_input = torch.nn.functional.pad(torch_input_nchw, (pl, pr, pt, pb))
        golden_padding = (0, 0)
    else:
        golden_input = torch_input_nchw
        golden_padding = (pad_h, pad_w)

    torch_golden = torch.nn.functional.conv2d(
        golden_input,
        torch_weight,
        bias=torch_bias.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=golden_padding,
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )

    if conv_config.activation is not None:
        act_str = str(conv_config.activation)
        if "RELU" in act_str:
            torch_golden = torch.nn.functional.relu(torch_golden)
        elif "GELU" in act_str:
            torch_golden = torch.nn.functional.gelu(torch_golden)
        elif "SILU" in act_str:
            torch_golden = torch.nn.functional.silu(torch_golden)
        elif "SIGMOID" in act_str:
            torch_golden = torch.sigmoid(torch_golden)

    # --- Create ttnn tensors ---
    is_mesh_device = hasattr(device, "get_num_devices")
    input_a_tensor_placement = kwargs.get("input_tensor_tensor_placement", None)

    # BFLOAT8_B/BFLOAT4_B require TILE_LAYOUT for from_torch
    effective_input_layout = input_layout
    if input_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        effective_input_layout = ttnn.TILE_LAYOUT

    effective_mem_config = input_memory_config or ttnn.DRAM_MEMORY_CONFIG

    if is_mesh_device and input_a_tensor_placement:
        tt_input = create_tensor_on_mesh(
            torch_input_nhwc,
            device,
            input_dtype,
            effective_input_layout,
            effective_mem_config,
            input_a_tensor_placement,
        )
    else:
        tt_input = ttnn.from_torch(
            torch_input_nhwc,
            dtype=input_dtype,
            layout=effective_input_layout,
            device=device,
            memory_config=effective_mem_config,
        )

    # conv2d requires weight/bias in ROW_MAJOR - it tilizes internally.
    # The traced layout (TILE) reflects model pipeline state, not the API expectation.
    effective_weight_dtype = weight_dtype
    if effective_weight_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        effective_weight_dtype = ttnn.float32
    tt_weight = ttnn.from_torch(torch_weight, effective_weight_dtype)

    tt_bias = None
    if has_bias:
        effective_bias_dtype = bias_dtype
        if effective_bias_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            effective_bias_dtype = ttnn.float32
        tt_bias = ttnn.from_torch(torch_bias, effective_bias_dtype)

    # --- Call ttnn.conv2d ---
    return_output_dim = bool(kwargs.get("return_output_dim", False))
    return_weights_and_bias = bool(kwargs.get("return_weights_and_bias", False))

    start_time = start_measuring_time()

    padding_arg = full_padding if full_padding is not None else (pad_h, pad_w)

    result = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=padding_arg,
        dilation=(dilation_h, dilation_w),
        groups=groups,
        bias_tensor=tt_bias,
        conv_config=conv_config,
        compute_config=compute_config,
        dtype=output_dtype,
        return_output_dim=return_output_dim,
        return_weights_and_bias=return_weights_and_bias,
    )

    e2e_perf = stop_measuring_time(start_time)

    # --- Extract output tensor from result ---
    # Return type depends on return_output_dim and return_weights_and_bias:
    #   both True  -> (tensor, (h, w), (weight, bias))
    #   output_dim -> (tensor, (h, w))
    #   weights    -> (tensor, (weight, bias))
    #   neither    -> tensor
    if return_output_dim and return_weights_and_bias:
        tt_output = result[0]
    elif return_output_dim or return_weights_and_bias:
        tt_output = result[0]
    else:
        tt_output = result

    # --- Extract output ---
    if is_mesh_device:
        device_tensors = ttnn.get_device_tensors(tt_output)
        torch_output = ttnn.to_torch(device_tensors[0])
    else:
        torch_output = ttnn.to_torch(tt_output)

    # Reshape output to NHWC then compare
    out_h = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if full_padding is not None:
        pt, pb, pl, pr = full_padding
        padded_h = input_height + pt + pb
        padded_w = input_width + pl + pr
        out_h = (padded_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (padded_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    torch_output = torch_output.reshape(batch_size, out_h, out_w, -1)
    torch_output = torch_output[:, :, :, :out_channels]

    torch_golden = torch_golden.permute(0, 2, 3, 1)

    pcc_passed, pcc_value = check_with_pcc_without_tensor_printout(torch_output, torch_golden, pcc=0.985)

    return [(bool(pcc_passed), f"PCC: {pcc_value:.6f}"), e2e_perf]
