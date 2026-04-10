# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.sweep_framework.sweep_utils.conv2d_common import (
    run_conv2d_short_sweep,
    run_conv1d_short_sweep,
)
import ttnn
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
# Conv2d operations can be slow, especially with large kernels/channels
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("conv2d")

parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_specs": [
            # Contains following params
            # [batch_size, output_channels, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w, groups, dilation_h, dilation_w, bias]
            # Use tuple so it serializes as a string for proper deserialization
            (1, 16, 8, 4, 4, 1, 1, 1, 1, 0, 0, 1, 1, 1, False),
        ],
        "is_conv1d": [False],
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
            device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*mesh_shape),
                dispatch_core_config=ttnn.DispatchCoreConfig(),
                l1_small_size=79104,
            )
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
    input_specs=None,
    is_conv1d=False,
    compute_config=None,
    dtype=None,
    config_tensors_in_dram=False,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    # Build input_specs from flat kwargs when not provided directly
    full_padding = None
    if input_specs is None:
        batch_size = kwargs.get("batch_size")
        out_channels = kwargs.get("out_channels")
        in_channels = kwargs.get("in_channels")
        input_height = kwargs.get("input_height") or kwargs.get("input_h")
        input_width = kwargs.get("input_width") or kwargs.get("input_w")
        kernel_size = kwargs.get("kernel_size")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")
        dilation = kwargs.get("dilation")
        groups = kwargs.get("groups")
        # Check if we have enough params to construct input_specs
        if batch_size is not None and out_channels is not None and in_channels is not None:
            if isinstance(kernel_size, (list, tuple)) and len(kernel_size) >= 2:
                kh, kw = kernel_size[0], kernel_size[1]
            elif isinstance(kernel_size, int):
                kh = kw = kernel_size
            else:
                kh = kw = kernel_size[0] if kernel_size else 1
            if isinstance(stride, (list, tuple)) and len(stride) >= 2:
                sh, sw = stride[0], stride[1]
            elif isinstance(stride, int):
                sh = sw = stride
            else:
                sh = sw = stride[0] if stride else 1
            full_padding = None
            if isinstance(padding, (list, tuple)) and len(padding) == 4:
                ph, pw = 0, 0
                full_padding = tuple(padding)
            elif isinstance(padding, (list, tuple)) and len(padding) >= 2:
                ph, pw = padding[0], padding[1]
            elif isinstance(padding, int):
                ph = pw = padding
            else:
                ph = pw = padding[0] if padding else 0
            if isinstance(dilation, (list, tuple)) and len(dilation) >= 2:
                dh, dw = dilation[0], dilation[1]
            elif isinstance(dilation, int):
                dh = dw = dilation
            else:
                dh = dw = dilation[0] if dilation else 1
            ih = input_height or 4
            iw = input_width or 4
            g = groups or 1
            has_bias = bool(
                kwargs.get("bias_tensor_shape") and kwargs.get("bias_tensor_shape") not in (None, "None", "")
            )
            input_specs = (
                int(batch_size),
                int(out_channels),
                int(in_channels),
                int(ih),
                int(iw),
                int(kh),
                int(kw),
                int(sh),
                int(sw),
                int(ph),
                int(pw),
                int(g),
                int(dh),
                int(dw),
                has_bias,
            )
        else:
            return [(False, "Cannot construct input_specs: missing batch_size/out_channels/in_channels"), 0.0]

    # Parse compute_kernel_config from dict to ttnn object via build_op_kwargs or manually
    parsed_compute_config = None
    if compute_config and isinstance(compute_config, dict):
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        parsed_compute_config = parse_dict_value("compute_config", compute_config)
    elif compute_config is not None:
        parsed_compute_config = compute_config

    # Parse output_dtype from string/dict to ttnn dtype
    parsed_dtype = None
    if dtype and isinstance(dtype, (str, dict)):
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        if isinstance(dtype, dict):
            parsed_dtype = parse_dict_value("dtype", dtype)
        else:
            dtype_map = {
                "DataType.BFLOAT16": ttnn.bfloat16,
                "DataType.BFLOAT8_B": ttnn.bfloat8_b,
                "DataType.BFLOAT4_B": ttnn.bfloat4_b,
                "DataType.FLOAT32": ttnn.float32,
                "DataType.UINT16": ttnn.uint16,
                "DataType.UINT32": ttnn.uint32,
                "DataType.INT32": ttnn.int32,
                "bfloat16": ttnn.bfloat16,
                "bfloat8_b": ttnn.bfloat8_b,
                "bfloat4_b": ttnn.bfloat4_b,
                "float32": ttnn.float32,
                "uint16": ttnn.uint16,
                "uint32": ttnn.uint32,
                "int32": ttnn.int32,
            }
            parsed_dtype = dtype_map.get(dtype, ttnn.bfloat16)

    # Build Conv2dConfig from traced kwargs.
    # In V2 format, conv_config may be a serialized dict {"type": "Conv2dConfig", "value": "Conv2dConfig(...)"}
    # or individual flat kwargs (shard_layout, act_block_h_override, etc.).
    conv_config = None
    traced_conv_config = kwargs.get("conv_config")
    if traced_conv_config and isinstance(traced_conv_config, dict) and traced_conv_config.get("type") == "Conv2dConfig":
        import re

        value_str = traced_conv_config.get("value", "")
        conv_config = ttnn.Conv2dConfig()
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
        int_attrs = {"act_block_h_override", "act_block_w_div"}

        # Extract shard_layout
        sl_m = re.search(r"shard_layout=TensorMemoryLayout::(\w+)", value_str)
        if sl_m:
            sl_map = {
                "HEIGHT_SHARDED": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "BLOCK_SHARDED": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                "WIDTH_SHARDED": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            }
            sl_val = sl_map.get(sl_m.group(1))
            if sl_val:
                conv_config.shard_layout = sl_val

        # Extract weights_dtype
        wdt_m = re.search(r"weights_dtype=DataType::(\w+)", value_str)
        if wdt_m:
            wdt_map = {
                "BFLOAT8_B": ttnn.bfloat8_b,
                "BFLOAT16": ttnn.bfloat16,
                "BFLOAT4_B": ttnn.bfloat4_b,
                "FLOAT32": ttnn.float32,
            }
            wdt_val = wdt_map.get(wdt_m.group(1))
            if wdt_val:
                conv_config.weights_dtype = wdt_val

        # Extract output_layout
        ol_m = re.search(r"output_layout=Layout::(\w+)", value_str)
        if ol_m:
            ol_map = {
                "TILE": ttnn.TILE_LAYOUT,
                "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT,
            }
            ol_val = ol_map.get(ol_m.group(1))
            if ol_val:
                conv_config.output_layout = ol_val

        # Extract activation (e.g. UnaryWithParam(op_type=UnaryOpType::RELU))
        act_m = re.search(r"activation=UnaryWithParam\(op_type=UnaryOpType::(\w+)", value_str)
        if act_m:
            act_map = {
                "RELU": ttnn.UnaryOpType.RELU,
                "GELU": ttnn.UnaryOpType.GELU,
                "SILU": ttnn.UnaryOpType.SILU,
                "SIGMOID": ttnn.UnaryOpType.SIGMOID,
            }
            act_op = act_map.get(act_m.group(1))
            if act_op:
                conv_config.activation = ttnn.UnaryWithParam(act_op)

        for attr in bool_attrs:
            m = re.search(rf"{attr}=(\w+)", value_str)
            if m:
                setattr(conv_config, attr, m.group(1).lower() in ("true", "1"))
        for attr in int_attrs:
            m = re.search(rf"{attr}=(\d+)", value_str)
            if m:
                setattr(conv_config, attr, int(m.group(1)))
    else:
        conv_config_attrs = {
            "shard_layout": kwargs.get("shard_layout"),
            "act_block_h_override": kwargs.get("act_block_h_override"),
            "act_block_w_div": kwargs.get("act_block_w_div"),
            "transpose_shards": kwargs.get("transpose_shards"),
            "enable_act_double_buffer": kwargs.get("enable_act_double_buffer"),
            "enable_weights_double_buffer": kwargs.get("enable_weights_double_buffer"),
            "deallocate_activation": kwargs.get("deallocate_activation"),
            "reshard_if_not_optimal": kwargs.get("reshard_if_not_optimal"),
            "override_sharding_config": kwargs.get("override_sharding_config"),
            "output_layout": kwargs.get("output_layout"),
            "enable_kernel_stride_folding": kwargs.get("enable_kernel_stride_folding"),
            "enable_activation_reuse": kwargs.get("enable_activation_reuse"),
            "full_inner_dim": kwargs.get("full_inner_dim"),
            "config_tensors_in_dram": kwargs.get("config_tensors_in_dram"),
        }
        conv_config_attrs = {k: v for k, v in conv_config_attrs.items() if v is not None and v != "__ABSENT__"}

        if conv_config_attrs:
            conv_config = ttnn.Conv2dConfig()
            for attr, value in conv_config_attrs.items():
                if attr in ("act_block_h_override", "act_block_w_div"):
                    value = int(value)
                setattr(conv_config, attr, value)

    # Parse slice_config from traced args.
    # Only pass non-default DRAM slice configs. When slice_config is L1_FULL/0
    # (the default), leave as None so conv2d can auto-determine DRAM slicing
    # when the input tensor is in DRAM (sweep tests create inputs in DRAM,
    # unlike models where inputs are already in L1 from previous ops).
    parsed_slice_config = None
    traced_slice_config = kwargs.get("slice_config")
    if (
        traced_slice_config
        and isinstance(traced_slice_config, dict)
        and traced_slice_config.get("type") == "Op2DSliceConfig"
    ):
        import re

        sc_value_str = traced_slice_config.get("value", "")
        slice_type_map = {
            "L1_FULL": ttnn.Op2DSliceConfig.SliceTypeEnum.L1Full,
            "DRAM_SLICE_HEIGHT": ttnn.Op2DSliceConfig.SliceTypeEnum.DRAMSliceHeight,
            "DRAM_SLICE_WIDTH": ttnn.Op2DSliceConfig.SliceTypeEnum.DRAMSliceWidth,
        }
        st_m = re.search(r"slice_type=SliceType::(\w+)", sc_value_str)
        ns_m = re.search(r"num_slices=(\d+)", sc_value_str)
        if st_m:
            st_val = slice_type_map.get(st_m.group(1))
            if st_val is not None:
                num_slices = int(ns_m.group(1)) if ns_m else 0
                if st_val != ttnn.Op2DSliceConfig.SliceTypeEnum.L1Full or num_slices > 0:
                    sc_kwargs = {"slice_type": st_val}
                    if num_slices > 0:
                        sc_kwargs["num_slices"] = num_slices
                    parsed_slice_config = ttnn.Op2DSliceConfig(**sc_kwargs)

    # Parse tensor dtypes from traced args
    dtype_str_map = {
        "DataType.BFLOAT16": ttnn.bfloat16,
        "DataType.BFLOAT8_B": ttnn.bfloat8_b,
        "DataType.BFLOAT4_B": ttnn.bfloat4_b,
        "DataType.FLOAT32": ttnn.float32,
        "DataType.UINT16": ttnn.uint16,
        "DataType.UINT32": ttnn.uint32,
        "DataType.INT32": ttnn.int32,
    }
    parsed_input_dtype = dtype_str_map.get(kwargs.get("input_tensor_dtype"))
    parsed_weight_dtype = dtype_str_map.get(kwargs.get("weight_tensor_dtype"))
    parsed_bias_dtype = dtype_str_map.get(kwargs.get("bias_tensor_dtype"))

    # Parse input_tensor_layout from traced args
    layout_str_map = {
        "Layout.ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT,
        "Layout.TILE": ttnn.TILE_LAYOUT,
    }
    parsed_input_layout = layout_str_map.get(kwargs.get("input_tensor_layout"))

    # Call the short sweep function with parsed ttnn objects
    if is_conv1d:
        result = run_conv1d_short_sweep(input_specs, device)
    else:
        result = run_conv2d_short_sweep(
            input_specs,
            device,
            config_tensors_in_dram=config_tensors_in_dram,
            output_dtype=parsed_dtype,
            compute_config=parsed_compute_config,
            conv_config=conv_config,
            input_dtype=parsed_input_dtype,
            weight_dtype=parsed_weight_dtype,
            bias_dtype=parsed_bias_dtype,
            slice_config=parsed_slice_config,
            input_layout=parsed_input_layout,
            padding_override=full_padding,
        )

    # Convert short_sweep format [pcc_bool, pcc_value, e2e_perf, output_tensor, expected_tensor]
    # to model_traced format [pcc_tuple, e2e_perf]
    # result[0]: bool (PCC passed/failed)
    # result[1]: float (actual PCC value)
    # result[2]: int/float (e2e performance time)

    pcc_passed = bool(result[0])
    pcc_value = float(result[1])
    e2e_perf = result[2]

    # Format as (bool, message) tuple expected by sweep framework
    pcc_result = (pcc_passed, f"PCC: {pcc_value:.6f}")

    return [pcc_result, e2e_perf]
