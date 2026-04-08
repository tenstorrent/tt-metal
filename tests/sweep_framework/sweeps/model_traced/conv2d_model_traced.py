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
        # Create mesh device based on env var
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
        # Single device (default)
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
            kh = kw = kernel_size if isinstance(kernel_size, int) else (kernel_size[0] if kernel_size else 1)
            sh = sw = stride if isinstance(stride, int) else (stride[0] if stride else 1)
            ph = pw = padding if isinstance(padding, int) else (padding[0] if padding else 0)
            dh = dw = dilation if isinstance(dilation, int) else (dilation[0] if dilation else 1)
            ih = input_height or 4
            iw = input_width or 4
            g = groups or 1
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
                False,
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
                "bfloat16": ttnn.bfloat16,
                "bfloat8_b": ttnn.bfloat8_b,
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
        # Parse Conv2dConfig from string representation
        import re

        value_str = traced_conv_config.get("value", "")
        conv_config = ttnn.Conv2dConfig()
        # Extract key=value pairs from the string repr
        bool_attrs = {
            "deallocate_activation",
            "reallocate_halo_output",
            "reshard_if_not_optimal",
            "override_sharding_config",
            "transpose_shards",
            "enable_act_double_buffer",
            "enable_weights_double_buffer",
            "enable_kernel_stride_folding",
            "enable_activation_reuse",
            "full_inner_dim",
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
        # Extract boolean and integer attributes
        for attr in bool_attrs:
            m = re.search(rf"{attr}=(\w+)", value_str)
            if m:
                setattr(conv_config, attr, m.group(1).lower() in ("true", "1"))
        for attr in int_attrs:
            m = re.search(rf"{attr}=(\d+)", value_str)
            if m:
                setattr(conv_config, attr, int(m.group(1)))
    else:
        # Fallback: build from individual flat kwargs
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
        }
        # Filter out None/absent values
        conv_config_attrs = {k: v for k, v in conv_config_attrs.items() if v is not None and v != "__ABSENT__"}

        if conv_config_attrs:
            conv_config = ttnn.Conv2dConfig()
            for attr, value in conv_config_attrs.items():
                if attr in ("act_block_h_override", "act_block_w_div"):
                    value = int(value)
                setattr(conv_config, attr, value)

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
