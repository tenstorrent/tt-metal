# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from models.experimental.SSD512.tt.layers.tt_vgg_backbone import (
    build_vgg_backbone,
    apply_vgg_backbone,
)
from models.experimental.SSD512.tt.layers.tt_extras_backbone import (
    build_extras_backbone,
)
from models.experimental.SSD512.tt.layers.tt_multibox_heads import (
    build_multibox_heads,
    apply_multibox_head,
)
from models.experimental.SSD512.tt.layers.tt_l2norm import TtL2Norm
from models.common.utility_functions import tt_to_torch_tensor
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
    WidthSliceStrategyConfiguration,
)


# Extra layers configuration for SSD
extras = {
    "512": [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128],
}


def build_extras(cfg, in_channels=1024, device=None):
    """
    Build extra layers configuration for SSD.
    """
    layers = []
    flag = False

    for k, v in enumerate(cfg):
        if in_channels != "S":
            if v == "S":
                # Strided 3x3 conv (downsampling)
                kernel_size = (1, 3)[flag]
                layers.append(
                    {
                        "type": "conv",
                        "config": {
                            "kernel_size": (kernel_size, kernel_size),
                            "stride": (2, 2),
                            "padding": (1, 1),
                            "dilation": (1, 1),
                            "groups": 1,
                        },
                        "in_channels": in_channels,
                        "out_channels": cfg[k + 1],
                    }
                )
            else:
                # Regular conv (1x1 or 3x3)
                kernel_size = (1, 3)[flag]
                layers.append(
                    {
                        "type": "conv",
                        "config": {
                            "kernel_size": (kernel_size, kernel_size),
                            "stride": (1, 1),
                            "padding": (0, 0) if kernel_size == 1 else (1, 1),
                            "dilation": (1, 1),
                            "groups": 1,
                        },
                        "in_channels": in_channels,
                        "out_channels": v,
                    }
                )

            # Add ReLU after each conv
            layers.append({"type": "relu"})
            flag = not flag
        in_channels = v

    # For SSD512, add final conv layer
    if len(cfg) == 13:
        layers.append(
            {
                "type": "conv",
                "config": {
                    "kernel_size": (4, 4),
                    "stride": (1, 1),
                    "padding": (1, 1),
                    "dilation": (1, 1),
                    "groups": 1,
                },
                "in_channels": in_channels,
                "out_channels": 256,
            }
        )
        layers.append({"type": "relu"})

    return layers


def load_extras_weights_from_torch(extras_config, torch_extras, device, dtype=ttnn.bfloat16, weight_device=None):
    """
    Load weights from PyTorch extras layers to TTNN configuration.
    """
    torch_idx = 0
    weight_device_placement = weight_device if weight_device is not None else device

    for layer in extras_config:
        if layer["type"] == "conv":
            torch_layer = torch_extras[torch_idx]

            weight = torch_layer.weight.data.clone()
            bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

            # convert to TTNN format
            weight_ttnn = ttnn.from_torch(
                weight,
                device=weight_device_placement,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            if bias is not None:
                bias_reshaped = bias.reshape((1, 1, 1, -1))
                bias_ttnn = ttnn.from_torch(
                    bias_reshaped,
                    device=weight_device_placement,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                bias_ttnn = None

            layer["weight"] = weight_ttnn
            layer["bias"] = bias_ttnn
            torch_idx += 1

    return extras_config


def load_vgg_weights_from_torch(vgg_config, torch_vgg, device, dtype=ttnn.bfloat16, weight_device=None):
    """
    Load VGG weights from PyTorch base layers to TTNN configuration.
    """
    torch_idx = 0
    weight_device_placement = weight_device if weight_device is not None else device

    for layer in vgg_config:
        if layer["type"] == "conv":
            while torch_idx < len(torch_vgg):
                torch_layer = torch_vgg[torch_idx]
                if isinstance(torch_layer, torch.nn.Conv2d):
                    weight = torch_layer.weight.data.clone()
                    bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

                    weight_ttnn = ttnn.from_torch(
                        weight,
                        device=weight_device_placement,
                        dtype=dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )

                    if bias is not None:
                        bias_reshaped = bias.reshape((1, 1, 1, -1))
                        bias_ttnn = ttnn.from_torch(
                            bias_reshaped,
                            device=weight_device_placement,
                            dtype=dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                    else:
                        bias_ttnn = None

                    layer["weight"] = weight_ttnn
                    if bias_ttnn is not None:
                        layer["bias"] = bias_ttnn

                    torch_idx += 1
                    break
                torch_idx += 1

    return vgg_config


def load_multibox_weights_from_torch(
    loc_layers_config, conf_layers_config, torch_loc, torch_conf, device, dtype=ttnn.bfloat16, weight_device=None
):
    """
    Load weights from PyTorch multibox layers to TTNN configuration.
    """
    torch_idx = 0
    weight_device_placement = weight_device if weight_device is not None else device

    # process location layers
    for layer_idx, layer in enumerate(loc_layers_config):
        if layer["type"] == "conv":
            if torch_idx >= len(torch_loc):
                break

            torch_layer = torch_loc[torch_idx]

            weight = torch_layer.weight.data.clone()
            bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

            expected_in_channels = layer.get("in_channels", layer.get("config", {}).get("in_channels", None))
            actual_weight_in_channels = weight.shape[1]

            if expected_in_channels is not None and actual_weight_in_channels != expected_in_channels:
                layer["in_channels"] = actual_weight_in_channels
                if "config" in layer:
                    layer["config"]["in_channels"] = actual_weight_in_channels

            # convert to TTNN format
            weight_ttnn = ttnn.from_torch(
                weight,
                device=weight_device_placement,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            if bias is not None:
                bias_reshaped = bias.reshape((1, 1, 1, -1))
                bias_ttnn = ttnn.from_torch(
                    bias_reshaped,
                    device=weight_device_placement,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                bias_ttnn = None

            layer["weight"] = weight_ttnn
            layer["bias"] = bias_ttnn
            torch_idx += 1

    torch_idx = 0

    # Process confidence layers
    for layer in conf_layers_config:
        if layer["type"] == "conv":
            # check bounds to avoid IndexError
            if torch_idx >= len(torch_conf):
                break

            torch_layer = torch_conf[torch_idx]

            weight = torch_layer.weight.data.clone()
            bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

            weight_ttnn = ttnn.from_torch(
                weight,
                device=weight_device_placement,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            if bias is not None:
                bias_reshaped = bias.reshape((1, 1, 1, -1))
                bias_ttnn = ttnn.from_torch(
                    bias_reshaped,
                    device=weight_device_placement,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                bias_ttnn = None

            layer["weight"] = weight_ttnn
            layer["bias"] = bias_ttnn
            torch_idx += 1

    return loc_layers_config, conf_layers_config


_extras_weight_device_cache = {}


def clear_extras_weight_cache():
    """Clear the global weight cache. Call this at the start of each test."""
    global _extras_weight_device_cache
    _extras_weight_device_cache.clear()


def forward_extras(
    x, extras_config, batch_size, input_height, input_width, device, dtype=ttnn.bfloat8_b, memory_config=None
):
    """
    Forward pass through extra layers.
    """
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    sources = []
    current_h = input_height
    current_w = input_width
    current_c = extras_config[0]["in_channels"]

    # Track conv_count as we iterate
    conv_count = 0

    for layer_idx, layer in enumerate(extras_config):
        if layer["type"] == "conv":
            weight = layer["weight"]
            bias = layer["bias"]
            config = layer["config"]

            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            padding = config["padding"]
            dilation = config["dilation"]
            groups = config["groups"]

            # Initialize cache_key for pre-formatted weight check
            cache_key = None
            used_cache_fallback = False
            if isinstance(weight, ttnn.Tensor):
                weight_id = id(weight)
                input_mem_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
                input_layout = ttnn.TILE_LAYOUT
                cache_key = (
                    weight_id,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    id(input_mem_config) if input_mem_config else None,
                    input_layout,
                )

                if cache_key in _extras_weight_device_cache:
                    weight_ttnn, bias_ttnn = _extras_weight_device_cache[cache_key]
                    used_cache_fallback = True
                else:
                    try:
                        if ttnn.is_tensor_storage_on_device(weight):
                            weight_torch = ttnn.to_torch(weight)
                        else:
                            weight_torch = ttnn.to_torch(weight)

                        bias_torch = None
                        if bias is not None:
                            if isinstance(bias, ttnn.Tensor):
                                if ttnn.is_tensor_storage_on_device(bias):
                                    bias_torch = ttnn.to_torch(bias).reshape(-1)
                                else:
                                    bias_torch = ttnn.to_torch(bias).reshape(-1)
                            elif isinstance(bias, torch.Tensor):
                                bias_torch = bias.reshape(-1)
                            else:
                                bias_torch = ttnn.to_torch(bias).reshape(-1) if hasattr(bias, "shape") else None

                        weight_ttnn, bias_ttnn = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
                            weight_torch, bias_torch
                        )
                        if device is not None:
                            weight_ttnn = ttnn.to_device(weight_ttnn, device)
                            if bias_ttnn is not None:
                                bias_ttnn = ttnn.to_device(bias_ttnn, device)

                        try:
                            weight_host = ttnn.from_torch(
                                weight_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=None
                            )
                            if bias_torch is not None:
                                bias_host = ttnn.from_torch(
                                    bias_torch.reshape((1, 1, 1, -1)),
                                    dtype=ttnn.bfloat16,
                                    layout=ttnn.ROW_MAJOR_LAYOUT,
                                    device=None,
                                )
                            else:
                                bias_host = None

                            conv_config_for_weights = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
                            weight_prep = ttnn.prepare_conv_weights(
                                weight_tensor=weight_host,
                                weights_format="OIHW",
                                input_memory_config=input_mem_config,
                                input_layout=input_layout,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                batch_size=batch_size,
                                input_height=current_h,
                                input_width=current_w,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                has_bias=bias is not None,
                                groups=groups,
                                device=device,
                                input_dtype=dtype,
                                output_dtype=dtype,
                                conv_config=conv_config_for_weights,
                                compute_config=None,
                            )

                            if bias_host is not None:
                                conv_config_for_bias = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
                                bias_prep = ttnn.prepare_conv_bias(
                                    bias_tensor=bias_host,
                                    input_memory_config=input_mem_config,
                                    input_layout=input_layout,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    batch_size=batch_size,
                                    input_height=current_h,
                                    input_width=current_w,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=groups,
                                    device=device,
                                    input_dtype=dtype,
                                    output_dtype=dtype,
                                    conv_config=conv_config_for_bias,
                                    compute_config=None,
                                )
                            else:
                                bias_prep = None

                            _extras_weight_device_cache[cache_key] = (weight_prep, bias_prep)
                            # weight_ttnn = weight_prep
                            # bias_ttnn = bias_prep
                            # used_cache_fallback = True
                        except (RuntimeError, ValueError):
                            pass
                    except RuntimeError as e:
                        error_msg = str(e) if e else ""
                        if (
                            "trace" in error_msg.lower()
                            or "Reads are not supported" in error_msg
                            or "Writes are not supported" in error_msg
                        ):
                            if cache_key in _extras_weight_device_cache:
                                weight_ttnn, bias_ttnn = _extras_weight_device_cache[cache_key]
                                used_cache_fallback = True
                            else:
                                raise RuntimeError(
                                    f"Weight cache not populated during warmup. Original error: {error_msg}"
                                ) from e
                        else:
                            raise

            else:
                if isinstance(weight, torch.Tensor):
                    weight_torch = weight
                else:
                    weight_torch = ttnn.to_torch(weight) if hasattr(weight, "shape") else None

                bias_torch = None
                if bias is not None:
                    if isinstance(bias, torch.Tensor):
                        bias_torch = bias
                    else:
                        bias_torch = ttnn.to_torch(bias) if hasattr(bias, "shape") else None

                if weight_torch is not None:
                    weight_ttnn, bias_ttnn = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
                        weight_torch, bias_torch
                    )
                else:
                    raise ValueError("Weight must be either a TTNN tensor or torch tensor")

            tensor_size_estimate = batch_size * current_h * current_w * in_channels
            is_very_small = current_h <= 2 or current_w <= 2
            is_1x1_conv = kernel_size == (1, 1) or (kernel_size[0] == 1 and kernel_size[1] == 1)
            is_l1_memory = memory_config is not None and memory_config.buffer_type == ttnn.BufferType.L1
            force_l1_slice = is_very_small or is_1x1_conv

            use_dram_slicing = (
                not force_l1_slice
                and not is_l1_memory
                and ((tensor_size_estimate > 1024 * 1024 or current_h > 64 or current_w > 64))
            )

            if use_dram_slicing:
                slice_count = max(1, (batch_size * current_h * current_w) // (1024))
                slice_count = min(slice_count, 64)
                slice_strategy = WidthSliceStrategyConfiguration(num_slices=slice_count)
                sharding_strategy = AutoShardedStrategyConfiguration()
                enable_act_double_buffer = False
                deallocate_activation = False
            elif is_l1_memory or force_l1_slice:
                sharding_strategy = AutoShardedStrategyConfiguration()
                slice_strategy = L1FullSliceStrategyConfiguration()
                enable_act_double_buffer = False
                deallocate_activation = False
            else:
                sharding_strategy = AutoShardedStrategyConfiguration()
                slice_strategy = None
                enable_act_double_buffer = False
                deallocate_activation = False

            if used_cache_fallback:
                cached_weight, cached_bias = _extras_weight_device_cache[cache_key]
                dummy_weight_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
                dummy_weight = ttnn.zeros(
                    dummy_weight_shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=None
                )
                dummy_bias = None
                if cached_bias is not None:
                    dummy_bias = ttnn.zeros(
                        (1, 1, 1, out_channels), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None
                    )

                conv_config = Conv2dConfiguration(
                    input_height=current_h,
                    input_width=current_w,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    batch_size=batch_size,
                    kernel_size=kernel_size,
                    weight=dummy_weight,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=dummy_bias,
                    activation_dtype=dtype,
                    weights_dtype=ttnn.bfloat16,
                    output_dtype=dtype,
                    sharding_strategy=sharding_strategy,
                    slice_strategy=slice_strategy,
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                    enable_act_double_buffer=enable_act_double_buffer,
                    enable_weights_double_buffer=False,
                    deallocate_activation=deallocate_activation,
                    reallocate_halo_output=True,
                    config_tensors_in_dram=not is_l1_memory and not force_l1_slice,
                )
                object.__setattr__(conv_config, "weight", cached_weight)
                if cached_bias is not None:
                    object.__setattr__(conv_config, "bias", cached_bias)
            else:
                # Normal path for non-pre-formatted weights
                conv_config = Conv2dConfiguration(
                    input_height=current_h,
                    input_width=current_w,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    batch_size=batch_size,
                    kernel_size=kernel_size,
                    weight=weight_ttnn,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias_ttnn,
                    activation_dtype=dtype,
                    weights_dtype=ttnn.bfloat16,
                    output_dtype=dtype,
                    sharding_strategy=sharding_strategy,
                    slice_strategy=slice_strategy,
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                    enable_act_double_buffer=enable_act_double_buffer,
                    enable_weights_double_buffer=False,
                    deallocate_activation=deallocate_activation,
                    reallocate_halo_output=True,
                    config_tensors_in_dram=not is_l1_memory and not force_l1_slice,
                )

            conv_layer = TtConv2d(conv_config, device)
            output_tensor = conv_layer(x)

            if output_tensor.is_sharded():
                output_tensor = ttnn.sharded_to_interleaved(output_tensor, memory_config)

            padding_h, padding_w = padding
            dilation_h, dilation_w = dilation
            stride_h, stride_w = stride
            kernel_h, kernel_w = kernel_size

            output_height = (current_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
            output_width = (current_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

            x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])
            current_h = output_height
            current_w = output_width
            current_c = out_channels

            # Increment conv_count after each conv layer
            conv_count += 1

        elif layer["type"] == "relu":
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.relu(x)

            should_extract = conv_count >= 2 and conv_count % 2 == 0
            if should_extract:
                sources.append(x)

    return sources


class SSD512Network:
    def __init__(self, num_classes=21, device=None):
        self.num_classes = num_classes
        self.device = device
        self.size = 512

        # Build network components
        self.vgg_config = build_vgg_backbone(size=512, input_channels=3, device=device)
        self.extras_config = build_extras_backbone(size=512, input_channels=1024, device=device)

        # L2Norm for conv4_3
        self.l2norm = TtL2Norm(n_channels=512, scale=20, device=device)

        # Multibox heads
        self.loc_config, self.conf_config = build_multibox_heads(
            size=512,
            num_classes=num_classes,
            vgg_channels=[512, 1024],
            extra_channels=[512, 256, 256, 256, 256, 256],
            device=device,
        )

    def load_weights_from_torch(self, torch_model, dtype=ttnn.bfloat16, weight_device=None):
        """
        Load weights from PyTorch model to TTNN configuration.
        """
        weight_device_placement = weight_device if weight_device is not None else self.device

        self.vgg_config = load_vgg_weights_from_torch(
            self.vgg_config, torch_model.base, self.device, dtype, weight_device=weight_device_placement
        )

        # Load L2Norm weights
        l2norm_weight = torch_model.L2Norm.weight.data
        self.l2norm.weight = ttnn.from_torch(
            l2norm_weight.reshape(1, -1, 1, 1),
            device=weight_device_placement,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Load extras weights
        self.extras_config = load_extras_weights_from_torch(
            self.extras_config, torch_model.extras, self.device, dtype, weight_device=weight_device_placement
        )

        # Load multibox head weights
        self.loc_config, self.conf_config = load_multibox_weights_from_torch(
            self.loc_config,
            self.conf_config,
            torch_model.loc,
            torch_model.conf,
            self.device,
            dtype=ttnn.bfloat16,
            weight_device=weight_device_placement,
        )

    def forward(self, x, dtype=ttnn.bfloat8_b, memory_config=None, debug=False):
        """
        Forward pass of SSD512 network.
        """
        memory_config = ttnn.L1_MEMORY_CONFIG

        batch_size = x.shape[0]

        vgg_result = apply_vgg_backbone(
            x,
            self.vgg_config,
            device=self.device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_sources=[22],
        )

        if isinstance(vgg_result, tuple):
            conv7, vgg_sources = vgg_result
            conv4_3 = vgg_sources[0]
        else:
            conv7 = vgg_result
            conv4_3 = None

        if conv4_3 is not None:
            conv4_3_norm = self.l2norm(conv4_3)
            sources = [conv4_3_norm, conv7]
        else:
            sources = [conv7]

        # Forward through extras
        conv7_h = conv7.shape[1]
        conv7_w = conv7.shape[2]
        extra_sources = forward_extras(
            conv7,
            self.extras_config,
            batch_size=batch_size,
            input_height=conv7_h,
            input_width=conv7_w,
            device=self.device,
            dtype=dtype,
            memory_config=memory_config,
        )
        sources.extend(extra_sources)

        expected_channels = [512, 1024, 512, 256, 256, 256, 256]
        processed_sources = []
        for idx, source in enumerate(sources):
            source_shape = source.shape
            if len(source_shape) == 4:
                expected_c = expected_channels[idx] if idx < len(expected_channels) else None
                dim1_val = source_shape[1]
                dim3_val = source_shape[3]

                if expected_c is not None:
                    if dim1_val == expected_c:
                        source = ttnn.permute(source, (0, 2, 3, 1))
                    elif dim3_val == expected_c:
                        pass
                    else:
                        if dim3_val > dim1_val:
                            if dim3_val == expected_c:
                                pass
                            else:
                                source = ttnn.permute(source, (0, 2, 3, 1))
                else:
                    if dim3_val > dim1_val:
                        pass
                    else:
                        source = ttnn.permute(source, (0, 2, 3, 1))
            processed_sources.append(source)

        loc_outputs = []
        conf_outputs = []
        for idx, source in enumerate(processed_sources):
            source_h, source_w = source.shape[1], source.shape[2]
            source_channels = source.shape[3]
            tensor_size_estimate = batch_size * source_h * source_w * source_channels
            use_l1_for_this_layer = source_h <= 128 and source_w <= 128 and tensor_size_estimate <= 2 * 1024 * 1024
            source_memory_config = ttnn.L1_MEMORY_CONFIG

            if idx < len(self.loc_config):
                actual_weight_channels = None
                if "weight" in self.loc_config[idx]:
                    weight = self.loc_config[idx]["weight"]
                    if isinstance(weight, ttnn.Tensor):
                        weight_shape = weight.shape
                        if len(weight_shape) >= 2:
                            actual_weight_channels = weight_shape[1] if len(weight_shape) == 4 else None
                    else:
                        actual_weight_channels = None

                if source_channels != self.loc_config[idx].get("in_channels", 0):
                    if actual_weight_channels is not None and actual_weight_channels == source_channels:
                        self.loc_config[idx]["in_channels"] = source_channels
                        if "config" in self.loc_config[idx]:
                            self.loc_config[idx]["config"]["in_channels"] = source_channels

            if idx < len(self.conf_config):
                actual_weight_channels = None
                if "weight" in self.conf_config[idx]:
                    weight = self.conf_config[idx]["weight"]
                    if isinstance(weight, ttnn.Tensor):
                        weight_shape = weight.shape
                        if len(weight_shape) >= 2:
                            actual_weight_channels = weight_shape[1] if len(weight_shape) == 4 else None
                    else:
                        actual_weight_channels = None

                if source_channels != self.conf_config[idx].get("in_channels", 0):
                    if actual_weight_channels is not None and actual_weight_channels == source_channels:
                        self.conf_config[idx]["in_channels"] = source_channels
                        if "config" in self.conf_config[idx]:
                            self.conf_config[idx]["config"]["in_channels"] = source_channels

            loc_out = apply_multibox_head(
                source, self.loc_config[idx], device=self.device, dtype=dtype, memory_config=source_memory_config
            )

            loc_out_flat = ttnn.reshape(loc_out, (batch_size, -1, 4))
            loc_outputs.append(loc_out_flat)

            conf_out = apply_multibox_head(
                source, self.conf_config[idx], device=self.device, dtype=dtype, memory_config=source_memory_config
            )

            conf_out_flat = ttnn.reshape(conf_out, (batch_size, -1, self.num_classes))
            conf_outputs.append(conf_out_flat)

        if len(loc_outputs) > 1:
            loc = ttnn.concat(loc_outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            loc = loc_outputs[0]

        if len(conf_outputs) > 1:
            conf = ttnn.concat(conf_outputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            conf = conf_outputs[0]

        # Return TTNN tensors directly to avoid device-to-host reads during trace capture
        # The pipeline model function will handle conversion if needed
        if debug:
            debug_sources = [tt_to_torch_tensor(s) for s in processed_sources]
            debug_dict = {
                "sources": debug_sources,
                "loc_preds": [tt_to_torch_tensor(l) for l in loc_outputs],
                "conf_preds": [tt_to_torch_tensor(c) for c in conf_outputs],
            }
            loc_torch = tt_to_torch_tensor(loc)
            conf_torch = tt_to_torch_tensor(conf)
            return loc_torch, conf_torch, debug_dict

        # Return TTNN tensors directly (not converted to torch)
        # This avoids device-to-host reads during trace capture
        return loc, conf


def build_ssd512(num_classes=21, device=None):
    """
    Build SSD512 network.
    """
    return SSD512Network(num_classes=num_classes, device=device)
