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

            # Prepare weight and bias
            if isinstance(weight, torch.Tensor):
                weight_torch = weight
            else:
                weight_torch = ttnn.to_torch(weight)

            weight = ttnn.from_torch(
                weight_torch,
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            if bias is not None:
                if isinstance(bias, torch.Tensor):
                    bias_torch = bias
                else:
                    bias_torch = ttnn.to_torch(bias)
                bias_reshaped = bias_torch.reshape((1, 1, 1, -1))
                bias = ttnn.from_torch(
                    bias_reshaped,
                    device=None,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

            compute_config = None
            if device is not None:
                compute_config = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                    math_approx_mode=False,
                )

            # Conv2d
            output_tensor, [output_height, output_width] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=bias,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                batch_size=batch_size,
                input_height=current_h,
                input_width=current_w,
                device=device,
                return_output_dim=True,
                dtype=ttnn.bfloat8_b,
                memory_config=memory_config,
                compute_config=compute_config,
            )

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

        if self.device is not None:
            ttnn.synchronize_device(self.device)
            import gc

            gc.collect()
            ttnn.synchronize_device(self.device)

        vgg_result = apply_vgg_backbone(
            x,
            self.vgg_config,
            device=self.device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_sources=[22],
        )

        if self.device is not None:
            ttnn.synchronize_device(self.device)

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
        torch_sources = []
        for idx, source in enumerate(sources):
            source_torch = ttnn.to_torch(source)
            if source_torch.dim() == 4:
                expected_c = expected_channels[idx] if idx < len(expected_channels) else None
                dim1_val = source_torch.shape[1]
                dim3_val = source_torch.shape[3]

                if expected_c is not None:
                    if dim1_val == expected_c:
                        pass
                    elif dim3_val == expected_c:
                        source_torch = source_torch.permute(0, 3, 1, 2)
                    else:
                        if dim3_val > dim1_val:
                            source_torch = source_torch.permute(0, 3, 1, 2)
                else:
                    if dim3_val > dim1_val:
                        source_torch = source_torch.permute(0, 3, 1, 2)
            torch_sources.append(source_torch)
            if hasattr(source, "is_allocated") and source.is_allocated():
                try:
                    ttnn.deallocate(source)
                except:
                    pass

        loc_outputs = []
        conf_outputs = []
        for idx, source in enumerate(torch_sources):
            source_h, source_w = source.shape[2], source.shape[3]
            source_channels = source.shape[1]
            tensor_size_estimate = batch_size * source_h * source_w * source_channels
            use_l1_for_this_layer = source_h <= 128 and source_w <= 128 and tensor_size_estimate <= 2 * 1024 * 1024
            source_memory_config = ttnn.L1_MEMORY_CONFIG

            if idx < len(self.loc_config):
                actual_weight_channels = None
                if "weight" in self.loc_config[idx]:
                    weight_torch = ttnn.to_torch(self.loc_config[idx]["weight"])
                    actual_weight_channels = weight_torch.shape[1]

                if source_channels != self.loc_config[idx].get("in_channels", 0):
                    if actual_weight_channels == source_channels:
                        self.loc_config[idx]["in_channels"] = source_channels
                        if "config" in self.loc_config[idx]:
                            self.loc_config[idx]["config"]["in_channels"] = source_channels

            if idx < len(self.conf_config):
                actual_weight_channels = None
                if "weight" in self.conf_config[idx]:
                    weight_torch = ttnn.to_torch(self.conf_config[idx]["weight"])
                    actual_weight_channels = weight_torch.shape[1]

                if source_channels != self.conf_config[idx].get("in_channels", 0):
                    if actual_weight_channels == source_channels:
                        self.conf_config[idx]["in_channels"] = source_channels
                        if "config" in self.conf_config[idx]:
                            self.conf_config[idx]["config"]["in_channels"] = source_channels

            loc_out = apply_multibox_head(
                source, self.loc_config[idx], device=self.device, dtype=dtype, memory_config=source_memory_config
            )

            loc_torch = tt_to_torch_tensor(loc_out)
            if hasattr(loc_out, "is_allocated") and loc_out.is_allocated():
                try:
                    ttnn.deallocate(loc_out)
                except:
                    pass

            if self.device is not None:
                ttnn.synchronize_device(self.device)

            conf_out = apply_multibox_head(
                source, self.conf_config[idx], device=self.device, dtype=dtype, memory_config=source_memory_config
            )

            conf_torch = tt_to_torch_tensor(conf_out)
            if hasattr(conf_out, "is_allocated") and conf_out.is_allocated():
                try:
                    ttnn.deallocate(conf_out)
                except:
                    pass

            if self.device is not None:
                ttnn.synchronize_device(self.device)

            loc_torch = loc_torch.reshape(batch_size, -1, 4)
            conf_torch = conf_torch.reshape(batch_size, -1, self.num_classes)

            loc_outputs.append(loc_torch)
            conf_outputs.append(conf_torch)

        loc = torch.cat([o.view(batch_size, -1) for o in loc_outputs], dim=1)
        conf = torch.cat([o.view(batch_size, -1) for o in conf_outputs], dim=1)

        if debug:
            debug_dict = {
                "sources": torch_sources,
                "loc_preds": loc_outputs,
                "conf_preds": conf_outputs,
            }
            return loc, conf, debug_dict

        return loc, conf


def build_ssd512(num_classes=21, device=None):
    """
    Build SSD512 network.
    """
    return SSD512Network(num_classes=num_classes, device=device)
