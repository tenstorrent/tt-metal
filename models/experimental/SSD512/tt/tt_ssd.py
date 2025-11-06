# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from models.experimental.SSD512.tt.layers.vgg_backbone import (
    build_vgg_backbone,
    apply_vgg_backbone,
)
from models.experimental.SSD512.tt.layers.tt_extras_backbone import (
    build_extras_backbone,
)
from models.experimental.SSD512.tt.layers.multibox_heads import (
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
    flag = False  # Alternates between 1x1 and 3x3 kernels

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

    Args:
        extras_config: Extras layer configuration
        torch_extras: PyTorch extras layers
        device: TTNN device (for backward compatibility)
        dtype: Data type for tensors
        weight_device: Device to place weights on (None = host, allows conv2d to prepare correctly)
                      If None, uses device parameter for backward compatibility
    """
    torch_idx = 0
    # Use weight_device if provided, otherwise use device (backward compatibility)
    weight_device_placement = weight_device if weight_device is not None else device

    for layer in extras_config:
        if layer["type"] == "conv":
            torch_layer = torch_extras[torch_idx]

            # Get weight and bias - use .clone() to avoid sharing memory with PyTorch model
            weight = torch_layer.weight.data.clone()
            bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

            # Convert to TTNN format
            # Place weights on specified device (None = host, allows conv2d to prepare correctly)
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

    Args:
        vgg_config: VGG layer configuration
        torch_vgg: PyTorch base (VGG) layers
        device: TTNN device (for backward compatibility)
        dtype: Data type for tensors
        weight_device: Device to place weights on (None = host, allows conv2d to prepare correctly)
                      If None, uses device parameter for backward compatibility

    Returns:
        vgg_config with loaded weights
    """
    torch_idx = 0
    # Use weight_device if provided, otherwise use device (backward compatibility)
    weight_device_placement = weight_device if weight_device is not None else device

    for layer in vgg_config:
        if layer["type"] == "conv":
            while torch_idx < len(torch_vgg):
                torch_layer = torch_vgg[torch_idx]
                if isinstance(torch_layer, torch.nn.Conv2d):
                    weight = torch_layer.weight.data.clone()
                    bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

                    # Place weights on specified device (None = host, allows conv2d to prepare correctly)
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

    Args:
        loc_layers_config: Location layer configuration
        conf_layers_config: Confidence layer configuration
        torch_loc: PyTorch location layers
        torch_conf: PyTorch confidence layers
        device: TTNN device (for backward compatibility)
        dtype: Data type for tensors
        weight_device: Device to place weights on (None = host, allows conv2d to prepare correctly)
                      If None, uses device parameter for backward compatibility
    """
    torch_idx = 0
    # Use weight_device if provided, otherwise use device (backward compatibility)
    weight_device_placement = weight_device if weight_device is not None else device

    # Process location layers
    for layer_idx, layer in enumerate(loc_layers_config):
        if layer["type"] == "conv":
            if torch_idx >= len(torch_loc):
                break

            torch_layer = torch_loc[torch_idx]

            # Get weight and bias - use .clone() to avoid sharing memory with PyTorch model
            weight = torch_layer.weight.data.clone()
            bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

            expected_in_channels = layer.get("in_channels", layer.get("config", {}).get("in_channels", None))
            actual_weight_in_channels = weight.shape[
                1
            ]  # PyTorch weight shape: [out_channels, in_channels, kernel_h, kernel_w]

            if expected_in_channels is not None and actual_weight_in_channels != expected_in_channels:
                layer["in_channels"] = actual_weight_in_channels
                if "config" in layer:
                    layer["config"]["in_channels"] = actual_weight_in_channels

            # Convert to TTNN format
            # Place weights on specified device (None = host, allows conv2d to prepare correctly)
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

    torch_idx = 0

    # Process confidence layers
    for layer in conf_layers_config:
        if layer["type"] == "conv":
            # Check bounds to avoid IndexError
            if torch_idx >= len(torch_conf):
                break

            torch_layer = torch_conf[torch_idx]

            # Get weight and bias - use .clone() to avoid sharing memory with PyTorch model
            weight = torch_layer.weight.data.clone()
            bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

            # Convert to TTNN format
            # Place weights on specified device (None = host, allows conv2d to prepare correctly)
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

    return loc_layers_config, conf_layers_config


def forward_extras(
    x, extras_config, batch_size, input_height, input_width, device, dtype=ttnn.bfloat16, memory_config=None
):
    """
    Forward pass through extra layers.
    """
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

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
                dtype=dtype,
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
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

            # Create compute_config with HiFi4 and fp32 accumulator for higher precision
            compute_config = None
            if device is not None:
                compute_config = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,  # High fidelity for better precision
                    fp32_dest_acc_en=True,  # Use fp32 accumulator for higher precision
                    packer_l1_acc=False,
                    math_approx_mode=False,  # Disable math approximation for maximum precision
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
                dtype=dtype,
                memory_config=memory_config,
                compute_config=compute_config,  # HiFi4 with fp32 accumulator for higher precision
            )

            x = output_tensor.reshape([batch_size, output_height, output_width, out_channels])
            current_h = output_height
            current_w = output_width
            current_c = out_channels

            # Increment conv_count after each conv layer
            conv_count += 1

        elif layer["type"] == "relu":
            x = ttnn.relu(x, memory_config=memory_config)

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
        # SSD512 has 8 sources: 2 from VGG (conv4_3, conv7) + 6 from extras
        self.loc_config, self.conf_config = build_multibox_heads(
            size=512,
            num_classes=num_classes,
            vgg_channels=[512, 1024],
            extra_channels=[512, 256, 256, 256, 256, 256],  # 6 extra feature maps
            device=device,
        )

    def load_weights_from_torch(self, torch_model, dtype=ttnn.bfloat16, weight_device=None):
        """
        Load weights from PyTorch model to TTNN configuration.

        Args:
            torch_model: PyTorch SSD model
            dtype: Data type for tensors
            weight_device: Device to place weights on (None = host, allows conv2d to prepare correctly)
                          If None, uses self.device for backward compatibility
        """
        # Use weight_device if provided, otherwise use self.device (backward compatibility)
        weight_device_placement = weight_device if weight_device is not None else self.device

        self.vgg_config = load_vgg_weights_from_torch(
            self.vgg_config, torch_model.base, self.device, dtype, weight_device=weight_device_placement
        )

        # Load L2Norm weights
        l2norm_weight = torch_model.L2Norm.weight.data
        self.l2norm.weight = ttnn.from_torch(
            l2norm_weight.reshape(1, -1, 1, 1),
            device=weight_device_placement,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
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
            dtype,
            weight_device=weight_device_placement,
        )

    def forward(self, x, dtype=ttnn.bfloat16, memory_config=None, debug=False):
        """
        Forward pass of SSD512 network.

        Args:
            x: Input tensor (torch.Tensor) with shape [batch, 3, 512, 512]
            dtype: TTNN data type (default: bfloat16)
            memory_config: Optional memory config for TTNN operations
            debug: If True, return intermediate results for debugging

        Returns:
            If debug=False: Tuple of (location_predictions, confidence_predictions)
            If debug=True: Tuple of (location_predictions, confidence_predictions, debug_dict)
                where debug_dict contains:
                - 'sources': List of intermediate source tensors (torch tensors)
                - 'loc_preds': List of location predictions per source (torch tensors)
                - 'conf_preds': List of confidence predictions per source (torch tensors)
        """
        memory_config = ttnn.DRAM_MEMORY_CONFIG

        batch_size = x.shape[0]

        if self.device is not None:
            ttnn.synchronize_device(self.device)
            import gc

            gc.collect()
            ttnn.synchronize_device(self.device)

        vgg_result = apply_vgg_backbone(
            x,  # Pass torch tensor directly, not pre-converted TTNN tensor
            self.vgg_config,
            device=self.device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Force DRAM to avoid L1 exhaustion
            return_sources=[22],  # Layer 22 is conv4_3 relu output (after 10th conv)
        )
        # print(f"vgg backbone done")

        # Synchronize device after VGG to free any L1 memory
        if self.device is not None:
            ttnn.synchronize_device(self.device)

        # Extract conv4_3 and conv7 from VGG output
        if isinstance(vgg_result, tuple):
            conv7, vgg_sources = vgg_result
            conv4_3 = vgg_sources[0]  # conv4_3 is the first (and only) requested source
        else:
            # Backward compatibility: if return_sources wasn't used, only conv7 is available
            conv7 = vgg_result
            conv4_3 = None

        # Apply L2Norm to conv4_3 if available
        if conv4_3 is not None:
            # Convert conv4_3 to torch temporarily to check shape, then apply L2Norm
            conv4_3_norm = self.l2norm(conv4_3)
            sources = [conv4_3_norm, conv7]
        else:
            # Fallback: use only conv7 if conv4_3 not available
            sources = [conv7]

        # Forward through extras
        # forward_extras returns a list of sources from extras layers
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

        # # Debug: Print source information
        # print(f"\n=== DEBUG: Sources before multibox ===")
        # for idx, source in enumerate(sources):
        #     if hasattr(source, 'shape'):
        #         print(f"Source {idx}: shape {source.shape}")
        #     else:
        #         source_torch = ttnn.to_torch(source)
        #         if source_torch.dim() == 4:
        #             source_torch = source_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
        #         print(f"Source {idx}: shape {source_torch.shape} (converted from TTNN)")
        # print(f"===============================\n")

        # Convert TTNN sources to torch tensors to free TTNN memory
        # This helps prevent out-of-memory errors by freeing TTNN tensors before multibox heads
        # Expected channel counts per source (for SSD512):
        # Source 0: 512 (conv4_3, after L2Norm - may be NCHW)
        # Source 1: 1024 (conv7)
        # Source 2: 512 (extras[1])
        # Source 3: 256 (extras[3])
        # Source 4: 256 (extras[5])
        # Source 5: 256 (extras[7])
        # Source 6: 256 (extras[9])
        expected_channels = [512, 1024, 512, 256, 256, 256, 256]
        torch_sources = []
        for idx, source in enumerate(sources):
            # Convert TTNN tensor to torch tensor
            source_torch = ttnn.to_torch(source)
            # print(f"Source {idx} after ttnn.to_torch: shape {source_torch.shape}")

            # Convert from NHWC to NCHW format if needed
            # IMPORTANT: Most ttnn.to_torch calls return NHWC format: [N, H, W, C]
            # However, L2Norm output might be in NCHW format already
            # apply_multibox_head expects NCHW format: [N, C, H, W]
            # So we check expected channel count to determine format
            if source_torch.dim() == 4:
                expected_c = expected_channels[idx] if idx < len(expected_channels) else None
                dim1_val = source_torch.shape[1]
                dim3_val = source_torch.shape[3]

                # Determine format based on which dimension matches expected channel count
                if expected_c is not None:
                    if dim1_val == expected_c:
                        # Already in NCHW format (channels at dim 1)
                        pass  # No conversion needed
                    elif dim3_val == expected_c:
                        # NHWC format (channels at dim 3), need to permute
                        source_torch = source_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
                    else:
                        # Ambiguous - use heuristic: if dim3 > dim1, likely NHWC
                        if dim3_val > dim1_val:
                            source_torch = source_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
                else:
                    # No expected channel count - use heuristic
                    if dim3_val > dim1_val:
                        source_torch = source_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
            torch_sources.append(source_torch)
            # Deallocate TTNN tensor to free memory
            if hasattr(source, "is_allocated") and source.is_allocated():
                try:
                    ttnn.deallocate(source)
                except:
                    pass
        print("extras done")
        # Apply multibox heads to all sources
        loc_outputs = []
        conf_outputs = []
        for idx, source in enumerate(torch_sources):
            # Determine memory config based on source size
            # Use L1 for very small feature maps (1x1, 2x2) to avoid weight preparation issues
            # Use DRAM for larger feature maps to avoid L1 exhaustion
            source_h, source_w = source.shape[2], source.shape[3]
            source_channels = source.shape[1]  # NCHW format
            is_very_small = source_h <= 2 or source_w <= 2
            source_memory_config = ttnn.L1_MEMORY_CONFIG if is_very_small else ttnn.DRAM_MEMORY_CONFIG

            # Update layer config to match actual source channels
            # This fixes mismatches where config has wrong channel count
            if idx < len(self.loc_config):
                actual_weight_channels = None
                if "weight" in self.loc_config[idx]:
                    weight_torch = ttnn.to_torch(self.loc_config[idx]["weight"])
                    actual_weight_channels = weight_torch.shape[1]  # PyTorch: [out, in, h, w]

                # If source channels don't match config, but weight matches source, update config
                if source_channels != self.loc_config[idx].get("in_channels", 0):
                    if actual_weight_channels == source_channels:
                        # print(f"Updating loc layer {idx} in_channels from {self.loc_config[idx].get('in_channels', 'None')} to {source_channels} to match source")
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
                        # print(f"Updating conf layer {idx} in_channels from {self.conf_config[idx].get('in_channels', 'None')} to {source_channels} to match source")
                        self.conf_config[idx]["in_channels"] = source_channels
                        if "config" in self.conf_config[idx]:
                            self.conf_config[idx]["config"]["in_channels"] = source_channels

            # print(f"\n>>> Processing multibox head {idx}: Location head")
            # print(f"Source shape: {source.shape}, H: {source_h}, W: {source_w}, C: {source_channels}")
            # print(f"Using memory_config: {source_memory_config}")

            # Location head - use appropriate memory config based on source size
            loc_out = apply_multibox_head(
                source, self.loc_config[idx], device=self.device, dtype=dtype, memory_config=source_memory_config
            )

            # Convert to torch immediately to free TTNN memory
            loc_torch = tt_to_torch_tensor(loc_out)
            # Deallocate TTNN tensor
            if hasattr(loc_out, "is_allocated") and loc_out.is_allocated():
                try:
                    ttnn.deallocate(loc_out)
                except:
                    pass

            # Synchronize device to ensure memory is freed
            if self.device is not None:
                ttnn.synchronize_device(self.device)

            # print(f"\n>>> Processing multibox head {idx}: Confidence head")

            # Confidence head - use appropriate memory config based on source size
            conf_out = apply_multibox_head(
                source, self.conf_config[idx], device=self.device, dtype=dtype, memory_config=source_memory_config
            )

            # Convert to torch immediately to free TTNN memory
            conf_torch = tt_to_torch_tensor(conf_out)
            # Deallocate TTNN tensor
            if hasattr(conf_out, "is_allocated") and conf_out.is_allocated():
                try:
                    ttnn.deallocate(conf_out)
                except:
                    pass

            # Synchronize device to ensure memory is freed
            if self.device is not None:
                ttnn.synchronize_device(self.device)

            # Reshape: [B, H, W, boxes*4] -> [B, H*W*boxes, 4]
            loc_torch = loc_torch.reshape(batch_size, -1, 4)
            # Reshape: [B, H, W, boxes*classes] -> [B, H*W*boxes, classes]
            conf_torch = conf_torch.reshape(batch_size, -1, self.num_classes)

            loc_outputs.append(loc_torch)
            conf_outputs.append(conf_torch)
        print("multibox heads done")
        # Concatenate all outputs
        # Match PyTorch format: flatten each output and concatenate
        # PyTorch does: torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # This results in [batch, total_priors*4] for location and [batch, total_priors*num_classes] for confidence
        loc = torch.cat([o.view(batch_size, -1) for o in loc_outputs], dim=1)  # [batch, total_priors*4]
        conf = torch.cat([o.view(batch_size, -1) for o in conf_outputs], dim=1)  # [batch, total_priors*num_classes]
        print("ssd done")

        if debug:
            debug_dict = {
                "sources": torch_sources,  # List of source tensors (NCHW format)
                "loc_preds": loc_outputs,  # List of location predictions per source
                "conf_preds": conf_outputs,  # List of confidence predictions per source
            }
            return loc, conf, debug_dict

        return loc, conf


def build_ssd512(num_classes=21, device=None):
    """
    Build SSD512 network.

    Args:
        num_classes: Number of object classes
        device: TTNN device

    Returns:
        SSD512Network instance
    """
    return SSD512Network(num_classes=num_classes, device=device)
