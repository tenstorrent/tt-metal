# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from typing import List


class TtSSDMultiboxHeadHybrid(nn.Module):
    """
    Final hybrid SSD Multibox Head implementation

    Strategy:
    - Sources 0-3 (64×64, 32×32, 16×16, 8×8): PyTorch convolutions
      (TTNN conv2d has limitations with large feature maps)
    - Sources 4-6 (4×4, 2×2, 1×1): TTNN convolutions
      (Works reliably with small feature maps)

    """

    def __init__(
        self,
        num_classes: int,
        mbox: List[int],
        source_channels: List[int],
        state_dict=None,
        base_address="",
        device=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mbox = mbox
        self.source_channels = source_channels
        self.device = device
        self.num_sources = len(mbox)

        print(f"\n=== TtSSDMultiboxHeadHybrid ===")
        print(f"Sources 0-3: PyTorch convolutions")
        print(f"Sources 4-6: TTNN convolutions")

        # PyTorch conv layers for ALL sources (as fallback)
        self.loc_layers_pytorch = nn.ModuleList()
        self.conf_layers_pytorch = nn.ModuleList()

        for i, (num_boxes, in_channels) in enumerate(zip(mbox, source_channels)):
            layer_addr_loc = f"loc.{i}" if not base_address else f"{base_address}.loc.{i}"
            layer_addr_conf = f"conf.{i}" if not base_address else f"{base_address}.conf.{i}"

            # Location layer (PyTorch)
            loc = nn.Conv2d(in_channels, num_boxes * 4, kernel_size=3, padding=1)
            loc.weight.data = state_dict[f"{layer_addr_loc}.weight"]
            loc.bias.data = state_dict[f"{layer_addr_loc}.bias"]
            loc.eval()
            self.loc_layers_pytorch.append(loc)

            # Confidence layer (PyTorch)
            conf = nn.Conv2d(in_channels, num_boxes * num_classes, kernel_size=3, padding=1)
            conf.weight.data = state_dict[f"{layer_addr_conf}.weight"]
            conf.bias.data = state_dict[f"{layer_addr_conf}.bias"]
            conf.eval()
            self.conf_layers_pytorch.append(conf)

        # Store raw weights for TTNN conv (sources 4-6)
        self.loc_weights_raw = []
        self.loc_biases_raw = []
        self.conf_weights_raw = []
        self.conf_biases_raw = []

        for i, (num_boxes, in_channels) in enumerate(zip(mbox, source_channels)):
            layer_addr_loc = f"loc.{i}" if not base_address else f"{base_address}.loc.{i}"
            layer_addr_conf = f"conf.{i}" if not base_address else f"{base_address}.conf.{i}"

            self.loc_weights_raw.append(state_dict[f"{layer_addr_loc}.weight"])
            self.loc_biases_raw.append(state_dict[f"{layer_addr_loc}.bias"])
            self.conf_weights_raw.append(state_dict[f"{layer_addr_conf}.weight"])
            self.conf_biases_raw.append(state_dict[f"{layer_addr_conf}.bias"])

        print(f"=== Initialization Complete ===\n")

    def _ttnn_conv(self, source, weight, bias, out_channels, in_channels):
        """
        TTNN convolution for small feature maps (4×4, 2×2, 1×1)
        """
        batch_size = source.shape[0]
        input_height = source.shape[2]
        input_width = source.shape[3]

        # Convert weights to TTNN
        weight_ttnn = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        bias_ttnn = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Conv config
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            deallocate_activation=False,
            reallocate_halo_output=False,
            config_tensors_in_dram=True,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Run convolution
        result = ttnn.conv2d(
            input_tensor=source,
            weight_tensor=weight_ttnn,
            bias_tensor=bias_ttnn,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            groups=1,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        output_tensor, (out_h, out_w), (weight_back, bias_back) = result

        # Deallocate weights
        ttnn.deallocate(weight_ttnn)
        ttnn.deallocate(bias_ttnn)

        # Reshape from [B, 1, H*W, C] to [B, C, H, W]
        output_tensor = ttnn.reshape(output_tensor, (batch_size, out_h, out_w, out_channels))
        output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))

        # Convert to PyTorch
        result_torch = ttnn.to_torch(output_tensor).float()
        ttnn.deallocate(output_tensor)

        return result_torch

    def forward(self, sources: List[ttnn.Tensor]):
        """
        Process all sources with hybrid approach

        Args:
            sources: List of 7 TTNN tensors

        Returns:
            loc: [B, 24532, 4]
            conf: [B, 24532, 21]
        """
        loc_preds = []
        conf_preds = []

        for i in range(self.num_sources):
            source = sources[i]

            # Determine which implementation to use
            # Sources 0-3: Use PyTorch (large feature maps)
            # Sources 4-6: Use TTNN (small feature maps)
            use_ttnn = i >= 4

            if use_ttnn:
                # TTNN path for small sources
                print(f"Source {i}: Using TTNN conv")

                try:
                    # Location conv
                    out_channels_loc = self.mbox[i] * 4
                    loc_pred = self._ttnn_conv(
                        source,
                        self.loc_weights_raw[i],
                        self.loc_biases_raw[i],
                        out_channels_loc,
                        self.source_channels[i],
                    )

                    # Confidence conv
                    out_channels_conf = self.mbox[i] * self.num_classes
                    conf_pred = self._ttnn_conv(
                        source,
                        self.conf_weights_raw[i],
                        self.conf_biases_raw[i],
                        out_channels_conf,
                        self.source_channels[i],
                    )

                    # Deallocate source
                    ttnn.deallocate(source)

                except Exception as e:
                    print(f"  TTNN failed for source {i}: {e}, falling back to PyTorch")
                    # Fallback to PyTorch if TTNN fails
                    source_torch = ttnn.to_torch(source).float()
                    ttnn.deallocate(source)

                    with torch.no_grad():
                        loc_pred = self.loc_layers_pytorch[i](source_torch)
                        conf_pred = self.conf_layers_pytorch[i](source_torch)

            else:
                # PyTorch path for large sources
                print(f"Source {i}: Using PyTorch conv")

                # Convert TTNN -> PyTorch
                source_torch = ttnn.to_torch(source).float()
                ttnn.deallocate(source)

                # Run PyTorch convs
                with torch.no_grad():
                    loc_pred = self.loc_layers_pytorch[i](source_torch)
                    conf_pred = self.conf_layers_pytorch[i](source_torch)

            # Reshape to [B, num_priors, 4] and [B, num_priors, num_classes]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(loc_pred.shape[0], -1, 4)

            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_pred.view(conf_pred.shape[0], -1, self.num_classes)

            loc_preds.append(loc_pred)
            conf_preds.append(conf_pred)

        # Concatenate all predictions
        loc = torch.cat(loc_preds, dim=1)
        conf = torch.cat(conf_preds, dim=1)

        print(f"\nFinal output shapes:")
        print(f"  Location: {loc.shape}")
        print(f"  Confidence: {conf.shape}")

        return loc, conf


# Helper function to calculate expected number of priors
def calculate_num_priors(mbox: List[int], feature_sizes: List[tuple]) -> int:
    """
    Calculate total number of prior boxes

    Args:
        mbox: List of boxes per location [4, 6, 6, 6, 4, 4, 4]
        feature_sizes: List of (H, W) for each feature map

    Returns:
        Total number of prior boxes (24532 for SSD512)
    """
    total = 0
    for num_boxes, (h, w) in zip(mbox, feature_sizes):
        total += h * w * num_boxes
    return total


# SSD512 configuration
SSD512_MBOX = [4, 6, 6, 6, 4, 4, 4]
SSD512_SOURCE_CHANNELS = [512, 1024, 512, 256, 256, 256, 256]
SSD512_FEATURE_SIZES = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
SSD512_NUM_PRIORS = 24532
