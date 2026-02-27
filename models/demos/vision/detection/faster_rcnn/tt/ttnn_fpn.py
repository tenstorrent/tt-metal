# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Feature Pyramid Network (FPN) implementation in TTNN for Faster-RCNN.
Takes multi-scale feature maps from the ResNet-50 backbone and produces
a set of feature maps at different scales with 256 channels each.
"""

import math

import torch

import ttnn

from models.demos.vision.detection.faster_rcnn.tt.ttnn_resnet50_backbone import TtConv2D


class TtFPN:
    """Feature Pyramid Network in TTNN.

    Architecture:
        - inner_blocks: 1x1 convolutions to reduce each backbone output to 256 channels
        - layer_blocks: 3x3 convolutions on each FPN level (256 -> 256)
        - Top-down pathway: upsample + element-wise add
        - Extra block: stride-2 maxpool from P5 to generate P6
    """

    def __init__(self, parameters, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size

        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256

        self.inner_convs = []
        self.layer_convs = []

        for idx, in_ch in enumerate(in_channels_list):
            inner_conv = TtConv2D(
                weight=parameters[f"fpn.inner_blocks.{idx}.weight"],
                bias=parameters[f"fpn.inner_blocks.{idx}.bias"],
                device=device,
                batch_size=batch_size,
                in_channels=in_ch,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=None,
                deallocate_activation=False,
                enable_act_double_buffer=True,
            )
            self.inner_convs.append(inner_conv)

            layer_conv = TtConv2D(
                weight=parameters[f"fpn.layer_blocks.{idx}.weight"],
                bias=parameters[f"fpn.layer_blocks.{idx}.bias"],
                device=device,
                batch_size=batch_size,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=None,
                deallocate_activation=False,
                enable_act_double_buffer=True,
            )
            self.layer_convs.append(layer_conv)

    def _upsample_and_add(self, top, lateral, target_h, target_w):
        """Upsample top feature map and add to lateral feature map.

        Falls back to CPU for upsample since it needs precise spatial control.
        """
        top_torch = ttnn.to_torch(ttnn.from_device(top))
        nhw = top_torch.shape[2]
        c = top_torch.shape[3]
        h_top = int(math.sqrt(nhw // self.batch_size))
        w_top = h_top

        top_4d = top_torch.reshape(self.batch_size, h_top, w_top, c).permute(0, 3, 1, 2)
        upsampled = torch.nn.functional.interpolate(top_4d, size=(target_h, target_w), mode="nearest")
        upsampled = upsampled.permute(0, 2, 3, 1).contiguous()
        upsampled = upsampled.reshape(1, 1, self.batch_size * target_h * target_w, c)

        upsampled_ttnn = ttnn.from_torch(upsampled, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        upsampled_ttnn = ttnn.to_device(upsampled_ttnn, self.device)
        upsampled_ttnn = ttnn.to_layout(upsampled_ttnn, ttnn.TILE_LAYOUT)

        lateral_tiled = ttnn.to_layout(lateral, ttnn.TILE_LAYOUT)
        if upsampled_ttnn.memory_config() != lateral_tiled.memory_config():
            upsampled_ttnn = ttnn.to_memory_config(upsampled_ttnn, lateral_tiled.memory_config())

        result = ttnn.add(upsampled_ttnn, lateral_tiled)
        ttnn.deallocate(upsampled_ttnn)

        return result

    def _get_spatial_dims(self, tensor):
        """Infer H, W from a flattened [1, 1, N*H*W, C] tensor."""
        nhw = tensor.shape[2]
        spatial = nhw // self.batch_size
        h = int(math.sqrt(spatial))
        w = h
        return h, w

    def __call__(self, features):
        """Run FPN on backbone features.

        Args:
            features: dict with keys "0", "1", "2", "3" mapping to C2, C3, C4, C5

        Returns:
            dict mapping "0"-"4" to P2, P3, P4, P5, pool (feature levels)
        """
        last_inner, _, _ = self.inner_convs[3](features["3"])

        results = [None, None, None, None]
        results[3], _, _ = self.layer_convs[3](last_inner)

        for idx in range(2, -1, -1):
            lateral, _, _ = self.inner_convs[idx](features[str(idx)])
            target_h, target_w = self._get_spatial_dims(lateral)

            inner = self._upsample_and_add(last_inner, lateral, target_h, target_w)
            ttnn.deallocate(lateral)

            results[idx], _, _ = self.layer_convs[idx](inner)
            last_inner = inner

        fpn_output = {}
        for idx in range(4):
            fpn_output[str(idx)] = results[idx]

        p5_torch = ttnn.to_torch(ttnn.from_device(results[3]))
        nhw = p5_torch.shape[2]
        c = p5_torch.shape[3]
        h = int(math.sqrt(nhw // self.batch_size))
        w = h
        p5_4d = p5_torch.reshape(self.batch_size, h, w, c).permute(0, 3, 1, 2)
        p6 = torch.nn.functional.max_pool2d(p5_4d, kernel_size=1, stride=2, padding=0)
        p6_nhwc = p6.permute(0, 2, 3, 1).contiguous()
        p6_flat = p6_nhwc.reshape(1, 1, self.batch_size * p6.shape[2] * p6.shape[3], c)
        p6_ttnn = ttnn.from_torch(p6_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        p6_ttnn = ttnn.to_device(p6_ttnn, self.device)
        fpn_output["pool"] = p6_ttnn

        return fpn_output
