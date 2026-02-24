# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Hybrid DyHead: Scale-aware and Task-aware attention on TTNN,
Spatial-aware (DCNv2) on CPU/PyTorch.

DyHead has three attention mechanisms per block:
  1. Spatial-aware  -- DCNv2 (stays on CPU, no native TTNN kernel)
  2. Scale-aware    -- AvgPool -> Conv(256,1) -> ReLU -> HSigmoid  (TTNN)
  3. Task-aware     -- DyReLU: AvgPool -> Conv(256,64) -> ReLU ->
                       Conv(64,1024) -> HSigmoid -> element-wise   (TTNN)

This module runs scale and task attention on device while using the
existing PyTorch DyHead for DCNv2 spatial convolutions.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

import ttnn


class TtScaleAttn:
    """TTNN scale-aware attention: global_avg_pool -> linear(C->1) -> ReLU -> hardsigmoid."""

    def __init__(self, device, conv_weight: Tensor, conv_bias: Tensor):
        self.device = device
        C = conv_weight.shape[1]
        w = conv_weight.reshape(1, C).T.contiguous()
        b = conv_bias.reshape(1, 1)
        self.weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.bias = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, feat_nchw):
        B, C, H, W = feat_nchw.shape
        pooled = ttnn.mean(feat_nchw, dim=(-2, -1), keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        pooled = ttnn.reshape(pooled, (B, C))

        out = ttnn.matmul(pooled, self.weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.add(out, self.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.relu(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.hardsigmoid(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out, (B, 1, 1, 1))
        return out


class TtDyReLU:
    """TTNN task-aware attention (DyReLU).

    Learns piecewise-linear activation parameters (a1, b1, a2, b2)
    per channel from the global feature via squeeze-and-excitation.
    """

    def __init__(self, device, conv1_weight, conv1_bias, conv2_weight, conv2_bias, channels=256):
        self.device = device
        self.channels = channels

        ratio_ch = conv1_weight.shape[0]
        exp_ch = conv2_weight.shape[0]

        w1 = conv1_weight.reshape(ratio_ch, channels).T.contiguous()
        b1 = conv1_bias.reshape(1, ratio_ch)
        w2 = conv2_weight.reshape(exp_ch, ratio_ch).T.contiguous()
        b2 = conv2_bias.reshape(1, exp_ch)

        self.weight1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.bias1 = ttnn.from_torch(b1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.weight2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.bias2 = ttnn.from_torch(b2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, feat_nchw):
        B, C, H, W = feat_nchw.shape

        pooled = ttnn.mean(feat_nchw, dim=(-2, -1), keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        pooled = ttnn.reshape(pooled, (B, C))

        h = ttnn.matmul(pooled, self.weight1, memory_config=ttnn.L1_MEMORY_CONFIG)
        h = ttnn.add(h, self.bias1, memory_config=ttnn.L1_MEMORY_CONFIG)
        h = ttnn.relu(h, memory_config=ttnn.L1_MEMORY_CONFIG)

        coeffs = ttnn.matmul(h, self.weight2, memory_config=ttnn.L1_MEMORY_CONFIG)
        coeffs = ttnn.add(coeffs, self.bias2, memory_config=ttnn.L1_MEMORY_CONFIG)
        coeffs = ttnn.hardsigmoid(coeffs, memory_config=ttnn.L1_MEMORY_CONFIG)
        coeffs = ttnn.add(coeffs, -0.5, memory_config=ttnn.L1_MEMORY_CONFIG)

        coeffs_cpu = ttnn.to_torch(ttnn.from_device(coeffs)).float()
        a1_t, b1_t, a2_t, b2_t = torch.split(coeffs_cpu, C, dim=1)
        a1_t = (a1_t * 2.0 + 1.0).reshape(B, C, 1, 1).contiguous()
        b1_t = b1_t.reshape(B, C, 1, 1).contiguous()
        a2_t = (a2_t * 2.0).reshape(B, C, 1, 1).contiguous()
        b2_t = b2_t.reshape(B, C, 1, 1).contiguous()

        def _to_dev(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        a1, b1, a2, b2 = _to_dev(a1_t), _to_dev(b1_t), _to_dev(a2_t), _to_dev(b2_t)

        feat = ttnn.to_layout(feat_nchw, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        branch1 = ttnn.multiply(feat, a1, memory_config=ttnn.L1_MEMORY_CONFIG)
        branch1 = ttnn.add(branch1, b1, memory_config=ttnn.L1_MEMORY_CONFIG)
        branch2 = ttnn.multiply(feat, a2, memory_config=ttnn.L1_MEMORY_CONFIG)
        branch2 = ttnn.add(branch2, b2, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = ttnn.maximum(branch1, branch2)
        return output


class TtHybridDyHead:
    """Hybrid DyHead: DCNv2 spatial on CPU, scale/task attention on TTNN.

    Wraps a fully-loaded PyTorch DyHead and replaces the scale_attn_module
    and task_attn_module forward paths with TTNN equivalents.
    """

    def __init__(self, device, pt_dyhead):
        self.device = device
        self.pt_dyhead = pt_dyhead
        self.num_blocks = pt_dyhead.num_blocks

        self.scale_attns: List[TtScaleAttn] = []
        self.task_attns: List[TtDyReLU] = []

        for i in range(self.num_blocks):
            block = pt_dyhead.dyhead_blocks[i]

            scale_conv = block.scale_attn_module[1]
            self.scale_attns.append(TtScaleAttn(device, scale_conv.weight.data, scale_conv.bias.data))

            dyrelu = block.task_attn_module
            self.task_attns.append(
                TtDyReLU(
                    device,
                    dyrelu.conv1[0].weight.data,
                    dyrelu.conv1[0].bias.data,
                    dyrelu.conv2[0].weight.data,
                    dyrelu.conv2[0].bias.data,
                    channels=dyrelu.channels,
                )
            )

    def __call__(self, inputs_torch: List[Tensor]) -> List[Tensor]:
        # Convert input layout from NHWC (FPN output) to NCHW
        x = list([feat.permute(0, 3, 1, 2).contiguous() for feat in inputs_torch])

        for block_idx in range(self.num_blocks):
            x = self._forward_block(block_idx, x)

        # Prepare for ATSS Head (NCHW -> NHWC)
        x = [feat.permute(0, 2, 3, 1).contiguous() for feat in x]
        return x

    def _to_device(self, feat_torch: Tensor):
        return ttnn.from_torch(
            feat_torch.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _to_host(self, feat_ttnn) -> Tensor:
        return ttnn.to_torch(ttnn.from_device(feat_ttnn)).float()

    def _forward_block(self, block_idx: int, x: List[Tensor]) -> List[Tensor]:
        block = self.pt_dyhead.dyhead_blocks[block_idx]
        scale_attn = self.scale_attns[block_idx]
        task_attn = self.task_attns[block_idx]

        outs: List[Tensor] = []
        for level in range(len(x)):
            offset_and_mask = block.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, : block.offset_dim, :, :]
            mask = offset_and_mask[:, block.offset_dim :, :, :].sigmoid()

            mid_feat = block.spatial_conv_mid(x[level], offset, mask)

            mid_tt = self._to_device(mid_feat)
            scale_w = scale_attn(mid_tt)
            sum_feat_tt = ttnn.multiply(mid_tt, scale_w, memory_config=ttnn.L1_MEMORY_CONFIG)
            summed_levels = 1

            if level > 0:
                low_input = x[level - 1]
                expected_low_out = (low_input.shape[2] // 2, low_input.shape[3] // 2)
                low_offset, low_mask = offset, mask
                if expected_low_out != offset.shape[-2:]:
                    low_offset, low_mask = block._resize_offset_mask(offset, mask, expected_low_out)
                low_feat = block.spatial_conv_low(low_input, low_offset, low_mask)

                low_tt = self._to_device(low_feat)
                scale_w_low = scale_attn(low_tt)
                weighted_low = ttnn.multiply(low_tt, scale_w_low, memory_config=ttnn.L1_MEMORY_CONFIG)
                sum_feat_tt = ttnn.add(sum_feat_tt, weighted_low, memory_config=ttnn.L1_MEMORY_CONFIG)
                summed_levels += 1

            if level < len(x) - 1:
                high_input = x[level + 1]
                high_target = (high_input.shape[2], high_input.shape[3])
                high_offset, high_mask = block._resize_offset_mask(offset, mask, high_target)
                high_feat = F.interpolate(
                    block.spatial_conv_high(high_input, high_offset, high_mask),
                    size=x[level].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

                high_tt = self._to_device(high_feat)
                scale_w_high = scale_attn(high_tt)
                weighted_high = ttnn.multiply(high_tt, scale_w_high, memory_config=ttnn.L1_MEMORY_CONFIG)
                sum_feat_tt = ttnn.add(sum_feat_tt, weighted_high, memory_config=ttnn.L1_MEMORY_CONFIG)
                summed_levels += 1

            if summed_levels > 1:
                sum_feat_tt = ttnn.multiply(sum_feat_tt, 1.0 / summed_levels, memory_config=ttnn.L1_MEMORY_CONFIG)

            out_tt = task_attn(sum_feat_tt)
            outs.append(self._to_host(out_tt))

        return outs
