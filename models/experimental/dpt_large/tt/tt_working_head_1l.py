# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Working head with 1 fusion layer - maximum speed.
Uses only the finest features for fastest inference.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import ttnn


class WorkingConv2d:
    """Conv2d with fused ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        weight: torch.Tensor = None,
        bias: Optional[torch.Tensor] = None,
        device=None,
        fused_relu: bool = False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        if weight is not None:
            self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)
        else:
            self.weight = None

        if bias is not None:
            bias_reshaped = bias.view(1, 1, 1, out_channels)
            self.bias = ttnn.from_torch(bias_reshaped, dtype=ttnn.float32)
        else:
            self.bias = None

        self._weight_tt = None
        self._bias_tt = None
        self._first_call = True

        activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if fused_relu else None
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=activation,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
        )

    def __call__(self, x: ttnn.Tensor, batch_size: int, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        if self._first_call:
            self._weight_tt = self.weight
            self._bias_tt = self.bias

        result = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._weight_tt,
            bias_tensor=self._bias_tt,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            batch_size=batch_size,
            input_height=h,
            input_width=w,
            conv_config=self.conv_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=self._first_call,
        )

        if self._first_call:
            output, (out_h, out_w), (self._weight_tt, self._bias_tt) = result
            self._first_call = False
        else:
            output, (out_h, out_w) = result

        output = ttnn.reshape(output, (batch_size, out_h, out_w, self.out_channels))
        return output, out_h, out_w


class WorkingPreActResidual:
    """PreAct Residual with fused ReLU."""

    def __init__(
        self,
        channels: int,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        device,
    ):
        self.channels = channels
        self.device = device

        conv1_w = state_dict[f"{prefix}.convolution1.weight"]
        conv1_b = state_dict.get(f"{prefix}.convolution1.bias")
        self.conv1 = WorkingConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            weight=conv1_w,
            bias=conv1_b,
            device=device,
            fused_relu=True,
        )

        conv2_w = state_dict[f"{prefix}.convolution2.weight"]
        conv2_b = state_dict.get(f"{prefix}.convolution2.bias")
        self.conv2 = WorkingConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            weight=conv2_w,
            bias=conv2_b,
            device=device,
            fused_relu=False,
        )

    def __call__(self, x: ttnn.Tensor, batch_size: int, h: int, w: int) -> ttnn.Tensor:
        residual = x
        x, h, w = self.conv1(x, batch_size, h, w)
        x, h, w = self.conv2(x, batch_size, h, w)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        residual = ttnn.to_layout(residual, ttnn.TILE_LAYOUT)
        x = ttnn.add(x, residual)
        return x


class WorkingFusionStage1L:
    """
    1-Layer Fusion Stage - uses only the finest feature.

    Takes 96x96 feature, applies 1 fusion layer -> 192x192
    This is the fastest possible fusion but may lose some accuracy.
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        device,
        channels: int = 256,
    ):
        self.device = device
        self.channels = channels

        # Use only layer 3 (the final fusion layer)
        base = f"neck.fusion_stage.layers.3"

        self.residual1 = WorkingPreActResidual(channels, state_dict, f"{base}.residual_layer1", device)
        self.residual2 = WorkingPreActResidual(channels, state_dict, f"{base}.residual_layer2", device)

        proj_w = state_dict[f"{base}.projection.weight"]
        proj_b = state_dict[f"{base}.projection.bias"]
        self.projection = WorkingConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            weight=proj_w,
            bias=proj_b,
            device=device,
            fused_relu=False,
        )

    def __call__(
        self,
        hidden_states: List[Tuple[ttnn.Tensor, int, int]],
        batch_size: int,
    ) -> List[Tuple[ttnn.Tensor, int, int]]:
        # Use only the finest feature (96x96)
        feat_96, h, w = hidden_states[0]

        # Apply residual blocks
        hidden_state = self.residual2(feat_96, batch_size, h, w)

        # Upsample 2x -> 192x192
        hidden_state = ttnn.to_layout(hidden_state, ttnn.ROW_MAJOR_LAYOUT)
        hidden_state = ttnn.upsample(hidden_state, (2, 2))
        new_h = h * 2
        new_w = w * 2

        # Projection
        hidden_state, out_h, out_w = self.projection(hidden_state, batch_size, new_h, new_w)

        return [(hidden_state, out_h, out_w)]


class WorkingDepthHead:
    """Depth head."""

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        device,
        channels: int = 256,
    ):
        self.device = device

        head0_w = state_dict["head.head.0.weight"]
        head0_b = state_dict["head.head.0.bias"]
        self.conv0 = WorkingConv2d(
            in_channels=channels,
            out_channels=channels // 2,
            kernel_size=3,
            padding=1,
            weight=head0_w,
            bias=head0_b,
            device=device,
            fused_relu=False,
        )

        head2_w = state_dict["head.head.2.weight"]
        head2_b = state_dict["head.head.2.bias"]
        self.conv1 = WorkingConv2d(
            in_channels=channels // 2,
            out_channels=32,
            kernel_size=3,
            padding=1,
            weight=head2_w,
            bias=head2_b,
            device=device,
            fused_relu=True,
        )

        head4_w = state_dict["head.head.4.weight"]
        head4_b = state_dict["head.head.4.bias"]
        self.conv2 = WorkingConv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            padding=0,
            weight=head4_w,
            bias=head4_b,
            device=device,
            fused_relu=True,
        )

    def __call__(self, fused_states: List[Tuple[ttnn.Tensor, int, int]], batch_size: int) -> ttnn.Tensor:
        x, h, w = fused_states[-1]

        x, h, w = self.conv0(x, batch_size, h, w)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.upsample(x, (2, 2))
        h *= 2
        w *= 2

        x, h, w = self.conv1(x, batch_size, h, w)
        x, h, w = self.conv2(x, batch_size, h, w)

        return x


class WorkingHead1L:
    """Working DPT head with 1 fusion layer - fastest."""

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        device,
        channels: int = 256,
    ):
        self.device = device
        self.fusion_stage = WorkingFusionStage1L(state_dict, device, channels)
        self.depth_head = WorkingDepthHead(state_dict, device, channels)

    def __call__(
        self,
        pyramid_feats: List[Tuple[ttnn.Tensor, int, int]],
        batch_size: int = 1,
    ) -> ttnn.Tensor:
        fused = self.fusion_stage(pyramid_feats, batch_size)
        depth = self.depth_head(fused, batch_size)
        return depth
