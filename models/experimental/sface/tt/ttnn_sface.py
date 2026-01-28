# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Optimized TTNN SFace (MobileFaceNet) Face Recognition Model.

Key optimizations:
- DRAM slicing for early layers with large spatial dimensions (112x112)
- NO reshape between consecutive convs in DepthwiseSeparable blocks
- Track dimensions separately instead of reshaping to get shape
- bfloat8_b weights for memory bandwidth improvement

Input: [1, 112, 112, 3] NHWC normalized face image
Output: [1, 128] L2-normalized embedding vector
"""

import ttnn
import torch
from typing import Tuple
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    AutoShardedStrategyConfiguration,
    HeightSliceStrategyConfiguration,
)


class TTNNConvBNPReLU:
    """
    TTNN Conv2d + BatchNorm + PReLU block.

    Uses DRAM slicing for large spatial dimensions to avoid L1 overflow.
    """

    def __init__(
        self,
        device,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        name: str = "",
        use_dram_slicing: bool = False,
        num_slices: int = 2,
        high_precision: bool = False,  # Use bfloat16 weights for better accuracy
    ):
        self.device = device
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel, self.stride, self.padding = kernel, stride, padding
        self.groups = groups
        self.name = name
        self.use_dram_slicing = use_dram_slicing
        self.num_slices = num_slices
        self.high_precision = high_precision
        self.weight = self.bias = None
        self.prelu_weight = None  # Per-channel PReLU slopes
        self._conv_cache = {}

    def load_weights(self, weight, bias, prelu_weight=None):
        """Load fused Conv+BN weights and PReLU slope."""
        self.weight, self.bias = weight, bias
        if prelu_weight is not None:
            # Store per-channel PReLU slopes
            self.prelu_weight = prelu_weight.clone()

    def _get_conv(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv_cache:
            # Use DRAM slicing for large spatial dimensions
            slice_strategy = None
            if self.use_dram_slicing:
                slice_strategy = HeightSliceStrategyConfiguration(num_slices=self.num_slices)

            # Use auto sharding - TTNN will determine optimal strategy
            # Height sharding fails when height > num_cores (98)
            sharding_strategy = AutoShardedStrategyConfiguration()

            # High precision mode: disable double buffering for more stable computation
            # Note: bfloat8_b works better than bfloat16 for weights in this model
            weights_dtype = ttnn.bfloat8_b
            enable_double_buffer = not self.high_precision

            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                batch_size=batch_size,
                kernel_size=(self.kernel, self.kernel),
                stride=(self.stride, self.stride),
                padding=(self.padding, self.padding),
                groups=self.groups,
                dilation=(1, 1),
                weight=self.weight,
                bias=self.bias,
                sharding_strategy=sharding_strategy,
                slice_strategy=slice_strategy,
                # No activation - PReLU applied separately
                weights_dtype=weights_dtype,
                enable_act_double_buffer=enable_double_buffer,
                enable_weights_double_buffer=enable_double_buffer,
            )
            self._conv_cache[key] = TtConv2d(config, self.device)
        return self._conv_cache[key]

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        conv = self._get_conv(batch_size, height, width)
        x, h_w = conv(x, return_output_dim=True)

        # Apply per-channel PReLU: prelu(x) = max(0, x) + slope * min(0, x)
        # Manual implementation is faster than ttnn.prelu (avoids permute overhead)
        if self.prelu_weight is not None:
            # Cache the slopes tensor on device
            if not hasattr(self, "_slopes_tt"):
                slopes = self.prelu_weight.view(1, 1, 1, -1)
                self._slopes_tt = ttnn.from_torch(
                    slopes,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )

            pos = ttnn.relu(x)  # max(0, x)
            neg = ttnn.neg(ttnn.relu(ttnn.neg(x)))  # min(0, x)
            neg_scaled = ttnn.mul(neg, self._slopes_tt)
            x = ttnn.add(pos, neg_scaled)

        return x, h_w[0], h_w[1]


class TTNNDepthwiseSeparableBlock:
    """
    TTNN Depthwise Separable Convolution block: DW 3x3 + PW 1x1.

    Key optimization: NO reshape between DW and PW convs!
    Uses DRAM slicing for large spatial dimensions.
    """

    def __init__(
        self,
        device,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        name: str = "",
        use_dram_slicing: bool = False,
        num_slices: int = 2,
        high_precision: bool = False,  # Use higher precision for this block
    ):
        self.device = device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.name = name

        # Depthwise 3x3 with stride
        self.dw = TTNNConvBNPReLU(
            device,
            in_ch,
            in_ch,
            kernel=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            name=f"{name}.dw",
            use_dram_slicing=use_dram_slicing,
            num_slices=num_slices,
            high_precision=high_precision,
        )

        # Pointwise 1x1 (always stride=1)
        self.pw = TTNNConvBNPReLU(
            device,
            in_ch,
            out_ch,
            kernel=1,
            stride=1,
            padding=0,
            groups=1,
            name=f"{name}.pw",
            use_dram_slicing=use_dram_slicing,
            num_slices=num_slices,
            high_precision=high_precision,
        )

    def load_weights(self, dw_weight, dw_bias, dw_prelu, pw_weight, pw_bias, pw_prelu):
        """Load weights for both DW and PW convolutions."""
        self.dw.load_weights(dw_weight, dw_bias, dw_prelu)
        self.pw.load_weights(pw_weight, pw_bias, pw_prelu)

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward - NO reshape between DW and PW!"""
        x, h, w = self.dw(x, batch_size, height, width)
        # NO RESHAPE - pass directly to PW
        x, h, w = self.pw(x, batch_size, h, w)
        return x, h, w


class TtSFace:
    """
    Optimized TTNN SFace (MobileFaceNet) for face recognition.

    Input: [B, 112, 112, 3] NHWC normalized face image
    Output: [B, 128] L2-normalized embedding vector

    Uses DRAM slicing for early layers (112x112, 56x56) to avoid L1 overflow.
    """

    def __init__(self, device, embedding_size: int = 128):
        self.device = device
        self.embedding_size = embedding_size

        # Early layers with large spatial dims use DRAM slicing
        # Conv1: 3 -> 32, stride 1, 112x112 -> 112x112
        self.conv1 = TTNNConvBNPReLU(
            device, 3, 32, kernel=3, stride=1, padding=1, name="conv1", use_dram_slicing=True, num_slices=4
        )

        # Block 2: 32 -> 64, DW stride=1, 112x112 -> 112x112
        self.block2 = TTNNDepthwiseSeparableBlock(
            device, 32, 64, stride=1, name="block2", use_dram_slicing=True, num_slices=4
        )

        # Block 3: 64 -> 128, DW stride=2, 112x112 -> 56x56
        self.block3 = TTNNDepthwiseSeparableBlock(
            device, 64, 128, stride=2, name="block3", use_dram_slicing=True, num_slices=4
        )

        # Block 4: 128 -> 128, DW stride=1, 56x56 -> 56x56
        self.block4 = TTNNDepthwiseSeparableBlock(
            device, 128, 128, stride=1, name="block4", use_dram_slicing=True, num_slices=2
        )

        # Block 5: 128 -> 256, DW stride=2, 56x56 -> 28x28
        self.block5 = TTNNDepthwiseSeparableBlock(
            device, 128, 256, stride=2, name="block5", use_dram_slicing=True, num_slices=2
        )

        # Block 6: 256 -> 256, DW stride=1, 28x28 -> 28x28 (smaller spatial, no slicing needed)
        self.block6 = TTNNDepthwiseSeparableBlock(device, 256, 256, stride=1, name="block6")

        # Block 7: 256 -> 512, DW stride=2, 28x28 -> 14x14
        self.block7 = TTNNDepthwiseSeparableBlock(device, 256, 512, stride=2, name="block7")

        # Blocks 8-12: 512 -> 512, DW stride=1, 14x14 -> 14x14 (5 blocks)
        self.block8 = TTNNDepthwiseSeparableBlock(device, 512, 512, stride=1, name="block8")
        self.block9 = TTNNDepthwiseSeparableBlock(device, 512, 512, stride=1, name="block9")
        self.block10 = TTNNDepthwiseSeparableBlock(device, 512, 512, stride=1, name="block10")
        self.block11 = TTNNDepthwiseSeparableBlock(device, 512, 512, stride=1, name="block11")
        self.block12 = TTNNDepthwiseSeparableBlock(device, 512, 512, stride=1, name="block12")

        # Block 13: 512 -> 1024, DW stride=2, 14x14 -> 7x7
        self.block13 = TTNNDepthwiseSeparableBlock(device, 512, 1024, stride=2, name="block13")

        # Block 14: 1024 -> 1024, DW stride=1, 7x7 -> 7x7
        self.block14 = TTNNDepthwiseSeparableBlock(device, 1024, 1024, stride=1, name="block14")

        # Global BN weights (applied as scale/bias after flatten)
        self.bn1_weight = None
        self.bn1_bias = None
        self.bn1_mean = None
        self.bn1_var = None

        # FC layer: 1024 * 7 * 7 = 50176 -> 128
        self.fc_weight = None
        self.fc_bias = None

        # Final BN on embedding
        self.bn2_weight = None
        self.bn2_bias = None
        self.bn2_mean = None
        self.bn2_var = None

    def load_weights_from_torch(self, torch_model):
        """Load weights from PyTorch SFace model."""

        def to_ttnn(w, b):
            return Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(w, b)

        def fuse_conv_bn(conv, bn):
            """Fuse Conv2d and BatchNorm2d weights."""
            w_conv = conv.weight.clone()
            b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)

            # BN params
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps

            # Fuse: w_fused = w * gamma / sqrt(var + eps)
            #       b_fused = (b - mean) * gamma / sqrt(var + eps) + beta
            std = torch.sqrt(var + eps)
            scale = gamma / std

            w_fused = w_conv * scale.view(-1, 1, 1, 1)
            b_fused = (b_conv - mean) * scale + beta

            return w_fused, b_fused

        def load_conv_bn_prelu(ttnn_module, torch_conv_bn_prelu):
            """Load fused Conv+BN weights and PReLU."""
            w_fused, b_fused = fuse_conv_bn(torch_conv_bn_prelu.conv, torch_conv_bn_prelu.bn)
            w, b = to_ttnn(w_fused, b_fused)
            prelu = torch_conv_bn_prelu.prelu.weight.clone()
            ttnn_module.load_weights(w, b, prelu)

        def load_dw_sep_block(ttnn_block, torch_block):
            """Load weights for DepthwiseSeparableBlock."""
            # DW
            dw_w, dw_b = fuse_conv_bn(torch_block.dw.conv, torch_block.dw.bn)
            dw_w, dw_b = to_ttnn(dw_w, dw_b)
            dw_prelu = torch_block.dw.prelu.weight.clone()

            # PW
            pw_w, pw_b = fuse_conv_bn(torch_block.pw.conv, torch_block.pw.bn)
            pw_w, pw_b = to_ttnn(pw_w, pw_b)
            pw_prelu = torch_block.pw.prelu.weight.clone()

            ttnn_block.load_weights(dw_w, dw_b, dw_prelu, pw_w, pw_b, pw_prelu)

        # Load conv1
        load_conv_bn_prelu(self.conv1, torch_model.conv1)

        # Load blocks 2-14
        load_dw_sep_block(self.block2, torch_model.block2)
        load_dw_sep_block(self.block3, torch_model.block3)
        load_dw_sep_block(self.block4, torch_model.block4)
        load_dw_sep_block(self.block5, torch_model.block5)
        load_dw_sep_block(self.block6, torch_model.block6)
        load_dw_sep_block(self.block7, torch_model.block7)
        load_dw_sep_block(self.block8, torch_model.block8)
        load_dw_sep_block(self.block9, torch_model.block9)
        load_dw_sep_block(self.block10, torch_model.block10)
        load_dw_sep_block(self.block11, torch_model.block11)
        load_dw_sep_block(self.block12, torch_model.block12)
        load_dw_sep_block(self.block13, torch_model.block13)
        load_dw_sep_block(self.block14, torch_model.block14)

        # Load global BN1
        self.bn1_weight = torch_model.bn1.weight.clone()
        self.bn1_bias = torch_model.bn1.bias.clone()
        self.bn1_mean = torch_model.bn1.running_mean.clone()
        self.bn1_var = torch_model.bn1.running_var.clone()

        # Load FC
        self.fc_weight = torch_model.fc.weight.clone()
        self.fc_bias = torch_model.fc.bias.clone()

        # Load final BN2
        self.bn2_weight = torch_model.bn2.weight.clone()
        self.bn2_bias = torch_model.bn2.bias.clone()
        self.bn2_mean = torch_model.bn2.running_mean.clone()
        self.bn2_var = torch_model.bn2.running_var.clone()

    def __call__(self, x: ttnn.Tensor):
        """
        Forward pass - ALL operations on TT device.

        Args:
            x: Input tensor [B, 112, 112, 3] NHWC (raw 0-255 pixel values)

        Returns:
            embedding: [B, 128] L2-normalized face embedding (TTNN tensor)
        """
        batch_size = x.shape[0]
        h, w = x.shape[1], x.shape[2]

        # ============== PREPROCESSING (matches ONNX model) ==============
        # Normalize: (x - 127.5) / 128.0 = (x - 127.5) * 0.0078125
        x = ttnn.sub(x, 127.5)
        x = ttnn.mul(x, 0.0078125)

        # ============== BACKBONE (TT Device) ==============
        x, h, w = self.conv1(x, batch_size, h, w)  # 112 -> 112
        x, h, w = self.block2(x, batch_size, h, w)  # 112 -> 112
        x, h, w = self.block3(x, batch_size, h, w)  # 112 -> 56
        x, h, w = self.block4(x, batch_size, h, w)  # 56 -> 56
        x, h, w = self.block5(x, batch_size, h, w)  # 56 -> 28
        x, h, w = self.block6(x, batch_size, h, w)  # 28 -> 28
        x, h, w = self.block7(x, batch_size, h, w)  # 28 -> 14
        x, h, w = self.block8(x, batch_size, h, w)  # 14 -> 14
        x, h, w = self.block9(x, batch_size, h, w)  # 14 -> 14
        x, h, w = self.block10(x, batch_size, h, w)  # 14 -> 14
        x, h, w = self.block11(x, batch_size, h, w)  # 14 -> 14
        x, h, w = self.block12(x, batch_size, h, w)  # 14 -> 14
        x, h, w = self.block13(x, batch_size, h, w)  # 14 -> 7
        x, h, w = self.block14(x, batch_size, h, w)  # 7 -> 7

        # ============== HEAD (TT Device) ==============
        # Backbone output: [batch, 1, h*w, channels] = [1, 1, 49, 1024]
        channels = 1024

        # Reshape to NHWC [batch, h, w, channels] = [1, 7, 7, 1024]
        x = ttnn.reshape(x, [batch_size, h, w, channels])

        # BN1: Apply as element-wise scale/bias (NHWC layout)
        if not hasattr(self, "_bn1_scale_tt"):
            eps = 0.001  # Match ONNX model's epsilon
            bn1_scale = self.bn1_weight / torch.sqrt(self.bn1_var + eps)
            bn1_bias = self.bn1_bias - self.bn1_mean * bn1_scale
            # Shape: [1, 1, 1, 1024] to broadcast over [B, H, W, C]
            self._bn1_scale_tt = ttnn.from_torch(
                bn1_scale.view(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self._bn1_bias_tt = ttnn.from_torch(
                bn1_bias.view(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        x = ttnn.mul(x, self._bn1_scale_tt)
        x = ttnn.add(x, self._bn1_bias_tt)

        # Permute NHWC -> NCHW for correct flatten order
        # [B, H, W, C] -> [B, C, H, W] = [1, 1024, 7, 7]
        x = ttnn.permute(x, [0, 3, 1, 2])

        # Flatten NCHW: [B, C, H, W] -> [B, C*H*W] = [1, 50176]
        x = ttnn.reshape(x, [batch_size, channels * h * w])

        # FC layer: [B, 50176] @ [50176, 128] + bias using ttnn.linear (fused & optimized)
        if not hasattr(self, "_fc_weight_tt"):
            # Weight for ttnn.linear: [50176, 128] (transposed from PyTorch [128, 50176])
            self._fc_weight_tt = ttnn.from_torch(
                self.fc_weight.T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self._fc_bias_tt = ttnn.from_torch(
                self.fc_bias.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        x = ttnn.linear(x, self._fc_weight_tt, bias=self._fc_bias_tt)

        # BN2: Apply as element-wise scale/bias
        if not hasattr(self, "_bn2_scale_tt"):
            eps = 0.001  # Match ONNX model's epsilon
            bn2_scale = self.bn2_weight / torch.sqrt(self.bn2_var + eps)
            bn2_bias = self.bn2_bias - self.bn2_mean * bn2_scale
            self._bn2_scale_tt = ttnn.from_torch(
                bn2_scale.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self._bn2_bias_tt = ttnn.from_torch(
                bn2_bias.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        x = ttnn.mul(x, self._bn2_scale_tt)
        x = ttnn.add(x, self._bn2_bias_tt)

        # L2 normalize: x / ||x||_2
        x_sq = ttnn.mul(x, x)
        x_sum = ttnn.sum(x_sq, dim=1, keepdim=True)
        x_norm = ttnn.sqrt(x_sum)
        x_norm = ttnn.add(x_norm, 1e-12)
        x = ttnn.div(x, x_norm)

        return x


def create_sface_model(device, torch_model=None) -> TtSFace:
    """Factory function to create SFace model."""
    model = TtSFace(device)
    if torch_model is not None:
        model.load_weights_from_torch(torch_model)
    return model
