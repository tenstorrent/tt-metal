# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LLVC (Low-Latency Low-Resource Voice Conversion) — TTNN Implementation
=======================================================================
Architecture reference: KoeAI/LLVC (https://github.com/KoeAI/LLVC)
Paper: "Low-latency Real-time Voice Conversion on CPU" (arXiv:2311.00873)

Pipeline:
    Raw PCM 16kHz [N,1,L,1] → Prenet (12 CausalConv blocks, 1→512)
    → Encoder (8 Depthwise-Separable Dilated blocks, dilations [1,2,4,8,16,32,1,2])
    → Decoder (13-frame Causal Cross-Attention, 512→256)
    → Vocoder (ConvTranspose1d upsampling [8,8,2,2])
    → Output Waveform [N,1,L_out,1]

All tensors use NHWC layout [batch, 1, seq_len, channels].
Conv1d is mapped to ttnn.conv2d with kernel_size=(1, K), input_height=1.
"""

import math
import torch
import ttnn
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

L1_MEMORY_CONFIG = ttnn.MemoryConfig(
    tensor_memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.L1,
)

DRAM_MEMORY_CONFIG = ttnn.MemoryConfig(
    tensor_memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
)


@dataclass
class LLVCConfig:
    sample_rate: int = 16000
    encoder_dim: int = 512
    decoder_dim: int = 256
    prenet_blocks: int = 12
    encoder_blocks: int = 8
    encoder_dilations: list = None
    decoder_frames: int = 13
    decoder_num_heads: int = 8
    prenet_kernel_size: int = 3
    encoder_kernel_size: int = 3
    vocoder_upsample_rates: list = None

    def __post_init__(self):
        if self.encoder_dilations is None:
            self.encoder_dilations = [1, 2, 4, 8, 16, 32, 1, 2]
        if self.vocoder_upsample_rates is None:
            self.vocoder_upsample_rates = [8, 8, 2, 2]


# =============================================================================
# Causal 1D Convolution (via ttnn.conv2d, NHWC layout)
# =============================================================================

class TtCausalConv1d:
    """
    Causal 1D Convolution with manual left-only padding.
    Maps to ttnn.conv2d with input_height=1, kernel_size=(1, K).
    Ensures zero lookahead for real-time streaming (<50ms latency).
    """

    def __init__(self, device, in_channels, out_channels, kernel_size,
                 dilation=1, groups=1, parameters=None):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.padding_size = (kernel_size - 1) * dilation

        self.weight = parameters.weight
        self.bias = getattr(parameters, "bias", None)

    def __call__(self, x, batch_size, seq_len):
        """
        x: NHWC tensor [batch, 1, seq_len, in_channels]
        Returns: (output_tensor, output_seq_len)
        """
        # Manual left-only padding for causal convolution
        if self.padding_size > 0:
            pad_tensor = ttnn.zeros(
                [batch_size, 1, self.padding_size, self.in_channels],
                dtype=x.dtype,
                layout=x.layout,
                device=self.device,
                memory_config=x.memory_config(),
            )
            x = ttnn.concat([pad_tensor, x], dim=2)
            padded_seq_len = seq_len + self.padding_size
        else:
            padded_seq_len = seq_len

        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            activation="",
            math_approx_mode_enabled=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        out = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=padded_seq_len,
            kernel_size=(1, self.kernel_size),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, self.dilation),
            groups=self.groups,
            conv_config=conv_config,
        )

        # ttnn.conv2d may return a tuple (output, [out_h, out_w], [w, b])
        if isinstance(out, tuple):
            out = out[0]

        return out, seq_len


class TtCausalConvTranspose1d:
    """
    Transposed 1D Convolution for vocoder upsampling.
    Implemented as repeat_interleave (upsample) + causal conv2d.
    """

    def __init__(self, device, in_channels, out_channels, kernel_size,
                 stride=1, parameters=None):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_size = kernel_size - 1

        self.weight = parameters.weight
        self.bias = getattr(parameters, "bias", None)

    def __call__(self, x, batch_size, seq_len):
        # Upsample by repeating each time step `stride` times
        if self.stride > 1:
            x = ttnn.repeat_interleave(x, self.stride, dim=2)
            upsampled_len = seq_len * self.stride
        else:
            upsampled_len = seq_len

        # Causal left padding
        if self.padding_size > 0:
            pad_tensor = ttnn.zeros(
                [batch_size, 1, self.padding_size, self.in_channels],
                dtype=x.dtype,
                layout=x.layout,
                device=self.device,
                memory_config=x.memory_config(),
            )
            x = ttnn.concat([pad_tensor, x], dim=2)
            padded_len = upsampled_len + self.padding_size
        else:
            padded_len = upsampled_len

        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            activation="",
        )

        out = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=padded_len,
            kernel_size=(1, self.kernel_size),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            conv_config=conv_config,
        )

        if isinstance(out, tuple):
            out = out[0]

        return out, upsampled_len


# =============================================================================
# Prenet: 12 Causal Conv Blocks (1 → 512 dim)
# =============================================================================

class TtLLVCPrenet:
    """
    12-block causal convolution prenet.
    Projects raw 1-channel PCM audio into 512-dim latent space.
    Each block: CausalConv1d → LayerNorm → LeakyReLU(0.2)
    """

    def __init__(self, device, in_channels=1, out_channels=512,
                 num_blocks=12, kernel_size=3, parameters=None):
        self.device = device
        self.num_blocks = num_blocks
        self.blocks = []

        for i in range(num_blocks):
            ch_in = in_channels if i == 0 else out_channels
            block_params = parameters.blocks[i] if parameters else None
            conv_params = block_params.conv if block_params else None
            norm_params = block_params.norm if block_params else None

            conv = TtCausalConv1d(
                device=device,
                in_channels=ch_in,
                out_channels=out_channels,
                kernel_size=kernel_size,
                parameters=conv_params,
            )
            self.blocks.append((conv, norm_params))

    def __call__(self, x, batch_size, seq_len):
        for conv, norm_params in self.blocks:
            x, seq_len = conv(x, batch_size, seq_len)
            if norm_params is not None:
                x = ttnn.layer_norm(
                    x, weight=norm_params.weight, bias=norm_params.bias,
                    memory_config=L1_MEMORY_CONFIG,
                )
            else:
                x = ttnn.layer_norm(x, memory_config=L1_MEMORY_CONFIG)
            x = ttnn.leaky_relu(x, negative_slope=0.2)
        return x, seq_len


# =============================================================================
# Encoder: 8 Depthwise-Separable Dilated Blocks
# =============================================================================

class TtDepthwiseSeparableBlock:
    """
    Depthwise-separable dilated causal block.
    DepthwiseConv(groups=C) → PointwiseConv(1×1) → LayerNorm → LeakyReLU → Residual
    """

    def __init__(self, device, channels, kernel_size=3, dilation=1,
                 parameters=None):
        self.device = device

        self.dw_conv = TtCausalConv1d(
            device, in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, dilation=dilation,
            groups=channels,  # depthwise
            parameters=parameters.dw_conv if parameters else None,
        )
        self.pw_conv = TtCausalConv1d(
            device, in_channels=channels, out_channels=channels,
            kernel_size=1,
            parameters=parameters.pw_conv if parameters else None,
        )

        self.norm_weight = parameters.norm.weight if parameters else None
        self.norm_bias = parameters.norm.bias if parameters else None

    def __call__(self, x, batch_size, seq_len):
        residual = x
        h, seq_len = self.dw_conv(x, batch_size, seq_len)
        h, seq_len = self.pw_conv(h, batch_size, seq_len)
        h = ttnn.layer_norm(
            h, weight=self.norm_weight, bias=self.norm_bias,
            memory_config=L1_MEMORY_CONFIG,
        )
        h = ttnn.leaky_relu(h, negative_slope=0.2)
        out = ttnn.add(h, residual, memory_config=L1_MEMORY_CONFIG)
        return out, seq_len


class TtLLVCEncoder:
    """8-block depthwise-separable dilated encoder with dilations [1,2,4,8,16,32,1,2]."""

    def __init__(self, device, channels=512,
                 dilations=None, parameters=None):
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 1, 2]
        self.blocks = []
        for i, d in enumerate(dilations):
            block_params = getattr(parameters, f"block_{i}") if parameters else None
            block = TtDepthwiseSeparableBlock(
                device, channels=channels, kernel_size=3,
                dilation=d, parameters=block_params,
            )
            self.blocks.append(block)

    def __call__(self, x, batch_size, seq_len):
        for block in self.blocks:
            x, seq_len = block(x, batch_size, seq_len)
        return x, seq_len


# =============================================================================
# Decoder: 13-Frame Causal Cross-Attention + Dimension Reduction (512 → 256)
# =============================================================================

class TtCausalCrossAttention:
    """
    Multi-head scaled dot-product attention with 13-frame causal window.
    Each position attends only to the current and 12 previous frames.
    """

    def __init__(self, device, channels, num_heads=8, frames=13,
                 parameters=None):
        self.device = device
        self.channels = channels
        self.num_heads = num_heads
        self.frames = frames
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj_weight = parameters.q_proj.weight if parameters else None
        self.k_proj_weight = parameters.k_proj.weight if parameters else None
        self.v_proj_weight = parameters.v_proj.weight if parameters else None
        self.out_proj_weight = parameters.out_proj.weight if parameters else None

    def __call__(self, x, batch_size, seq_len):
        # x: [N, 1, L, C] — ttnn.linear operates on last dim
        q = ttnn.linear(x, self.q_proj_weight, memory_config=L1_MEMORY_CONFIG)
        k = ttnn.linear(x, self.k_proj_weight, memory_config=L1_MEMORY_CONFIG)
        v = ttnn.linear(x, self.v_proj_weight, memory_config=L1_MEMORY_CONFIG)

        # Scale Q
        q = ttnn.multiply(q, self.scale, memory_config=L1_MEMORY_CONFIG)

        # Reshape to multi-head: [N, num_heads, L, head_dim]
        q = ttnn.reshape(q, (batch_size, self.num_heads, seq_len, self.head_dim))
        k = ttnn.reshape(k, (batch_size, self.num_heads, seq_len, self.head_dim))
        v = ttnn.reshape(v, (batch_size, self.num_heads, seq_len, self.head_dim))

        # Attention scores: Q @ K^T
        k_t = ttnn.transpose(k, -2, -1)
        scores = ttnn.matmul(q, k_t, memory_config=L1_MEMORY_CONFIG)

        # 13-frame causal mask (additive: 0 for allowed, -inf for blocked)
        host_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
        for i in range(seq_len):
            start = max(0, i - self.frames + 1)
            host_mask[0, 0, i, start : i + 1] = 0.0
        mask = ttnn.from_torch(
            host_mask, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        scores = ttnn.add(scores, mask, memory_config=L1_MEMORY_CONFIG)

        # Softmax → weighted sum
        attn_weights = ttnn.softmax(scores, dim=-1, memory_config=L1_MEMORY_CONFIG)
        attn_out = ttnn.matmul(attn_weights, v, memory_config=L1_MEMORY_CONFIG)

        # Reshape back to [N, 1, L, C]
        attn_out = ttnn.reshape(attn_out, (batch_size, 1, seq_len, self.channels))
        out = ttnn.linear(attn_out, self.out_proj_weight, memory_config=L1_MEMORY_CONFIG)
        return out


class TtLLVCDecoder:
    """Decoder: Cross-attention + residual + LayerNorm + projection 512 → 256."""

    def __init__(self, device, in_channels=512, out_channels=256,
                 num_heads=8, frames=13, parameters=None):
        self.attention = TtCausalCrossAttention(
            device, in_channels, num_heads=num_heads, frames=frames,
            parameters=parameters.attention if parameters else None,
        )
        self.proj_down = TtCausalConv1d(
            device, in_channels=in_channels, out_channels=out_channels,
            kernel_size=1,
            parameters=parameters.proj_down if parameters else None,
        )
        self.norm_weight = parameters.norm.weight if parameters else None
        self.norm_bias = parameters.norm.bias if parameters else None

    def __call__(self, x, batch_size, seq_len):
        h = self.attention(x, batch_size, seq_len)
        h = ttnn.add(h, x, memory_config=L1_MEMORY_CONFIG)  # residual
        h = ttnn.layer_norm(
            h, weight=self.norm_weight, bias=self.norm_bias,
            memory_config=L1_MEMORY_CONFIG,
        )
        h, seq_len = self.proj_down(h, batch_size, seq_len)
        return h, seq_len


# =============================================================================
# Full LLVC Generator
# =============================================================================

class TtLLVCGenerator:
    """
    Full LLVC Voice Conversion Generator on TTNN.
    Architecture from KoeAI/LLVC mapped to Tenstorrent Wormhole/Blackhole hardware.

    Components:
        1. Prenet  — 12 causal conv blocks (1 → 512 dim)
        2. Encoder — 8 depthwise-separable dilated blocks (512 dim)
        3. Decoder — 13-frame causal cross-attention (512 → 256 dim)
        4. Vocoder — ConvTranspose1d upsampling (256 → 1 waveform)
    """

    def __init__(self, device, config=None, parameters=None):
        if config is None:
            config = LLVCConfig()
        self.device = device
        self.config = config

        # 1. Prenet: 12 causal conv blocks (1 → 512)
        self.prenet = TtLLVCPrenet(
            device, in_channels=1, out_channels=config.encoder_dim,
            num_blocks=config.prenet_blocks,
            kernel_size=config.prenet_kernel_size,
            parameters=parameters.prenet if parameters else None,
        )

        # 2. Encoder: 8 depthwise-separable dilated blocks
        self.encoder = TtLLVCEncoder(
            device, channels=config.encoder_dim,
            dilations=config.encoder_dilations,
            parameters=parameters.encoder if parameters else None,
        )

        # 3. Decoder: cross-attention + dim reduction (512 → 256)
        self.decoder = TtLLVCDecoder(
            device, in_channels=config.encoder_dim,
            out_channels=config.decoder_dim,
            num_heads=config.decoder_num_heads,
            frames=config.decoder_frames,
            parameters=parameters.decoder if parameters else None,
        )

        # 4. Vocoder: upsampling via transposed convolutions
        self.vocoder_layers = []
        ch = config.decoder_dim
        for i, rate in enumerate(config.vocoder_upsample_rates):
            out_ch = max(ch // 2, 1)
            layer = TtCausalConvTranspose1d(
                device, in_channels=ch, out_channels=out_ch,
                kernel_size=rate * 2, stride=rate,
                parameters=getattr(parameters.vocoder, f"{i}") if parameters else None,
            )
            self.vocoder_layers.append(layer)
            ch = out_ch

        # 5. Output projection
        self.out_conv = TtCausalConv1d(
            device, in_channels=ch, out_channels=1, kernel_size=7,
            parameters=parameters.out_conv if parameters else None,
        )

    def __call__(self, audio_input, batch_size=1, seq_len=16000):
        """
        Forward pass for voice conversion.

        Args:
            audio_input: NHWC tensor [batch, 1, seq_len, 1] — raw PCM 16kHz
            batch_size: batch dimension
            seq_len: input sequence length (samples)

        Returns:
            Converted waveform tensor [batch, 1, seq_len_out, 1]
        """
        # 1. Prenet: extract 512-dim acoustic features
        h, seq_len = self.prenet(audio_input, batch_size, seq_len)

        # 2. Encoder: capture temporal dependencies via dilated convolutions
        h, seq_len = self.encoder(h, batch_size, seq_len)

        # 3. Decoder: cross-attention context + dimension reduction
        h, seq_len = self.decoder(h, batch_size, seq_len)

        # 4. Vocoder: upsample to target sample rate
        for layer in self.vocoder_layers:
            h, seq_len = layer(h, batch_size, seq_len)
            h = ttnn.leaky_relu(h, negative_slope=0.2)

        # 5. Output waveform synthesis
        waveform, seq_len = self.out_conv(h, batch_size, seq_len)
        waveform = ttnn.tanh(waveform)

        return waveform
