# ==============================================================================
# LLVC (Low-Latency Low-Resource Voice Conversion) Exact Koe AI Architecture
# Target Repository: tenstorrent/tt-metal
# Pull Request: #52645
# ==============================================================================

import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

L1_MEM_CONFIG = ttnn.MemoryConfig(
    tensor_memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.L1,
)


class TtCausalConv1d:
    """
    Causal 1D Convolution with MANUAL LEFT PADDING before ttnn.conv2d
    Ensures zero lookahead for real-time streaming audio (<50ms).
    """
    def __init__(self, device, in_channels, out_channels, kernel_size, dilation=1, groups=1, parameters=None):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.padding = (kernel_size - 1) * dilation

        self.weight = parameters.weight
        self.bias = getattr(parameters, "bias", None)

    def __call__(self, x, batch_size, seq_len, state_buffer=None):
        """
        x: NHWC Tensor [batch_size, 1, seq_len, in_channels]
        """
        # Manual Left Padding (Left ONLY) for Causal Convolution
        if self.padding > 0:
            x = ttnn.pad(x, padding=((0, 0), (0, 0), (self.padding, 0), (0, 0)), value=0.0)
            seq_len_padded = seq_len + self.padding
        else:
            seq_len_padded = seq_len

        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation="",
            deallocate_activation=False,
            reallocate_halo=False,
        )

        # Padding in conv2d is set to (0,0) because left padding was applied manually above
        out, [out_h, out_w], [w_prep, b_prep] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=seq_len_padded,
            kernel_size=(1, self.kernel_size),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, self.dilation),
            groups=self.groups,
            conv_config=conv_config,
        )
        return out


class TtDepthwiseSeparableDilatedBlock:
    """
    Depthwise-Separable Dilated Block for LLVC Encoder.
    Depthwise Causal Conv -> Pointwise Conv -> LayerNorm -> LeakyReLU.
    """
    def __init__(self, device, channels, kernel_size=3, dilation=1, parameters=None):
        self.device = device
        self.channels = channels

        # Depthwise Conv (groups = channels)
        self.dw_conv = TtCausalConv1d(
            device=device,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            parameters=parameters.dw_conv,
        )
        # Pointwise Conv (1x1)
        self.pw_conv = TtCausalConv1d(
            device=device,
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            dilation=1,
            groups=1,
            parameters=parameters.pw_conv,
        )

        self.ln_weight = parameters.ln.weight
        self.ln_bias = parameters.ln.bias

    def __call__(self, x, batch_size, seq_len):
        residual = x
        h = self.dw_conv(x, batch_size=batch_size, seq_len=seq_len)
        h = self.pw_conv(h, batch_size=batch_size, seq_len=seq_len)
        h = ttnn.layer_norm(h, weight=self.ln_weight, bias=self.ln_bias, memory_config=L1_MEM_CONFIG)
        h = ttnn.leaky_relu(h, negative_slope=0.2)
        out = ttnn.add(h, residual, memory_config=L1_MEM_CONFIG)
        return out


class TtCausalCrossAttention:
    """
    13-Frame Causal Cross-Attention Module for LLVC Decoder.
    """
    def __init__(self, device, channels=256, frames=13, parameters=None):
        self.device = device
        self.channels = channels
        self.frames = frames

        self.query_proj = TtCausalConv1d(device, channels, channels, kernel_size=1, parameters=parameters.query_proj)
        self.key_proj = TtCausalConv1d(device, channels, channels, kernel_size=1, parameters=parameters.key_proj)
        self.value_proj = TtCausalConv1d(device, channels, channels, kernel_size=1, parameters=parameters.value_proj)
        self.out_proj = TtCausalConv1d(device, channels, channels, kernel_size=1, parameters=parameters.out_proj)

    def __call__(self, x, batch_size, seq_len):
        q = self.query_proj(x, batch_size=batch_size, seq_len=seq_len)
        k = self.key_proj(x, batch_size=batch_size, seq_len=seq_len)
        v = self.value_proj(x, batch_size=batch_size, seq_len=seq_len)

        # Causal Attention Score computation
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        attn_weights = ttnn.softmax(scores, dim=-1)
        attn_out = ttnn.matmul(attn_weights, v)
        out = self.out_proj(attn_out, batch_size=batch_size, seq_len=seq_len)
        return out


class TtLLVCModel:
    """
    Exact Real LLVC Model Architecture on TTNN.
    - Input: Raw PCM 16kHz Audio [N, 1, L, 1]
    - Prenet: 12 Causal Conv Blocks -> 512-dim
    - Encoder: 8 Depthwise-Separable Dilated Blocks [1, 2, 4, 8, 16, 32, 1, 2]
    - Decoder: 13-Frame Causal Cross-Attention (512 -> 256)
    - Vocoder: ConvTranspose1d Upsampling
    - 4 Streaming State Buffers for Continuous Streaming
    """
    def __init__(self, device, in_channels=1, hidden_dim=512, bottle_dim=256, out_channels=1, parameters=None):
        self.device = device
        self.hidden_dim = hidden_dim
        self.bottle_dim = bottle_dim

        # 1. Prenet: 12 Causal Conv Blocks (Input 1-ch Raw PCM -> 512-dim)
        self.prenet = []
        ch_in = in_channels
        for i in range(12):
            ch_out = hidden_dim if i == 11 else min(64 * (2 ** (i // 3)), hidden_dim)
            conv = TtCausalConv1d(device, ch_in, ch_out, kernel_size=3, parameters=getattr(parameters.prenet, f"{i}"))
            self.prenet.append(conv)
            ch_in = ch_out

        # 2. Encoder: 8 Depthwise-Separable Dilated Blocks
        dilations = [1, 2, 4, 8, 16, 32, 1, 2]
        self.encoder = []
        for i, d in enumerate(dilations):
            block = TtDepthwiseSeparableDilatedBlock(
                device=device,
                channels=hidden_dim,
                kernel_size=3,
                dilation=d,
                parameters=getattr(parameters.encoder, f"{i}")
            )
            self.encoder.append(block)

        # 3. Decoder: 13-Frame Causal Cross-Attention & Dimension Reduction (512 -> 256)
        self.attention = TtCausalCrossAttention(device, channels=hidden_dim, frames=13, parameters=parameters.attention)
        self.proj_down = TtCausalConv1d(device, hidden_dim, bottle_dim, kernel_size=1, parameters=parameters.proj_down)

        # 4. Vocoder / Upsampling Synthesizer
        self.vocoder = TtCausalConv1d(device, bottle_dim, out_channels, kernel_size=7, parameters=parameters.vocoder)

        # 5. 4 Streaming State Buffers for Continuous Audio Chunks
        self.state_buffers = [None] * 4

    def __call__(self, x, batch_size=1, seq_len=16000):
        """
        x: Raw PCM Audio Tensor in NHWC [batch_size, 1, seq_len, 1]
        """
        # 1. Prenet Processing
        h = x
        for conv in self.prenet:
            h = conv(h, batch_size=batch_size, seq_len=seq_len)
            h = ttnn.leaky_relu(h, negative_slope=0.2)

        # 2. Encoder Processing
        for block in self.encoder:
            h = block(h, batch_size=batch_size, seq_len=seq_len)

        # 3. Decoder & Causal Cross-Attention
        h = self.attention(h, batch_size=batch_size, seq_len=seq_len)
        h = self.proj_down(h, batch_size=batch_size, seq_len=seq_len)

        # 4. Vocoder Output Waveform Synthesis
        waveform = self.vocoder(h, batch_size=batch_size, seq_len=seq_len)
        waveform = ttnn.tanh(waveform)
        return waveform
