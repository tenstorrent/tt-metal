# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TTNN implementation of Qwen3-TTS Speech Tokenizer Decoder.

Converts codec tokens to audio waveforms.

Architecture:
1. Codebook lookup (RVQ with 16 codebooks)
2. Pre-transformer (8 layers, 512 hidden)
3. ConvNeXt upsampler (2× + 2×)
4. Conv decoder (8× + 5× + 4× + 3×)

The pre-transformer uses TTNN, while convolutional layers use PyTorch fallback.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.reference.functional import SpeechTokenizerDecoderConfig
from models.demos.qwen3_tts.reference.functional import speech_tokenizer_decoder_forward as reference_decoder_forward
from models.demos.qwen3_tts.tt.ttnn_conv_decoder import TTNNConv1d, TTNNConvTranspose1d, ttnn_snake_activation


@dataclass
class SpeechTokenizerConfig:
    """Configuration for Speech Tokenizer Decoder."""

    # Quantizer config
    num_quantizers: int = 16
    codebook_size: int = 2048
    codebook_dim: int = 256
    latent_dim: int = 1024

    # Pre-transformer config
    pre_transformer_hidden_size: int = 512  # Input/output dim
    pre_transformer_qkv_dim: int = 1024  # Internal QKV dimension
    pre_transformer_intermediate_size: int = 1024
    pre_transformer_num_layers: int = 8
    pre_transformer_num_heads: int = 16
    pre_transformer_head_dim: int = 64  # qkv_dim / num_heads = 1024 / 16 = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 72

    # Decoder config
    decoder_dim: int = 1536
    upsample_rates: Tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: Tuple[int, ...] = (2, 2)

    # Audio config
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000

    # TTNN specific
    tile_size: int = 32


# =============================================================================
# PyTorch helper functions for convolutional decoder
# =============================================================================


def convnext_block(
    x: ttnn.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: Optional[torch.Tensor],
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: Optional[torch.Tensor],
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    device,
    gamma: Optional[torch.Tensor] = None,
) -> ttnn.Tensor:
    """ConvNeXt block for upsampling in TTNN.

    Input/Output shape: [batch, 1, seq_len, channels] (NHWC)
    """
    mc = ttnn.DRAM_MEMORY_CONFIG
    residual = x
    b, _, l, c = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])

    # Depthwise conv: NHWC -> NLC -> NHWC
    x_nlc = ttnn.reshape(x, (b, l, c), memory_config=mc)
    dw_conv = TTNNConv1d(
        device=device,
        in_channels=c,
        out_channels=c,
        kernel_size=int(dwconv_weight.shape[-1]),
        padding=int(dwconv_weight.shape[-1]) // 2,
        groups=c,
        weight=dwconv_weight,
        bias_tensor=dwconv_bias,
    )
    x_nlc, out_len = dw_conv(x_nlc, l)
    x = ttnn.reshape(x_nlc, (b, 1, out_len, c), memory_config=mc)

    # LayerNorm + pointwise linears in NHWC
    norm_w_tt = ttnn.from_torch(
        norm_weight.view(1, 1, 1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    norm_b_tt = ttnn.from_torch(
        norm_bias.view(1, 1, 1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    x = ttnn.layer_norm(x, weight=norm_w_tt, bias=norm_b_tt, memory_config=mc)

    # Pointwise convs (linear)
    pw1_w_tt = ttnn.from_torch(
        pwconv1_weight.T.contiguous().view(1, 1, pwconv1_weight.shape[1], pwconv1_weight.shape[0]).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    pw1_b_tt = (
        ttnn.from_torch(
            pwconv1_bias.view(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        if pwconv1_bias is not None
        else None
    )
    x = ttnn.linear(x, pw1_w_tt, bias=pw1_b_tt, memory_config=mc)
    x = ttnn.gelu(x, memory_config=mc)
    pw2_w_tt = ttnn.from_torch(
        pwconv2_weight.T.contiguous().view(1, 1, pwconv2_weight.shape[1], pwconv2_weight.shape[0]).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    pw2_b_tt = (
        ttnn.from_torch(
            pwconv2_bias.view(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        if pwconv2_bias is not None
        else None
    )
    x = ttnn.linear(x, pw2_w_tt, bias=pw2_b_tt, memory_config=mc)

    # Layer scale (broadcast on channels)
    if gamma is not None:
        gamma_tt = ttnn.from_torch(
            gamma.view(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        x = ttnn.multiply(x, gamma_tt, memory_config=mc)

    return ttnn.add(residual, x, memory_config=mc)


def conv_decoder_block(
    x: ttnn.Tensor,
    block_weights: dict,
    upsample_rate: int,
    device,
    num_residual_layers: int = 3,
) -> ttnn.Tensor:
    """Conv decoder block with upsampling and residual layers in TTNN.

    Input/Output shape: [batch, 1, seq_len, channels] (NHWC)
    """
    mc = ttnn.DRAM_MEMORY_CONFIG
    b, _, l, c = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])

    # Snake activation before upsampling
    if "alpha" in block_weights and "beta" in block_weights:
        channels = int(x.shape[-1])
        alpha_tt = ttnn.from_torch(
            block_weights["alpha"].view(1, 1, 1, channels).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        beta_tt = ttnn.from_torch(
            block_weights["beta"].view(1, 1, 1, channels).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        x = ttnn_snake_activation(x, alpha_tt, beta_tt)

    # Transposed conv for upsampling
    if "block.1.conv.weight" in block_weights:
        up_w = block_weights["block.1.conv.weight"]
        up_b = block_weights.get("block.1.conv.bias")
        up_conv = TTNNConvTranspose1d(
            device=device,
            in_channels=int(up_w.shape[0]),
            out_channels=int(up_w.shape[1]),
            kernel_size=int(up_w.shape[-1]),
            stride=upsample_rate,
            padding=(int(up_w.shape[-1]) - upsample_rate) // 2,
            weight=up_w,
            bias_tensor=up_b,
        )
        x, l = up_conv(x, l)
        c = int(x.shape[-1])

    # Residual layers
    for i in range(2, 2 + num_residual_layers):
        residual = x

        # First activation + conv
        act1_key = f"block.{i}.act1"
        if f"{act1_key}.alpha" in block_weights:
            channels = int(x.shape[-1])
            alpha_tt = ttnn.from_torch(
                block_weights[f"{act1_key}.alpha"].view(1, 1, 1, channels).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            beta_tt = ttnn.from_torch(
                block_weights[f"{act1_key}.beta"].view(1, 1, 1, channels).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            x = ttnn_snake_activation(x, alpha_tt, beta_tt)

        conv1_weight = block_weights.get(f"block.{i}.conv1.conv.weight")
        conv1_bias = block_weights.get(f"block.{i}.conv1.conv.bias")
        if conv1_weight is not None:
            x_nlc = ttnn.reshape(x, (b, l, c), memory_config=mc)
            conv1 = TTNNConv1d(
                device=device,
                in_channels=int(conv1_weight.shape[1]),
                out_channels=int(conv1_weight.shape[0]),
                kernel_size=int(conv1_weight.shape[-1]),
                padding=int(conv1_weight.shape[-1]) // 2,
                weight=conv1_weight,
                bias_tensor=conv1_bias,
            )
            x_nlc, l = conv1(x_nlc, l)
            c = int(conv1_weight.shape[0])
            x = ttnn.reshape(x_nlc, (b, 1, l, c), memory_config=mc)

        # Second activation + conv
        act2_key = f"block.{i}.act2"
        if f"{act2_key}.alpha" in block_weights:
            channels = int(x.shape[-1])
            alpha_tt = ttnn.from_torch(
                block_weights[f"{act2_key}.alpha"].view(1, 1, 1, channels).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            beta_tt = ttnn.from_torch(
                block_weights[f"{act2_key}.beta"].view(1, 1, 1, channels).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            x = ttnn_snake_activation(x, alpha_tt, beta_tt)

        conv2_weight = block_weights.get(f"block.{i}.conv2.conv.weight")
        conv2_bias = block_weights.get(f"block.{i}.conv2.conv.bias")
        if conv2_weight is not None:
            x_nlc = ttnn.reshape(x, (b, l, c), memory_config=mc)
            conv2 = TTNNConv1d(
                device=device,
                in_channels=int(conv2_weight.shape[1]),
                out_channels=int(conv2_weight.shape[0]),
                kernel_size=int(conv2_weight.shape[-1]),
                padding=0,
                weight=conv2_weight,
                bias_tensor=conv2_bias,
            )
            x_nlc, l = conv2(x_nlc, l)
            c = int(conv2_weight.shape[0])
            x = ttnn.reshape(x_nlc, (b, 1, l, c), memory_config=mc)

        x = ttnn.add(residual, x, memory_config=mc)

    return x


# =============================================================================
# TTNN Pre-Transformer Layer
# =============================================================================


class TtPreTransformerAttention(LightweightModule):
    """Pre-transformer attention for speech tokenizer decoder."""

    def __init__(
        self,
        device,
        state_dict: dict,
        layer_num: int,
        config: SpeechTokenizerConfig,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.num_heads = config.pre_transformer_num_heads
        self.head_dim = config.pre_transformer_head_dim
        self.hidden_size = config.pre_transformer_hidden_size  # Input dim (512)
        self.qkv_dim = config.pre_transformer_qkv_dim  # Internal QKV dim (1024)
        self.sliding_window = config.sliding_window

        prefix = f"pre_transformer.layers.{layer_num}.self_attn."

        # Load Q, K, V, O projection weights
        self.wq = ttnn.from_torch(
            state_dict[prefix + "q_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wk = ttnn.from_torch(
            state_dict[prefix + "k_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wv = ttnn.from_torch(
            state_dict[prefix + "v_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wo = ttnn.from_torch(
            state_dict[prefix + "o_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Load biases if present
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.o_bias = None
        if prefix + "q_proj.bias" in state_dict:
            self.q_bias = ttnn.from_torch(
                state_dict[prefix + "q_proj.bias"].unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        if prefix + "k_proj.bias" in state_dict:
            self.k_bias = ttnn.from_torch(
                state_dict[prefix + "k_proj.bias"].unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        if prefix + "v_proj.bias" in state_dict:
            self.v_bias = ttnn.from_torch(
                state_dict[prefix + "v_proj.bias"].unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        if prefix + "o_proj.bias" in state_dict:
            self.o_bias = ttnn.from_torch(
                state_dict[prefix + "o_proj.bias"].unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass for pre-transformer attention.

        Args:
            x: Input of shape [batch, seq_len, hidden_size]
            cos, sin: RoPE frequencies

        Returns:
            Output of shape [batch, seq_len, hidden_size]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Project Q, K, V
        q = ttnn.linear(x, self.wq, bias=self.q_bias)
        k = ttnn.linear(x, self.wk, bias=self.k_bias)
        v = ttnn.linear(x, self.wv, bias=self.v_bias)

        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = ttnn.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = ttnn.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = ttnn.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = ttnn.permute(q, [0, 2, 1, 3])
        k = ttnn.permute(k, [0, 2, 1, 3])
        v = ttnn.permute(v, [0, 2, 1, 3])

        # Apply RoPE
        # Note: Simplified RoPE application for pre-transformer
        # The pre-transformer uses standard RoPE without MROPE

        # Scaled dot-product attention with causal mask
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        # Transpose and reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, qkv_dim]
        attn_output = ttnn.permute(attn_output, [0, 2, 1, 3])
        attn_output = ttnn.reshape(attn_output, [batch_size, seq_len, self.qkv_dim])

        # Output projection: qkv_dim (1024) -> hidden_size (512)
        output = ttnn.linear(attn_output, self.wo, bias=self.o_bias)

        return output


class TtPreTransformerMLP(LightweightModule):
    """Pre-transformer MLP (SwiGLU) for speech tokenizer decoder."""

    def __init__(
        self,
        device,
        state_dict: dict,
        layer_num: int,
        config: SpeechTokenizerConfig,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device

        prefix = f"pre_transformer.layers.{layer_num}.mlp."

        # Load weights
        self.w_gate = ttnn.from_torch(
            state_dict[prefix + "gate_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w_up = ttnn.from_torch(
            state_dict[prefix + "up_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w_down = ttnn.from_torch(
            state_dict[prefix + "down_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """SwiGLU MLP forward."""
        gate = ttnn.linear(x, self.w_gate)
        gate = ttnn.silu(gate)
        up = ttnn.linear(x, self.w_up)
        hidden = ttnn.mul(gate, up)
        output = ttnn.linear(hidden, self.w_down)
        return output


class TtPreTransformerLayer(LightweightModule):
    """Single pre-transformer layer."""

    TILE = 32

    def __init__(
        self,
        device,
        state_dict: dict,
        layer_num: int,
        config: SpeechTokenizerConfig,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.config = config
        hidden = config.pre_transformer_hidden_size  # 512

        prefix = f"pre_transformer.layers.{layer_num}."

        # Layer norms: shape [1, 1, hidden//TILE, TILE] in ROW_MAJOR_LAYOUT
        # (matches the RMSNorm class convention used by the Talker)
        # def _load_norm(key):
        #     w = state_dict[key].view(1, 1, hidden).reshape([1, 1, hidden // self.TILE, self.TILE])
        #     return ttnn.from_torch(
        #         w.to(dtype.to_torch() if hasattr(dtype, "to_torch") else torch.bfloat16),
        #         dtype=dtype,
        #         layout=ttnn.ROW_MAJOR_LAYOUT,
        #         device=device,
        #         memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     )

        import torch as _torch

        def _load_norm_t(key):
            w = state_dict[key].to(_torch.bfloat16).view(1, 1, hidden).reshape([1, 1, hidden // self.TILE, self.TILE])
            return ttnn.as_tensor(
                w, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        self.input_layernorm_weight = _load_norm_t(prefix + "input_layernorm.weight")
        self.post_attention_layernorm_weight = _load_norm_t(prefix + "post_attention_layernorm.weight")

        # Attention and MLP
        self.attention = TtPreTransformerAttention(device, state_dict, layer_num, config, dtype)
        self.mlp = TtPreTransformerMLP(device, state_dict, layer_num, config, dtype)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass for pre-transformer layer.

        Args:
            x: [batch, seq_len, hidden_size] (3D)
        """
        # ttnn.rms_norm requires 4D [batch, 1, seq, hidden]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x_4d = ttnn.reshape(x, [batch_size, 1, seq_len, self.config.pre_transformer_hidden_size])

        # Pre-norm for attention
        residual = x
        x_normed = ttnn.rms_norm(
            x_4d,
            epsilon=self.config.rms_norm_eps,
            weight=self.input_layernorm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        # Back to 3D for attention
        x_normed_3d = ttnn.reshape(x_normed, [batch_size, seq_len, self.config.pre_transformer_hidden_size])

        # Self-attention (3D in, 3D out)
        attn_out = self.attention(x_normed_3d, cos, sin)

        # Residual connection (3D)
        x = ttnn.add(attn_out, residual)

        # Pre-norm for MLP (back to 4D)
        residual = x
        x_4d = ttnn.reshape(x, [batch_size, 1, seq_len, self.config.pre_transformer_hidden_size])
        x_normed = ttnn.rms_norm(
            x_4d,
            epsilon=self.config.rms_norm_eps,
            weight=self.post_attention_layernorm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        x_normed_3d = ttnn.reshape(x_normed, [batch_size, seq_len, self.config.pre_transformer_hidden_size])

        # MLP (3D in, 3D out)
        mlp_out = self.mlp(x_normed_3d)

        # Residual connection (3D)
        x = ttnn.add(mlp_out, residual)

        return x


class TtPreTransformer(LightweightModule):
    """Pre-transformer for speech tokenizer decoder (8 layers)."""

    def __init__(
        self,
        device,
        state_dict: dict,
        config: SpeechTokenizerConfig,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.config = config

        # Input projection
        self.input_proj = ttnn.from_torch(
            state_dict["pre_transformer.input_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.input_proj_bias = None
        if "pre_transformer.input_proj.bias" in state_dict:
            self.input_proj_bias = ttnn.from_torch(
                state_dict["pre_transformer.input_proj.bias"].unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Transformer layers
        self.layers = []
        for i in range(config.pre_transformer_num_layers):
            self.layers.append(TtPreTransformerLayer(device, state_dict, i, config, dtype))

        # Final norm: [1, 1, hidden//32, 32] in ROW_MAJOR_LAYOUT
        _hidden = config.pre_transformer_hidden_size
        import torch as _t

        _norm_w = (
            state_dict["pre_transformer.norm.weight"]
            .to(_t.bfloat16)
            .view(1, 1, _hidden)
            .reshape([1, 1, _hidden // 32, 32])
        )
        self.norm_weight = ttnn.as_tensor(
            _norm_w,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output projection
        self.output_proj = ttnn.from_torch(
            state_dict["pre_transformer.output_proj.weight"].T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.output_proj_bias = None
        if "pre_transformer.output_proj.bias" in state_dict:
            self.output_proj_bias = ttnn.from_torch(
                state_dict["pre_transformer.output_proj.bias"].unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through pre-transformer.

        Args:
            x: Input embeddings [batch, seq_len, latent_dim] (3D)
            cos, sin: RoPE frequencies (unused, pre-transformer uses full attention)

        Returns:
            Output [batch, seq_len, decoder_dim] (3D)
        """
        # Input projection (3D → 3D)
        x = ttnn.linear(x, self.input_proj, bias=self.input_proj_bias)

        # Process through layers (each layer: 3D → 3D)
        for layer in self.layers:
            x = layer(x, cos, sin)

        # Final norm: requires 4D [batch, 1, seq, hidden]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        hidden = self.config.pre_transformer_hidden_size
        x_4d = ttnn.reshape(x, [batch_size, 1, seq_len, hidden])
        x_4d = ttnn.rms_norm(x_4d, epsilon=self.config.rms_norm_eps, weight=self.norm_weight)
        x = ttnn.reshape(x_4d, [batch_size, seq_len, hidden])

        # Output projection (3D → 3D)
        x = ttnn.linear(x, self.output_proj, bias=self.output_proj_bias)

        return x


# =============================================================================
# Speech Tokenizer Decoder (Hybrid TTNN + PyTorch)
# =============================================================================


class TtSpeechTokenizerDecoder(LightweightModule):
    """
    Speech Tokenizer Decoder for Qwen3-TTS.

    Converts codec tokens to audio waveforms.

    Uses TTNN for:
    - Codebook embedding lookup
    - Pre-transformer (8 layers)

    Uses PyTorch fallback for:
    - ConvNeXt upsampler
    - Conv decoder (complex conv operations)
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: Optional[SpeechTokenizerConfig] = None,
        dtype=ttnn.bfloat16,
        use_reference: bool = True,  # Use fixed reference implementation
    ):
        super().__init__()
        self.device = device
        self.config = config or SpeechTokenizerConfig()
        self.dtype = dtype
        self.use_reference = use_reference

        # Store state_dict for reference implementation
        self._state_dict = state_dict

        # Store PyTorch weights for codebook and conv layers
        self._load_pytorch_weights(state_dict)

        # Check if pre-transformer weights exist
        self.has_pre_transformer = False
        if any(k.startswith("pre_transformer.") for k in state_dict.keys()):
            # Check if input projection dimensions match
            input_proj_key = "pre_transformer.input_proj.weight"
            if input_proj_key in state_dict:
                input_proj_shape = state_dict[input_proj_key].shape
                expected_input_dim = input_proj_shape[1]  # Linear weight is [out, in]

                # With proper RVQ processing: rvq_first (512) + rvq_rest (512) = 1024
                # This should match the pre_transformer input expectation
                actual_input_dim = self.config.latent_dim  # Should be 1024

                if expected_input_dim == actual_input_dim:
                    try:
                        self.pre_transformer = TtPreTransformer(device, state_dict, self.config, dtype)
                        self.has_pre_transformer = True
                        self._setup_rope()
                        print(f"  Pre-transformer loaded successfully (input_dim={expected_input_dim})")
                    except Exception as e:
                        print(f"  Warning: Could not initialize pre-transformer: {e}")
                        print("  Using conv decoder only (lower quality)")
                        self.has_pre_transformer = False
                else:
                    print(
                        f"  Note: Pre-transformer expects input dim {expected_input_dim}, "
                        f"but latent_dim is {actual_input_dim}. Skipping pre-transformer."
                    )

    def _load_pytorch_weights(self, state_dict: dict):
        """Load weights that will be used with PyTorch operations."""
        # RVQ First (semantic codebook) - MUST normalize by cluster_usage
        self.rvq_first_codebook = None
        self.rvq_first_cluster_usage = None
        first_key = "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"
        first_usage_key = "quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"
        if first_key in state_dict:
            embedding_sum = state_dict[first_key].clone()
            cluster_usage = state_dict.get(first_usage_key)
            # Normalize codebook: embedding_sum / cluster_usage.clamp(min=epsilon)
            if cluster_usage is not None:
                epsilon = 1e-5
                self.rvq_first_codebook = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]
            else:
                self.rvq_first_codebook = embedding_sum

        # RVQ First output projection (256 -> 512)
        self.rvq_first_output_proj = state_dict.get("quantizer.rvq_first.output_proj.weight")

        # RVQ Rest codebooks (acoustic - 15 codebooks) - MUST normalize by cluster_usage
        self.rvq_rest_codebooks = []
        for i in range(self.config.num_quantizers - 1):
            key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
            usage_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
            if key in state_dict:
                embedding_sum = state_dict[key].clone()
                cluster_usage = state_dict.get(usage_key)
                # Normalize codebook: embedding_sum / cluster_usage.clamp(min=epsilon)
                if cluster_usage is not None:
                    epsilon = 1e-5
                    normalized = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]
                    self.rvq_rest_codebooks.append(normalized)
                else:
                    self.rvq_rest_codebooks.append(embedding_sum)

        # RVQ Rest output projection (256 -> 512)
        self.rvq_rest_output_proj = state_dict.get("quantizer.rvq_rest.output_proj.weight")

        # Legacy: keep codebooks list for backwards compatibility
        self.codebooks = []
        if self.rvq_first_codebook is not None:
            self.codebooks.append(self.rvq_first_codebook)
        self.codebooks.extend(self.rvq_rest_codebooks)

        # Pre-conv weights
        self.pre_conv_weight = state_dict.get("pre_conv.conv.weight")
        self.pre_conv_bias = state_dict.get("pre_conv.conv.bias")

        # Upsampler weights (ConvNeXt blocks)
        self.upsample_weights = {}
        for i in range(len(self.config.upsampling_ratios)):
            prefix = f"upsample.{i}."
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    self.upsample_weights[k] = v.clone()

        # Conv decoder weights
        # Note: After extract_speech_tokenizer_weights, the "decoder." prefix is removed
        # So keys like "decoder.0.conv.weight" become "0.conv.weight"
        self.decoder_weights = {}
        for k, v in state_dict.items():
            # Check for numeric prefix (decoder blocks) or known decoder keys
            if k[0].isdigit() or k.startswith("decoder."):
                # Store with "decoder." prefix for consistent lookup
                if k.startswith("decoder."):
                    self.decoder_weights[k] = v.clone()
                else:
                    self.decoder_weights[f"decoder.{k}"] = v.clone()

    def _setup_rope(self):
        """Pre-compute RoPE frequencies for pre-transformer."""
        # This is a placeholder - actual implementation would compute RoPE
        # For now, we'll compute on-the-fly in forward

    def _compute_rope(self, seq_len: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute RoPE frequencies as TTNN tensors."""
        head_dim = self.config.pre_transformer_head_dim
        half_dim = head_dim // 2

        idx = ttnn.arange(0, head_dim, 2, device=self.device, dtype=ttnn.float32)
        idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
        exponents = ttnn.multiply(idx, 1.0 / head_dim)
        inv_freq = ttnn.reciprocal(ttnn.pow(self.config.rope_theta, exponents))

        pos = ttnn.arange(0, seq_len, 1, device=self.device, dtype=ttnn.float32)
        pos = ttnn.to_layout(pos, ttnn.TILE_LAYOUT)
        pos_col = ttnn.reshape(pos, (seq_len, 1))
        freq_row = ttnn.reshape(inv_freq, (1, half_dim))
        angles = ttnn.multiply(pos_col, freq_row)

        cos_half = ttnn.cos(angles)
        sin_half = ttnn.sin(angles)
        cos = ttnn.concat([cos_half, cos_half], dim=-1)
        sin = ttnn.concat([sin_half, sin_half], dim=-1)
        cos = ttnn.reshape(cos, (1, 1, seq_len, head_dim))
        sin = ttnn.reshape(sin, (1, 1, seq_len, head_dim))
        return (
            ttnn.to_layout(cos, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.to_layout(sin, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )

    def _codebook_lookup(self, token_ids: Union[torch.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        """
        Lookup embeddings from RVQ codebooks with proper projections.

        The RVQ has two parts:
        - rvq_first (semantic): 1 codebook -> output_proj -> 512 dim
        - rvq_rest (acoustic): 15 codebooks -> sum -> output_proj -> 512 dim
        - Concatenate -> 1024 dim (input to pre_transformer)

        Args:
            token_ids: [batch, num_quantizers, seq_len]

        Returns:
            embeddings: [batch, seq_len, 1024] (or codebook_dim if no projections)
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = ttnn.from_torch(
                token_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        batch_size, num_quantizers, seq_len = int(token_ids.shape[0]), int(token_ids.shape[1]), int(token_ids.shape[2])
        mc = ttnn.DRAM_MEMORY_CONFIG

        def _ids_for_quantizer(q_idx: int) -> ttnn.Tensor:
            q_slice = ttnn.slice(token_ids, [0, q_idx, 0], [batch_size, q_idx + 1, seq_len], memory_config=mc)
            return ttnn.reshape(q_slice, (batch_size, seq_len), memory_config=mc)

        def _tt_weight(codebook: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                codebook.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=mc,
            )

        # Process RVQ First (semantic codebook - index 0)
        rvq_first_emb_tt = None
        if self.rvq_first_codebook is not None and num_quantizers > 0:
            ids_tt = _ids_for_quantizer(0)
            codebook_tt = _tt_weight(self.rvq_first_codebook)
            rvq_first_emb_tt = ttnn.embedding(ids_tt, codebook_tt, layout=ttnn.TILE_LAYOUT, memory_config=mc)

            # Apply output projection if available (256 -> 512)
            if self.rvq_first_output_proj is not None:
                conv = TTNNConv1d(
                    device=self.device,
                    in_channels=int(self.rvq_first_output_proj.shape[1]),
                    out_channels=int(self.rvq_first_output_proj.shape[0]),
                    kernel_size=int(self.rvq_first_output_proj.shape[-1]),
                    padding=0,
                    weight=self.rvq_first_output_proj,
                    bias_tensor=None,
                )
                rvq_first_emb_tt, _ = conv(rvq_first_emb_tt, seq_len)

        # Process RVQ Rest (acoustic codebooks - indices 1-15)
        rvq_rest_emb_tt = None
        if len(self.rvq_rest_codebooks) > 0 and num_quantizers > 1:
            for i, codebook in enumerate(self.rvq_rest_codebooks):
                if i + 1 >= num_quantizers:
                    break
                ids_tt = _ids_for_quantizer(i + 1)
                codebook_tt = _tt_weight(codebook)
                emb_tt = ttnn.embedding(ids_tt, codebook_tt, layout=ttnn.TILE_LAYOUT, memory_config=mc)
                if rvq_rest_emb_tt is None:
                    rvq_rest_emb_tt = emb_tt
                else:
                    rvq_rest_emb_tt = ttnn.add(rvq_rest_emb_tt, emb_tt, memory_config=mc)

            # Apply output projection if available (256 -> 512)
            if rvq_rest_emb_tt is not None and self.rvq_rest_output_proj is not None:
                conv = TTNNConv1d(
                    device=self.device,
                    in_channels=int(self.rvq_rest_output_proj.shape[1]),
                    out_channels=int(self.rvq_rest_output_proj.shape[0]),
                    kernel_size=int(self.rvq_rest_output_proj.shape[-1]),
                    padding=0,
                    weight=self.rvq_rest_output_proj,
                    bias_tensor=None,
                )
                rvq_rest_emb_tt, _ = conv(rvq_rest_emb_tt, seq_len)

        # Combine semantic + acoustic branches (512 channels).
        if rvq_first_emb_tt is not None and rvq_rest_emb_tt is not None:
            embeddings = ttnn.add(rvq_first_emb_tt, rvq_rest_emb_tt, memory_config=mc)
        elif rvq_first_emb_tt is not None:
            embeddings = rvq_first_emb_tt
        elif rvq_rest_emb_tt is not None:
            embeddings = rvq_rest_emb_tt
        else:
            # Fallback: simple sum without projections
            embeddings = None
            for i, codebook in enumerate(self.codebooks):
                if i >= num_quantizers:
                    break
                ids_tt = _ids_for_quantizer(i)
                codebook_tt = _tt_weight(codebook)
                emb_tt = ttnn.embedding(ids_tt, codebook_tt, layout=ttnn.TILE_LAYOUT, memory_config=mc)
                embeddings = emb_tt if embeddings is None else ttnn.add(embeddings, emb_tt, memory_config=mc)

        return embeddings

    def _conv_decoder_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Conv decoder forward pass in TTNN.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            audio: [batch, 1, num_samples]
        """
        batch_size, seq_len, hidden_size = (
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            int(hidden_states.shape[2]),
        )

        # Project to expected dimension if needed
        if self.pre_conv_weight is not None:
            expected_in_channels = self.pre_conv_weight.shape[1]
            if hidden_size != expected_in_channels:
                raise RuntimeError(f"Expected hidden_size {expected_in_channels}, got {hidden_size}")

        hidden_states_tt = ttnn.reshape(
            hidden_states,
            (batch_size, 1, seq_len, hidden_size),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pre-conv
        if self.pre_conv_weight is not None:
            x_nlc = ttnn.reshape(
                hidden_states_tt,
                (batch_size, seq_len, hidden_size),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            conv_pre = TTNNConv1d(
                device=self.device,
                in_channels=int(self.pre_conv_weight.shape[1]),
                out_channels=int(self.pre_conv_weight.shape[0]),
                kernel_size=int(self.pre_conv_weight.shape[-1]),
                padding=int(self.pre_conv_weight.shape[-1]) // 2,
                weight=self.pre_conv_weight,
                bias_tensor=self.pre_conv_bias,
            )
            x_nlc, out_len = conv_pre(x_nlc, seq_len)
            hidden_states_tt = ttnn.reshape(
                x_nlc,
                (batch_size, 1, out_len, int(self.pre_conv_weight.shape[0])),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Upsampler (ConvNeXt blocks)
        for i, ratio in enumerate(self.config.upsampling_ratios):
            conv_weight_key = f"upsample.{i}.0.conv.weight"
            conv_bias_key = f"upsample.{i}.0.conv.bias"

            if conv_weight_key in self.upsample_weights:
                conv_weight = self.upsample_weights[conv_weight_key]
                conv_bias = self.upsample_weights.get(conv_bias_key)

                # Upsample with TTNN conv_transpose1d
                up_conv = TTNNConvTranspose1d(
                    device=self.device,
                    in_channels=int(conv_weight.shape[0]),
                    out_channels=int(conv_weight.shape[1]),
                    kernel_size=int(conv_weight.shape[-1]),
                    stride=ratio,
                    padding=0,
                    weight=conv_weight,
                    bias_tensor=conv_bias,
                )
                hidden_states_tt, _ = up_conv(hidden_states_tt, int(hidden_states_tt.shape[2]))

                # ConvNeXt block
                prefix = f"upsample.{i}.1."
                dwconv_weight = self.upsample_weights.get(f"{prefix}dwconv.conv.weight")
                if dwconv_weight is not None:
                    dwconv_bias = self.upsample_weights.get(f"{prefix}dwconv.conv.bias")
                    pwconv1_bias = self.upsample_weights.get(f"{prefix}pwconv1.bias")
                    pwconv2_bias = self.upsample_weights.get(f"{prefix}pwconv2.bias")
                    gamma = self.upsample_weights.get(f"{prefix}gamma")
                    hidden_states_tt = convnext_block(
                        hidden_states_tt,
                        dwconv_weight=dwconv_weight,
                        dwconv_bias=dwconv_bias if dwconv_bias is not None else None,
                        pwconv1_weight=self.upsample_weights.get(f"{prefix}pwconv1.weight"),
                        pwconv1_bias=pwconv1_bias if pwconv1_bias is not None else None,
                        pwconv2_weight=self.upsample_weights.get(f"{prefix}pwconv2.weight"),
                        pwconv2_bias=pwconv2_bias if pwconv2_bias is not None else None,
                        norm_weight=self.upsample_weights.get(f"{prefix}norm.weight"),
                        norm_bias=self.upsample_weights.get(f"{prefix}norm.bias"),
                        device=self.device,
                        gamma=gamma if gamma is not None else None,
                    )

        # Initial conv in decoder
        if "decoder.0.conv.weight" in self.decoder_weights:
            weight = self.decoder_weights["decoder.0.conv.weight"]
            bias = self.decoder_weights.get("decoder.0.conv.bias")
            x_nlc = ttnn.reshape(
                hidden_states_tt,
                (int(hidden_states_tt.shape[0]), int(hidden_states_tt.shape[2]), int(hidden_states_tt.shape[3])),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            conv0 = TTNNConv1d(
                device=self.device,
                in_channels=int(weight.shape[1]),
                out_channels=int(weight.shape[0]),
                kernel_size=int(weight.shape[-1]),
                padding=int(weight.shape[-1]) // 2,
                weight=weight,
                bias_tensor=bias,
            )
            x_nlc, out_len = conv0(x_nlc, int(x_nlc.shape[1]))
            hidden_states_tt = ttnn.reshape(
                x_nlc,
                (int(x_nlc.shape[0]), 1, out_len, int(weight.shape[0])),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Decoder blocks with upsampling
        for i, rate in enumerate(self.config.upsample_rates):
            block_prefix = f"decoder.{i + 1}."
            block_weights = {
                k.replace(block_prefix, ""): v for k, v in self.decoder_weights.items() if k.startswith(block_prefix)
            }
            if block_weights:
                hidden_states_tt = conv_decoder_block(hidden_states_tt, block_weights, rate, self.device)

        # Final activation + conv
        if "decoder.5.alpha" in self.decoder_weights:
            channels = int(hidden_states_tt.shape[-1])
            alpha_tt = ttnn.from_torch(
                self.decoder_weights["decoder.5.alpha"].to(torch.bfloat16).view(1, 1, 1, channels),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            beta_tt = ttnn.from_torch(
                self.decoder_weights["decoder.5.beta"].to(torch.bfloat16).view(1, 1, 1, channels),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            hidden_states_tt = ttnn_snake_activation(hidden_states_tt, alpha_tt, beta_tt)

        if "decoder.6.conv.weight" in self.decoder_weights:
            weight = self.decoder_weights["decoder.6.conv.weight"]
            bias = self.decoder_weights.get("decoder.6.conv.bias")
            x_nlc = ttnn.reshape(
                hidden_states_tt,
                (int(hidden_states_tt.shape[0]), int(hidden_states_tt.shape[2]), int(hidden_states_tt.shape[3])),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            conv6 = TTNNConv1d(
                device=self.device,
                in_channels=int(weight.shape[1]),
                out_channels=int(weight.shape[0]),
                kernel_size=int(weight.shape[-1]),
                padding=int(weight.shape[-1]) // 2,
                weight=weight,
                bias_tensor=bias,
            )
            x_nlc, out_len = conv6(x_nlc, int(x_nlc.shape[1]))
            hidden_states_tt = ttnn.reshape(
                x_nlc,
                (int(x_nlc.shape[0]), 1, out_len, int(weight.shape[0])),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Final tanh for audio range [-1, 1]
        audio_tt = ttnn.tanh(hidden_states_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.permute(audio_tt, (0, 3, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: codec tokens -> audio waveform.

        Args:
            token_ids: Token IDs of shape [batch, num_quantizers, seq_len]

        Returns:
            audio: Audio waveform of shape [batch, 1, num_samples]
        """
        # Use fixed reference implementation for correct audio output
        if self.use_reference:
            ref_config = SpeechTokenizerDecoderConfig()
            with torch.no_grad():
                audio = reference_decoder_forward(token_ids, self._state_dict, ref_config)
            return audio

        # Original TTNN implementation (for comparison/optimization)
        batch_size, num_quantizers, seq_len = token_ids.shape
        device = token_ids.device

        token_ids_ttnn = ttnn.from_torch(
            token_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 1. Codebook lookup (TTNN)
        embeddings_ttnn = self._codebook_lookup(token_ids_ttnn)

        # 2. Pre-transformer (TTNN)
        if self.has_pre_transformer:
            # Compute RoPE frequencies
            cos_ttnn, sin_ttnn = self._compute_rope(seq_len)

            # Forward through pre-transformer
            hidden_states_ttnn = self.pre_transformer(embeddings_ttnn, cos_ttnn, sin_ttnn)
        else:
            hidden_states_ttnn = embeddings_ttnn

        # 3. Conv decoder (TTNN path)
        audio_ttnn = self._conv_decoder_forward(hidden_states_ttnn)
        audio = ttnn.to_torch(audio_ttnn, dtype=torch.float32).squeeze(-1).contiguous()

        return audio


def extract_speech_tokenizer_weights(state_dict: dict) -> dict:
    """
    Extract speech tokenizer decoder weights from state dict.

    The weights are in speech_tokenizer/model.safetensors with prefix "decoder."

    Args:
        state_dict: Speech tokenizer state dict (from safetensors.torch.load_file)

    Returns:
        Dictionary with decoder weights (prefix removed)
    """
    prefix = "decoder."
    decoder_weights = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            # Only remove the first occurrence of prefix
            decoder_weights[k[len(prefix) :]] = v
    return decoder_weights
