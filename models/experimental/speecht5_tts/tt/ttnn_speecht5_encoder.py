# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of SpeechT5 Encoder.
Adapted from models/experimental/stable_diffusion_35_large/tt/t5_encoder.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import ttnn


@dataclass
class TtSpeechT5Config:
    """Configuration for SpeechT5 encoder"""

    vocab_size: int = 81
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    layer_norm_epsilon: float = 1e-5

    @classmethod
    def from_hf_config(cls, hf_config):
        """Create from HuggingFace config"""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.encoder_layers,
            num_heads=hf_config.encoder_attention_heads,
            ffn_dim=hf_config.encoder_ffn_dim,
            layer_norm_epsilon=getattr(hf_config, "layer_norm_epsilon", 1e-5),
        )


def from_torch_tensor(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    device: ttnn.Device,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Convert PyTorch tensor to TTNN tensor"""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


@dataclass
class TtLinearParameters:
    """Parameters for a linear layer"""

    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor] = None


class TtLinear:
    """TTNN Linear layer wrapper"""

    def __init__(self, parameters: TtLinearParameters):
        self.weight = parameters.weight
        self.bias = parameters.bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        output = ttnn.linear(x, self.weight)
        if self.bias is not None:
            output = ttnn.add(output, self.bias)
        return output


@dataclass
class TtLayerNormParameters:
    """Parameters for layer normalization"""

    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor] = None


class TtLayerNorm:
    """TTNN Layer Normalization"""

    def __init__(self, parameters: TtLayerNormParameters, eps: float = 1e-5):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.eps = eps

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Layer norm: (x - mean) / sqrt(var + eps) * weight + bias
        mean = ttnn.mean(x, dim=-1, keepdim=True)
        variance = ttnn.var(x, dim=-1, keepdim=True)
        x_normalized = (x - mean) / ttnn.sqrt(variance + self.eps)
        output = x_normalized * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


@dataclass
class TtSpeechT5AttentionParameters:
    """Parameters for SpeechT5 attention"""

    q_proj: TtLinearParameters
    k_proj: TtLinearParameters
    v_proj: TtLinearParameters
    out_proj: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state_dict: dict,
        prefix: str,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
        device: ttnn.Device,
    ) -> TtSpeechT5AttentionParameters:
        """Load attention parameters from PyTorch state dict"""

        def load_linear(name: str) -> TtLinearParameters:
            weight = state_dict[f"{prefix}.{name}.weight"].T  # Transpose for ttnn
            bias = state_dict.get(f"{prefix}.{name}.bias", None)

            weight_tt = from_torch_tensor(weight, dtype=dtype, device=device)
            bias_tt = from_torch_tensor(bias, dtype=dtype, device=device) if bias is not None else None

            return TtLinearParameters(weight=weight_tt, bias=bias_tt)

        return cls(
            q_proj=load_linear("q_proj"),
            k_proj=load_linear("k_proj"),
            v_proj=load_linear("v_proj"),
            out_proj=load_linear("out_proj"),
        )


class TtSpeechT5Attention:
    """TTNN implementation of SpeechT5 Attention"""

    def __init__(
        self,
        parameters: TtSpeechT5AttentionParameters,
        num_heads: int,
        hidden_size: int,
    ):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = TtLinear(parameters.q_proj)
        self.k_proj = TtLinear(parameters.k_proj)
        self.v_proj = TtLinear(parameters.v_proj)
        self.out_proj = TtLinear(parameters.out_proj)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        q = ttnn.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        # scores = Q @ K^T / sqrt(head_dim)
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        scores = ttnn.matmul(q, k_t)
        scores = scores / (self.head_dim**0.5)

        # Softmax
        attn_weights = ttnn.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = ttnn.matmul(attn_weights, v)

        # Transpose back and reshape
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))

        # Output projection
        output = self.out_proj(attn_output)

        return output


@dataclass
class TtSpeechT5FeedForwardParameters:
    """Parameters for feed-forward network"""

    intermediate_dense: TtLinearParameters
    output_dense: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state_dict: dict,
        prefix: str,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
        device: ttnn.Device,
    ) -> TtSpeechT5FeedForwardParameters:
        """Load FFN parameters from PyTorch state dict"""

        def load_linear(name: str) -> TtLinearParameters:
            weight = state_dict[f"{prefix}.{name}.weight"].T
            bias = state_dict.get(f"{prefix}.{name}.bias", None)

            weight_tt = from_torch_tensor(weight, dtype=dtype, device=device)
            bias_tt = from_torch_tensor(bias, dtype=dtype, device=device) if bias is not None else None

            return TtLinearParameters(weight=weight_tt, bias=bias_tt)

        return cls(
            intermediate_dense=load_linear("intermediate_dense"),
            output_dense=load_linear("output_dense"),
        )


class TtSpeechT5FeedForward:
    """TTNN implementation of SpeechT5 Feed-Forward Network"""

    def __init__(self, parameters: TtSpeechT5FeedForwardParameters):
        self.intermediate_dense = TtLinear(parameters.intermediate_dense)
        self.output_dense = TtLinear(parameters.output_dense)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Intermediate projection + GELU activation
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = ttnn.gelu(hidden_states)

        # Output projection
        hidden_states = self.output_dense(hidden_states)

        return hidden_states


@dataclass
class TtSpeechT5EncoderLayerParameters:
    """Parameters for a single encoder layer"""

    attention: TtSpeechT5AttentionParameters
    attention_norm: TtLayerNormParameters
    feed_forward: TtSpeechT5FeedForwardParameters
    ffn_norm: TtLayerNormParameters

    @classmethod
    def from_torch(
        cls,
        state_dict: dict,
        layer_idx: int,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
        device: ttnn.Device,
    ) -> TtSpeechT5EncoderLayerParameters:
        """Load encoder layer parameters from PyTorch state dict"""
        prefix = f"speecht5.encoder.wrapped_encoder.layers.{layer_idx}"

        # Load layer norm parameters
        def load_norm(name: str) -> TtLayerNormParameters:
            weight = state_dict[f"{prefix}.{name}.weight"]
            bias = state_dict.get(f"{prefix}.{name}.bias", None)

            weight_tt = from_torch_tensor(weight, dtype=dtype, device=device)
            bias_tt = from_torch_tensor(bias, dtype=dtype, device=device) if bias is not None else None

            return TtLayerNormParameters(weight=weight_tt, bias=bias_tt)

        return cls(
            attention=TtSpeechT5AttentionParameters.from_torch(
                state_dict, f"{prefix}.attention", dtype=dtype, device=device
            ),
            attention_norm=load_norm("layer_norm"),
            feed_forward=TtSpeechT5FeedForwardParameters.from_torch(
                state_dict, f"{prefix}.feed_forward", dtype=dtype, device=device
            ),
            ffn_norm=load_norm("final_layer_norm"),
        )


class TtSpeechT5EncoderLayer:
    """TTNN implementation of SpeechT5 Encoder Layer"""

    def __init__(
        self,
        parameters: TtSpeechT5EncoderLayerParameters,
        num_heads: int,
        hidden_size: int,
        layer_norm_epsilon: float,
    ):
        self.attention = TtSpeechT5Attention(parameters.attention, num_heads, hidden_size)
        self.attention_norm = TtLayerNorm(parameters.attention_norm, eps=layer_norm_epsilon)
        self.feed_forward = TtSpeechT5FeedForward(parameters.feed_forward)
        self.ffn_norm = TtLayerNorm(parameters.ffn_norm, eps=layer_norm_epsilon)

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward with pre-norm and residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class TtSpeechT5EncoderParameters:
    """Parameters for complete SpeechT5 encoder"""

    embed_tokens: ttnn.Tensor
    layers: list[TtSpeechT5EncoderLayerParameters]
    layer_norm: TtLayerNormParameters

    @classmethod
    def from_torch(
        cls,
        state_dict: dict,
        config: TtSpeechT5Config,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
        device: ttnn.Device,
    ) -> TtSpeechT5EncoderParameters:
        """Load encoder parameters from PyTorch state dict"""
        # Load embedding
        embed_tokens = state_dict["speecht5.encoder.prenet.embed_tokens.weight"]
        embed_tokens_tt = from_torch_tensor(embed_tokens, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        # Load encoder layers
        layers = [
            TtSpeechT5EncoderLayerParameters.from_torch(state_dict, i, dtype=dtype, device=device)
            for i in range(config.num_layers)
        ]

        # Load final layer norm
        layer_norm_weight = state_dict["speecht5.encoder.wrapped_encoder.layer_norm.weight"]
        layer_norm_bias = state_dict.get("speecht5.encoder.wrapped_encoder.layer_norm.bias", None)

        layer_norm_weight_tt = from_torch_tensor(layer_norm_weight, dtype=dtype, device=device)
        layer_norm_bias_tt = (
            from_torch_tensor(layer_norm_bias, dtype=dtype, device=device) if layer_norm_bias is not None else None
        )

        layer_norm = TtLayerNormParameters(weight=layer_norm_weight_tt, bias=layer_norm_bias_tt)

        return cls(
            embed_tokens=embed_tokens_tt,
            layers=layers,
            layer_norm=layer_norm,
        )


class TtSpeechT5Encoder:
    """TTNN implementation of SpeechT5 Encoder"""

    def __init__(
        self,
        parameters: TtSpeechT5EncoderParameters,
        config: TtSpeechT5Config,
    ):
        self.embed_tokens = parameters.embed_tokens
        self.layers = [
            TtSpeechT5EncoderLayer(
                param,
                num_heads=config.num_heads,
                hidden_size=config.hidden_size,
                layer_norm_epsilon=config.layer_norm_epsilon,
            )
            for param in parameters.layers
        ]
        self.layer_norm = TtLayerNorm(parameters.layer_norm, eps=config.layer_norm_epsilon)
        self.config = config

    def __call__(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through the encoder.

        Args:
            input_ids: [batch, seq_len] - text token IDs

        Returns:
            hidden_states: [batch, seq_len, hidden_size] - encoded representations
        """
        # Embedding lookup
        # Convert to row major for embedding, then back to tile
        input_ids = ttnn.to_layout(input_ids, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.embedding(input_ids, self.embed_tokens)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        # Pass through encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
        device: ttnn.Device,
    ) -> TtSpeechT5Encoder:
        """
        Load pre-trained SpeechT5 encoder from HuggingFace.

        Args:
            model_name: HuggingFace model name (e.g., "microsoft/speecht5_tts")
            dtype: TTNN data type
            device: TTNN device

        Returns:
            encoder: TtSpeechT5Encoder instance with loaded weights
        """
        from transformers import SpeechT5ForTextToSpeech

        # Load HuggingFace model
        hf_model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        state_dict = hf_model.state_dict()

        # Create config
        config = TtSpeechT5Config.from_hf_config(hf_model.config)

        # Load parameters
        parameters = TtSpeechT5EncoderParameters.from_torch(state_dict, config, dtype=dtype, device=device)

        # Create encoder
        return cls(parameters, config)
