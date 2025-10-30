# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Encoder for SpeechT5 TTS model.
Adapted from transformers.models.speecht5.modeling_speecht5
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .speecht5_attention import SpeechT5Attention
from .speecht5_feedforward import SpeechT5FeedForward
from .speecht5_config import SpeechT5Config


class SpeechT5ScaledPositionalEncoding(nn.Module):
    """
    Scaled sinusoidal positional encoding with learnable scaling parameter.
    """

    def __init__(self, dropout: float, max_length: int = 5000, hidden_size: int = 768):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Create fixed sinusoidal positional encoding
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))

        pe = torch.zeros(1, max_length, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            encoded: [batch, seq_len, hidden_size]
        """
        seq_len = hidden_states.size(1)
        hidden_states = hidden_states + self.alpha * self.pe[:, :seq_len, :]
        return self.dropout(hidden_states)


class SpeechT5RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for self-attention.
    Similar to T5's relative position bias but with different embedding structure.
    """

    def __init__(self, num_heads: int, max_distance: int = 160):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.head_dim = 64  # Fixed head dimension for position encoding

        # Embedding for relative positions
        # HuggingFace uses 2*max_distance buckets (not +1)
        num_embeddings = 2 * max_distance
        self.pe_k = nn.Embedding(num_embeddings, self.head_dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate relative position bias.

        Args:
            seq_len: sequence length
            device: torch device

        Returns:
            position_bias: [1, num_heads, seq_len, seq_len]
        """
        # Compute relative positions: position[i, j] = j - i
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions[None, :] - positions[:, None]  # [seq_len, seq_len]

        # Clip to max_distance (but use max_distance - 1 for indexing)
        relative_positions = torch.clamp(relative_positions, -self.max_distance + 1, self.max_distance - 1)

        # Shift to positive indices (0 to 2*max_distance - 1)
        relative_position_bucket = relative_positions + self.max_distance - 1

        # Get embeddings: [seq_len, seq_len, head_dim]
        position_embeddings = self.pe_k(relative_position_bucket)

        # For now, return zeros as position bias
        # Full implementation would compute attention with these embeddings
        # This is a simplified version - full version requires integration into attention
        position_bias = torch.zeros(1, self.num_heads, seq_len, seq_len, device=device)

        return position_bias


class SpeechT5TextEncoderPrenet(nn.Module):
    """
    Pre-net for text encoder: embedding + positional encoding.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1)
        self.encode_positions = SpeechT5ScaledPositionalEncoding(
            dropout=config.positional_dropout,
            max_length=config.max_text_positions,
            hidden_size=config.hidden_size,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.encode_positions(inputs_embeds)
        return hidden_states


class SpeechT5EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()

        self.attention = SpeechT5Attention(
            hidden_size=config.hidden_size,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.feed_forward = SpeechT5FeedForward(
            hidden_size=config.hidden_size,
            ffn_dim=config.encoder_ffn_dim,
            dropout=config.activation_dropout,
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len]
            position_bias: [1, num_heads, seq_len, seq_len]
            output_attentions: whether to return attention weights

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            attn_weights: optional attention weights
        """
        # Self-attention block with pre-layer norm
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward block with pre-layer norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class SpeechT5Encoder(nn.Module):
    """
    SpeechT5 Encoder: stack of encoder layers.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([SpeechT5EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        self.embed_positions = SpeechT5RelativePositionalEncoding(
            num_heads=config.encoder_attention_heads,
            max_distance=config.encoder_max_relative_position,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] - output from prenet
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding
            output_attentions: whether to return attention weights
            output_hidden_states: whether to return all hidden states

        Returns:
            last_hidden_state: [batch, seq_len, hidden_size]
            all_hidden_states: tuple of hidden states from each layer (if requested)
            all_attentions: tuple of attention weights from each layer (if requested)
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Convert attention mask to bias
        if attention_mask is not None:
            # attention_mask: [batch, seq_len]
            # Expand to [batch, 1, 1, seq_len] for broadcasting
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Get position bias for relative positional encoding
        seq_len = hidden_states.size(1)
        position_bias = self.embed_positions(seq_len, hidden_states.device)

        # Process through encoder layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class SpeechT5EncoderWithTextPrenet(nn.Module):
    """
    Complete encoder with text pre-net.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()
        self.prenet = SpeechT5TextEncoderPrenet(config)
        self.wrapped_encoder = SpeechT5Encoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            input_ids: [batch, seq_len] - text token IDs
            attention_mask: [batch, seq_len] - attention mask
            output_attentions: whether to return attention weights
            output_hidden_states: whether to return all hidden states

        Returns:
            encoder_outputs: tuple of (last_hidden_state, all_hidden_states, all_attentions)
        """
        hidden_states = self.prenet(input_ids)
        return self.wrapped_encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
