# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Production Op-by-Op PyTorch Reference for SpeechT5 Encoder.

Adapted from models/experimental/stable_diffusion_35_large/reference/t5_encoder.py
This follows the same clean pattern as T5, making it easy to translate to TTNN.

Every operation is explicit and maps 1:1 to TTNN operations.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeechT5Config:
    vocab_size: int = 81
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 600
    max_relative_distance: int = 160


def create_sinusoidal_positions(max_length: int, hidden_size: int) -> torch.Tensor:
    """Create sinusoidal positional encoding buffer."""
    position = torch.arange(max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))

    pe = torch.zeros(1, max_length, hidden_size)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe


class SpeechT5Encoder(nn.Module):
    """
    Complete SpeechT5 Encoder following T5's clean pattern.

    Forward flow:
    1. Embedding
    2. Scaled positional encoding
    3. Compute relative position bias (once)
    4. Pass through layers (with position bias)
    5. Final layer norm
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1)

        # Scaled positional encoding
        self.encode_positions_alpha = nn.Parameter(torch.ones(1))
        pe = create_sinusoidal_positions(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("positional_encoding", pe)

        # Relative positional encoding
        num_embeddings = 2 * config.max_relative_distance
        self.relative_pe_k = nn.Embedding(num_embeddings, 64)
        self.max_relative_distance = config.max_relative_distance

        # Pre-encoder layer norm and dropout (from wrapped_encoder)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout_p = config.dropout

        # Encoder blocks
        self.blocks = nn.ModuleList([SpeechT5EncoderBlock(config) for _ in range(config.num_layers)])

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass with explicit operations.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            tuple: (hidden_states,) where hidden_states is [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Op 1: Embedding lookup
        hidden_states = self.embed_tokens(input_ids)

        # Op 2: Add scaled positional encoding
        pe_slice = self.positional_encoding[:, :seq_len, :]
        hidden_states = hidden_states + self.encode_positions_alpha * pe_slice

        # Op 3: Prenet dropout (from encode_positions)
        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_p, training=True)

        # Op 4: Pre-encoder layer norm (from wrapped_encoder)
        hidden_states = self.layer_norm(hidden_states)

        # Op 5: Pre-encoder dropout (from wrapped_encoder)
        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_p, training=True)

        # Op 6: Compute relative position bias (from wrapped_encoder)
        position_bias = self._compute_position_bias(seq_len, hidden_states.device)

        # Ops 6-N: Pass through encoder blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, position_bias=position_bias)

        return (hidden_states,)

    def _compute_position_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias following SpeechT5's approach.

        Returns:
            position_bias: [1, 1, seq_len, seq_len] for broadcasting
        """
        # Create position sequence
        pos_seq = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # Compute relative positions
        relative_positions = pos_seq[:, None] - pos_seq[None, :]

        # Clamp to max distance
        relative_positions = torch.clamp(
            relative_positions, -self.max_relative_distance, self.max_relative_distance - 1
        )

        # Shift to positive indices
        relative_positions = relative_positions + self.max_relative_distance

        # Embedding lookup: [seq_len, seq_len, 64]
        position_embeddings = self.relative_pe_k(relative_positions)

        # For T5-style usage, we need [1, 1, seq_len, seq_len]
        # But SpeechT5 uses a different approach - it projects through Q
        # For now, return as [seq_len, seq_len, 64] and handle in attention
        return position_embeddings


class SpeechT5EncoderBlock(nn.Module):
    """
    Single encoder block following SpeechT5's POST-NORM pattern.
    Different from T5's pre-norm!
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()

        self.attention = SpeechT5Attention(config)
        self.dropout_p = config.dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = SpeechT5FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        """
        POST-NORM architecture (matching HuggingFace SpeechT5).

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_bias: [seq_len, seq_len, 64]

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Attention sub-layer with POST-NORM (HF style)
        residual = hidden_states

        # HF: hidden_states, attn_weights, _ = self.attention(...)
        hidden_states = self.attention(hidden_states, position_bias=position_bias)

        # HF: hidden_states = self.dropout(hidden_states)
        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_p, training=True)

        # HF: hidden_states = residual + hidden_states
        hidden_states = residual + hidden_states

        # HF: hidden_states = self.layer_norm(hidden_states)  ← POST-NORM
        hidden_states = self.layer_norm(hidden_states)

        # FFN sub-layer with POST-NORM
        # HF: hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)

        # HF: hidden_states = self.final_layer_norm(hidden_states)  ← POST-NORM
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class SpeechT5Attention(nn.Module):
    """
    Multi-head self-attention following T5's clean pattern.

    Key difference from T5: SpeechT5 uses position_bias shape [seq, seq, 64]
    and projects it through Q, rather than T5's [1, num_heads, seq, seq] direct add.
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__()

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)  # Key addition!

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        """
        Forward following SpeechT5 HuggingFace pattern.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_bias: [seq_len, seq_len, 64]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Op 1-3: Project to Q, K, V
        query_states = self.q_proj(hidden_states) * self.scaling  # Scale Q (HF line 19)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Op 4-6: Reshape for multi-head attention
        # HF uses: [batch, seq_len, hidden] -> [batch*num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2).contiguous()  # [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size * self.num_heads, seq_len, self.head_dim)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).contiguous()
        key_states = key_states.view(batch_size * self.num_heads, seq_len, self.head_dim)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).contiguous()
        value_states = value_states.view(batch_size * self.num_heads, seq_len, self.head_dim)

        # Op 7: Compute attention scores
        # query_states: [batch*num_heads, seq_len, head_dim]
        # key_states: [batch*num_heads, seq_len, head_dim]
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # attn_weights: [batch*num_heads, seq_len, seq_len]

        # Op 8-11: Add relative position bias (SpeechT5-specific)
        # Following HF lines 66-71 exactly
        if position_bias is not None:
            # Op 8: Transpose query for position bias matmul
            # query_states is [batch*num_heads, seq_len, head_dim]
            # Transpose to [seq_len, batch*num_heads, head_dim]
            reshape_q = query_states.transpose(0, 1)

            # Op 9: position_bias is [seq_len, seq_len, 64]
            # Transpose to [seq_len, 64, seq_len]
            position_bias_t = position_bias.transpose(-2, -1)

            # Op 10: Matmul [seq_len, batch*num_heads, 64] @ [seq_len, 64, seq_len]
            # Result: [seq_len, batch*num_heads, seq_len]
            rel_pos_bias = torch.matmul(reshape_q, position_bias_t)

            # Op 11: Transpose and reshape to [batch*num_heads, seq_len, seq_len]
            rel_pos_bias = rel_pos_bias.transpose(0, 1).view(batch_size * self.num_heads, seq_len, seq_len)

            # Op 12: Add to attention weights
            attn_weights = attn_weights + rel_pos_bias

        # Op 13: Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Op 14: Apply attention to values
        # attn_weights: [batch*num_heads, seq_len, seq_len]
        # value_states: [batch*num_heads, seq_len, head_dim]
        attn_output = torch.bmm(attn_weights, value_states)
        # attn_output: [batch*num_heads, seq_len, head_dim]

        # Op 15-16: Reshape back
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Op 17: Output projection
        return self.out_proj(attn_output)


class SpeechT5FeedForward(nn.Module):
    """Feed-forward network (simpler than T5's gated version)."""

    def __init__(self, config: SpeechT5Config):
        super().__init__()

        self.intermediate_dense = nn.Linear(config.hidden_size, config.ffn_dim)
        self.output_dense = nn.Linear(config.ffn_dim, config.hidden_size)
        self.dropout_p = config.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, ffn_dim]
        """
        # Intermediate
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)

        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_p, training=True)

        # Output
        hidden_states = self.output_dense(hidden_states)

        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_p, training=True)

        return hidden_states


def load_from_huggingface(model_name: str = "microsoft/speecht5_tts") -> SpeechT5Encoder:
    """
    Load SpeechT5 encoder from HuggingFace checkpoint.

    Returns:
        encoder: SpeechT5Encoder with loaded weights
    """
    from transformers import SpeechT5ForTextToSpeech

    # Load HF model
    hf_model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
    hf_encoder = hf_model.speecht5.encoder
    hf_config = hf_model.config

    # Create config
    config = SpeechT5Config(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.encoder_layers,
        num_heads=hf_config.encoder_attention_heads,
        ffn_dim=hf_config.encoder_ffn_dim,
        dropout=hf_config.positional_dropout,
        layer_norm_eps=1e-5,
        max_position_embeddings=600,
        max_relative_distance=hf_config.encoder_max_relative_position,
    )

    # Create encoder
    encoder = SpeechT5Encoder(config)

    # Load weights
    encoder.embed_tokens.weight.data.copy_(hf_encoder.prenet.embed_tokens.weight.data)
    encoder.encode_positions_alpha.data.copy_(hf_encoder.prenet.encode_positions.alpha.data.unsqueeze(0))
    encoder.relative_pe_k.weight.data.copy_(hf_encoder.wrapped_encoder.embed_positions.pe_k.weight.data)

    # Pre-encoder layer norm
    encoder.layer_norm.weight.data.copy_(hf_encoder.wrapped_encoder.layer_norm.weight.data)
    encoder.layer_norm.bias.data.copy_(hf_encoder.wrapped_encoder.layer_norm.bias.data)

    # Load layer weights
    for i, block in enumerate(encoder.blocks):
        hf_layer = hf_encoder.wrapped_encoder.layers[i]

        # Attention
        block.attention.q_proj.weight.data.copy_(hf_layer.attention.q_proj.weight.data)
        block.attention.q_proj.bias.data.copy_(hf_layer.attention.q_proj.bias.data)
        block.attention.k_proj.weight.data.copy_(hf_layer.attention.k_proj.weight.data)
        block.attention.k_proj.bias.data.copy_(hf_layer.attention.k_proj.bias.data)
        block.attention.v_proj.weight.data.copy_(hf_layer.attention.v_proj.weight.data)
        block.attention.v_proj.bias.data.copy_(hf_layer.attention.v_proj.bias.data)
        block.attention.out_proj.weight.data.copy_(hf_layer.attention.out_proj.weight.data)
        block.attention.out_proj.bias.data.copy_(hf_layer.attention.out_proj.bias.data)

        # Layer norms
        block.layer_norm.weight.data.copy_(hf_layer.layer_norm.weight.data)
        block.layer_norm.bias.data.copy_(hf_layer.layer_norm.bias.data)
        block.final_layer_norm.weight.data.copy_(hf_layer.final_layer_norm.weight.data)
        block.final_layer_norm.bias.data.copy_(hf_layer.final_layer_norm.bias.data)

        # FFN
        block.feed_forward.intermediate_dense.weight.data.copy_(hf_layer.feed_forward.intermediate_dense.weight.data)
        block.feed_forward.intermediate_dense.bias.data.copy_(hf_layer.feed_forward.intermediate_dense.bias.data)
        block.feed_forward.output_dense.weight.data.copy_(hf_layer.feed_forward.output_dense.weight.data)
        block.feed_forward.output_dense.bias.data.copy_(hf_layer.feed_forward.output_dense.bias.data)

    encoder.eval()
    return encoder


def test_encoder():
    """Test the production encoder."""
    from transformers import SpeechT5ForTextToSpeech
    import torch

    print("=== Testing Production SpeechT5 Encoder ===\n")

    # Load HF model
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    hf_model.eval()
    hf_encoder = hf_model.speecht5.encoder

    # Load our encoder
    our_encoder = load_from_huggingface()
    our_encoder.eval()

    # Test input
    torch.manual_seed(42)
    input_ids = torch.randint(0, 81, (1, 20))

    print(f"Input shape: {input_ids.shape}\n")

    with torch.no_grad():
        # HF output
        hf_output = hf_encoder(input_ids)
        hf_hidden = hf_output.last_hidden_state

        # Our output
        our_output = our_encoder(input_ids)
        our_hidden = our_output[0]

    # Compute metrics
    max_diff = torch.max(torch.abs(hf_hidden - our_hidden)).item()
    mean_diff = torch.mean(torch.abs(hf_hidden - our_hidden)).item()

    hf_flat = hf_hidden.flatten()
    our_flat = our_hidden.flatten()
    pcc = torch.corrcoef(torch.stack([hf_flat, our_flat]))[0, 1].item()

    print("=== Results ===\n")
    print(f"Max absolute difference: {max_diff:.10f}")
    print(f"Mean absolute difference: {mean_diff:.10f}")
    print(f"PCC: {pcc:.10f}")

    if pcc > 0.9999:
        print(f"\n✓✓✓ PERFECT! PCC ≈ 1.0")
    elif pcc > 0.99:
        print(f"\n✓✓ EXCELLENT! PCC > 0.99")
    elif pcc > 0.94:
        print(f"\n✓ GOOD! PCC > 0.94")
    else:
        print(f"\n⏳ PCC = {pcc:.6f} - May need refinement in relative PE handling")

    print("\nThis encoder follows T5's clean pattern and is ready for TTNN translation!")
    return pcc


if __name__ == "__main__":
    test_encoder()
