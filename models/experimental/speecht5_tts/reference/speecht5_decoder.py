# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Production Op-by-Op PyTorch Reference for SpeechT5 Decoder.

Similar to speecht5_encoder_production.py, this follows a clean pattern
making it easy to translate to TTNN.

Every operation is explicit and maps 1:1 to TTNN operations.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeechT5DecoderConfig:
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 4000
    max_relative_distance: int = 160

    # Speech-specific
    num_mel_bins: int = 80
    reduction_factor: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_prenet_layers: int = 2
    speaker_embedding_dim: int = 512


def create_sinusoidal_positions(max_length: int, hidden_size: int) -> torch.Tensor:
    """Create sinusoidal positional encoding buffer."""
    position = torch.arange(max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))

    pe = torch.zeros(1, max_length, hidden_size)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe


class SpeechT5Decoder(nn.Module):
    """
    Complete SpeechT5 Decoder with Speech Prenet.

    Forward flow:
    1. Speech prenet (mel -> hidden with positional encoding)
    2. Pass through decoder layers (self-attn + cross-attn + FFN)
    3. Output projection
    """

    def __init__(self, config: SpeechT5DecoderConfig, hf_prenet=None):
        super().__init__()

        # Speech decoder prenet - use HF prenet if provided
        if hf_prenet is not None:
            self.prenet = hf_prenet
        else:
            self.prenet = SpeechDecoderPrenet(config)

        # Decoder layers (no relative positional encoding - uses absolute only)
        self.layers = nn.ModuleList([SpeechT5DecoderLayer(config) for _ in range(config.num_layers)])

    def forward(
        self,
        decoder_input_values: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with explicit operations.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins] - mel spectrograms
            encoder_hidden_states: [batch, enc_seq_len, hidden_size] - from encoder
            speaker_embeddings: [batch, speaker_embedding_dim] - optional speaker info
            attention_mask: [batch, seq_len] - optional mask

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Check if using HF prenet (which handles speaker embeddings internally)
        if hasattr(self.prenet, "encode_positions"):  # HF prenet has this attribute
            # HF prenet: handles everything internally
            hidden_states = self.prenet(decoder_input_values, speaker_embeddings=speaker_embeddings)
        else:
            # Our prenet: separate speaker processing
            hidden_states = self.prenet(decoder_input_values, speaker_embeddings=speaker_embeddings)

        batch_size, seq_len, _ = hidden_states.shape

        # Op 2: Create causal attention mask (following HF's _prepare_4d_causal_attention_mask)
        # Lower triangular mask: future positions are masked with -inf
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)

        # Op 3-N: Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=causal_mask,
            )

        return hidden_states


class SpeechDecoderPrenet(nn.Module):
    """Speech decoder prenet: processes mel spectrograms to hidden states."""

    def __init__(self, config: SpeechT5DecoderConfig):
        super().__init__()

        # Prenet layers (mel -> intermediate)
        self.layers = nn.ModuleList()
        input_dim = config.num_mel_bins
        for _ in range(config.speech_decoder_prenet_layers):
            self.layers.append(nn.Linear(input_dim, config.speech_decoder_prenet_units))
            input_dim = config.speech_decoder_prenet_units

        # Final projection to hidden_size
        self.final_layer = nn.Linear(config.speech_decoder_prenet_units, config.hidden_size)

        # Scaled positional encoding
        self.encode_positions_alpha = nn.Parameter(torch.ones(1))
        pe = create_sinusoidal_positions(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("positional_encoding", pe)

        # Speaker embedding projection
        self.speaker_embeds_layer = nn.Linear(config.hidden_size + config.speaker_embedding_dim, config.hidden_size)

        self.dropout_p = config.dropout
        self.prenet_dropout_p = 0.5  # Fixed dropout for prenet layers
        self.positional_dropout_p = config.dropout  # Dropout for positional encoding

    def _consistent_dropout(self, inputs_embeds: torch.Tensor, p: float) -> torch.Tensor:
        """
        Consistent dropout: creates mask from first batch element, applies to all.
        From HuggingFace SpeechT5SpeechDecoderPrenet._consistent_dropout
        """
        # HF implementation: torch.bernoulli(inputs_embeds[0], p=p)
        # This creates a mask by sampling with probability p for each element
        mask = torch.bernoulli(torch.full_like(inputs_embeds[0], p))
        all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
        return torch.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)

    def forward(
        self,
        decoder_input_values: torch.Tensor,
        speaker_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            speaker_embeddings: [batch, speaker_embedding_dim]

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        hidden_states = decoder_input_values

        # Op 1-N: Prenet layers with ReLU and consistent dropout
        # NOTE: Dropout is ALWAYS applied (even in eval mode) for prenet!
        # See https://huggingface.co/papers/1712.05884 §2.2
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            hidden_states = F.relu(hidden_states)
            # Always apply consistent dropout (HF's special dropout pattern)
            hidden_states = self._consistent_dropout(hidden_states, self.prenet_dropout_p)

        # Op N+1: Final projection
        hidden_states = self.final_layer(hidden_states)

        # Op N+2: Add scaled positional encoding (matches HF SpeechT5ScaledPositionalEncoding)
        seq_len = hidden_states.size(1)
        pe_slice = self.positional_encoding[:, :seq_len, :]
        hidden_states = hidden_states + self.encode_positions_alpha * pe_slice

        # Op N+3: Dropout (HF always applies dropout, even in eval mode - see §2.2)
        hidden_states = F.dropout(hidden_states, p=self.positional_dropout_p, training=True)

        # Op N+4: Add speaker embeddings if provided
        if speaker_embeddings is not None:
            # Normalize speaker embeddings (L2 normalization)
            speaker_embeddings = F.normalize(speaker_embeddings, p=2, dim=-1)

            # Expand speaker embeddings to match sequence length
            speaker_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

            # Concatenate and project
            hidden_states = torch.cat([hidden_states, speaker_embeddings], dim=-1)
            hidden_states = self.speaker_embeds_layer(hidden_states)

            # Apply ReLU after speaker projection (HF does this)
            hidden_states = F.relu(hidden_states)

        return hidden_states


class SpeechT5DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention, cross-attention, and FFN.
    Uses POST-NORM architecture like the encoder.
    """

    def __init__(self, config: SpeechT5DecoderConfig):
        super().__init__()

        # Self-attention
        self.self_attn = SpeechT5Attention(config)
        self.dropout_p = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Cross-attention (encoder-decoder)
        self.encoder_attn = SpeechT5Attention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Feed-forward
        self.feed_forward = SpeechT5FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] - causal mask for self-attn
            encoder_attention_mask: [batch, 1, seq_len, enc_seq_len] - mask for cross-attn

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Self-attention sub-layer (POST-NORM, with causal masking)
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_p, training=True)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-attention sub-layer (POST-NORM)
        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states,
            attention_mask=encoder_attention_mask,
            key_value_states=encoder_hidden_states,
        )
        if self.training:
            hidden_states = F.dropout(hidden_states, p=self.dropout_p, training=True)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Feed-forward sub-layer (POST-NORM)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class SpeechT5Attention(nn.Module):
    """
    Multi-head attention (same as encoder, but supports cross-attention).
    """

    def __init__(self, config: SpeechT5DecoderConfig):
        super().__init__()

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with support for both self-attention and cross-attention.
        Note: Decoder does NOT use relative position bias (uses absolute PE only).

        Args:
            hidden_states: [batch, seq_len, hidden_size] - queries
            attention_mask: [batch, 1, seq_len, seq_len or kv_seq_len] - attention mask
            key_value_states: [batch, kv_seq_len, hidden_size] - keys/values (if cross-attn)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Determine if this is cross-attention
        is_cross_attention = key_value_states is not None

        # Op 1-3: Project to Q, K, V
        query_states = self.q_proj(hidden_states) * self.scaling

        if is_cross_attention:
            # Cross-attention: K,V from encoder
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
            kv_seq_len = key_value_states.size(1)
        else:
            # Self-attention: K,V from same input
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            kv_seq_len = seq_len

        # Op 4-6: Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2).contiguous()
        query_states = query_states.view(batch_size * self.num_heads, seq_len, self.head_dim)

        key_states = key_states.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).contiguous()
        key_states = key_states.view(batch_size * self.num_heads, kv_seq_len, self.head_dim)

        value_states = value_states.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).contiguous()
        value_states = value_states.view(batch_size * self.num_heads, kv_seq_len, self.head_dim)

        # Op 7: Compute attention scores
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # Op 8: Apply attention mask if provided (for causal masking)
        if attention_mask is not None:
            # attention_mask is [batch, 1, seq, kv_seq], expand for all heads
            # attn_weights is [batch*heads, seq, kv_seq]
            attention_mask = attention_mask.view(batch_size, 1, seq_len, kv_seq_len)
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, kv_seq_len)
            attention_mask = attention_mask.reshape(batch_size * self.num_heads, seq_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        # Op 9: Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Op 9: Apply attention to values
        attn_output = torch.bmm(attn_weights, value_states)

        # Op 10-11: Reshape back
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Op 12: Output projection
        return self.out_proj(attn_output)


class SpeechT5FeedForward(nn.Module):
    """Feed-forward network (same as encoder)."""

    def __init__(self, config: SpeechT5DecoderConfig):
        super().__init__()

        self.intermediate_dense = nn.Linear(config.hidden_size, config.ffn_dim)
        self.output_dense = nn.Linear(config.ffn_dim, config.hidden_size)
        self.dropout_p = config.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
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


def load_from_huggingface(model_name: str = "microsoft/speecht5_tts") -> SpeechT5Decoder:
    """
    Load SpeechT5 decoder from HuggingFace checkpoint.

    Returns:
        decoder: SpeechT5Decoder with loaded weights
    """
    from transformers import SpeechT5ForTextToSpeech

    # Load HF model
    hf_model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
    hf_decoder = hf_model.speecht5.decoder
    hf_prenet = hf_decoder.prenet  # Get prenet before creating decoder
    hf_config = hf_model.config

    # Create config
    config = SpeechT5DecoderConfig(
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.decoder_layers,
        num_heads=hf_config.decoder_attention_heads,
        ffn_dim=hf_config.decoder_ffn_dim,
        dropout=hf_config.positional_dropout,
        layer_norm_eps=1e-5,
        max_position_embeddings=hf_config.max_speech_positions,
        max_relative_distance=160,  # Decoder doesn't use relative PE
        num_mel_bins=hf_config.num_mel_bins,
        reduction_factor=hf_config.reduction_factor,
        speech_decoder_prenet_units=hf_config.speech_decoder_prenet_units,
        speech_decoder_prenet_layers=hf_config.speech_decoder_prenet_layers,
        speaker_embedding_dim=hf_config.speaker_embedding_dim,
    )

    # Create decoder with HF prenet for exact matching
    decoder = SpeechT5Decoder(config, hf_prenet=hf_prenet)

    # Load layer weights (decoder does NOT have relative position encoding)
    hf_wrapped = hf_decoder.wrapped_decoder
    for i, layer in enumerate(decoder.layers):
        hf_layer = hf_wrapped.layers[i]

        # Self-attention
        layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        layer.self_attn.q_proj.bias.data.copy_(hf_layer.self_attn.q_proj.bias.data)
        layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        layer.self_attn.k_proj.bias.data.copy_(hf_layer.self_attn.k_proj.bias.data)
        layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        layer.self_attn.v_proj.bias.data.copy_(hf_layer.self_attn.v_proj.bias.data)
        layer.self_attn.out_proj.weight.data.copy_(hf_layer.self_attn.out_proj.weight.data)
        layer.self_attn.out_proj.bias.data.copy_(hf_layer.self_attn.out_proj.bias.data)
        layer.self_attn_layer_norm.weight.data.copy_(hf_layer.self_attn_layer_norm.weight.data)
        layer.self_attn_layer_norm.bias.data.copy_(hf_layer.self_attn_layer_norm.bias.data)

        # Cross-attention (encoder-decoder)
        layer.encoder_attn.q_proj.weight.data.copy_(hf_layer.encoder_attn.q_proj.weight.data)
        layer.encoder_attn.q_proj.bias.data.copy_(hf_layer.encoder_attn.q_proj.bias.data)
        layer.encoder_attn.k_proj.weight.data.copy_(hf_layer.encoder_attn.k_proj.weight.data)
        layer.encoder_attn.k_proj.bias.data.copy_(hf_layer.encoder_attn.k_proj.bias.data)
        layer.encoder_attn.v_proj.weight.data.copy_(hf_layer.encoder_attn.v_proj.weight.data)
        layer.encoder_attn.v_proj.bias.data.copy_(hf_layer.encoder_attn.v_proj.bias.data)
        layer.encoder_attn.out_proj.weight.data.copy_(hf_layer.encoder_attn.out_proj.weight.data)
        layer.encoder_attn.out_proj.bias.data.copy_(hf_layer.encoder_attn.out_proj.bias.data)
        layer.encoder_attn_layer_norm.weight.data.copy_(hf_layer.encoder_attn_layer_norm.weight.data)
        layer.encoder_attn_layer_norm.bias.data.copy_(hf_layer.encoder_attn_layer_norm.bias.data)

        # FFN
        layer.feed_forward.intermediate_dense.weight.data.copy_(hf_layer.feed_forward.intermediate_dense.weight.data)
        layer.feed_forward.intermediate_dense.bias.data.copy_(hf_layer.feed_forward.intermediate_dense.bias.data)
        layer.feed_forward.output_dense.weight.data.copy_(hf_layer.feed_forward.output_dense.weight.data)
        layer.feed_forward.output_dense.bias.data.copy_(hf_layer.feed_forward.output_dense.bias.data)
        layer.final_layer_norm.weight.data.copy_(hf_layer.final_layer_norm.weight.data)
        layer.final_layer_norm.bias.data.copy_(hf_layer.final_layer_norm.bias.data)

    decoder.eval()
    return decoder


def test_decoder():
    """Test the production decoder."""
    from transformers import SpeechT5ForTextToSpeech
    import torch

    print("=== Testing Production SpeechT5 Decoder ===\n")

    # Load HF model
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    hf_model.eval()

    # Load our decoder
    our_decoder = load_from_huggingface()
    our_decoder.eval()

    # Create test inputs
    torch.manual_seed(42)
    batch_size = 1
    mel_seq_len = 10
    enc_seq_len = 20

    # Mel spectrograms (decoder input)
    decoder_input_values = torch.randn(batch_size, mel_seq_len, 80)

    # Encoder hidden states (from encoder output)
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, 768)

    # Speaker embeddings (optional)
    speaker_embeddings = torch.randn(batch_size, 512)

    print(f"Decoder input shape: {decoder_input_values.shape}")
    print(f"Encoder hidden shape: {encoder_hidden_states.shape}")
    print(f"Speaker embedding shape: {speaker_embeddings.shape}\n")

    with torch.no_grad():
        # HF decoder forward
        # Note: HF decoder expects specific format, we test our implementation
        our_output = our_decoder(
            decoder_input_values=decoder_input_values,
            encoder_hidden_states=encoder_hidden_states,
            speaker_embeddings=speaker_embeddings,
        )

        print(f"Output shape: {our_output.shape}")
        print(f"Output mean: {our_output.mean():.6f}")
        print(f"Output std: {our_output.std():.6f}")

    print("\n✓ Decoder forward pass successful!")
    print("\nNote: Full PCC validation requires matching HF's exact input format.")
    print("This will be done in the comprehensive validation script.")


if __name__ == "__main__":
    test_decoder()
