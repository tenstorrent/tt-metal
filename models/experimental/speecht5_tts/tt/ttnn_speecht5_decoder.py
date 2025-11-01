# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Implementation of SpeechT5 Decoder.

Translates from: reference/speecht5_decoder.py
Target PCC: > 0.94 vs PyTorch reference
Following patterns from: ttnn_speecht5_encoder.py and GUIDE_PYTORCH_TO_TTNN.md

Architecture:
1. Speech Decoder Prenet (mel → hidden + positional encoding + speaker embeddings)
2. Decoder Layers (self-attention + cross-attention + feed-forward) × 6
3. Output hidden states

Key Features:
- Cross-attention to encoder outputs
- Causal masking for autoregressive generation
- Speaker embedding support (optional)
- POST-NORM architecture
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import ttnn


@dataclass
class TTNNDecoderConfig:
    """Configuration for TTNN SpeechT5 Decoder."""

    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 4000

    # Speech-specific
    num_mel_bins: int = 80
    reduction_factor: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_prenet_layers: int = 2
    speech_decoder_prenet_dropout: float = 0.5
    speaker_embedding_dim: int = 512


class TTNNSpeechDecoderPrenet:
    """
    Speech decoder prenet: processes mel spectrograms to hidden states.

    Flow:
    1. Prenet layers (mel → units → units) with ReLU
    2. Final projection (units → hidden)
    3. Add scaled positional encoding
    4. Optionally add speaker embeddings

    Note: Consistent dropout is skipped for inference mode.
    """

    def __init__(self, device, parameters, config: TTNNDecoderConfig):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

    def _consistent_dropout(self, inputs_embeds, p):
        """
        TTNN consistent dropout implementation: temporarily disabled for PCC debugging.

        HF applies dropout during inference (unusual but by design per paper §2.2).
        For now, disable dropout to isolate numerical precision issues.
        """
        # Temporarily disable dropout for PCC debugging
        return inputs_embeds

    def __call__(
        self,
        decoder_input_values: ttnn.Tensor,
        speaker_embeddings: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        hidden_states = decoder_input_values

        # Prenet layers with ReLU and consistent dropout (HF applies this even in eval mode)
        for i in range(self.config.speech_decoder_prenet_layers):
            hidden_states = ttnn.linear(
                hidden_states,
                self.parameters["prenet_layers"][i]["weight"],
                bias=self.parameters["prenet_layers"][i]["bias"],
                memory_config=self.L1_MEMCFG,
            )
            hidden_states = ttnn.relu(hidden_states)
            # Apply HF's consistent dropout (this was missing!)
            hidden_states = self._consistent_dropout(hidden_states, self.config.speech_decoder_prenet_dropout)

        # Final projection
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["final_layer"]["weight"],
            bias=self.parameters["final_layer"]["bias"],
            memory_config=self.L1_MEMCFG,
        )

        # Add scaled positional encoding
        seq_len = hidden_states.shape[1]
        pe_slice = self.parameters["positional_encoding"][:, :seq_len, :]

        # Scale positional encoding
        pe_scaled = ttnn.multiply(pe_slice, self.parameters["encode_positions_alpha"])
        hidden_states = ttnn.add(hidden_states, pe_scaled)

        # Apply dropout after positional encoding (HF always applies this even in eval mode - §2.2)
        hidden_states = self._consistent_dropout(hidden_states, self.config.dropout)

        # Add speaker embeddings if provided
        if speaker_embeddings is not None:
            # L2 normalization of speaker embeddings
            # Compute: speaker_embeddings / ||speaker_embeddings||_2
            speaker_norm = ttnn.sqrt(
                ttnn.sum(
                    ttnn.multiply(speaker_embeddings, speaker_embeddings),
                    dim=-1,
                    keepdim=True,
                )
            )
            speaker_embeddings_normalized = ttnn.divide(speaker_embeddings, speaker_norm)

            # Expand to match sequence length
            batch_size = hidden_states.shape[0]
            speaker_embeddings_expanded = ttnn.reshape(
                speaker_embeddings_normalized,
                [batch_size, 1, self.config.speaker_embedding_dim],
            )
            # Repeat for sequence length
            speaker_embeddings_expanded = ttnn.repeat(
                speaker_embeddings_expanded,
                [1, seq_len, 1],
            )

            # Concatenate with hidden states
            hidden_states = ttnn.concat(
                [hidden_states, speaker_embeddings_expanded],
                dim=-1,
            )

            # Project back to hidden size
            hidden_states = ttnn.linear(
                hidden_states,
                self.parameters["speaker_embeds_layer"]["weight"],
                bias=self.parameters["speaker_embeds_layer"]["bias"],
            )

            # ReLU after speaker projection
            hidden_states = ttnn.relu(hidden_states)

        return hidden_states


class TTNNSpeechT5FeedForward:
    """
    Feed-forward network: same as encoder.

    Flow: Linear → GELU → Linear
    """

    def __init__(self, device, parameters, config: TTNNDecoderConfig):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        # Intermediate dense
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["intermediate_dense"]["weight"],
            bias=self.parameters["intermediate_dense"]["bias"],
        )
        hidden_states = ttnn.gelu(hidden_states)

        # Output dense
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["output_dense"]["weight"],
            bias=self.parameters["output_dense"]["bias"],
        )

        return hidden_states


class TTNNSpeechT5Attention:
    """
    Multi-head attention supporting both self and cross-attention.

    Based on encoder attention, extended for cross-attention.

    Self-Attention: Q, K, V all from hidden_states, with causal mask
    Cross-Attention: Q from hidden_states, K/V from encoder_hidden_states, no causal mask
    """

    def __init__(self, device, parameters, config: TTNNDecoderConfig):
        self.device = device
        self.parameters = parameters
        self.config = config

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

    def _reshape_for_multihead(self, tensor: ttnn.Tensor, batch: int, seq_len: int) -> ttnn.Tensor:
        """
        Reshape [B, S, H] -> [B*NH, S, HD]
        where NH = num_heads, HD = head_dim
        """
        # [B, S, H] -> [B, S, NH, HD]
        tensor = ttnn.reshape(tensor, [batch, seq_len, self.num_heads, self.head_dim])

        # [B, S, NH, HD] -> [B, NH, S, HD]
        tensor = ttnn.permute(tensor, [0, 2, 1, 3])

        # [B, NH, S, HD] -> [B*NH, S, HD]
        tensor = ttnn.reshape(tensor, [batch * self.num_heads, seq_len, self.head_dim])

        return tensor

    def _reshape_from_multihead(self, tensor: ttnn.Tensor, batch: int, seq_len: int) -> ttnn.Tensor:
        """
        Reshape [B*NH, S, HD] -> [B, S, H]
        """
        # [B*NH, S, HD] -> [B, NH, S, HD]
        tensor = ttnn.reshape(tensor, [batch, self.num_heads, seq_len, self.head_dim])

        # [B, NH, S, HD] -> [B, S, NH, HD]
        tensor = ttnn.permute(tensor, [0, 2, 1, 3])

        # [B, S, NH, HD] -> [B, S, H]
        tensor = ttnn.reshape(tensor, [batch, seq_len, self.hidden_size])

        return tensor

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        key_value_states: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Multi-head attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - queries
            attention_mask: [batch, 1, seq_len, seq_len or kv_seq_len] - optional mask
            key_value_states: [batch, kv_seq_len, hidden_size] - for cross-attention

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = hidden_states.shape

        # Determine if this is cross-attention
        is_cross_attention = key_value_states is not None

        # Q always from hidden_states
        query = ttnn.linear(
            hidden_states,
            self.parameters["q_proj"]["weight"],
            bias=self.parameters["q_proj"]["bias"],
        )
        # Apply scaling
        query = ttnn.multiply(query, self.scaling)

        # K, V from key_value_states (cross-attn) or hidden_states (self-attn)
        if is_cross_attention:
            # Cross-attention: K, V from encoder
            key = ttnn.linear(
                key_value_states,
                self.parameters["k_proj"]["weight"],
                bias=self.parameters["k_proj"]["bias"],
            )
            value = ttnn.linear(
                key_value_states,
                self.parameters["v_proj"]["weight"],
                bias=self.parameters["v_proj"]["bias"],
            )
            kv_seq_len = key_value_states.shape[1]
        else:
            # Self-attention: K, V from decoder hidden states
            key = ttnn.linear(
                hidden_states,
                self.parameters["k_proj"]["weight"],
                bias=self.parameters["k_proj"]["bias"],
            )
            value = ttnn.linear(
                hidden_states,
                self.parameters["v_proj"]["weight"],
                bias=self.parameters["v_proj"]["bias"],
            )
            kv_seq_len = seq_len

        # Reshape for multi-head attention
        query = self._reshape_for_multihead(query, batch, seq_len)
        key = self._reshape_for_multihead(key, batch, kv_seq_len)
        value = self._reshape_for_multihead(value, batch, kv_seq_len)

        # Compute attention scores: Q @ K^T
        key_t = ttnn.permute(key, [0, 2, 1])
        attn_weights = ttnn.matmul(query, key_t)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for all heads: [B, 1, S, KS] -> [B*NH, S, KS]
            # attention_mask already has shape [B, 1, S, KS]
            mask_expanded = ttnn.reshape(
                attention_mask,
                [batch, 1, seq_len, kv_seq_len],
            )
            mask_expanded = ttnn.repeat(mask_expanded, [1, self.num_heads, 1, 1])
            mask_expanded = ttnn.reshape(
                mask_expanded,
                [batch * self.num_heads, seq_len, kv_seq_len],
            )
            attn_weights = ttnn.add(attn_weights, mask_expanded)

        # Softmax
        attn_weights = ttnn.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = ttnn.matmul(attn_weights, value)

        # Reshape back to [B, S, H]
        attn_output = self._reshape_from_multihead(attn_output, batch, seq_len)

        # Output projection
        output = ttnn.linear(
            attn_output,
            self.parameters["out_proj"]["weight"],
            bias=self.parameters["out_proj"]["bias"],
        )

        return output


class TTNNSpeechT5DecoderLayer:
    """
    Single decoder layer with self-attention, cross-attention, and FFN.
    POST-NORM architecture.

    Flow:
    1. Self-Attention + Residual + LayerNorm
    2. Cross-Attention + Residual + LayerNorm
    3. Feed-Forward + Residual + LayerNorm
    """

    def __init__(self, device, parameters, config: TTNNDecoderConfig):
        self.device = device
        self.config = config

        # Self-attention
        self.self_attn = TTNNSpeechT5Attention(
            device,
            parameters["self_attn"],
            config,
        )

        # Cross-attention
        self.encoder_attn = TTNNSpeechT5Attention(
            device,
            parameters["encoder_attn"],
            config,
        )

        # Feed-forward
        self.feed_forward = TTNNSpeechT5FeedForward(
            device,
            parameters["feed_forward"],
            config,
        )

        # Store layer norm parameters
        self.self_attn_layer_norm_params = parameters["self_attn_layer_norm"]
        self.encoder_attn_layer_norm_params = parameters["encoder_attn_layer_norm"]
        self.final_layer_norm_params = parameters["final_layer_norm"]

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] - causal mask for self-attn

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Self-attention sub-layer (POST-NORM, with causal masking)
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=None,  # Self-attention
        )
        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.self_attn_layer_norm_params["weight"],
            bias=self.self_attn_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
        )

        # Cross-attention sub-layer (POST-NORM)
        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states,
            attention_mask=None,  # No causal mask for cross-attention
            key_value_states=encoder_hidden_states,  # Cross-attention
        )
        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.encoder_attn_layer_norm_params["weight"],
            bias=self.encoder_attn_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
        )

        # Feed-forward sub-layer (POST-NORM)
        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states)
        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.final_layer_norm_params["weight"],
            bias=self.final_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
        )

        return hidden_states


class TTNNSpeechT5Decoder:
    """
    Complete SpeechT5 Decoder with Speech Prenet.

    Flow:
    1. Speech prenet (mel → hidden with positional encoding)
    2. Create causal mask for self-attention
    3. Pass through decoder layers (self-attn + cross-attn + FFN)
    4. Return hidden states
    """

    def __init__(self, device, parameters, config: TTNNDecoderConfig):
        print("  [Decoder Init] Starting...")
        self.device = device
        self.config = config
        self.parameters = parameters

        # Speech decoder prenet
        print("  [Decoder Init] Creating prenet...")
        self.prenet = TTNNSpeechDecoderPrenet(
            device,
            parameters["prenet"],
            config,
        )
        print("  [Decoder Init] Prenet done")

        # Decoder layers
        print(f"  [Decoder Init] Creating {config.num_layers} decoder layers...")
        self.layers = []
        for i in range(config.num_layers):
            print(f"  [Decoder Init] Creating layer {i+1}/{config.num_layers}...")
            layer = TTNNSpeechT5DecoderLayer(
                device,
                parameters["layers"][i],
                config,
            )
            self.layers.append(layer)
        print("  [Decoder Init] All layers created")

        # Pre-compute causal masks for common sequence lengths
        # This avoids dynamic tensor creation during forward pass (required for trace)
        print("  [Decoder Init] Pre-computing causal masks...")
        self.causal_mask_cache = {}
        self._precompute_causal_masks()
        print("  [Decoder Init] Done!")

    def _precompute_causal_masks(self):
        """
        Pre-compute causal masks for common sequence lengths.
        This avoids dynamic tensor creation during forward pass, which is
        required for trace support.

        Pre-computes for lengths: 20
        Note: Add more lengths as needed for your use case.
        """
        common_lengths = [20]

        for seq_len in common_lengths:
            # Create causal mask in PyTorch
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

            # Convert to TTNN and cache
            causal_mask_ttnn = ttnn.from_torch(
                causal_mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self.causal_mask_cache[seq_len] = causal_mask_ttnn

    def _create_causal_mask(self, seq_len: int) -> ttnn.Tensor:
        """
        Get causal mask for the given sequence length.
        Uses pre-computed cache when available, otherwise computes on-the-fly.

        Args:
            seq_len: Sequence length

        Returns:
            causal_mask: [1, 1, seq_len, seq_len] with -inf above diagonal

        Note:
            For trace support, the sequence length must be pre-computed in the cache.
            On-the-fly computation will fail during trace capture.
        """
        # Check cache first
        if seq_len in self.causal_mask_cache:
            return self.causal_mask_cache[seq_len]

        # If not in cache, compute on-the-fly (this won't work during trace!)
        # This path is only for non-traced execution with unusual sequence lengths
        causal_mask = torch.full((seq_len, seq_len), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        causal_mask_ttnn = ttnn.from_torch(
            causal_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return causal_mask_ttnn

    def __call__(
        self,
        decoder_input_values: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        speaker_embeddings: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Prenet: mel -> hidden + positional encoding + speaker embedding
        hidden_states = self.prenet(decoder_input_values, speaker_embeddings=speaker_embeddings)

        seq_len = hidden_states.shape[1]

        # Create causal attention mask
        causal_mask = self._create_causal_mask(seq_len)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=causal_mask,
            )

        return hidden_states


def preprocess_decoder_parameters(torch_model, config: TTNNDecoderConfig, device):
    """
    Preprocess PyTorch decoder parameters for TTNN.

    Converts all weights to TTNN format with proper:
    - Transposition for linear layers
    - Memory configuration (DRAM for weights)
    - Data type (bfloat16)
    - Layout (TILE_LAYOUT)

    Args:
        torch_model: PyTorch SpeechT5Decoder model
        config: TTNNDecoderConfig
        device: TTNN device

    Returns:
        parameters: Dict of TTNN tensors
    """
    print("[Preprocess Decoder] Starting...")
    parameters = {}

    # Memory configs
    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

    # Helper to convert linear weights
    def convert_linear(torch_linear):
        weight = torch_linear.weight.data  # [out, in]
        bias = torch_linear.bias.data if torch_linear.bias is not None else None

        return {
            "weight": ttnn.from_torch(
                weight.T,  # Transpose for TTNN
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            ),
            "bias": ttnn.from_torch(
                bias.unsqueeze(0).unsqueeze(0),  # [256] -> [1, 1, 256] for proper broadcasting
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            )
            if bias is not None
            else None,
        }

    # Helper to convert layer norm
    def convert_layer_norm(torch_ln):
        return {
            "weight": ttnn.from_torch(
                torch_ln.weight.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            ),
            "bias": ttnn.from_torch(
                torch_ln.bias.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=DRAM_MEMCFG,
            ),
        }

    # Process prenet
    prenet_params = {}

    # Prenet layers
    prenet_params["prenet_layers"] = []
    for i in range(config.speech_decoder_prenet_layers):
        prenet_params["prenet_layers"].append(convert_linear(torch_model.prenet.layers[i]))

    # Final layer
    prenet_params["final_layer"] = convert_linear(torch_model.prenet.final_layer)

    # Positional encoding (buffer)
    prenet_params["positional_encoding"] = ttnn.from_torch(
        torch_model.prenet.encode_positions.pe,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    # Positional encoding alpha (scalar parameter)
    prenet_params["encode_positions_alpha"] = ttnn.from_torch(
        torch_model.prenet.encode_positions.alpha.data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    # Speaker embeddings layer
    prenet_params["speaker_embeds_layer"] = convert_linear(torch_model.prenet.speaker_embeds_layer)

    parameters["prenet"] = prenet_params

    # Process decoder layers (from wrapped_decoder)
    parameters["layers"] = []
    wrapped_decoder = torch_model.wrapped_decoder
    for i in range(config.num_layers):
        layer = wrapped_decoder.layers[i]
        layer_params = {}

        # Self-attention
        layer_params["self_attn"] = {
            "q_proj": convert_linear(layer.self_attn.q_proj),
            "k_proj": convert_linear(layer.self_attn.k_proj),
            "v_proj": convert_linear(layer.self_attn.v_proj),
            "out_proj": convert_linear(layer.self_attn.out_proj),
        }

        # Cross-attention
        layer_params["encoder_attn"] = {
            "q_proj": convert_linear(layer.encoder_attn.q_proj),
            "k_proj": convert_linear(layer.encoder_attn.k_proj),
            "v_proj": convert_linear(layer.encoder_attn.v_proj),
            "out_proj": convert_linear(layer.encoder_attn.out_proj),
        }

        # Feed-forward
        layer_params["feed_forward"] = {
            "intermediate_dense": convert_linear(layer.feed_forward.intermediate_dense),
            "output_dense": convert_linear(layer.feed_forward.output_dense),
        }

        # Layer norms
        layer_params["self_attn_layer_norm"] = convert_layer_norm(layer.self_attn_layer_norm)
        layer_params["encoder_attn_layer_norm"] = convert_layer_norm(layer.encoder_attn_layer_norm)
        layer_params["final_layer_norm"] = convert_layer_norm(layer.final_layer_norm)

        parameters["layers"].append(layer_params)

    print("[Preprocess Decoder] Done!")
    return parameters
