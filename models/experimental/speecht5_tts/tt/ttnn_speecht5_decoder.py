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


# ============================================================================
# Memory Management Utilities - Comprehensive L1 Optimization
# ============================================================================


def ensure_l1_memory(tensor):
    """
    Ensure tensor is in L1 memory for optimal performance.
    Moves tensor to L1 if not already there.
    """
    return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)


def move_to_l1_if_dram(tensor):
    """
    Conditionally move tensor to L1 only if it's currently in DRAM.
    Avoids unnecessary moves if already in L1.
    """
    try:
        if hasattr(tensor, "memory_config") and tensor.memory_config.buffer_type == ttnn.BufferType.DRAM:
            return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
    except:
        # If we can't check memory config, assume it's DRAM and move to L1
        pass
    return ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)


def l1_reshape(tensor, *args, **kwargs):
    """Reshape with L1 memory output"""
    return ttnn.reshape(tensor, *args, memory_config=ttnn.L1_MEMORY_CONFIG, **kwargs)


def l1_permute(tensor, *args, **kwargs):
    """Permute with L1 memory output"""
    return ttnn.permute(tensor, *args, memory_config=ttnn.L1_MEMORY_CONFIG, **kwargs)


def l1_concat(tensors, *args, **kwargs):
    """Concat with L1 memory output"""
    return ttnn.concat(tensors, *args, memory_config=ttnn.L1_MEMORY_CONFIG, **kwargs)


# ============================================================================
# High-Performance Compute Kernel Configs - Maximum Core Utilization
# ============================================================================


def get_high_perf_compute_config():
    """
    Get compute kernel config optimized for maximum core utilization and performance.
    Uses HiFi4 for speed while maintaining L1 memory optimization.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,  # Keep L1 accumulation for memory efficiency
    )


def l1_matmul(a, b, *args, **kwargs):
    """Matmul with L1 memory config and high-performance compute kernel"""
    if "compute_kernel_config" not in kwargs:
        kwargs["compute_kernel_config"] = get_high_perf_compute_config()
    if "memory_config" not in kwargs:
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    return ttnn.matmul(a, b, *args, **kwargs)


def l1_linear(input_tensor, weight, bias=None, *args, **kwargs):
    """Linear layer with L1 memory config and high-performance compute kernel"""
    if "compute_kernel_config" not in kwargs:
        kwargs["compute_kernel_config"] = get_high_perf_compute_config()
    if "memory_config" not in kwargs:
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    return ttnn.linear(input_tensor, weight, bias=bias, *args, **kwargs)


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
        TTNN consistent dropout implementation.

        HF applies dropout during inference (unusual but by design per paper §2.2).
        """
        # Create probability tensor from first element in batch
        prob_tensor = ttnn.full(
            inputs_embeds[0].shape, p, dtype=ttnn.bfloat16, device=inputs_embeds.device(), layout=ttnn.TILE_LAYOUT
        )

        # Generate Bernoulli mask (0s and 1s)
        mask = ttnn.bernoulli(prob_tensor, 0, dtype=ttnn.bfloat16)

        # Repeat mask across batch dimension
        batch_size = inputs_embeds.shape[0]
        all_masks = ttnn.repeat(ttnn.unsqueeze(mask, 0), [batch_size, 1, 1])

        # Apply mask and scale
        scale = 1.0 / (1.0 - p)
        return ttnn.multiply(ttnn.where(all_masks == 1, inputs_embeds, 0), scale)

    def __call__(
        self,
        decoder_input_values: ttnn.Tensor,
        speaker_embeddings: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Speech decoder prenet with comprehensive L1 memory management.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # PHASE 1: Ensure inputs are in L1
        hidden_states = ensure_l1_memory(decoder_input_values)
        if speaker_embeddings is not None:
            speaker_embeddings = ensure_l1_memory(speaker_embeddings)

        # PHASE 2: Prenet layers with ReLU and consistent dropout
        for i in range(self.config.speech_decoder_prenet_layers):
            hidden_states = ttnn.linear(
                hidden_states,
                self.parameters["prenet_layers"][i]["weight"],
                bias=self.parameters["prenet_layers"][i]["bias"],
                memory_config=self.L1_MEMCFG,
            )
            hidden_states = ensure_l1_memory(hidden_states)

            hidden_states = ttnn.relu(hidden_states)
            hidden_states = ensure_l1_memory(hidden_states)

            # Apply HF's consistent dropout (L1 output)
            hidden_states = self._consistent_dropout(hidden_states, self.config.speech_decoder_prenet_dropout)
            hidden_states = ensure_l1_memory(hidden_states)

        # PHASE 3: Final projection (L1 output)
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["final_layer"]["weight"],
            bias=self.parameters["final_layer"]["bias"],
            memory_config=self.L1_MEMCFG,
        )
        hidden_states = ensure_l1_memory(hidden_states)

        # PHASE 4: Add scaled positional encoding (ensure slice uses L1)
        seq_len = hidden_states.shape[1]
        pe_slice = self.parameters["positional_encoding"][:, :seq_len, :]
        pe_slice = ensure_l1_memory(pe_slice)  # Ensure sliced positional encoding is in L1

        # Scale positional encoding (L1 output)
        pe_scaled = ttnn.multiply(
            pe_slice, self.parameters["encode_positions_alpha"], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        pe_scaled = ensure_l1_memory(pe_scaled)

        # Add positional encoding (L1 output)
        hidden_states = ttnn.add(hidden_states, pe_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ensure_l1_memory(hidden_states)

        # PHASE 5: Add speaker embeddings if provided
        if speaker_embeddings is not None:
            # L2 normalization of speaker embeddings (L1 outputs)
            speaker_squared = ttnn.multiply(speaker_embeddings, speaker_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG)
            speaker_squared = ensure_l1_memory(speaker_squared)

            speaker_sum = ttnn.sum(speaker_squared, dim=-1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
            speaker_sum = ensure_l1_memory(speaker_sum)

            speaker_norm = ttnn.sqrt(speaker_sum)
            speaker_norm = ensure_l1_memory(speaker_norm)

            speaker_embeddings_normalized = ttnn.divide(
                speaker_embeddings, speaker_norm, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            speaker_embeddings_normalized = ensure_l1_memory(speaker_embeddings_normalized)

            # Expand to match sequence length (L1 output)
            batch_size = hidden_states.shape[0]
            speaker_embeddings_expanded = l1_reshape(
                speaker_embeddings_normalized,
                [batch_size, 1, self.config.speaker_embedding_dim],
            )

            # Repeat for sequence length (L1 output)
            speaker_embeddings_expanded = ttnn.repeat(
                speaker_embeddings_expanded,
                [1, seq_len, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            speaker_embeddings_expanded = ensure_l1_memory(speaker_embeddings_expanded)

            # Concatenate with hidden states (L1 output)
            hidden_states = l1_concat(
                [hidden_states, speaker_embeddings_expanded],
                dim=-1,
            )

            # Project back to hidden size (L1 output)
            hidden_states = ttnn.linear(
                hidden_states,
                self.parameters["speaker_embeds_layer"]["weight"],
                bias=self.parameters["speaker_embeds_layer"]["bias"],
                memory_config=self.L1_MEMCFG,
            )
            hidden_states = ensure_l1_memory(hidden_states)

            # ReLU after speaker projection (L1 output)
            hidden_states = ttnn.relu(hidden_states)
            hidden_states = ensure_l1_memory(hidden_states)

        # PHASE 6: Final output must be in L1
        return ensure_l1_memory(hidden_states)


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
        """
        Feed-forward network with comprehensive L1 memory management.
        """
        # PHASE 1: Ensure input is in L1
        hidden_states = ensure_l1_memory(hidden_states)

        # Intermediate dense (L1 output)
        hidden_states = l1_linear(
            hidden_states,
            self.parameters["intermediate_dense"]["weight"],
            bias=self.parameters["intermediate_dense"]["bias"],
        )

        # GELU activation (L1 output)
        hidden_states = ttnn.gelu(hidden_states)
        hidden_states = ensure_l1_memory(hidden_states)

        # Output dense (high-performance compute kernel)
        hidden_states = l1_linear(
            hidden_states,
            self.parameters["output_dense"]["weight"],
            bias=self.parameters["output_dense"]["bias"],
        )

        # PHASE 2: Final output must be in L1
        return ensure_l1_memory(hidden_states)


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
        Reshape [B, S, H] -> [B*NH, S, HD] with L1 memory management.
        where NH = num_heads, HD = head_dim
        """
        # PHASE 1: Ensure input is in L1
        tensor = ensure_l1_memory(tensor)

        # [B, S, H] -> [B, S, NH, HD] (L1)
        tensor = l1_reshape(tensor, [batch, seq_len, self.num_heads, self.head_dim])

        # [B, S, NH, HD] -> [B, NH, S, HD] (L1)
        tensor = l1_permute(tensor, [0, 2, 1, 3])

        # [B, NH, S, HD] -> [B*NH, S, HD] (L1)
        tensor = l1_reshape(tensor, [batch * self.num_heads, seq_len, self.head_dim])

        # PHASE 2: Output must be in L1
        return ensure_l1_memory(tensor)

    def _reshape_from_multihead(self, tensor: ttnn.Tensor, batch: int, seq_len: int) -> ttnn.Tensor:
        """
        Reshape [B*NH, S, HD] -> [B, S, H] with L1 memory management.
        """
        # PHASE 1: Ensure input is in L1
        tensor = ensure_l1_memory(tensor)

        # [B*NH, S, HD] -> [B, NH, S, HD] (L1)
        tensor = l1_reshape(tensor, [batch, self.num_heads, seq_len, self.head_dim])

        # [B, NH, S, HD] -> [B, S, NH, HD] (L1)
        tensor = l1_permute(tensor, [0, 2, 1, 3])

        # [B, S, NH, HD] -> [B, S, H] (L1)
        tensor = l1_reshape(tensor, [batch, seq_len, self.hidden_size])

        # PHASE 2: Output must be in L1
        return ensure_l1_memory(tensor)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        key_value_states: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Multi-head attention with comprehensive L1 memory management.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - queries
            attention_mask: [batch, 1, seq_len, seq_len or kv_seq_len] - optional mask
            key_value_states: [batch, kv_seq_len, hidden_size] - for cross-attention

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = hidden_states.shape

        # PHASE 1: Ensure inputs are in L1
        hidden_states = ensure_l1_memory(hidden_states)
        if key_value_states is not None:
            key_value_states = ensure_l1_memory(key_value_states)
        if attention_mask is not None:
            attention_mask = ensure_l1_memory(attention_mask)

        # Determine if this is cross-attention
        is_cross_attention = key_value_states is not None

        # PHASE 2: Q always from hidden_states (high-performance compute kernel)
        query = l1_linear(
            hidden_states,
            self.parameters["q_proj"]["weight"],
            bias=self.parameters["q_proj"]["bias"],
        )

        # Apply scaling (L1 output)
        query = ttnn.multiply(query, self.scaling, memory_config=ttnn.L1_MEMORY_CONFIG)
        query = ensure_l1_memory(query)

        # PHASE 3: K, V computation
        if is_cross_attention:
            # Cross-attention: K, V from encoder
            key = l1_linear(
                key_value_states,
                self.parameters["k_proj"]["weight"],
                bias=self.parameters["k_proj"]["bias"],
            )

            value = l1_linear(
                key_value_states,
                self.parameters["v_proj"]["weight"],
                bias=self.parameters["v_proj"]["bias"],
            )

            kv_seq_len = key_value_states.shape[1]
        else:
            # Self-attention: K, V from decoder hidden states
            key = l1_linear(
                hidden_states,
                self.parameters["k_proj"]["weight"],
                bias=self.parameters["k_proj"]["bias"],
            )

            value = l1_linear(
                hidden_states,
                self.parameters["v_proj"]["weight"],
                bias=self.parameters["v_proj"]["bias"],
            )

            kv_seq_len = seq_len

        # PHASE 4: Reshape for multi-head attention (L1 outputs)
        query = self._reshape_for_multihead(query, batch, seq_len)
        key = self._reshape_for_multihead(key, batch, kv_seq_len)
        value = self._reshape_for_multihead(value, batch, kv_seq_len)

        # PHASE 5: Compute attention scores: Q @ K^T (L1 outputs)
        key_t = l1_permute(key, [0, 2, 1])
        attn_weights = l1_matmul(query, key_t)
        attn_weights = ensure_l1_memory(attn_weights)

        # PHASE 6: Apply attention mask if provided (L1 outputs)
        if attention_mask is not None:
            # Expand mask for all heads: [B, 1, S, KS] -> [B*NH, S, KS]
            # attention_mask already has shape [B, 1, S, KS]
            mask_expanded = l1_reshape(attention_mask, [batch, 1, seq_len, kv_seq_len])
            mask_expanded = ttnn.repeat(mask_expanded, [1, self.num_heads, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
            mask_expanded = ensure_l1_memory(mask_expanded)
            mask_expanded = l1_reshape(mask_expanded, [batch * self.num_heads, seq_len, kv_seq_len])

            # Add mask to attention weights (L1 output)
            attn_weights = ttnn.add(attn_weights, mask_expanded, memory_config=ttnn.L1_MEMORY_CONFIG)
            attn_weights = ensure_l1_memory(attn_weights)

        # PHASE 7: Softmax (L1 output)
        attn_weights = ttnn.softmax(attn_weights, dim=-1)
        attn_weights = ensure_l1_memory(attn_weights)

        # PHASE 8: Apply attention to values (L1 output)
        attn_output = l1_matmul(attn_weights, value)
        attn_output = ensure_l1_memory(attn_output)

        # PHASE 9: Reshape back to [B, S, H] (L1 output)
        attn_output = self._reshape_from_multihead(attn_output, batch, seq_len)

        # PHASE 10: Output projection (L1 output)
        output = l1_linear(
            attn_output,
            self.parameters["out_proj"]["weight"],
            bias=self.parameters["out_proj"]["bias"],
        )
        output = ensure_l1_memory(output)

        # PHASE 11: Final output must be in L1
        return ensure_l1_memory(output)


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
        Decoder layer with comprehensive L1 memory management.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] - causal mask for self-attn

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # PHASE 1: Ensure all inputs are in L1
        hidden_states = ensure_l1_memory(hidden_states)
        encoder_hidden_states = ensure_l1_memory(encoder_hidden_states)
        if attention_mask is not None:
            attention_mask = ensure_l1_memory(attention_mask)

        # PHASE 2: Self-attention sub-layer (POST-NORM, with causal masking)
        residual = hidden_states
        residual = ensure_l1_memory(residual)

        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=None,  # Self-attention
        )
        hidden_states = ensure_l1_memory(hidden_states)

        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ensure_l1_memory(hidden_states)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.self_attn_layer_norm_params["weight"],
            bias=self.self_attn_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = ensure_l1_memory(hidden_states)

        # PHASE 3: Cross-attention sub-layer (POST-NORM)
        residual = hidden_states
        residual = ensure_l1_memory(residual)

        hidden_states = self.encoder_attn(
            hidden_states,
            attention_mask=None,  # No causal mask for cross-attention
            key_value_states=encoder_hidden_states,  # Cross-attention
        )
        hidden_states = ensure_l1_memory(hidden_states)

        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ensure_l1_memory(hidden_states)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.encoder_attn_layer_norm_params["weight"],
            bias=self.encoder_attn_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = ensure_l1_memory(hidden_states)

        # PHASE 4: Feed-forward sub-layer (POST-NORM)
        residual = hidden_states
        residual = ensure_l1_memory(residual)

        hidden_states = self.feed_forward(hidden_states)
        hidden_states = ensure_l1_memory(hidden_states)

        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = ensure_l1_memory(hidden_states)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.final_layer_norm_params["weight"],
            bias=self.final_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = ensure_l1_memory(hidden_states)

        # PHASE 5: Final output must be in L1
        return ensure_l1_memory(hidden_states)


class TTNNSpeechT5Decoder:
    """
    Complete SpeechT5 Decoder with Speech Prenet.

    Flow:
    1. Speech prenet (mel → hidden with positional encoding)
    2. Create causal mask for self-attention
    3. Pass through decoder layers (self-attn + cross-attn + FFN)
    4. Return hidden states
    """

    def __init__(self, device, parameters, config: TTNNDecoderConfig, max_sequence_length: int = 20):
        print("  [Decoder Init] Starting...")
        self.device = device
        self.config = config
        self.parameters = parameters
        self.max_sequence_length = max_sequence_length  # Store the max length

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
        print(f"  [Decoder Init] Pre-computing causal masks up to length {max_sequence_length}...")
        self.causal_mask_cache = {}
        self._precompute_causal_masks()
        print("  [Decoder Init] Done!")

    def _precompute_causal_masks(self):
        """
        Pre-compute causal masks for common sequence lengths.
        This avoids dynamic tensor creation during forward pass, which is
        required for trace support.

        Pre-computes for lengths: 20 and the configured max_sequence_length
        """
        common_lengths = [20, self.max_sequence_length]

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
        timing_details: bool = False,
    ) -> ttnn.Tensor:
        """
        Forward pass with comprehensive L1 memory management.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional
            timing_details: If True, return (output, timing_dict)

        Returns:
            hidden_states: [batch, seq_len, hidden_size] or (hidden_states, timing_dict)
        """
        import time

        timing = {}

        # PHASE 1: Ensure all inputs are in L1
        start_time = time.time()
        decoder_input_values = ensure_l1_memory(decoder_input_values)
        encoder_hidden_states = ensure_l1_memory(encoder_hidden_states)
        if speaker_embeddings is not None:
            speaker_embeddings = ensure_l1_memory(speaker_embeddings)
        timing["memory_input"] = time.time() - start_time

        # PHASE 2: Prenet processing (L1 output)
        start_time = time.time()
        hidden_states = self.prenet(decoder_input_values, speaker_embeddings=speaker_embeddings)
        hidden_states = ensure_l1_memory(hidden_states)
        timing["prenet"] = time.time() - start_time

        seq_len = hidden_states.shape[1]

        # PHASE 3: Create causal attention mask (L1 output)
        start_time = time.time()
        causal_mask = self._create_causal_mask(seq_len)
        causal_mask = ensure_l1_memory(causal_mask)
        timing["causal_mask"] = time.time() - start_time

        # PHASE 4: Pass through decoder layers with L1 management
        start_time = time.time()
        layer_times = []
        for i, layer in enumerate(self.layers):
            layer_start = time.time()
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=causal_mask,
            )
            hidden_states = ensure_l1_memory(hidden_states)
            layer_times.append(time.time() - layer_start)

        timing["decoder_layers"] = time.time() - start_time
        timing["layer_times"] = layer_times

        # PHASE 5: Final output must be in L1
        start_time = time.time()
        final_output = ensure_l1_memory(hidden_states)
        timing["memory_output"] = time.time() - start_time

        if timing_details:
            return final_output, timing
        return final_output


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
