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


def l1_width_sharded_memory(hidden_states):
    """L1 width sharded memory"""
    # Calculate shard dimensions for width sharding
    if len(hidden_states.shape) == 4:
        batch_size, __, seq_len, hidden_size = hidden_states.padded_shape
    elif len(hidden_states.shape) == 3:
        batch_size, seq_len, hidden_size = hidden_states.padded_shape
    else:
        raise ValueError(f"Unsupported shape: {hidden_states.shape}")

    num_cores = hidden_size // ttnn.TILE_SIZE

    if num_cores % 8 == 0:
        core_grid = ttnn.CoreGrid(y=num_cores // 8, x=8)
    else:
        core_grid = ttnn.CoreGrid(y=1, x=num_cores)
    num_cores = core_grid.x * core_grid.y
    shard_width = (hidden_size + num_cores - 1) // num_cores  # Divide width across cores
    # Create width sharded memory config for tensor
    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=[batch_size * seq_len, shard_width],  # Height is batch*seq_len, width is sharded
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Apply width sharding to hidden_states
    return ttnn.to_memory_config(hidden_states, width_sharded_mem_config)


# ============================================================================
# High-Performance Compute Kernel Configs - Maximum Core Utilization
# ============================================================================


def get_high_perf_compute_config():
    """
    Get compute kernel config optimized for maximum core utilization and performance.
    Uses HiFi4 for speed while maintaining L1 memory optimization.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,  # Keep L1 accumulation for memory efficiency
    )


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

        # Precomputed constants (always available for performance)
        self.dropout_scale = (
            1.0 / (1.0 - self.config.speech_decoder_prenet_dropout)
            if self.config.speech_decoder_prenet_dropout > 0.0
            else 1.0
        )

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
        hidden_states = ttnn.to_memory_config(decoder_input_values, ttnn.L1_MEMORY_CONFIG)
        if speaker_embeddings is not None:
            speaker_embeddings = ttnn.to_memory_config(speaker_embeddings, ttnn.L1_MEMORY_CONFIG)

        # PHASE 2: Prenet layers with ReLU and consistent dropout
        for i in range(self.config.speech_decoder_prenet_layers):
            hidden_states = ttnn.linear(
                hidden_states,
                self.parameters["prenet_layers"][i]["weight"],
                bias=self.parameters["prenet_layers"][i]["bias"],
                memory_config=self.L1_MEMCFG,
            )

            hidden_states = ttnn.relu(hidden_states)

            # Apply HF's consistent dropout (L1 output)
            # Use precomputed dropout mask for performance
            mask = self.parameters["dropout_masks"][i]
            # Slice mask to match current sequence length [seq_len, hidden_size]
            seq_len = hidden_states.shape[1]
            mask_sliced = mask[:seq_len, :]
            # Expand mask to match batch dimension [batch, seq_len, hidden_size]
            batch_size = hidden_states.shape[0]
            mask_expanded = ttnn.unsqueeze(mask_sliced, 0)  # [1, seq_len, hidden_size]
            mask_expanded = ttnn.repeat(mask_expanded, [batch_size, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
            # Apply mask: hidden_states = (mask == 1) * hidden_states * scale
            hidden_states = ttnn.multiply(
                ttnn.where(mask_expanded == 1, hidden_states, 0),
                self.dropout_scale,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # PHASE 3: Final projection (L1 output)
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["final_layer"]["weight"],
            bias=self.parameters["final_layer"]["bias"],
            memory_config=self.L1_MEMCFG,
        )

        # PHASE 4: Add scaled positional encoding (ensure slice uses L1)
        seq_len = hidden_states.shape[1]
        pe_slice = self.parameters["positional_encoding"][:, :seq_len, :]
        pe_slice = ttnn.to_memory_config(pe_slice, ttnn.L1_MEMORY_CONFIG)  # Ensure sliced positional encoding is in L1

        # Scale positional encoding (L1 output)
        pe_scaled = ttnn.multiply(
            pe_slice, self.parameters["encode_positions_alpha"], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # Add positional encoding (L1 output)
        hidden_states = ttnn.add(hidden_states, pe_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

        # PHASE 5: Add speaker embeddings (precomputed for performance)
        # Expand precomputed speaker embeddings to match sequence length (L1 output)
        speaker_embeddings_expanded = ttnn.repeat(
            self.parameters["speaker_embeddings_normalized"],
            [1, seq_len, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Concatenate with hidden states (L1 output)
        hidden_states = ttnn.concat(
            [hidden_states, speaker_embeddings_expanded],
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Project back to hidden size (L1 output)
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["speaker_embeds_layer"]["weight"],
            bias=self.parameters["speaker_embeds_layer"]["bias"],
            memory_config=self.L1_MEMCFG,
        )

        # ReLU after speaker projection (L1 output)
        hidden_states = ttnn.relu(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        # PHASE 6: Final output must be in L1
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
        """
        Feed-forward network with comprehensive L1 memory management.
        """
        # PHASE 1: Ensure input is in L1
        # hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

        # Intermediate dense (L1 output)
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["intermediate_dense"]["weight"],
            bias=self.parameters["intermediate_dense"]["bias"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=get_high_perf_compute_config(),
        )

        # GELU activation (L1 output)
        hidden_states = ttnn.gelu(hidden_states)  # , memory_config=ttnn.L1_MEMORY_CONFIG)

        # Output dense (high-performance compute kernel)
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters["output_dense"]["weight"],
            bias=self.parameters["output_dense"]["bias"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=get_high_perf_compute_config(),
        )

        # PHASE 2: Final output must be in L1
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
        Reshape [B, S, H] -> [B*NH, S, HD] with L1 memory management.
        where NH = num_heads, HD = head_dim
        """
        # PHASE 1: Ensure input is in L1
        tensor = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)

        # [B, S, H] -> [B, S, NH, HD] (L1)
        tensor = ttnn.reshape(
            tensor, [batch, seq_len, self.num_heads, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # [B, S, NH, HD] -> [B, NH, S, HD] (L1)
        tensor = ttnn.permute(tensor, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)

        # [B, NH, S, HD] -> [B*NH, S, HD] (L1)
        tensor = ttnn.reshape(
            tensor, [batch * self.num_heads, seq_len, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # PHASE 2: Output must be in L1
        return tensor

    def _reshape_from_multihead(self, tensor: ttnn.Tensor, batch: int, seq_len: int) -> ttnn.Tensor:
        """
        Reshape [B*NH, S, HD] -> [B, S, H] with L1 memory management.
        """
        # PHASE 1: Ensure input is in L1
        tensor = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)

        # [B*NH, S, HD] -> [B, NH, S, HD] (L1)
        tensor = ttnn.reshape(
            tensor, [batch, self.num_heads, seq_len, self.head_dim], memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # [B, NH, S, HD] -> [B, S, NH, HD] (L1)
        tensor = ttnn.permute(tensor, [0, 2, 1, 3], memory_config=ttnn.L1_MEMORY_CONFIG)

        # [B, S, NH, HD] -> [B, S, H] (L1)
        tensor = ttnn.reshape(tensor, [batch, seq_len, self.hidden_size], memory_config=ttnn.L1_MEMORY_CONFIG)

        return tensor

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

        # Determine if this is cross-attention
        is_cross_attention = key_value_states is not None

        if not is_cross_attention:
            qkv_states = ttnn.linear(
                hidden_states,
                self.parameters["qkv_proj"]["weight"],
                bias=self.parameters["qkv_proj"]["bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=get_high_perf_compute_config(),
            )
            qkv_states = ttnn.unsqueeze(qkv_states, dim=1)

            (
                query,
                key,
                value,
            ) = ttnn.experimental.nlp_create_qkv_heads(
                qkv_states,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                transpose_k_heads=True,
            )
            ttnn.deallocate(qkv_states)

        else:
            # PHASE 2: Q always from hidden_states (high-performance compute kernel)
            query = ttnn.linear(
                hidden_states,
                self.parameters["q_proj"]["weight"],
                bias=self.parameters["q_proj"]["bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=get_high_perf_compute_config(),
            )

            # Cross-attention: K, V from encoder

            kv_states = ttnn.linear(
                key_value_states,
                self.parameters["kv_proj"]["weight"],
                bias=self.parameters["kv_proj"]["bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=get_high_perf_compute_config(),
            )
            kv_states = ttnn.unsqueeze(kv_states, dim=1)
            kv_states_hidden_size = kv_states.shape[3]
            key = kv_states[:, :, :, : kv_states_hidden_size // 2]
            value = kv_states[:, :, :, kv_states_hidden_size // 2 :]
            ttnn.deallocate(kv_states)

            query = ttnn.unsqueeze(query, dim=1)
            query = ttnn.experimental.nlp_create_qkv_heads(
                query,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=self.num_heads,
                num_kv_heads=0,
            )[0]

            key = ttnn.experimental.nlp_create_qkv_heads(
                key,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=self.num_heads,
                num_kv_heads=0,
            )[0]

            value = ttnn.experimental.nlp_create_qkv_heads(
                value,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_heads=self.num_heads,
                num_kv_heads=0,
            )[0]
            key = ttnn.permute(key, [0, 1, 3, 2])

        # Apply scaling (L1 output)
        query = ttnn.multiply(query, self.scaling, memory_config=ttnn.L1_MEMORY_CONFIG)

        attn_weights = ttnn.matmul(
            query, key, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=get_high_perf_compute_config()
        )

        # PHASE 6: Apply attention mask if provided (L1 outputs)
        if attention_mask is not None:
            attn_weights = ttnn.add(attn_weights, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

        # PHASE 7: Softmax (L1 output)
        attn_weights = ttnn.softmax(attn_weights, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # PHASE 8: Apply attention to values (L1 output)
        attn_output = ttnn.matmul(
            attn_weights,
            value,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=get_high_perf_compute_config(),
        )

        attn_output = ttnn.transformer.concatenate_heads(
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # PHASE 10: Output projection (L1 output)
        output = ttnn.linear(
            attn_output,
            self.parameters["out_proj"]["weight"],
            bias=self.parameters["out_proj"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=get_high_perf_compute_config(),
        )
        # PHASE 11: Final output must be in L1
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

    def __init__(self, device, parameters, config: TTNNDecoderConfig, max_sequence_length: int = 32):
        self.device = device
        self.config = config
        self.max_sequence_length = max_sequence_length

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

        """
        # Match the sharding configuration from l1_width_sharded_memory
        # For hidden_size=768 (24 tiles), l1_width_sharded_memory creates:
        # CoreGrid(y=3, x=8) with width sharding
        num_cores = config.hidden_size // ttnn.TILE_SIZE  # 24 cores
        if num_cores % 8 == 0:
            core_grid = ttnn.CoreGrid(y=num_cores // 8, x=8)  # (3, 8) for 768
        else:
            core_grid = ttnn.CoreGrid(y=1, x=num_cores)

        print("max_sequence_length", self.max_sequence_length)
        per_core_M = math.ceil(self.max_sequence_length / ttnn.TILE_SIZE)

        # For width sharding, each core gets a portion of the width
        # Total width in tiles = hidden_size / TILE_SIZE = 24
        # Width per core = 24 / (3*8) = 1 tile per core
        K_tiles = config.hidden_size // ttnn.TILE_SIZE
        total_cores = core_grid.x * core_grid.y
        block_w = K_tiles // total_cores  # 24 / 24 = 1

        self.program_configs = {
            "layernorm_1_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid.x, core_grid.y),
                subblock_w=1,
                block_h=per_core_M,
                block_w=block_w,
                inplace=False,
            ),
        }
        """

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
        hidden_states = l1_width_sharded_memory(hidden_states)
        encoder_hidden_states = l1_width_sharded_memory(encoder_hidden_states)

        if attention_mask is not None:
            attention_mask = ttnn.to_memory_config(attention_mask, ttnn.L1_MEMORY_CONFIG)

        # PHASE 2: Self-attention sub-layer (POST-NORM, with causal masking)
        residual = hidden_states

        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=None,  # Self-attention
        )

        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states)  # , memory_config=ttnn.L1_MEMORY_CONFIG)

        # print("hidden_states", hidden_states.memory_config())
        # print("residual", residual.memory_config())
        # print("hidden_states", hidden_states.memory_config())
        # print("--------------------------------")

        # Match the sharding configuration from l1_width_sharded_memory
        # For hidden_size=768 (24 tiles), l1_width_sharded_memory creates:
        # CoreGrid(y=3, x=8) with width sharding
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_cores = hidden_size // ttnn.TILE_SIZE  # 24 cores
        if num_cores % 8 == 0:
            core_grid = ttnn.CoreGrid(y=num_cores // 8, x=8)  # (3, 8) for 768
        else:
            core_grid = ttnn.CoreGrid(y=1, x=num_cores)

        actual_seq_len = seq_len * batch_size
        per_core_M = math.ceil(actual_seq_len / ttnn.TILE_SIZE)

        # For width sharding, each core gets a portion of the width
        # Total width in tiles = hidden_size / TILE_SIZE = 24
        # Width per core = 24 / (3*8) = 1 tile per core
        K_tiles = hidden_size // ttnn.TILE_SIZE
        total_cores = core_grid.x * core_grid.y
        block_w = K_tiles // total_cores  # 24 / 24 = 1

        self.program_configs = {
            "layernorm_1_program_config": ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid.x, core_grid.y),
                subblock_w=1,
                block_h=per_core_M,
                block_w=block_w,
                inplace=False,
            ),
        }

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.self_attn_layer_norm_params["weight"],
            bias=self.self_attn_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.program_configs["layernorm_1_program_config"],
        )

        # PHASE 3: Cross-attention sub-layer (POST-NORM)
        residual = hidden_states

        hidden_states = self.encoder_attn(
            hidden_states,
            attention_mask=None,  # No causal mask for cross-attention
            key_value_states=encoder_hidden_states,  # Cross-attention
        )

        # Dropout skipped for inference
        hidden_states = ttnn.add(residual, hidden_states)  # , memory_config=ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.encoder_attn_layer_norm_params["weight"],
            bias=self.encoder_attn_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.program_configs["layernorm_1_program_config"],
        )

        # PHASE 4: Feed-forward sub-layer (POST-NORM)
        residual = hidden_states

        hidden_states = self.feed_forward(hidden_states)

        # Dropout skipped for inference
        # Ensure both tensors are in L1 before add
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        residual = ttnn.to_memory_config(residual, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Convert to width sharded for layer norm
        hidden_states = l1_width_sharded_memory(hidden_states)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.final_layer_norm_params["weight"],
            bias=self.final_layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.program_configs["layernorm_1_program_config"],
        )

        # PHASE 5: Final output must be in L1
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

    def __init__(self, device, parameters, config: TTNNDecoderConfig, max_sequence_length: int = 32):
        self.device = device
        self.config = config
        self.parameters = parameters
        self.max_sequence_length = max_sequence_length  # Store the max length

        # Speech decoder prenet
        self.prenet = TTNNSpeechDecoderPrenet(
            device,
            parameters["prenet"],
            config,
        )

        # Decoder layers
        self.layers = []
        for i in range(config.num_layers):
            layer = TTNNSpeechT5DecoderLayer(
                device,
                parameters["layers"][i],
                config,
                max_sequence_length=self.max_sequence_length,
            )
            self.layers.append(layer)

        # Pre-compute causal masks for common sequence lengths
        # This avoids dynamic tensor creation during forward pass (required for trace)
        self.causal_mask_cache = {}
        self._precompute_causal_masks()

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
    ):
        """
        Forward pass with comprehensive L1 memory management.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional
            timing_details: If True, return tuple (output, timing_dict)

        Returns:
            If timing_details=False:
                hidden_states: [batch, seq_len, hidden_size]
            If timing_details=True:
                Tuple of (hidden_states, timing_dict)
        """
        import time

        timing = {}

        # PHASE 1: Ensure all inputs are in L1
        start_time = time.time()
        decoder_input_values = ttnn.to_memory_config(decoder_input_values, ttnn.L1_MEMORY_CONFIG)
        encoder_hidden_states = ttnn.to_memory_config(encoder_hidden_states, ttnn.L1_MEMORY_CONFIG)
        if speaker_embeddings is not None:
            speaker_embeddings = ttnn.to_memory_config(speaker_embeddings, ttnn.L1_MEMORY_CONFIG)
        timing["memory_input"] = time.time() - start_time

        # PHASE 2: Prenet processing (L1 output)
        start_time = time.time()
        hidden_states = self.prenet(decoder_input_values, speaker_embeddings=speaker_embeddings)
        timing["prenet"] = time.time() - start_time

        seq_len = hidden_states.shape[1]

        # PHASE 3: Create causal attention mask (L1 output)
        start_time = time.time()
        causal_mask = self._create_causal_mask(seq_len)
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
            layer_times.append(time.time() - layer_start)

        timing["decoder_layers"] = time.time() - start_time
        timing["layer_times"] = layer_times

        # PHASE 5: Final output must be in L1
        start_time = time.time()
        timing["memory_output"] = time.time() - start_time
        final_output = hidden_states
        if timing_details:
            return final_output, timing
        return hidden_states

    def prepare_decode_inputs(
        self,
        decoder_input_values: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        speaker_embeddings: Optional[ttnn.Tensor] = None,
    ):
        """
        Prepare inputs for trace execution by ensuring proper memory config.

        This method separates input preparation from the forward pass to support trace capture.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional

        Returns:
            List of prepared input tensors
        """
        # Ensure all inputs are in L1 memory config for trace compatibility
        prepared_decoder_input = ttnn.to_memory_config(decoder_input_values, ttnn.L1_MEMORY_CONFIG)
        prepared_encoder_hidden = ttnn.to_memory_config(encoder_hidden_states, ttnn.L1_MEMORY_CONFIG)
        prepared_speaker_emb = None
        if speaker_embeddings is not None:
            prepared_speaker_emb = ttnn.to_memory_config(speaker_embeddings, ttnn.L1_MEMORY_CONFIG)

        return [prepared_decoder_input, prepared_encoder_hidden, prepared_speaker_emb]


def preprocess_decoder_parameters(
    torch_model, config: TTNNDecoderConfig, device, speaker_embeddings: Optional[torch.Tensor] = None
):
    """
    Preprocess PyTorch decoder parameters for TTNN.

    Converts all weights to TTNN format with proper:
    - Transposition for linear layers
    - Memory configuration (DRAM for weights)
    - Data type (bfloat16)
    - Layout (TILE_LAYOUT)

    Also precomputes constant values for performance:
    - Normalized speaker embeddings (if provided)
    - Bernoulli dropout masks for prenet layers

    Args:
        torch_model: PyTorch SpeechT5Decoder model
        config: TTNNDecoderConfig
        device: TTNN device
        speaker_embeddings: Optional speaker embeddings [batch, speaker_dim] to precompute

    Returns:
        parameters: Dict of TTNN tensors
    """
    parameters = {}

    # Memory configs
    DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
    L1_MEMCFG = ttnn.L1_MEMORY_CONFIG

    # Helper to convert linear weights

    def convert_linear_from_dict(torch_linear):
        weight = torch_linear["weight"]  # [out, in]
        if "bias" in torch_linear.keys():
            bias = torch_linear["bias"]
        else:
            bias = None

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

    # Precompute speaker embeddings (required for performance)
    # L2 normalization (exact same as HF implementation)
    speaker_embeddings_normalized = torch.nn.functional.normalize(speaker_embeddings, p=2, dim=-1)

    # Reshape to [batch, 1, speaker_dim] for easy expansion later
    batch_size = speaker_embeddings.shape[0]
    speaker_embeddings_normalized = speaker_embeddings_normalized.reshape(batch_size, 1, config.speaker_embedding_dim)

    # Convert to TTNN and store in DRAM (weights memory)
    prenet_params["speaker_embeddings_normalized"] = ttnn.from_torch(
        speaker_embeddings_normalized,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=DRAM_MEMCFG,
    )

    # Pre-generate dropout masks for prenet layers (required for performance)
    # Use same dropout probability as prenet layers (0.5 as per HF)
    prenet_dropout_p = config.speech_decoder_prenet_dropout
    prenet_params["dropout_masks"] = []

    # Generate mask for each prenet layer
    # We need to determine the expected input shape for each layer
    input_shapes = []
    input_shape = config.num_mel_bins  # First layer input
    for i in range(config.speech_decoder_prenet_layers):
        if i < config.speech_decoder_prenet_layers - 1:
            output_shape = config.speech_decoder_prenet_units
        else:
            output_shape = config.speech_decoder_prenet_units  # Last prenet layer before final projection

        input_shapes.append((config.max_position_embeddings, output_shape))  # Assume max seq len
        input_shape = output_shape

    # Generate masks for each prenet layer
    for seq_len, hidden_size in input_shapes:
        # Create probability tensor matching the expected output shape after ReLU
        prob_tensor = torch.full((seq_len, hidden_size), prenet_dropout_p, dtype=torch.float32)

        # Generate Bernoulli mask (consistent across batches as per HF)
        mask = torch.bernoulli(prob_tensor)

        # Convert to TTNN
        mask_ttnn = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM_MEMCFG,  # Store masks in DRAM like weights
        )
        prenet_params["dropout_masks"].append(mask_ttnn)

    parameters["prenet"] = prenet_params

    # Process decoder layers (from wrapped_decoder)
    parameters["layers"] = []
    wrapped_decoder = torch_model.wrapped_decoder
    for i in range(config.num_layers):
        layer = wrapped_decoder.layers[i]
        layer_params = {}

        # Self-attention
        qkv_weight = torch.cat(
            [
                layer.self_attn.q_proj.weight.data,
                layer.self_attn.k_proj.weight.data,
                layer.self_attn.v_proj.weight.data,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [
                layer.self_attn.q_proj.bias.data,
                layer.self_attn.k_proj.bias.data,
                layer.self_attn.v_proj.bias.data,
            ],
            dim=0,
        )

        qkv_proj = {}
        qkv_proj = {
            "weight": qkv_weight,
            "bias": qkv_bias,
        }

        layer_params["self_attn"] = {
            "q_proj": convert_linear(layer.self_attn.q_proj),
            "k_proj": convert_linear(layer.self_attn.k_proj),
            "v_proj": convert_linear(layer.self_attn.v_proj),
            "out_proj": convert_linear(layer.self_attn.out_proj),
            "qkv_proj": convert_linear_from_dict(qkv_proj),
        }

        # Cross-attention
        kv_weight = torch.cat(
            [
                layer.encoder_attn.k_proj.weight.data,
                layer.encoder_attn.v_proj.weight.data,
            ],
            dim=0,
        )
        kv_bias = torch.cat(
            [
                layer.encoder_attn.k_proj.bias.data,
                layer.encoder_attn.v_proj.bias.data,
            ],
            dim=0,
        )

        kv_proj = {}
        kv_proj = {
            "weight": kv_weight,
            "bias": kv_bias,
        }
        layer_params["encoder_attn"] = {
            "q_proj": convert_linear(layer.encoder_attn.q_proj),
            "k_proj": convert_linear(layer.encoder_attn.k_proj),
            "v_proj": convert_linear(layer.encoder_attn.v_proj),
            "out_proj": convert_linear(layer.encoder_attn.out_proj),
            "kv_proj": convert_linear_from_dict(kv_proj),
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

    return parameters
