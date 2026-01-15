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
from typing import Optional, List

import torch
import ttnn

from models.common.utility_functions import nearest_32


# ============================================================================
# KV Cache Initialization
# ============================================================================


def init_kv_cache(config, device, max_batch_size, max_seq_len, encoder_seq_len):
    """
    Initialize KV cache for decoder self-attention and cross-attention.

    Following Whisper pattern from models/demos/whisper/tt/ttnn_optimized_functional_whisper.py

    Args:
        config: TTNNDecoderConfig
        device: TTNN device
        max_batch_size: Maximum batch size
        max_seq_len: Maximum decoder sequence length
        encoder_seq_len: Encoder sequence length (for cross-attention cache)

    Returns:
        tuple: (kv_cache, cross_attn_cache)
            - kv_cache: List of [K, V] tensors per layer for self-attention
            - cross_attn_cache: List of [K, V] tensors per layer for cross-attention
    """
    kv_cache = []
    cross_attn_cache = []
    head_dim = config.hidden_size // config.num_heads
    num_layers = config.num_layers

    # SDPA decode requires K sequence length to be a multiple of chunk size (256)
    # Round up max_seq_len to nearest multiple of 256
    chunk_size = 256
    max_seq_len = ((max_seq_len + chunk_size - 1) // chunk_size) * chunk_size

    for layer_idx in range(num_layers):
        # Self-attention cache: [batch, num_heads, max_seq_len, head_dim]
        # Following Whisper pattern - both K and V have same format
        k_cache = torch.zeros((max_batch_size, config.num_heads, max_seq_len, head_dim))
        v_cache = torch.zeros((max_batch_size, config.num_heads, max_seq_len, head_dim))

        k_cache_ttnn = ttnn.from_torch(
            k_cache,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_cache_ttnn = ttnn.from_torch(
            v_cache,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kv_cache.append([k_cache_ttnn, v_cache_ttnn])

        # Cross-attention cache: Pre-allocated for encoder K/V
        # K is transposed: [batch, heads, head_dim, enc_seq_len]
        # V is normal: [batch, heads, enc_seq_len, head_dim]
        cross_k = torch.zeros((max_batch_size, config.num_heads, head_dim, encoder_seq_len))
        cross_v = torch.zeros((max_batch_size, config.num_heads, encoder_seq_len, head_dim))

        cross_k_ttnn = ttnn.from_torch(
            cross_k,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cross_v_ttnn = ttnn.from_torch(
            cross_v,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cross_attn_cache.append([cross_k_ttnn, cross_v_ttnn])

    return kv_cache, cross_attn_cache


def get_decode_sdpa_configs(config, bsz, device, max_seq_len=256):
    """
    Get sharded memory config and program config for SDPA decode.

    Following Whisper pattern - creates HEIGHT-sharded memory config required
    by paged_update_cache and scaled_dot_product_attention_decode.

    Args:
        config: TTNNDecoderConfig
        bsz: Batch size
        device: TTNN device
        max_seq_len: Maximum sequence length for KV cache (used to compute chunk sizes)

    Returns:
        tuple: (sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_kernel_config)
    """
    head_dim = config.hidden_size // config.num_heads
    padded_num_heads = nearest_32(config.num_heads)  # 12 -> 32

    # Batch-sharded across cores
    grid_size = device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(bsz, grid_size, row_wise=True)

    sdpa_batch_sharded_memcfg = ttnn.create_sharded_memory_config(
        shape=(padded_num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Compute appropriate chunk sizes based on max_seq_len
    # Chunk sizes must divide evenly into padded sequence length
    padded_seq_len = nearest_32(max_seq_len)
    # Use smaller of 256 or padded_seq_len to ensure divisibility
    k_chunk_size = min(256, padded_seq_len)
    q_chunk_size = min(256, padded_seq_len)

    compute_grid_size = device.compute_with_storage_grid_size()
    sdpa_decode_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(compute_grid_size.x, compute_grid_size.y),
        exp_approx_mode=False,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    sdpa_decode_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    return sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_kernel_config


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
        position_offset: int = 0,
        precomputed_pe: Optional[ttnn.Tensor] = None,
        precomputed_dropout_masks: Optional[List[ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        """
        Speech decoder prenet with comprehensive L1 memory management.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional
            position_offset: Offset for positional encoding (used in KV cache mode)
            precomputed_pe: Pre-computed positional embedding [1, 1, hidden_size] for trace support.
                           When provided, this is used directly instead of slicing from positional_encoding.
                           This enables trace by avoiding dynamic slicing operations.
            precomputed_dropout_masks: Pre-computed dropout masks for each prenet layer for trace support.
                           List of tensors, one per layer. When provided, bypasses dynamic slicing.

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
            # Skip dropout if p=0.0 (dropout disabled for testing)
            if self.config.speech_decoder_prenet_dropout > 0.0:
                seq_len = hidden_states.shape[1]
                batch_size = hidden_states.shape[0]

                if precomputed_dropout_masks is not None:
                    # Use pre-computed dropout mask (for trace support)
                    # precomputed_dropout_masks[i] is already sliced for this position
                    mask_sliced = precomputed_dropout_masks[i]
                else:
                    # Use precomputed dropout mask for performance
                    mask = self.parameters["dropout_masks"][i]
                    # Slice mask to match current sequence length and position [seq_len, hidden_size]
                    # For KV cache mode, use position_offset to get correct dropout mask position
                    mask_sliced = mask[position_offset : position_offset + seq_len, :]

                # Expand mask to match batch dimension [batch, seq_len, hidden_size]
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
        # For KV cache mode, use position_offset to get correct positional encoding
        seq_len = hidden_states.shape[1]

        if precomputed_pe is not None:
            # Use pre-computed positional embedding (for trace support)
            # precomputed_pe is already scaled and in L1
            pe_scaled = precomputed_pe
        else:
            # Standard path: slice positional encoding dynamically
            start_pos = position_offset
            end_pos = position_offset + seq_len
            pe_slice = self.parameters["positional_encoding"][:, start_pos:end_pos, :]
            pe_slice = ttnn.to_memory_config(
                pe_slice, ttnn.L1_MEMORY_CONFIG
            )  # Ensure sliced positional encoding is in L1

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

    def __init__(self, device, parameters, config: TTNNDecoderConfig, max_seq_len: int = 256):
        self.device = device
        self.parameters = parameters
        self.config = config
        self.max_seq_len = max_seq_len

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
        kv_cache: Optional[List[ttnn.Tensor]] = None,
        cross_attn_cache: Optional[List[ttnn.Tensor]] = None,
        cross_attn_cache_valid: bool = False,
        current_decode_pos: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Multi-head attention with KV cache support.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - queries
            attention_mask: [batch, 1, seq_len, seq_len or kv_seq_len] - optional mask
            key_value_states: [batch, kv_seq_len, hidden_size] - for cross-attention
            kv_cache: [K_cache, V_cache] - self-attention KV cache
            cross_attn_cache: [K_cache, V_cache] - cross-attention KV cache
            cross_attn_cache_valid: If True, use cached cross-attention K/V
            current_decode_pos: Current position tensor for cache update (must be in DRAM)
            encoder_attention_mask: [batch, 1, 1, encoder_seq_len] - mask for cross-attention padding

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Handle both 3D [B, S, H] and 4D [B, 1, S, H] tensors
        shape = hidden_states.shape
        if len(shape) == 4:
            batch, _, seq_len, _ = shape
        else:
            batch, seq_len, _ = shape

        # Determine if this is cross-attention
        is_cross_attention = key_value_states is not None

        # Check if we're in decode mode (KV cache enabled)
        is_decode = kv_cache is not None and current_decode_pos is not None

        if not is_cross_attention:
            # Self-attention path
            # For decode mode, don't transpose K since we need [B, H, S, d] format
            transpose_k = not is_decode

            qkv_states = ttnn.linear(
                hidden_states,
                self.parameters["qkv_proj"]["weight"],
                bias=self.parameters["qkv_proj"]["bias"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=get_high_perf_compute_config(),
            )
            # Only unsqueeze if 3D (from decode mode output might already be 4D)
            if len(qkv_states.shape) == 3:
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
                transpose_k_heads=transpose_k,
            )
            ttnn.deallocate(qkv_states)

            if is_decode:
                # Decode mode: update KV cache and use SDPA decode
                k_cache, v_cache = kv_cache

                # Get sharded config for SDPA decode
                sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_cfg = get_decode_sdpa_configs(
                    self.config, batch, self.device, self.max_seq_len
                )

                # CRITICAL: Transpose from [B, H, S, d] to [S, B, H, d] for SDPA decode
                query = ttnn.transpose(query, 0, 2)  # [B, H, S, d] -> [S, H, B, d]
                query = ttnn.transpose(query, 1, 2)  # [S, H, B, d] -> [S, B, H, d]
                key = ttnn.transpose(key, 0, 2)
                key = ttnn.transpose(key, 1, 2)
                value = ttnn.transpose(value, 0, 2)
                value = ttnn.transpose(value, 1, 2)

                # CRITICAL: Convert to sharded (required by paged_update_cache)
                query = ttnn.interleaved_to_sharded(query, sdpa_batch_sharded_memcfg)
                key = ttnn.interleaved_to_sharded(key, sdpa_batch_sharded_memcfg)
                value = ttnn.interleaved_to_sharded(value, sdpa_batch_sharded_memcfg)

                # Update KV cache
                ttnn.experimental.paged_update_cache(
                    k_cache, key, update_idxs_tensor=current_decode_pos, page_table=None
                )
                ttnn.experimental.paged_update_cache(
                    v_cache, value, update_idxs_tensor=current_decode_pos, page_table=None
                )

                # SDPA decode with cache
                attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                    query,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=current_decode_pos,
                    scale=self.scaling,
                    program_config=sdpa_decode_progcfg,
                    compute_kernel_config=sdpa_decode_compute_cfg,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )  # Output: [1, B, H, d]

                # Transpose back: [1, B, H, d] -> [B, H, 1, d]
                attn_output = ttnn.transpose(attn_output, 1, 2)  # [1, B, H, d] -> [1, H, B, d]
                attn_output = ttnn.transpose(attn_output, 0, 2)  # [1, H, B, d] -> [B, H, 1, d]

                # Concatenate heads
                attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)

            else:
                # Standard self-attention (prefill mode)
                # K is already transposed by nlp_create_qkv_heads

                # Apply scaling
                query = ttnn.multiply(query, self.scaling, memory_config=ttnn.L1_MEMORY_CONFIG)

                attn_weights = ttnn.matmul(
                    query,
                    key,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=get_high_perf_compute_config(),
                )

                # Apply attention mask if provided
                if attention_mask is not None:
                    attn_weights = ttnn.add(attn_weights, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

                # Softmax
                attn_weights = ttnn.softmax(attn_weights, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

                # Apply attention to values
                attn_output = ttnn.matmul(
                    attn_weights,
                    value,
                    compute_kernel_config=get_high_perf_compute_config(),
                )

                attn_output = ttnn.transformer.concatenate_heads(
                    attn_output,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

        else:
            # Cross-attention path
            # Check if we can use cached encoder K/V
            if cross_attn_cache is not None and cross_attn_cache_valid:
                # Reuse cached encoder K/V
                key = cross_attn_cache[0]
                value = cross_attn_cache[1]

                # Compute only Q from hidden_states
                query = ttnn.linear(
                    hidden_states,
                    self.parameters["q_proj"]["weight"],
                    bias=self.parameters["q_proj"]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=get_high_perf_compute_config(),
                )
                # Only unsqueeze if 3D (from decode mode it might be 4D)
                if len(query.shape) == 3:
                    query = ttnn.unsqueeze(query, dim=1)
                query = ttnn.experimental.nlp_create_qkv_heads(
                    query,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    num_heads=self.num_heads,
                    num_kv_heads=0,
                )[0]

            else:
                # Compute Q from hidden_states
                query = ttnn.linear(
                    hidden_states,
                    self.parameters["q_proj"]["weight"],
                    bias=self.parameters["q_proj"]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=get_high_perf_compute_config(),
                )

                # Compute K, V from encoder
                kv_states = ttnn.linear(
                    key_value_states,
                    self.parameters["kv_proj"]["weight"],
                    bias=self.parameters["kv_proj"]["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=get_high_perf_compute_config(),
                )
                # Only unsqueeze if 3D (encoder output is usually 3D)
                if len(kv_states.shape) == 3:
                    kv_states = ttnn.unsqueeze(kv_states, dim=1)
                kv_states_hidden_size = kv_states.shape[3]
                key = kv_states[:, :, :, : kv_states_hidden_size // 2]
                value = kv_states[:, :, :, kv_states_hidden_size // 2 :]
                ttnn.deallocate(kv_states)

                # Only unsqueeze query if it's 3D (from decode mode it might be 4D)
                if len(query.shape) == 3:
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

                # Transpose K for matmul: [B, H, S, d] -> [B, H, d, S]
                key = ttnn.permute(key, [0, 1, 3, 2])

                # Copy to pre-allocated cache (required for trace)
                if cross_attn_cache is not None:
                    ttnn.copy(key, cross_attn_cache[0])
                    ttnn.copy(value, cross_attn_cache[1])
                    key = cross_attn_cache[0]
                    value = cross_attn_cache[1]

            # Apply scaling
            query = ttnn.multiply(query, self.scaling, memory_config=ttnn.L1_MEMORY_CONFIG)

            attn_weights = ttnn.matmul(
                query, key, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=get_high_perf_compute_config()
            )

            # Apply encoder attention mask if provided (for padding)
            # Mask shape: [B, 1, 1, encoder_seq_len], attn_weights: [B, H, query_len, encoder_seq_len]
            if encoder_attention_mask is not None:
                attn_weights = ttnn.add(attn_weights, encoder_attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Softmax
            attn_weights = ttnn.softmax(attn_weights, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Apply attention to values
            attn_output = ttnn.matmul(
                attn_weights,
                value,
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
            max_seq_len=max_sequence_length,
        )

        # Cross-attention (doesn't use KV cache for self-attention, so max_seq_len not critical)
        self.encoder_attn = TTNNSpeechT5Attention(
            device,
            parameters["encoder_attn"],
            config,
            max_seq_len=max_sequence_length,
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
        kv_cache: Optional[List[ttnn.Tensor]] = None,
        cross_attn_cache: Optional[List[ttnn.Tensor]] = None,
        cross_attn_cache_valid: bool = False,
        current_decode_pos: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Decoder layer with KV cache support.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] - causal mask for self-attn
            kv_cache: [K_cache, V_cache] - self-attention KV cache for this layer
            cross_attn_cache: [K_cache, V_cache] - cross-attention KV cache for this layer
            cross_attn_cache_valid: If True, use cached cross-attention K/V
            current_decode_pos: Current position tensor for cache update
            encoder_attention_mask: [batch, 1, 1, encoder_seq_len] - mask for padding in cross-attn

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
            kv_cache=kv_cache,
            current_decode_pos=current_decode_pos,
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
        # Handle both 3D [B, S, H] and 4D [B, 1, S, H] tensors
        shape = hidden_states.shape
        if len(shape) == 4:
            batch_size, _, seq_len, hidden_size = shape
        else:
            batch_size, seq_len, hidden_size = shape
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
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=cross_attn_cache_valid,
            encoder_attention_mask=encoder_attention_mask,  # Mask for padding
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

    For trace support, preprocessing (prenet + PE) can be done separately via
    preprocess_decoder_inputs() method. This allows PE addition to happen
    OUTSIDE the traced decoder, following the Whisper pattern.
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

    def preprocess_decoder_inputs(
        self,
        decoder_input_values: ttnn.Tensor,
        position_offset: int = 0,
    ) -> ttnn.Tensor:
        """
        Preprocess decoder inputs OUTSIDE trace capture (following Whisper pattern).

        This method runs the prenet (which includes position-dependent operations like
        PE slicing and dropout mask slicing) BEFORE the traced decoder. This ensures
        that position-dependent operations happen outside the trace.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins] - mel frames
            position_offset: Position for positional encoding (decode step number)

        Returns:
            hidden_states: [batch, seq_len, hidden_size] - ready for decoder layers
        """
        # Run prenet with position-dependent operations
        # speaker_embeddings are pre-baked into prenet parameters, so None is passed
        hidden_states = self.prenet(
            decoder_input_values,
            speaker_embeddings=None,  # Pre-baked into parameters
            position_offset=position_offset,
            precomputed_pe=None,  # Will slice based on position_offset
            precomputed_dropout_masks=None,  # Will slice based on position_offset
        )
        return hidden_states

    def decoder_layers_forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        kv_cache: Optional[List] = None,
        cross_attn_cache: Optional[List] = None,
        cross_attn_cache_valid: bool = False,
        current_decode_pos: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass through decoder layers only (for trace support).

        This method is called AFTER preprocess_decoder_inputs() has run.
        It contains NO position-dependent operations, making it safe for trace capture.

        Args:
            hidden_states: [batch, seq_len, hidden_size] - from preprocess_decoder_inputs
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            kv_cache: List of [K_cache, V_cache] per layer for self-attention
            cross_attn_cache: List of [K_cache, V_cache] per layer for cross-attention
            cross_attn_cache_valid: If True, use cached cross-attention K/V
            current_decode_pos: Current position tensor for cache update (must be in DRAM)
            encoder_attention_mask: [batch, 1, 1, encoder_seq_len] - mask for padding in cross-attn

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Ensure inputs are in L1
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        encoder_hidden_states = ttnn.to_memory_config(encoder_hidden_states, ttnn.L1_MEMORY_CONFIG)

        seq_len = hidden_states.shape[1]

        # For decode mode with seq_len=1, no causal mask needed
        is_decode = kv_cache is not None and current_decode_pos is not None
        if is_decode and seq_len == 1:
            causal_mask = None
        else:
            causal_mask = self._create_causal_mask(seq_len)

        # Pass through decoder layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=causal_mask,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                cross_attn_cache=cross_attn_cache[i] if cross_attn_cache is not None else None,
                cross_attn_cache_valid=cross_attn_cache_valid,
                current_decode_pos=current_decode_pos,
                encoder_attention_mask=encoder_attention_mask,
            )

        return hidden_states

    def __call__(
        self,
        decoder_input_values: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        speaker_embeddings: Optional[ttnn.Tensor] = None,
        kv_cache: Optional[List] = None,
        cross_attn_cache: Optional[List] = None,
        cross_attn_cache_valid: bool = False,
        current_decode_pos: Optional[ttnn.Tensor] = None,
        position_offset: int = 0,
        precomputed_pe: Optional[ttnn.Tensor] = None,
        precomputed_dropout_masks: Optional[List[ttnn.Tensor]] = None,
        preprocessed_hidden_states: Optional[ttnn.Tensor] = None,
        timing_details: bool = False,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
    ):
        """
        Forward pass with KV cache support.

        For trace support, use preprocessed_hidden_states to bypass the prenet entirely.
        This allows position-dependent operations to happen OUTSIDE trace capture.

        Args:
            decoder_input_values: [batch, seq_len, num_mel_bins]
            encoder_hidden_states: [batch, enc_seq_len, hidden_size]
            speaker_embeddings: [batch, speaker_embedding_dim] - optional (ignored if preprocessed_hidden_states provided)
            kv_cache: List of [K_cache, V_cache] per layer for self-attention
            cross_attn_cache: List of [K_cache, V_cache] per layer for cross-attention
            cross_attn_cache_valid: If True, use cached cross-attention K/V
            current_decode_pos: Current position tensor for cache update (must be in DRAM)
            position_offset: Offset for positional encoding (used in KV cache mode, typically same as step)
            precomputed_pe: Pre-computed positional embedding [1, 1, hidden_size] for trace support.
                           When provided, bypasses dynamic slicing in prenet (required for trace).
            precomputed_dropout_masks: Pre-computed dropout masks for each prenet layer for trace support.
                           List of tensors, one per layer. When provided, bypasses dynamic slicing.
            preprocessed_hidden_states: [batch, seq_len, hidden_size] - If provided, skip prenet entirely.
                           Use this for trace mode: call preprocess_decoder_inputs() outside trace,
                           then pass result here during traced execution.
            timing_details: If True, return tuple (output, timing_dict)
            encoder_attention_mask: [batch, 1, 1, encoder_seq_len] - mask for padding in cross-attn

        Returns:
            If timing_details=False:
                hidden_states: [batch, seq_len, hidden_size]
            If timing_details=True:
                Tuple of (hidden_states, timing_dict)
        """
        import time

        timing = {}

        # Check if we're in decode mode (KV cache enabled)
        is_decode = kv_cache is not None and current_decode_pos is not None

        # TRACE MODE: Use preprocessed hidden states (prenet already run outside trace)
        if preprocessed_hidden_states is not None:
            start_time = time.time()
            hidden_states = preprocessed_hidden_states
            timing["prenet"] = 0.0  # Prenet was run outside
            timing["memory_input"] = time.time() - start_time
        else:
            # STANDARD MODE: Run prenet inside decoder
            # PHASE 1: Ensure all inputs are in L1
            start_time = time.time()
            decoder_input_values = ttnn.to_memory_config(decoder_input_values, ttnn.L1_MEMORY_CONFIG)
            if speaker_embeddings is not None:
                speaker_embeddings = ttnn.to_memory_config(speaker_embeddings, ttnn.L1_MEMORY_CONFIG)
            timing["memory_input"] = time.time() - start_time

            # PHASE 2: Prenet processing (L1 output)
            start_time = time.time()
            hidden_states = self.prenet(
                decoder_input_values,
                speaker_embeddings=speaker_embeddings,
                position_offset=position_offset,
                precomputed_pe=precomputed_pe,
                precomputed_dropout_masks=precomputed_dropout_masks,
            )
            timing["prenet"] = time.time() - start_time

        # Ensure encoder hidden states are in L1
        encoder_hidden_states = ttnn.to_memory_config(encoder_hidden_states, ttnn.L1_MEMORY_CONFIG)

        seq_len = hidden_states.shape[1]

        # PHASE 3: Create causal attention mask (L1 output)
        # For decode mode with seq_len=1, no causal mask needed
        start_time = time.time()
        if is_decode and seq_len == 1:
            causal_mask = None  # No mask needed for single token
        else:
            causal_mask = self._create_causal_mask(seq_len)
        timing["causal_mask"] = time.time() - start_time

        # PHASE 4: Pass through decoder layers with KV cache support
        start_time = time.time()
        layer_times = []
        for i, layer in enumerate(self.layers):
            layer_start = time.time()
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=causal_mask,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                cross_attn_cache=cross_attn_cache[i] if cross_attn_cache is not None else None,
                cross_attn_cache_valid=cross_attn_cache_valid,
                current_decode_pos=current_decode_pos,
                encoder_attention_mask=encoder_attention_mask,
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
