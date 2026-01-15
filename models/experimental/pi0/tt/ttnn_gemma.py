# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gemma transformer blocks - TTNN Implementation (Optimized).

This module implements Gemma 2B style transformer layers using TTNN operations:
    - RMSNorm (using native ttnn.rms_norm)
    - Multi-Query Attention (MQA) with fused QKV and native RoPE
    - GeGLU MLP (gated GELU activation)
    - Native head operations (nlp_create_qkv_heads, nlp_concat_heads)

Architecture configurations:
    - Gemma 2B (VLM): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1
    - Gemma 300M (Expert): width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1

Optimizations over baseline:
    1. Fused QKV projection (1 linear instead of 3)
    2. Native ttnn.experimental.nlp_create_qkv_heads
    3. Native ttnn.experimental.rotary_embedding (split-half pattern)
    4. Native ttnn.experimental.nlp_concat_heads for output
    5. Native ttnn.rms_norm (single fused kernel)
    6. Pure TTNN RoPE precomputation
"""

import math
from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import GemmaConfig


# ============================================================================
# RMSNorm (TTNN - Optimized)
# ============================================================================


def rms_norm_ttnn(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    eps: float = 1e-6,
) -> ttnn.Tensor:
    """
    OPTIMIZED: RMSNorm using ttnn.rms_norm fused operation.

    NOTE: The weight tensor should already have the Gemma-style +1 offset
    pre-applied during initialization (not computed here every forward pass).

    Args:
        x: TTNN tensor (batch_size, seq_len, hidden_dim)
        weight: TTNN weight tensor with +1 offset already applied (1, hidden_dim)
        eps: Epsilon for numerical stability

    Returns:
        Normalized TTNN tensor (bfloat16)
    """
    return ttnn.rms_norm(
        x,
        weight=weight,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


# ============================================================================
# Rotary Position Embeddings (TTNN Meta Format)
# ============================================================================


def precompute_freqs_cis_meta_format(
    head_dim: int,
    max_seq_len: int,
    device: ttnn.Device,
    base: float = 10000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Precompute cos and sin for rotary embeddings using pure TTNN operations.

    ttnn.experimental.rotary_embedding uses the split-half pattern (same as Gemma):
    - rotate_half(x) = cat(-x[..., dim/2:], x[..., :dim/2])
    - result = x * cos + rotate_half(x) * sin

    For this to work correctly, cos/sin must have shape [1, 1, max_seq_len, head_dim]
    where the values are repeated: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]

    This matches how the rotation pairs x[i] with x[i+dim/2] for i < dim/2.

    Args:
        head_dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        device: TTNN device
        base: Base for frequency computation

    Returns:
        Tuple of (cos, sin) each of shape (1, 1, max_seq_len, head_dim) as TTNN tensors
    """
    half_dim = head_dim // 2

    # Compute inverse frequencies using ttnn.arange
    # indices: [0, 2, 4, ..., head_dim-2]
    indices = ttnn.arange(0, head_dim, 2, device=device, dtype=ttnn.float32)
    # Convert to TILE_LAYOUT early (required for unary ops like pow, reciprocal, cos, sin)
    indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)

    # freqs = 1.0 / (base ** (indices / head_dim))
    exponents = ttnn.multiply(indices, 1.0 / head_dim)
    ttnn.deallocate(indices)
    base_powers = ttnn.pow(base, exponents)
    ttnn.deallocate(exponents)
    freqs = ttnn.reciprocal(base_powers)  # Shape: [half_dim]
    ttnn.deallocate(base_powers)

    # Compute positions: [0, 1, 2, ..., max_seq_len-1]
    t = ttnn.arange(0, max_seq_len, 1, device=device, dtype=ttnn.float32)  # Shape: [max_seq_len]
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    # Outer product: t[i] * freqs[j] -> [max_seq_len, half_dim]
    # Reshape for broadcasting: t -> [max_seq_len, 1], freqs -> [1, half_dim]
    t_col = ttnn.reshape(t, (max_seq_len, 1))
    ttnn.deallocate(t)
    freqs_row = ttnn.reshape(freqs, (1, half_dim))
    ttnn.deallocate(freqs)
    freqs_outer = ttnn.multiply(t_col, freqs_row)  # Shape: [max_seq_len, half_dim]
    ttnn.deallocate(t_col)
    ttnn.deallocate(freqs_row)

    # Compute cos/sin: [max_seq_len, half_dim]
    cos_half = ttnn.cos(freqs_outer)
    sin_half = ttnn.sin(freqs_outer)
    ttnn.deallocate(freqs_outer)

    # Repeat for full head_dim: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]
    # This matches the split-half rotation where x[i] pairs with x[i+dim/2]
    cos_2d = ttnn.concat([cos_half, cos_half], dim=-1)  # [seq, head_dim]
    sin_2d = ttnn.concat([sin_half, sin_half], dim=-1)  # [seq, head_dim]
    ttnn.deallocate(cos_half)
    ttnn.deallocate(sin_half)

    # Reshape to add batch and head dimensions: [1, 1, seq, head_dim]
    cos = ttnn.reshape(cos_2d, (1, 1, max_seq_len, head_dim))
    sin = ttnn.reshape(sin_2d, (1, 1, max_seq_len, head_dim))
    ttnn.deallocate(cos_2d)
    ttnn.deallocate(sin_2d)

    # Convert to bfloat16 for use with rotary_embedding
    cos = ttnn.typecast(cos, ttnn.bfloat16)
    sin = ttnn.typecast(sin, ttnn.bfloat16)

    return cos, sin


# ============================================================================
# Multi-Query Attention (TTNN - Optimized)
# ============================================================================


class GemmaAttentionTTNN:
    """
    Gemma Multi-Query Attention using TTNN operations.

    OPTIMIZED:
    1. Fused QKV projection (1 linear instead of 3)
    2. Native ttnn.experimental.nlp_create_qkv_heads
    3. Native ttnn.experimental.rotary_embedding (split-half pattern)
    4. Native ttnn.experimental.nlp_concat_heads for output
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
    ):
        """
        Initialize attention layer with TTNN weights.

        Args:
            config: Gemma configuration
            weights: TTNN weight tensors (including fused wqkv)
            layer_idx: Layer index
            device: TTNN device
            cos_meta: Precomputed cos for native TTNN RoPE [1, 1, max_seq, head_dim]
            sin_meta: Precomputed sin for native TTNN RoPE [1, 1, max_seq, head_dim]
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        # OPTIMIZATION: Use fused QKV weight (single linear instead of 3)
        self.wqkv = weights["self_attn.wqkv"]
        self.o_proj = weights["self_attn.o_proj.weight"]

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.width
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Store meta format cos/sin for native TTNN RoPE (split-half pattern)
        self.cos_meta = cos_meta
        self.sin_meta = sin_meta

        # Compute kernel config for attention (HiFi4 for precision-critical ops)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # HiFi2 config for projections (faster, less precision needed)
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        OPTIMIZED forward pass using fused QKV and native TTNN operations.

        Key optimizations:
        1. Single fused QKV linear (3x fewer linear ops)
        2. Native ttnn.experimental.nlp_create_qkv_heads
        3. Native ttnn.experimental.rotary_embedding (split-half pattern)
        4. Native ttnn.experimental.nlp_concat_heads for output

        Args:
            hidden_states: TTNN tensor (batch, seq_len, hidden_dim)
            cos, sin: Unused (kept for API compatibility, native RoPE uses self.cos_meta/sin_meta)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache

        Returns:
            Tuple of (output, optional_cache)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Reshape to 4D for nlp_create_qkv_heads: [batch, 1, seq, hidden]
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))

        # OPTIMIZATION 1: Single fused QKV linear (instead of 3 separate)
        # Output: [batch, 1, seq, Q_dim + K_dim + V_dim]

        # Use WIDTH_SHARDED L1 memory config
        xqkv = ttnn.linear(
            hidden_states,
            self.wqkv,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        # Convert back to interleaved for nlp_create_qkv_heads
        xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)

        # OPTIMIZATION 2: Native TTNN head splitting (no PyTorch transfers!)
        # This splits the fused QKV into separate Q, K, V with proper head layout
        # Output shapes: q=[batch, num_heads, seq, head_dim], k/v=[batch, num_kv_heads, seq, head_dim]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # OPTIMIZATION 3: Apply RoPE using native TTNN (split-half pattern)
        # Slice cos/sin to match actual sequence length
        cos_sliced = ttnn.slice(
            self.cos_meta,
            [0, 0, 0, 0],
            [1, 1, seq_len, self.head_dim],
        )
        sin_sliced = ttnn.slice(
            self.sin_meta,
            [0, 0, 0, 0],
            [1, 1, seq_len, self.head_dim],
        )

        # ttnn.experimental.rotary_embedding uses split-half pattern like Gemma
        q_rope_padded = ttnn.experimental.rotary_embedding(q, cos_sliced, sin_sliced)
        k_rope_padded = ttnn.experimental.rotary_embedding(k, cos_sliced, sin_sliced)

        ttnn.deallocate(cos_sliced)
        ttnn.deallocate(sin_sliced)

        # rotary_embedding pads output to tile boundary, slice back to original seq_len
        q_rope = ttnn.slice(q_rope_padded, [0, 0, 0, 0], [batch_size, self.num_heads, seq_len, self.head_dim])
        k_rope = ttnn.slice(k_rope_padded, [0, 0, 0, 0], [batch_size, self.num_kv_heads, seq_len, self.head_dim])

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_rope = ttnn.concat([past_k, k_rope], dim=2)
            v = ttnn.concat([past_v, v], dim=2)

        new_cache = (k_rope, v) if use_cache else None

        # Use TTNN scaled dot product attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_rope,
            k_rope,
            v,
            attn_mask=attention_mask,
            is_causal=False,  # Mask handles causality
            scale=self.scale,
        )

        # OPTIMIZATION 4: Native TTNN head concatenation (no PyTorch transfers!)
        # attn_output: [batch, num_heads, seq, head_dim] -> [batch, 1, seq, num_heads * head_dim]
        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Output projection - use bfloat16 for residual add compatibility
        output = ttnn.linear(
            attn_concat,
            self.o_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Reshape back to 3D: [batch, 1, seq, hidden] -> [batch, seq, hidden]
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

        return output, new_cache


# ============================================================================
# GeGLU MLP (TTNN)
# ============================================================================


class GemmaMLPTTNN:
    """
    Gemma MLP with GeGLU activation using TTNN.

    Uses chunking along sequence dimension combined with auto L1 sharding
    to fit large intermediate tensors (mlp_dim=16384) in L1 memory.

    Strategy:
    - Chunk input along sequence dimension (e.g., 544 → 3 chunks of 256)
    - Let matmul auto-compute optimal sharding for L1
    - Subsequent ops inherit the sharding from matmul output
    - Accumulate results in L1, concatenate at end
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize MLP with weights.

        Args:
            config: Gemma configuration
            weights: PyTorch weight tensors (will be converted to TTNN)
            device: TTNN device
        """
        self.config = config
        self.device = device

        # Convert weights to TTNN if they're PyTorch tensors
        def to_ttnn(w):
            if isinstance(w, torch.Tensor):
                return ttnn.from_torch(
                    w.T.contiguous(),  # Transpose for linear
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            return w

        self.gate_proj = to_ttnn(weights["mlp.gate_proj.weight"])
        self.up_proj = to_ttnn(weights["mlp.up_proj.weight"])
        self.down_proj = to_ttnn(weights["mlp.down_proj.weight"])

        # Chunk size must be tile-aligned (multiple of 32)
        # 256 = 32 × 8, optimal for 64-core auto-sharding (4 tokens/core)
        # 256 tokens × 16384 mlp_dim × 1 byte = ~4MB total
        # With auto-sharding across 64 cores = ~64KB per core (fits L1)
        self.chunk_size = 256

    def forward(self, x) -> ttnn.Tensor:
        """
        Forward pass using chunked processing with auto L1 sharding.

        Strategy:
        1. Chunk input along sequence dimension (e.g., 544 → 3 chunks of 256)
        2. Let matmul auto-compute sharding (typically WIDTH or BLOCK in L1)
        3. Subsequent ops inherit sharding from matmul output
        4. Accumulate results in L1, concatenate at end

        Args:
            x: Input tensor [batch, seq, hidden] or [batch, 1, seq, hidden] (PyTorch or TTNN)

        Returns:
            TTNN output tensor [batch, seq, hidden] or [batch, 1, seq, hidden]
        """
        # Convert PyTorch to TTNN if needed
        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            x = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        batch_size = x.shape[0]
        was_3d = len(x.shape) == 3

        # Always work with 4D tensors (ttnn.slice requires 4D coordinates)
        if was_3d:
            x = ttnn.reshape(x, [batch_size, 1, x.shape[1], x.shape[2]])

        seq_len = x.shape[2]
        hidden = x.shape[3]

        # Calculate number of chunks (tile-aligned)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        output_chunks = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            actual_chunk_size = chunk_end - chunk_start

            # Pad last chunk to tile alignment if needed
            needs_chunk_padding = actual_chunk_size < self.chunk_size
            padded_chunk_size = self.chunk_size if needs_chunk_padding else actual_chunk_size

            # Slice input chunk (always 4D)
            x_chunk = ttnn.slice(x, [0, 0, chunk_start, 0], [batch_size, 1, chunk_end, hidden])

            # Pad chunk if needed for tile alignment
            # Move to DRAM for multicore pad support (avoids L1 fallback warning)
            if needs_chunk_padding:
                pad_amount = padded_chunk_size - actual_chunk_size
                x_chunk = ttnn.to_memory_config(x_chunk, ttnn.DRAM_MEMORY_CONFIG)
                x_chunk = ttnn.pad(x_chunk, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)

            # Gate and up projections - use bfloat8_b for 2x memory savings
            # Use L1 interleaved (WIDTH_SHARDED incompatible with MLP dimensions)
            gate = ttnn.linear(x_chunk, self.gate_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            up = ttnn.linear(x_chunk, self.up_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(x_chunk)

            # GELU activation - inherits sharding and dtype
            gate_activated = ttnn.gelu(gate)
            ttnn.deallocate(gate)

            # Element-wise multiply - inherits sharding from inputs
            hidden_out = ttnn.multiply(gate_activated, up)
            ttnn.deallocate(gate_activated)
            ttnn.deallocate(up)

            # Down projection - keep in L1 interleaved (simpler, avoids conversion overhead)
            output_chunk = ttnn.linear(
                hidden_out, self.down_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(hidden_out)

            # Slice back to actual size if padded
            if needs_chunk_padding:
                output_chunk = ttnn.slice(output_chunk, [0, 0, 0, 0], [batch_size, 1, actual_chunk_size, hidden])

            output_chunks.append(output_chunk)

        # Concatenate all chunks along sequence dimension (always 4D now)
        if len(output_chunks) == 1:
            output = output_chunks[0]
        else:
            output = output_chunks[0]
            for i in range(1, len(output_chunks)):
                output = ttnn.concat([output, output_chunks[i]], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(output_chunks[i])

        # Move final output to L1
        output = ttnn.to_memory_config(output, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Reshape back to 3D if input was 3D
        if was_3d:
            output = ttnn.reshape(output, [batch_size, seq_len, hidden])

        # Convert back to PyTorch if input was PyTorch
        if was_torch:
            output = ttnn.to_torch(output)

        return output


# ============================================================================
# Full Transformer Block (TTNN)
# ============================================================================


class GemmaBlockTTNN:
    """
    Complete Gemma transformer block using TTNN.

    Architecture: Pre-LN with residual connections
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |______________________________|___________________|
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
    ):
        """
        Initialize transformer block with TTNN weights.

        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNN device
            cos_meta: Precomputed cos for native TTNN RoPE [1, 1, max_seq, head_dim]
            sin_meta: Precomputed sin for native TTNN RoPE [1, 1, max_seq, head_dim]
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

        self.attention = GemmaAttentionTTNN(config, weights, layer_idx, device, cos_meta, sin_meta)
        self.mlp = GemmaMLPTTNN(config, weights, device)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass using TTNN operations.

        Args:
            hidden_states: TTNN input tensor
            cos, sin: Unused (kept for API compatibility, passed through to attention)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache

        Returns:
            Tuple of (output, optional_cache)
        """
        # Pre-attention norm
        normed = rms_norm_ttnn(
            hidden_states,
            self.input_layernorm_weight,
            self.config.rms_norm_eps,
        )

        # Attention with residual
        attn_output, new_cache = self.attention.forward(
            normed,
            cos,
            sin,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )
        hidden_states = ttnn.add(hidden_states, attn_output)
        ttnn.deallocate(attn_output)
        ttnn.ReadDeviceProfiler(
            self.device
        )  # Clear device profiler buffer, this helps resolve a issue when building profiler perf sheets

        # Pre-MLP norm
        normed = rms_norm_ttnn(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )

        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_output)
        ttnn.deallocate(mlp_output)
        ttnn.ReadDeviceProfiler(
            self.device
        )  # Clear device profiler buffer, this helps resolve a issue when building profiler perf sheets

        return hidden_states, new_cache


# Default exports
GemmaAttention = GemmaAttentionTTNN
GemmaMLP = GemmaMLPTTNN
GemmaBlock = GemmaBlockTTNN
