# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

        # HiFi4 + fp32 accumulation for precise multicast matmul
        self._hifi4_fp32 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Multicast program configs for VLM (544→576 padded, 5x faster with 99.99% PCC)
        self._qkv_mc_config = None
        self._oproj_mc_config = None

        # Pre-sliced cos/sin cache: avoid 2 slice + 2 deallocate per block (saves 792 ops)
        self._cos_sin_cache = {}

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

        # OPTIMIZATION: QKV linear with multicast config for VLM (5x speedup)
        # Pad 544→576 (18 tiles), use (8,9) grid, HiFi4+fp32 for precision
        if seq_len >= 512 and self._qkv_mc_config is None:
            padded_tiles = ((seq_len + 31) // 32 + 8) // 9 * 9  # round up to multiple of 9
            qkv_out_tiles = (self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim) // 32
            self._qkv_mc_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 9),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=padded_tiles // 9,
                per_core_N=qkv_out_tiles // 8,
                transpose_mcast=False,
                fused_activation=None,
            )
            self._qkv_pad_to = padded_tiles * 32

        if seq_len >= 512 and self._qkv_mc_config is not None:
            pad_amount = self._qkv_pad_to - seq_len
            if pad_amount > 0:
                qkv_input = ttnn.pad(hidden_states, ((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)
            else:
                qkv_input = hidden_states
            xqkv = ttnn.linear(
                qkv_input,
                self.wqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=self._qkv_mc_config,
                compute_kernel_config=self._hifi4_fp32,
            )
            if pad_amount > 0:
                ttnn.deallocate(qkv_input)
                xqkv = ttnn.slice(xqkv, [0, 0, 0, 0], [batch_size, 1, seq_len, xqkv.shape[3]])
        else:
            xqkv = ttnn.linear(
                hidden_states,
                self.wqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

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

        # OPTIMIZATION: Pre-cached cos/sin slices (saves 4 ops per block × 198 blocks = 792 ops)
        if seq_len not in self._cos_sin_cache:
            self._cos_sin_cache[seq_len] = (
                ttnn.slice(self.cos_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim]),
                ttnn.slice(self.sin_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim]),
            )
        cos_sliced, sin_sliced = self._cos_sin_cache[seq_len]

        # Apply RoPE
        q_rope_padded = ttnn.experimental.rotary_embedding(q, cos_sliced, sin_sliced)
        k_rope_padded = ttnn.experimental.rotary_embedding(k, cos_sliced, sin_sliced)

        # rotary_embedding pads output to tile boundary — skip slice if already correct size
        if q_rope_padded.shape[2] == seq_len:
            q_rope = q_rope_padded
            k_rope = k_rope_padded
        else:
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
            is_causal=False,
            scale=self.scale,
        )

        # OPTIMIZATION 4: Native TTNN head concatenation (no PyTorch transfers!)
        # attn_output: [batch, num_heads, seq, head_dim] -> [batch, 1, seq, num_heads * head_dim]
        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Output projection with multicast config for VLM (5x speedup)
        if seq_len >= 512 and self._oproj_mc_config is None:
            padded_tiles = ((seq_len + 31) // 32 + 8) // 9 * 9
            o_tiles = self.hidden_size // 32
            self._oproj_mc_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 9),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=padded_tiles // 9,
                per_core_N=o_tiles // 8,
                transpose_mcast=False,
                fused_activation=None,
            )
            self._oproj_pad_to = padded_tiles * 32

        if seq_len >= 512 and self._oproj_mc_config is not None:
            pad_amount = self._oproj_pad_to - seq_len
            if pad_amount > 0:
                oproj_input = ttnn.pad(attn_concat, ((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)
            else:
                oproj_input = attn_concat
            output = ttnn.linear(
                oproj_input,
                self.o_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=self._oproj_mc_config,
                compute_kernel_config=self._hifi4_fp32,
            )
            if pad_amount > 0:
                ttnn.deallocate(oproj_input)
                output = ttnn.slice(output, [0, 0, 0, 0], [batch_size, 1, seq_len, self.hidden_size])
        else:
            output = ttnn.linear(
                attn_concat,
                self.o_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
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

        # Use bfloat8_b for small models (expert=1024), bfloat16 for large (VLM=2048)
        mlp_dtype = ttnn.bfloat8_b if config.width <= 1024 else ttnn.bfloat16

        def to_ttnn(w):
            if isinstance(w, torch.Tensor):
                return ttnn.from_torch(
                    w.T.contiguous(),
                    dtype=mlp_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            return w

        # Fuse gate+up weights for single matmul on small sequences
        gate_w = weights["mlp.gate_proj.weight"]
        up_w = weights["mlp.up_proj.weight"]
        if isinstance(gate_w, torch.Tensor) and isinstance(up_w, torch.Tensor):
            fused = torch.cat([gate_w, up_w], dim=0)  # [2*mlp_dim, hidden]
            self.fused_gate_up = ttnn.from_torch(
                fused.T.contiguous(),
                dtype=mlp_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        elif hasattr(gate_w, "shape") and hasattr(up_w, "shape"):
            # Both are already TTNN tensors — fuse on device
            gate_ttnn = to_ttnn(gate_w)
            up_ttnn = to_ttnn(up_w)
            self.fused_gate_up = ttnn.concat([gate_ttnn, up_ttnn], dim=-1)
        else:
            self.fused_gate_up = None
        self.gate_proj = to_ttnn(gate_w)
        self.up_proj = to_ttnn(up_w)
        self.down_proj = to_ttnn(weights["mlp.down_proj.weight"])
        self.mlp_dim = config.mlp_dim

        # Chunk size must be tile-aligned (multiple of 32)
        self.chunk_size = 256

        # Multicast config for down_proj (1.4x speedup on [256,16384]×[16384,hidden])
        # Only for large MLP (VLM: mlp_dim=16384)
        self._down_mc_config = None
        self._down_hifi4_fp32 = None
        if config.mlp_dim >= 8192:
            n_tiles = config.width // 32
            self._down_mc_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,  # 256/32/8 = 1
                per_core_N=n_tiles // 8,
                transpose_mcast=False,
                fused_activation=None,
            )
            self._down_hifi4_fp32 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

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

        # Fast path: small sequences (e.g., expert with ~50 tokens) — no chunking
        seq_dim = x.shape[1] if was_3d else x.shape[2]
        if seq_dim <= self.chunk_size:
            if self.fused_gate_up is not None:
                # Fused gate+up: single matmul instead of 2
                gate_up = ttnn.linear(x, self.fused_gate_up, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
                gate = ttnn.slice(gate_up, [0, 0, 0], [gate_up.shape[0], gate_up.shape[1], self.mlp_dim])
                up = ttnn.slice(gate_up, [0, 0, self.mlp_dim], [gate_up.shape[0], gate_up.shape[1], self.mlp_dim * 2])
                ttnn.deallocate(gate_up)
            else:
                gate = ttnn.linear(x, self.gate_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
                up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            # Fused GELU+multiply: 1 op instead of 3 (gelu + deallocate + multiply)
            _gelu_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)  # 0.0 = exact GELU
            hidden_out = ttnn.multiply(gate, up, input_tensor_a_activations=[_gelu_act])
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            output = ttnn.linear(hidden_out, self.down_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(hidden_out)
            if was_torch:
                output = ttnn.to_torch(output)
            return output

        # Chunked path for large sequences (VLM with 544 tokens)
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

            # Pad to nearest tile boundary (32) only if needed — NOT to chunk_size
            tile_aligned_size = ((actual_chunk_size + 31) // 32) * 32
            needs_tile_padding = tile_aligned_size > actual_chunk_size

            # Slice input chunk (always 4D)
            x_chunk = ttnn.slice(x, [0, 0, chunk_start, 0], [batch_size, 1, chunk_end, hidden])

            # Pad to tile alignment only (not chunk_size)
            if needs_tile_padding:
                pad_amount = tile_aligned_size - actual_chunk_size
                x_chunk = ttnn.to_memory_config(x_chunk, ttnn.DRAM_MEMORY_CONFIG)
                x_chunk = ttnn.pad(x_chunk, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)

            # Gate projection with FUSED GELU + LoFi compute
            gate_activated = ttnn.linear(
                x_chunk,
                self.gate_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                activation="gelu",
            )
            up = ttnn.linear(x_chunk, self.up_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(x_chunk)

            # Element-wise multiply
            hidden_out = ttnn.multiply(gate_activated, up)
            ttnn.deallocate(gate_activated)
            ttnn.deallocate(up)

            # Down projection — use multicast config if available (1.4x speedup for VLM)
            output_chunk = ttnn.linear(
                hidden_out,
                self.down_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=self._down_mc_config,
                compute_kernel_config=self._down_hifi4_fp32,
            )
            ttnn.deallocate(hidden_out)

            # Slice back to actual size if tile-padded
            if needs_tile_padding:
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

        # Move final output to L1 (skip if already there)
        if output.memory_config() != ttnn.L1_MEMORY_CONFIG:
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

        return hidden_states, new_cache


# Default exports
GemmaAttention = GemmaAttentionTTNN
GemmaMLP = GemmaMLPTTNN
GemmaBlock = GemmaBlockTTNN
