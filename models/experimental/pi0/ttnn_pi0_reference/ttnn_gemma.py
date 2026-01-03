# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gemma transformer blocks for TTNN PI0 implementation.

This module implements Gemma 2B style transformer layers with:
    - RMSNorm (pre-normalization)
    - Multi-Query Attention (MQA) with num_kv_heads=1
    - GeGLU MLP (gated GELU activation)
    - Rotary Position Embeddings (RoPE)

Architecture configurations:
    - Gemma 2B (VLM): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1
    - Gemma 300M (Expert): width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


@dataclass
class GemmaConfig:
    """Configuration for Gemma transformer."""
    width: int = 2048
    depth: int = 18
    mlp_dim: int = 16384
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    
    @classmethod
    def gemma_2b(cls) -> "GemmaConfig":
        """Gemma 2B configuration (VLM backbone)."""
        return cls(
            width=2048,
            depth=18,
            mlp_dim=16384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    
    @classmethod
    def gemma_300m(cls) -> "GemmaConfig":
        """Gemma 300M configuration (action expert)."""
        return cls(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )


# ============================================================================
# RMSNorm
# ============================================================================

def rms_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm with Gemma-style weight offset.
    
    Gemma uses (weight + 1) instead of just weight.
    
    Args:
        x: Input tensor (batch_size, seq_len, hidden_dim)
        weight: Learnable weight (hidden_dim,)
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    """
    # Compute RMS
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    
    # Apply Gemma-style weight (weight + 1)
    return x_normalized * (weight + 1.0)


def rms_norm_ttnn(
    x: "ttnn.Tensor",
    weight: "ttnn.Tensor",
    eps: float = 1e-6,
) -> "ttnn.Tensor":
    """
    OPTIMIZED: RMSNorm using ttnn.rms_norm fused operation.

    NOTE: The weight tensor should already have the Gemma-style +1 offset
    pre-applied during initialization (not computed here every forward pass).

    Args:
        x: TTNN tensor (batch_size, seq_len, hidden_dim)
        weight: TTNN weight tensor with +1 offset already applied (1, hidden_dim)
        eps: Epsilon for numerical stability

    Returns:
        Normalized TTNN tensor
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")

    # Use fused ttnn.rms_norm (single optimized kernel instead of 9 separate ops)
    return ttnn.rms_norm(
        x,
        weight=weight,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


# ============================================================================
# Rotary Position Embeddings
# ============================================================================


def precompute_freqs_cis_torch(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for rotary embeddings.

    Args:
        head_dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        dtype: Output dtype
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) each of shape (max_seq_len, head_dim // 2)
    """
    # Compute inverse frequencies
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))

    # Compute positions
    t = torch.arange(max_seq_len, device=device, dtype=dtype)

    # Outer product
    freqs_outer = torch.outer(t, freqs)

    return torch.cos(freqs_outer), torch.sin(freqs_outer)


def precompute_freqs_cis_meta_format(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for rotary embeddings for ttnn.experimental.rotary_embedding.

    ttnn.experimental.rotary_embedding uses the split-half pattern (same as Gemma):
    - rotate_half(x) = cat(-x[..., dim/2:], x[..., :dim/2])
    - result = x * cos + rotate_half(x) * sin

    For this to work correctly, cos/sin must have shape [1, 1, max_seq_len, head_dim]
    where the values are repeated: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]

    This matches how the rotation pairs x[i] with x[i+dim/2] for i < dim/2.

    Args:
        head_dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        dtype: Output dtype

    Returns:
        Tuple of (cos, sin) each of shape (1, 1, max_seq_len, head_dim)
    """
    # Compute inverse frequencies
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))

    # Compute positions
    t = torch.arange(max_seq_len, dtype=dtype)

    # Outer product: [max_seq_len, head_dim/2]
    freqs_outer = torch.outer(t, freqs)

    # Compute cos/sin: [max_seq_len, head_dim/2]
    cos = torch.cos(freqs_outer)
    sin = torch.sin(freqs_outer)

    # Repeat for full head_dim: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]
    # This matches the split-half rotation where x[i] pairs with x[i+dim/2]
    cos = torch.cat([cos, cos], dim=-1)  # [seq, dim]
    sin = torch.cat([sin, sin], dim=-1)  # [seq, dim]

    # Add batch and head dimensions: [1, 1, seq, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return cos, sin


def apply_rotary_emb_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.
    
    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine frequencies (max_seq_len, head_dim // 2)
        sin: Sine frequencies (max_seq_len, head_dim // 2)
        position_ids: Position indices for each position (batch, seq_len)
    
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    seq_len = q.shape[2]
    head_dim = q.shape[-1]
    
    # Select cos/sin based on position IDs or use sequential
    if position_ids is not None:
        cos = cos[position_ids]  # (batch, seq_len, head_dim // 2)
        sin = sin[position_ids]
        # Add head dimension
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim // 2)
        sin = sin.unsqueeze(1)
    else:
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    # Duplicate for full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    # Apply rotation
    # Split and interleave for rotation
    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
    
    q_rotated = torch.cat([q1 * cos[..., :head_dim // 2] - q2 * sin[..., :head_dim // 2],
                           q1 * sin[..., head_dim // 2:] + q2 * cos[..., head_dim // 2:]], dim=-1)
    k_rotated = torch.cat([k1 * cos[..., :head_dim // 2] - k2 * sin[..., :head_dim // 2],
                           k1 * sin[..., head_dim // 2:] + k2 * cos[..., head_dim // 2:]], dim=-1)
    
    return q_rotated, k_rotated


# ============================================================================
# Multi-Query Attention
# ============================================================================

class GemmaAttentionTorch:
    """
    Gemma Multi-Query Attention (PyTorch).
    
    Uses 8 query heads but only 1 key-value head, which is broadcast
    across all query heads during attention computation.
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        """
        Initialize attention layer.
        
        Args:
            config: Gemma configuration
            weights: Layer weights with keys:
                - self_attn.q_proj.weight
                - self_attn.k_proj.weight
                - self_attn.v_proj.weight
                - self_attn.o_proj.weight
            layer_idx: Layer index for caching
        """
        self.config = config
        self.layer_idx = layer_idx
        
        self.q_proj = weights["self_attn.q_proj.weight"]
        self.k_proj = weights["self_attn.k_proj.weight"]
        self.v_proj = weights["self_attn.v_proj.weight"]
        self.o_proj = weights["self_attn.o_proj.weight"]
        
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for attention.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            cos, sin: RoPE embeddings
            attention_mask: (batch, 1, seq_len, kv_len) additive mask
            position_ids: Position indices
            past_key_value: Cached (K, V) from previous forward
            use_cache: Whether to return updated cache
        
        Returns:
            Tuple of (output, optional_new_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V (ensure dtype compatibility)
        q_proj = self.q_proj.to(hidden_states.dtype)
        k_proj = self.k_proj.to(hidden_states.dtype)
        v_proj = self.v_proj.to(hidden_states.dtype)
        q = F.linear(hidden_states, q_proj)
        k = F.linear(hidden_states, k_proj)
        v = F.linear(hidden_states, v_proj)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_emb_torch(q, k, cos, sin, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        
        # Expand K, V for broadcast with Q
        # K, V: (batch, 1, kv_len, head_dim) -> (batch, num_heads, kv_len, head_dim)
        kv_len = k.shape[2]
        k_expanded = k.expand(batch_size, self.num_heads, kv_len, self.head_dim)
        v_expanded = v.expand(batch_size, self.num_heads, kv_len, self.head_dim)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_expanded)
        
        # Reshape and project output (ensure dtype compatibility)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        o_proj = self.o_proj.to(attn_output.dtype)
        output = F.linear(attn_output, o_proj)
        
        return output, new_cache


class GemmaAttentionTTNN:
    """
    Gemma Multi-Query Attention using TTNN operations.

    OPTIMIZED:
    1. Fused QKV projection (1 linear instead of 3)
    2. Native ttnn.experimental.nlp_create_qkv_heads (no PyTorch transfers)
    3. Native ttnn.experimental.rotary_embedding (split-half pattern)
    4. Native ttnn.experimental.nlp_concat_heads for output
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, "ttnn.Tensor"],
        layer_idx: int,
        device: "ttnn.Device",
        cos_meta: Optional["ttnn.Tensor"] = None,
        sin_meta: Optional["ttnn.Tensor"] = None,
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
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")

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

        # Enable native RoPE when precomputed tensors are available
        self.use_native_rope = cos_meta is not None and sin_meta is not None

        # Compute kernel config for attention
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    
    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        cos: "ttnn.Tensor",
        sin: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_value: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
        use_cache: bool = False,
    ) -> Tuple["ttnn.Tensor", Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """
        OPTIMIZED forward pass using fused QKV and native TTNN operations.

        Key optimizations:
        1. Single fused QKV linear (3x fewer linear ops)
        2. Native ttnn.experimental.nlp_create_qkv_heads (no PyTorch transfers)
        3. Native ttnn.experimental.rotary_embedding (split-half pattern)
        4. Native ttnn.experimental.nlp_concat_heads for output

        Args:
            hidden_states: TTNN tensor (batch, seq_len, hidden_dim)
            cos, sin: RoPE embeddings (unused when use_native_rope=True)
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
        xqkv = ttnn.linear(
            hidden_states,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # OPTIMIZATION 2: Native TTNN head splitting (no PyTorch transfers!)
        # This splits the fused QKV into separate Q, K, V with proper head layout
        # Output shapes: q=[batch, num_heads, seq, head_dim], k/v=[batch, num_kv_heads, seq, head_dim]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output projection (DRAM for large tensors)
        output = ttnn.linear(
            attn_concat,
            self.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Reshape back to 3D: [batch, 1, seq, hidden] -> [batch, seq, hidden]
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

        return output, new_cache


# ============================================================================
# GeGLU MLP
# ============================================================================

class GemmaMLPTorch:
    """
    Gemma MLP with GeGLU activation (PyTorch).
    
    Uses gated GELU: output = down_proj(gelu(gate_proj(x)) * up_proj(x))
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize MLP.
        
        Args:
            config: Gemma configuration
            weights: Layer weights with keys:
                - mlp.gate_proj.weight
                - mlp.up_proj.weight
                - mlp.down_proj.weight
        """
        self.config = config
        self.gate_proj = weights["mlp.gate_proj.weight"]
        self.up_proj = weights["mlp.up_proj.weight"]
        self.down_proj = weights["mlp.down_proj.weight"]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
        
        Returns:
            Output tensor of same shape
        """
        # Ensure dtype compatibility
        gate_proj = self.gate_proj.to(x.dtype)
        up_proj = self.up_proj.to(x.dtype)
        down_proj = self.down_proj.to(x.dtype)
        gate = F.linear(x, gate_proj)
        up = F.linear(x, up_proj)
        hidden = F.gelu(gate, approximate="tanh") * up
        return F.linear(hidden, down_proj)


class GemmaMLPTTNN:
    """
    Gemma MLP with GeGLU activation using TTNN.
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, "ttnn.Tensor"],
        device: "ttnn.Device",
    ):
        """
        Initialize MLP with TTNN weights.
        
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.device = device
        self.gate_proj = weights["mlp.gate_proj.weight"]
        self.up_proj = weights["mlp.up_proj.weight"]
        self.down_proj = weights["mlp.down_proj.weight"]
    
    def forward(self, x: "ttnn.Tensor") -> "ttnn.Tensor":
        """
        Forward pass using TTNN operations.
        
        Args:
            x: TTNN input tensor
        
        Returns:
            TTNN output tensor
        """
        # Use DRAM for large intermediate tensors to avoid OOM
        gate = ttnn.linear(x, self.gate_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.linear(x, self.up_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # GELU activation
        gate_activated = ttnn.gelu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Element-wise multiply
        hidden = ttnn.multiply(gate_activated, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Down projection
        return ttnn.linear(hidden, self.down_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ============================================================================
# Full Transformer Block
# ============================================================================

class GemmaBlockTorch:
    """
    Complete Gemma transformer block (PyTorch).
    
    Architecture: Pre-LN with residual connections
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |______________________________|___________________|
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        """
        Initialize transformer block.
        
        Args:
            config: Gemma configuration
            weights: All weights for this layer
            layer_idx: Layer index
        """
        self.config = config
        self.layer_idx = layer_idx
        
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]
        
        self.attention = GemmaAttentionTorch(config, weights, layer_idx)
        self.mlp = GemmaMLPTorch(config, weights)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: Input tensor
            cos, sin: RoPE embeddings
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache
        
        Returns:
            Tuple of (output, optional_cache)
        """
        # Pre-attention norm
        normed = rms_norm_torch(
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
        hidden_states = hidden_states + attn_output
        
        # Pre-MLP norm
        normed = rms_norm_torch(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        
        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        hidden_states = hidden_states + mlp_output
        
        return hidden_states, new_cache


class GemmaBlockTTNN:
    """
    Complete Gemma transformer block using TTNN.
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, "ttnn.Tensor"],
        layer_idx: int,
        device: "ttnn.Device",
        cos_meta: Optional["ttnn.Tensor"] = None,
        sin_meta: Optional["ttnn.Tensor"] = None,
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
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")

        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

        self.attention = GemmaAttentionTTNN(
            config, weights, layer_idx, device, cos_meta, sin_meta
        )
        self.mlp = GemmaMLPTTNN(config, weights, device)
    
    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        cos: "ttnn.Tensor",
        sin: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_value: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
        use_cache: bool = False,
    ) -> Tuple["ttnn.Tensor", Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """
        Forward pass using TTNN operations.
        
        Args:
            hidden_states: TTNN input tensor
            cos, sin: RoPE embeddings
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
        
        # Pre-MLP norm
        normed = rms_norm_ttnn(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        
        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        hidden_states = ttnn.add(hidden_states, mlp_output)
        
        return hidden_states, new_cache


# Default exports - Use TTNN when available for performance
if TTNN_AVAILABLE:
    GemmaAttention = GemmaAttentionTTNN
    GemmaMLP = GemmaMLPTTNN
    GemmaBlock = GemmaBlockTTNN
else:
    GemmaAttention = GemmaAttentionTorch
    GemmaMLP = GemmaMLPTorch
    GemmaBlock = GemmaBlockTorch

