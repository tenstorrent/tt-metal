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
    RMSNorm using TTNN operations.
    
    Args:
        x: TTNN tensor (batch_size, seq_len, hidden_dim)
        weight: TTNN weight tensor (1, hidden_dim)
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized TTNN tensor
    """
    if not TTNN_AVAILABLE:
        raise RuntimeError("TTNN not available")
    
    # Compute x^2
    x_squared = ttnn.pow(x, 2)
    
    # Mean across last dimension
    variance = ttnn.mean(x_squared, dim=-1, keepdim=True)
    
    # rsqrt(variance + eps)
    eps_tensor = ttnn.full_like(variance, eps)
    variance_eps = ttnn.add(variance, eps_tensor)
    rsqrt_var = ttnn.rsqrt(variance_eps)
    
    # Normalize
    x_normalized = ttnn.multiply(x, rsqrt_var)
    
    # Apply weight + 1
    ones = ttnn.ones_like(weight)
    weight_plus_one = ttnn.add(weight, ones)
    
    return ttnn.multiply(x_normalized, weight_plus_one)


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
    
    Leverages TTNN's optimized attention kernels.
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, "ttnn.Tensor"],
        layer_idx: int,
        device: "ttnn.Device",
    ):
        """
        Initialize attention layer with TTNN weights.
        
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        
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
            hidden_states: TTNN tensor (batch, seq_len, hidden_dim)
            cos, sin: RoPE embeddings
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache
        
        Returns:
            Tuple of (output, optional_cache)
        """
        # Project Q, K, V
        q = ttnn.linear(hidden_states, self.q_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.linear(hidden_states, self.k_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.linear(hidden_states, self.v_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        # Reshape and split heads using TTNN experimental ops
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        q = ttnn.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        
        # Transpose: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        
        # Apply RoPE using hybrid approach (convert to torch, apply RoPE, convert back)
        # This handles the complex broadcasting required for rotary embeddings
        q_torch = ttnn.to_torch(q)
        k_torch = ttnn.to_torch(k)
        cos_torch = ttnn.to_torch(cos) if hasattr(cos, 'shape') else cos
        sin_torch = ttnn.to_torch(sin) if hasattr(sin, 'shape') else sin
        
        # Apply rotary embeddings in torch
        def apply_rope_torch(x, cos_t, sin_t):
            # x: (batch, heads, seq, dim)
            # cos_t, sin_t: (seq, dim/2) - need to repeat for full dim
            if len(cos_t.shape) == 2:
                cos_t = cos_t.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim/2)
                sin_t = sin_t.unsqueeze(0).unsqueeze(0)
            elif len(cos_t.shape) == 3:
                cos_t = cos_t.unsqueeze(0)  # (1, 1, seq, dim/2)
                sin_t = sin_t.unsqueeze(0)
            
            # Slice to match sequence length
            seq_len = x.shape[2]
            cos_t = cos_t[:, :, :seq_len, :]
            sin_t = sin_t[:, :, :seq_len, :]
            
            # Repeat cos/sin to match full head_dim
            cos_t = torch.cat([cos_t, cos_t], dim=-1)  # (1, 1, seq, dim)
            sin_t = torch.cat([sin_t, sin_t], dim=-1)  # (1, 1, seq, dim)
            
            # Split x into two halves for rotation
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            
            # Rotate
            rotated = torch.cat([-x2, x1], dim=-1)
            
            # Apply RoPE
            return x * cos_t + rotated * sin_t
        
        q_rope_torch = apply_rope_torch(q_torch, cos_torch, sin_torch)
        k_rope_torch = apply_rope_torch(k_torch, cos_torch, sin_torch)
        
        # Convert back to TTNN
        q_rope = ttnn.from_torch(
            q_rope_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        k_rope = ttnn.from_torch(
            k_rope_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        
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
        
        # Transpose back and reshape
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, -1))
        
        # Output projection
        output = ttnn.linear(attn_output, self.o_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        
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
        gate = ttnn.linear(x, self.gate_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        up = ttnn.linear(x, self.up_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        # GELU activation
        gate_activated = ttnn.gelu(gate)
        
        # Element-wise multiply
        hidden = ttnn.multiply(gate_activated, up)
        
        # Down projection
        return ttnn.linear(hidden, self.down_proj, memory_config=ttnn.L1_MEMORY_CONFIG)


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
    ):
        """
        Initialize transformer block with TTNN weights.
        
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]
        
        self.attention = GemmaAttentionTTNN(config, weights, layer_idx, device)
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

