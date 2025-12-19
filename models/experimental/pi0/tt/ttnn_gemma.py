# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gemma transformer blocks - TTNN Implementation.

This module implements Gemma 2B style transformer layers using TTNN operations:
    - RMSNorm (pre-normalization)
    - Multi-Query Attention (MQA) with num_kv_heads=1
    - GeGLU MLP (gated GELU activation)
    - Rotary Position Embeddings (RoPE)

Architecture configurations:
    - Gemma 2B (VLM): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1
    - Gemma 300M (Expert): width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1
"""

import math
from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import GemmaConfig


# ============================================================================
# RMSNorm (TTNN)
# ============================================================================

def rms_norm(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    eps: float = 1e-6,
) -> ttnn.Tensor:
    """
    RMSNorm using TTNN operations.
    
    Args:
        x: TTNN tensor (batch_size, seq_len, hidden_dim)
        weight: TTNN weight tensor (1, hidden_dim)
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized TTNN tensor
    """
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
# Multi-Query Attention (TTNN)
# ============================================================================

class TtGemmaAttention:
    """
    Gemma Multi-Query Attention using TTNN operations.
    
    Leverages TTNN's optimized attention kernels.
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
    ):
        """
        Initialize attention layer with TTNN weights.
        
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNN device
        """
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
            seq_len_x = x.shape[2]
            cos_t = cos_t[:, :, :seq_len_x, :]
            sin_t = sin_t[:, :, :seq_len_x, :]
            
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
# GeGLU MLP (TTNN)
# ============================================================================

class TtGemmaMLP:
    """
    Gemma MLP with GeGLU activation using TTNN.
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize MLP with TTNN weights.
        
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            device: TTNN device
        """
        self.config = config
        self.device = device
        self.gate_proj = weights["mlp.gate_proj.weight"]
        self.up_proj = weights["mlp.up_proj.weight"]
        self.down_proj = weights["mlp.down_proj.weight"]
    
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
# Full Transformer Block (TTNN)
# ============================================================================

class TtGemmaBlock:
    """
    Complete Gemma transformer block using TTNN.
    """
    
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
    ):
        """
        Initialize transformer block with TTNN weights.
        
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNN device
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]
        
        self.attention = TtGemmaAttention(config, weights, layer_idx, device)
        self.mlp = TtGemmaMLP(config, weights, device)
    
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
            cos, sin: RoPE embeddings
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache
        
        Returns:
            Tuple of (output, optional_cache)
        """
        # Pre-attention norm
        normed = rms_norm(
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
        normed = rms_norm(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        
        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        hidden_states = ttnn.add(hidden_states, mlp_output)
        
        return hidden_states, new_cache
