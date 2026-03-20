# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Decoder layer implementation for Qwen3-TTS.

Supports both prefill mode (full sequence) and decode mode (single token with KV cache).
"""

from typing import Optional, Tuple

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.attention import Attention
from models.demos.qwen3_tts.tt.mlp import MLP
from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm


class DecoderLayer(LightweightModule):
    """
    Qwen3-TTS decoder layer.

    Architecture:
        x = x + attention(norm(x))
        x = x + mlp(norm(x))

    This is a simplified implementation for single device (N150/N300).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        state_dict: dict,
        layer_idx: int,
        layer_prefix: str,
        rms_norm_eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx

        full_prefix = f"{layer_prefix}.layers.{layer_idx}"

        # Input layernorm (pre-attention)
        self.input_layernorm = RMSNorm(
            device=device,
            dim=hidden_size,
            state_dict=state_dict,
            weight_key=f"{full_prefix}.input_layernorm.weight",
            eps=rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
            weight_cache_path=weight_cache_path,
        )

        # Self-attention
        self.attention = Attention(
            device=device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            state_dict=state_dict,
            layer_prefix=full_prefix,
            rms_norm_eps=rms_norm_eps,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
        )

        # Post-attention layernorm (pre-MLP)
        self.post_attention_layernorm = RMSNorm(
            device=device,
            dim=hidden_size,
            state_dict=state_dict,
            weight_key=f"{full_prefix}.post_attention_layernorm.weight",
            eps=rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
            weight_cache_path=weight_cache_path,
        )

        # MLP
        self.mlp = MLP(
            device=device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            state_dict=state_dict,
            layer_prefix=full_prefix,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_attn_mask: Optional[ttnn.Tensor] = None,
        cp_prefill_mask: Optional[ttnn.Tensor] = None,
        prefill_attn_mask: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Apply decoder layer.

        Supports both prefill (full sequence) and decode (single token) modes.

        Args:
            x: Input tensor of shape [batch, 1, seq_len, hidden_size]
            cos: Cosine frequencies for RoPE
            sin: Sine frequencies for RoPE
            transformation_mat: Transformation matrix for RoPE
            attention_mask: Optional attention mask
            kv_cache: Optional tuple of (k_cache, v_cache) for this layer
            start_pos: Starting position in sequence (for KV cache, non-trace path)
            mode: "prefill" for full sequence or "decode" for single token
            cur_pos_tensor: Optional int32 device tensor [1] for trace-compatible decode
            decode_attn_mask: Optional float32 device tensor [1,1,1,max_seq] for decode
            cp_prefill_mask: Optional float32 device tensor [1,1,seq,max_seq] for
                trace-compatible CP prefill (writes cache at constant positions 0,1)
            prefill_attn_mask: Optional float32 device tensor [1,heads,padded_seq,max_seq]
                for trace-compatible Talker prefill (writes full K/V at position 0)

        Returns:
            Tuple of (output, updated_kv_cache) where:
            - output: tensor of shape [batch, 1, seq_len, hidden_size]
            - updated_kv_cache: tuple of (k_cache, v_cache) or None
        """
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x, updated_kv_cache = self.attention(
            x,
            cos,
            sin,
            transformation_mat,
            attention_mask,
            kv_cache=kv_cache,
            start_pos=start_pos,
            mode=mode,
            cur_pos_tensor=cur_pos_tensor,
            decode_attn_mask=decode_attn_mask,
            cp_prefill_mask=cp_prefill_mask,
            prefill_attn_mask=prefill_attn_mask,
        )
        x = ttnn.add(residual, x)

        # Pre-norm MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = ttnn.add(residual, x)

        return x, updated_kv_cache
