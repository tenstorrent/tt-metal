# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reference CPU implementation of DeepSeek V3 Multi-Latent Attention (MLA) module.
This module can run with both downloaded weights from HuggingFace or random weights for testing.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    apply_rotary_pos_emb,
)


class MLAReference(nn.Module):
    """
    Reference CPU implementation of Multi-Latent Attention (MLA) for DeepSeek V3.

    This is a wrapper around the DeepseekV3Attention module that provides:
    - Easy initialization with random or pretrained weights
    - Simplified interface for testing
    - Support for both prefill and decode modes

    Args:
        config: DeepseekV3Config with model configuration
        layer_idx: Index of the layer (used for caching)
    """

    def __init__(self, config: DeepseekV3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Initialize the attention module for its weights and sub-modules (projections, rope, layernorms).
        # forward() is overridden below to use memory-efficient F.scaled_dot_product_attention.
        self.attention = DeepseekV3Attention(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[any]]:
        """
        Memory-efficient forward pass using F.scaled_dot_product_attention.

        Uses flash attention under the hood to avoid materializing the full [seq, seq]
        attention matrix, enabling runs at full 128k sequence length on CPU.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs of shape [batch_size, seq_len]
            past_key_value: Cache for past key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use caching

        Returns:
            Tuple of (output_tensor, attention_weights, updated_cache)
        """
        attn = self.attention
        bsz, q_len, _ = hidden_states.size()

        # Q projection
        if attn.q_lora_rank is None:
            q = attn.q_proj(hidden_states)
        else:
            q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

        # Absorbed Q: q_nope projected into latent space
        kv_b1_proj = attn.kv_b_proj.weight.view(attn.num_heads, -1, attn.kv_lora_rank)[:, : attn.qk_nope_head_dim]
        q_nope = torch.matmul(q_nope, kv_b1_proj)

        # KV projection
        compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, 1, q_len, attn.qk_rope_head_dim)
        k_nope = attn.kv_a_layernorm(compressed_kv).view(bsz, 1, q_len, attn.kv_lora_rank)

        # RoPE
        kv_seq_len = k_nope.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(attn.layer_idx)
        cos, sin = attn.rotary_emb(k_nope, seq_len=kv_seq_len, meta_style=True)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, meta_style=True)

        # Assemble query_states and key_states (KVPE)
        query_states = k_pe.new_empty(bsz, attn.num_heads, q_len, attn.kv_lora_rank + attn.qk_rope_head_dim)
        query_states[:, :, :, : attn.kv_lora_rank] = q_nope
        query_states[:, :, :, attn.kv_lora_rank :] = q_pe

        key_states = k_pe.new_empty(bsz, 1, q_len, attn.kv_lora_rank + attn.qk_rope_head_dim)
        key_states[:, :, :, : attn.kv_lora_rank] = k_nope
        key_states[:, :, :, attn.kv_lora_rank :] = k_pe

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, _ = past_key_value.update(
                key_states, torch.empty((*key_states.shape[:-1], 0)), attn.layer_idx, cache_kwargs
            )

        # V is just the latent part of K
        value_states = key_states[:, :, :, : attn.kv_lora_rank]

        # Memory-efficient chunked attention for CPU
        # On CPU, F.scaled_dot_product_attention uses the math kernel which materializes
        # the full [heads, seq, seq] attention matrix. To avoid OOM at long sequence lengths,
        # we chunk along both the query sequence and head dimensions.
        SEQ_CHUNK = 4096
        HEAD_CHUNK = 16

        if q_len <= SEQ_CHUNK and attn.num_heads <= HEAD_CHUNK:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states.expand(bsz, attn.num_heads, -1, -1),
                value_states.expand(bsz, attn.num_heads, -1, -1),
                is_causal=True,
                scale=attn.softmax_scale,
            )
        else:
            attn_output = torch.empty_like(query_states[:, :, :, : attn.kv_lora_rank])
            for h_start in range(0, attn.num_heads, HEAD_CHUNK):
                h_end = min(h_start + HEAD_CHUNK, attn.num_heads)
                q_heads = query_states[:, h_start:h_end, :, :]
                k_heads = key_states.expand(bsz, h_end - h_start, -1, -1)
                v_heads = value_states.expand(bsz, h_end - h_start, -1, -1)
                for seq_start in range(0, q_len, SEQ_CHUNK):
                    seq_end = min(seq_start + SEQ_CHUNK, q_len)
                    q_chunk = q_heads[:, :, seq_start:seq_end, :]
                    k_chunk = k_heads[:, :, :seq_end, :]
                    v_chunk = v_heads[:, :, :seq_end, :]
                    # Causal mask: Q at absolute position (seq_start + i) attends to K positions 0..(seq_start + i)
                    q_pos = torch.arange(seq_start, seq_end, device=q_chunk.device).unsqueeze(1)
                    k_pos = torch.arange(0, seq_end, device=q_chunk.device).unsqueeze(0)
                    causal_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                    attn_output[:, h_start:h_end, seq_start:seq_end, :] = F.scaled_dot_product_attention(
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        attn_mask=causal_mask,
                        scale=attn.softmax_scale,
                    )

        # KV b2 projection (V head expansion)
        kv_b2_proj = attn.kv_b_proj.weight.view(attn.num_heads, -1, attn.kv_lora_rank)[:, -attn.v_head_dim :].transpose(
            1, 2
        )
        attn_output = torch.matmul(attn_output, kv_b2_proj)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, attn.num_heads * attn.v_head_dim)
        attn_output = attn.o_proj(attn_output)

        return attn_output, None, past_key_value

    @classmethod
    def from_pretrained(
        cls,
        config: DeepseekV3Config,
        state_dict: dict[str, torch.Tensor],
        layer_idx: int = 0,
        module_path: str = "model.layers.0.self_attn",
    ) -> "MLAReference":
        """
        Create MLA module from pretrained weights.

        Args:
            config: Model configuration
            state_dict: State dict containing pretrained weights
            layer_idx: Layer index (default: 0)
            module_path: Path prefix in state dict (default: "model.layers.0.self_attn")

        Returns:
            MLAReference module with loaded weights
        """
        module = cls(config, layer_idx=layer_idx)

        # Extract module-specific state dict
        if module_path:
            prefix = module_path + "."
            module_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        else:
            module_state_dict = state_dict

        # Load weights
        module.attention.load_state_dict(module_state_dict)
        return module

    @classmethod
    def from_random(
        cls,
        config: DeepseekV3Config,
        layer_idx: int = 0,
        seed: int = 42,
    ) -> "MLAReference":
        """
        Create MLA module with random weights for testing.

        Args:
            config: Model configuration
            layer_idx: Layer index (default: 0)
            seed: Random seed for reproducibility (default: 42)

        Returns:
            MLAReference module with random weights
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Create module (weights are randomly initialized)
        module = cls(config, layer_idx=layer_idx)

        # Initialize weights using standard initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, DeepseekV3RMSNorm):
                if hasattr(m, "weight"):
                    nn.init.ones_(m.weight)

        module.apply(init_weights)
        return module


def create_mla_reference(
    config: DeepseekV3Config,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    layer_idx: int = 0,
    module_path: str = "model.layers.0.self_attn",
) -> MLAReference:
    """
    Convenient factory function to create MLA reference module.

    Args:
        config: Model configuration
        state_dict: State dict with weights (passed in, either pretrained weights or random weights)
        layer_idx: Layer index (default: 0)
        module_path: Path to module in state dict (default: "model.layers.0.self_attn")

    Returns:
        MLAReference module
    """

    if state_dict is None:
        raise ValueError("state_dict must be provided")
    return MLAReference.from_pretrained(config, state_dict, layer_idx, module_path)
