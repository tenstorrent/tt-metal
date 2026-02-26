# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention mechanism implementations for TTNN."""

from dataclasses import dataclass
import math
from typing import Optional, Union

import torch

try:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
except ImportError:
    print("Could not import sdpa_attention_forward from transformers.integrations.sdpa_attention. ")

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
)
from models.experimental.tt_symbiote.modules.rope import (
    TTNNRotaryPositionEmbedding,
)
from models.experimental.tt_symbiote.modules.normalization import (
    TTNNRMSNorm,
)

import os


class TorchSDPAAttention(torch.nn.Module):
    def forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        transpose_output: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        attn_output = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )[0]
        if not transpose_output:  # revert the transpose in sdpa_attention_forward
            attn_output = attn_output.transpose(1, 2)
        return attn_output


class TTNNSDPAAttention(TTNNModule):
    def __init__(self):
        super().__init__()
        self._fallback_torch_layer = TorchSDPAAttention()
        self.program_config = None
        self.compute_kernel_config = None
        self.memory_config = None

    def forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        transpose_output: bool = True,
        **kwargs,
    ) -> ttnn.Tensor:
        if query.layout != ttnn.TILE_LAYOUT:
            query = ttnn.to_layout(query, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if key.layout != ttnn.TILE_LAYOUT:
            key = ttnn.to_layout(key, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if value.layout != ttnn.TILE_LAYOUT:
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        assert len(query.shape) == 4, "Query tensor must be 4D"
        assert dropout == 0.0, "TTNNSDPAAttention does not support dropout"
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
        if attention_mask is not None:
            if attention_mask.layout != ttnn.TILE_LAYOUT:
                attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if attention_mask.dtype != query.dtype:
                attention_mask = ttnn.typecast(attention_mask, query.dtype)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scaling,
            program_config=self.program_config,
            attn_mask=attention_mask,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
        )
        if transpose_output:
            attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        return attn_output


class PytorchFusedQKVSelfAttention(torch.nn.Module):
    def __init__(self, linear1, linear2, linear3, num_attention_heads, hidden_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = hidden_size
        self.query = linear1
        self.key = linear2
        self.value = linear3

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        return query_layer, key_layer, value_layer


@dataclass
class SelfAttentionConfig:
    """Configuration dataclass for Self-Attention."""

    hidden_size: int = 768
    num_attention_heads: int = 12


class SelfAttention(torch.nn.Module):
    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False
        self.fused_qkv = PytorchFusedQKVSelfAttention(
            torch.nn.Linear(config.hidden_size, self.all_head_size, bias=True),
            torch.nn.Linear(config.hidden_size, self.all_head_size, bias=True),
            torch.nn.Linear(config.hidden_size, self.all_head_size, bias=True),
            num_attention_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
        )
        self.sdpa = TorchSDPAAttention()

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        query_layer, key_layer, value_layer = self.fused_qkv(hidden_states)
        context_layer = self.sdpa(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        if output_attentions:
            raise NotImplementedError("output_attentions is not implemented in this SelfAttention module.")

        return (context_layer,)


class TTNNFusedQKVSelfAttention(TTNNModule):
    @classmethod
    def from_torch(cls, fused_qkv: "PytorchFusedQKVSelfAttention"):
        """Create TTNNViTSelfAttention from PyTorch ViTSelfAttention."""
        new_fused_qkv = cls()
        new_fused_qkv._fallback_torch_layer = fused_qkv
        new_fused_qkv.num_attention_heads = fused_qkv.num_attention_heads
        new_fused_qkv.hidden_size = fused_qkv.hidden_size
        num_heads = fused_qkv.num_attention_heads
        hidden_size = fused_qkv.hidden_size
        head_size = hidden_size // num_heads
        hidden_size = num_heads * head_size * 3
        qkv_weight = torch.cat(
            [
                fused_qkv.query.weight,
                fused_qkv.key.weight,
                fused_qkv.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [
                fused_qkv.query.bias,
                fused_qkv.key.bias,
                fused_qkv.value.bias,
            ],
            dim=0,
        )
        torch_layer_query_key_value = torch.nn.Linear(
            in_features=fused_qkv.query.in_features,
            out_features=hidden_size,
            bias=True,
        )
        torch_layer_query_key_value.weight = torch.nn.Parameter(qkv_weight)
        torch_layer_query_key_value.bias = torch.nn.Parameter(qkv_bias)
        new_fused_qkv.linear = TTNNLinear.from_torch(torch_layer_query_key_value)
        return new_fused_qkv

    def forward(self, hidden_states):
        """Forward pass through fused QKV linear layer."""
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.unsqueeze(hidden_states, 1)
        query_key_value = self.linear(hidden_states).ttnn_tensor
        query_key_value = ttnn.to_memory_config(query_key_value, ttnn.L1_MEMORY_CONFIG)
        queries, keys, values = ttnn.experimental.nlp_create_qkv_heads(
            query_key_value,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            transpose_k_heads=False,
        )
        ttnn.deallocate(query_key_value)
        return queries, keys, values


class TTNNSelfAttention(TTNNModule):
    """TTNN-accelerated ViT Self-Attention layer."""

    def __init__(self, attention_config: SelfAttentionConfig) -> None:
        super().__init__()

        self.num_attention_heads = attention_config.num_attention_heads
        self.hidden_size = attention_config.hidden_size
        self.attention_head_size = int(attention_config.hidden_size / attention_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.core_grid = ttnn.CoreGrid(y=8, x=8)
        self.is_causal = False
        self.should_reallocate_in_attention = False

    @classmethod
    def from_torch(cls, self_attention: "SelfAttention"):
        """Create TTNNViTSelfAttention from PyTorch ViTSelfAttention."""
        new_self_attention = cls(
            attention_config=self_attention.config,
        )
        new_self_attention._fallback_torch_layer = self_attention
        new_self_attention.query_key_value = TTNNFusedQKVSelfAttention.from_torch(
            PytorchFusedQKVSelfAttention(
                self_attention.fused_qkv.query,
                self_attention.fused_qkv.key,
                self_attention.fused_qkv.value,
                self_attention.num_attention_heads,
                self_attention.config.hidden_size,
            ),
        )
        new_self_attention.sdpa = TTNNSDPAAttention()
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(new_self_attention.core_grid.x, new_self_attention.core_grid.y),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        new_self_attention.sdpa.program_config = program_config
        new_self_attention.sdpa.compute_kernel_config = compute_kernel_config
        return new_self_attention

    def forward(self, hidden_states, head_mask=None, output_attentions: bool = False):
        """Forward pass through ViT self-attention."""
        assert head_mask is None, "head_mask is not supported in TTNNViTSelfAttention"
        assert not output_attentions, "output_attentions is not supported in TTNNViTSelfAttention"
        original_dtype = hidden_states.dtype
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        query, key, value = self.query_key_value(hidden_states)

        ttnn.deallocate(hidden_states)
        if self.should_reallocate_in_attention:
            value = ttnn.reallocate(value.to_ttnn)

        context_layer = self.sdpa(
            self,
            query,
            key,
            value,
            attention_mask=head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            transpose_output=False,
        )
        context_layer = ttnn.experimental.nlp_concat_heads(context_layer.to_ttnn)
        # context_layer = ttnn.typecast(context_layer, original_dtype)
        context_layer = ttnn.typecast(context_layer, original_dtype)
        context_layer = ttnn.squeeze(context_layer, 1)
        return (context_layer,)


class TTNNViTSelfAttention(TTNNSelfAttention):
    """TTNN-accelerated ViT Self-Attention layer."""

    @classmethod
    def from_torch(cls, self_attention: "ViTSelfAttention"):
        """Create TTNNViTSelfAttention from PyTorch ViTSelfAttention."""
        new_self_attention = cls(
            attention_config=self_attention.config,
        )
        new_self_attention._fallback_torch_layer = self_attention
        new_self_attention.query_key_value = TTNNFusedQKVSelfAttention.from_torch(
            PytorchFusedQKVSelfAttention(
                self_attention.query,
                self_attention.key,
                self_attention.value,
                self_attention.num_attention_heads,
                self_attention.config.hidden_size,
            ),
        )
        new_self_attention.sdpa = TTNNSDPAAttention()
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(new_self_attention.core_grid.x, new_self_attention.core_grid.y),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        new_self_attention.sdpa.program_config = program_config
        new_self_attention.sdpa.compute_kernel_config = compute_kernel_config
        return new_self_attention


class TTNNWhisperAttention(TTNNModule):
    """Minimal TTNN Whisper Attention with KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        self.dropout = dropout

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.sdpa = TTNNSDPAAttention()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)
        self.sdpa.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        self.sdpa.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, whisper_attn: "WhisperAttention"):
        new_attn = cls(
            embed_dim=whisper_attn.embed_dim,
            num_heads=whisper_attn.num_heads,
            dropout=whisper_attn.dropout,
            is_causal=whisper_attn.is_causal,
            layer_idx=whisper_attn.layer_idx,
        )
        new_attn._fallback_torch_layer = whisper_attn

        # Fuse Q/K/V for self-attention (zero-pad K bias)
        qkv_weight = torch.cat([whisper_attn.q_proj.weight, whisper_attn.k_proj.weight, whisper_attn.v_proj.weight])
        qkv_bias = torch.cat(
            [whisper_attn.q_proj.bias, torch.zeros_like(whisper_attn.q_proj.bias), whisper_attn.v_proj.bias]
        )
        fused_qkv = torch.nn.Linear(whisper_attn.embed_dim, whisper_attn.embed_dim * 3, bias=True)
        fused_qkv.weight = torch.nn.Parameter(qkv_weight)
        fused_qkv.bias = torch.nn.Parameter(qkv_bias)
        new_attn.qkv_proj = TTNNLinear.from_torch(fused_qkv)
        new_attn.q_proj_ttnn = TTNNLinear.from_torch(whisper_attn.q_proj)
        # Separate K/V for cross-attention
        new_attn.k_proj_cross = TTNNLinear.from_torch(whisper_attn.k_proj)
        new_attn.v_proj_cross = TTNNLinear.from_torch(whisper_attn.v_proj)
        new_attn.out_proj = TTNNLinear.from_torch(whisper_attn.out_proj)

        return new_attn

    def _reshape_heads(self, x: ttnn.Tensor, seq_len: int, bsz: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (bsz, seq_len, self.num_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        key_value_states: Optional[ttnn.Tensor] = None,
        past_key_value=None,
        attention_mask=None,
        **kwargs,
    ):
        is_cross = key_value_states is not None
        bsz, tgt_len = hidden_states.shape[0], hidden_states.shape[1]

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Cache logic
        cache = None
        is_updated = False
        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx, False)
            cache = past_key_value.cross_attention_cache if is_cross else past_key_value.self_attention_cache
            if is_cross:
                past_key_value.is_updated[self.layer_idx] = True

        # Q/K/V projection
        if is_cross:
            # Cross-attention: extract Q from fused weights
            query = self.q_proj_ttnn(hidden_states)
            query = ttnn.multiply(query.to_ttnn, self.scaling)
            query = self._reshape_heads(query, tgt_len, bsz)

            if cache and is_updated:
                key, value = cache.key_cache[self.layer_idx], cache.value_cache[self.layer_idx]
            else:
                if key_value_states.layout != ttnn.TILE_LAYOUT:
                    key_value_states = ttnn.to_layout(
                        key_value_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                key = self.k_proj_cross(key_value_states).to_ttnn
                value = self.v_proj_cross(key_value_states).to_ttnn
                src_len = key.shape[1]
                key = self._reshape_heads(key, src_len, bsz)
                value = self._reshape_heads(value, src_len, bsz)
                if cache is not None:
                    key, value = cache.update(
                        TorchTTNNTensor(key), TorchTTNNTensor(value), self.layer_idx, {"cache_position": None}
                    )
        else:
            # Self-attention: fused QKV
            hidden_states = ttnn.unsqueeze(hidden_states, 1)
            query_key_value = self.qkv_proj(hidden_states).ttnn_tensor
            query_key_value = ttnn.to_memory_config(query_key_value, ttnn.L1_MEMORY_CONFIG)
            query, key, value = ttnn.experimental.nlp_create_qkv_heads(
                query_key_value,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                transpose_k_heads=False,
            )

            ttnn.deallocate(query_key_value)
            query = ttnn.multiply(query, self.scaling)
            if cache is not None:
                key, value = cache.update(
                    TorchTTNNTensor(key),
                    TorchTTNNTensor(value),
                    self.layer_idx,
                    {"cache_position": kwargs.get("cache_position")},
                )

        # SDPA with query padding for KV cache
        # print(f"Shape of query before SDPA: {query.shape}")  # --- IGNORE ---
        # print(f"Shape of key before SDPA: {key.shape}")  # --- IGNORE ---
        # print(f"Shape of value before SDPA: {value.shape}")  # --- IGNORE ---

        # Pad query if needed for causal SDPA with cache
        original_q_len = query.shape[2]
        kv_len = key.shape[2]
        use_causal = self.is_causal and not is_cross

        if use_causal and original_q_len < kv_len:
            # Pad query: [B, H, q_len, D] -> [B, H, kv_len, D]
            pad_len = kv_len - original_q_len
            # Create zero padding on device
            pad_shape = (query.shape[0], query.shape[1], pad_len, query.shape[3])
            zero_pad = ttnn.zeros(
                pad_shape,
                device=query.device(),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=query.dtype,
            )
            query = ttnn.concat([zero_pad, query], dim=2)

        attn_out = self.sdpa(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=1.0,
            is_causal=self.is_causal and not is_cross,
            transpose_output=True,
        )

        # Slice output if query was padded
        if use_causal and original_q_len < kv_len:
            # Slice: [B, kv_len, H, D] -> [B, q_len, H, D]
            attn_out = attn_out[:, -original_q_len:, :, :]

        attn_out = ttnn.reshape(attn_out.to_ttnn, (bsz, tgt_len, self.embed_dim))
        return self.out_proj(attn_out), None, past_key_value


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(TTNNModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
    ):
        super().__init__()
        self.sdpa = TTNNSDPAAttention()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)
        self.sdpa.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        self.sdpa.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, llama_attn: "LlamaAttention"):
        new_attn = cls()
        new_attn._fallback_torch_layer = llama_attn

        # Fuse Q/K/V for self-attention (zero-pad K bias)
        qkv_weight = torch.cat([llama_attn.q_proj.weight, llama_attn.k_proj.weight, llama_attn.v_proj.weight])
        assert not llama_attn.k_proj.bias, "LlamaAttention k_proj bias is expected to be None"
        assert not llama_attn.v_proj.bias, "LlamaAttention v_proj bias is expected to be None"
        assert not llama_attn.q_proj.bias, "LlamaAttention q_proj bias is expected to be None"
        fused_qkv = torch.nn.Linear(llama_attn.hidden_size, llama_attn.hidden_size * 3, bias=False)
        fused_qkv.weight = torch.nn.Parameter(qkv_weight)
        new_attn.qkv_proj = TTNNLinear.from_torch(fused_qkv)
        new_attn.o_proj = TTNNLinear.from_torch(llama_attn.o_proj)
        return new_attn

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,  # will become mandatory in v4.46
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        # Self-attention: fused QKV
        hidden_states = ttnn.unsqueeze(hidden_states, 1)
        query_key_value = self.qkv_proj(hidden_states).ttnn_tensor
        query_key_value = ttnn.to_memory_config(query_key_value, ttnn.L1_MEMORY_CONFIG)
        query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
            query_key_value,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.torch_layer.num_heads,
            num_kv_heads=self.torch_layer.num_key_value_heads,
            transpose_k_heads=False,
        )
        value_states = TorchTTNNTensor(value_states)

        if position_embeddings is None:
            cos, sin = self.torch_layer.rotary_emb(value_states.to_torch, TorchTTNNTensor(position_ids).to_torch)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            TorchTTNNTensor(query_states), TorchTTNNTensor(key_states), cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.torch_layer.layer_idx, cache_kwargs
            )

        query_states = ttnn.to_device(query_states.to_ttnn, device=hidden_states.device())
        query_states = ttnn.to_layout(query_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query_states = ttnn.multiply(query_states, math.sqrt(1 / self.torch_layer.head_dim))
        original_q_len = query_states.shape[2]
        kv_len = key_states.shape[2]

        if self.torch_layer.is_causal and original_q_len < kv_len:
            # Pad query: [B, H, q_len, D] -> [B, H, kv_len, D]
            pad_len = kv_len - original_q_len
            # Create zero padding on device
            pad_shape = (query_states.shape[0], query_states.shape[1], pad_len, query_states.shape[3])
            zero_pad = ttnn.zeros(
                pad_shape,
                device=query_states.device(),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=query_states.dtype,
            )
            query_states = ttnn.concat([zero_pad, query_states], dim=2)

        attn_out = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            None,
            dropout=0.0,
            scaling=1.0,
            is_causal=self.torch_layer.is_causal,
            transpose_output=True,
        )

        # Slice output if query was padded
        if self.torch_layer.is_causal and original_q_len < kv_len:
            # Slice: [B, kv_len, H, D] -> [B, q_len, H, D]
            attn_out = attn_out[:, -original_q_len:, :, :]

        attn_out = ttnn.reshape(attn_out.to_ttnn, (bsz, q_len, -1))
        return self.o_proj(attn_out), None, past_key_value


class TTNNGR00TSelfAttention(TTNNModule):
    """GR00T self-attention on TTNN: Q/K/V/O proj, optional Q/K RMSNorm and RoPE, SDPA with tile-aligned padding."""

    def __init__(self, config=None, torch_layer=None):
        super().__init__()
        self._torch_layer = torch_layer
        self._fallback_torch_layer = torch_layer

        if isinstance(torch_layer, TTNNGR00TSelfAttention):
            return

        if torch_layer is not None and hasattr(torch_layer, "heads") and hasattr(torch_layer, "inner_dim"):
            self.num_heads = torch_layer.heads
            dim_head = torch_layer.inner_dim // torch_layer.heads
            self.num_kv_heads = torch_layer.inner_kv_dim // dim_head
            self.hidden_size = torch_layer.inner_dim
        else:
            self.num_heads = getattr(torch_layer, "num_heads", getattr(config, "num_attention_heads", 16))
            self.num_kv_heads = getattr(
                torch_layer, "num_key_value_heads", getattr(config, "num_key_value_heads", self.num_heads)
            )
            self.hidden_size = getattr(config, "hidden_size", 1152)

        self.tt_q_proj = None
        self.tt_k_proj = None
        self.tt_v_proj = None
        self.tt_o_proj = None
        self.tt_q_norm = None
        self.tt_k_norm = None
        self.rope = TTNNRotaryPositionEmbedding()
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        if torch_layer is not None:
            self._map_weights(torch_layer)

    @classmethod
    def from_torch(cls, torch_layer, model_config=None):
        """Build from PyTorch attention module."""
        if isinstance(torch_layer, TTNNGR00TSelfAttention):
            return torch_layer
        config = getattr(torch_layer, "config", None)
        return cls(config=config, torch_layer=torch_layer)

    def _map_weights(self, torch_layer):
        """Map Q/K/V/O and optional Q/K RMSNorm from torch layer."""
        for name, m in torch_layer.named_children():
            lname = name.lower()
            if any(x in lname for x in ["q_proj", "query", "to_q"]) and hasattr(m, "weight"):
                self.tt_q_proj = TTNNLinear.from_torch(m)
            elif any(x in lname for x in ["k_proj", "key", "to_k"]) and hasattr(m, "weight"):
                self.tt_k_proj = TTNNLinear.from_torch(m)
            elif any(x in lname for x in ["v_proj", "value", "to_v"]) and hasattr(m, "weight"):
                self.tt_v_proj = TTNNLinear.from_torch(m)
            elif any(x in lname for x in ["o_proj", "out_proj", "to_out"]):
                out_m = m[0] if hasattr(m, "__getitem__") and len(m) > 0 and hasattr(m[0], "weight") else m
                if hasattr(out_m, "weight"):
                    self.tt_o_proj = TTNNLinear.from_torch(out_m)
            elif "q_norm" in lname and hasattr(m, "weight") and hasattr(m, "variance_epsilon"):
                self.tt_q_norm = TTNNRMSNorm.from_torch(m)
            elif "k_norm" in lname and hasattr(m, "weight") and hasattr(m, "variance_epsilon"):
                self.tt_k_norm = TTNNRMSNorm.from_torch(m)

    def _prepare_attention_mask_for_ttnn(
        self, attention_mask, batch_size, q_len, kv_len, q_pad, kv_pad, device, is_causal=False
    ):
        """Build additive mask for SDPA on device (encoder + optional causal)."""
        PAD_MASK_VALUE = -10000.0
        tt_additive = None

        if attention_mask is not None:
            raw = getattr(attention_mask, "elem", attention_mask)
            raw_tt = None
            mask_kv_len = None
            if isinstance(raw, ttnn.Tensor):
                raw_tt = raw
                raw_tt = ttnn.to_device(raw_tt, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if raw_tt.layout != ttnn.TILE_LAYOUT:
                    raw_tt = ttnn.to_layout(raw_tt, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if raw_tt.dtype != ttnn.bfloat16:
                    raw_tt = ttnn.typecast(raw_tt, ttnn.bfloat16)
                ndim = len(raw_tt.shape)
                if ndim == 3:
                    raw_tt = ttnn.squeeze(raw_tt, 1)
                if len(raw_tt.shape) == 2:
                    mask_kv_len = raw_tt.shape[1]
            elif isinstance(raw, torch.Tensor) and raw.dim() >= 2:
                if raw.dim() == 3:
                    raw = raw.squeeze(1)
                if raw.dim() == 2:
                    _, mask_kv_len = raw.shape
                    raw_tt = ttnn.from_torch(
                        raw.to(torch.bfloat16).contiguous(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
            if raw_tt is not None and mask_kv_len is not None:
                raw_tt = ttnn.unsqueeze(raw_tt, 1)
                raw_tt = ttnn.unsqueeze(raw_tt, 2)
                if mask_kv_len < kv_len:
                    raw_tt = ttnn.pad(raw_tt, [[0, 0], [0, 0], [0, 0], [0, kv_len - mask_kv_len]], value=0.0)
                elif mask_kv_len > kv_len:
                    raw_tt = ttnn.slice(raw_tt, [0, 0, 0, 0], [batch_size, 1, 1, kv_len])
                ones_tt = ttnn.ones(
                    (batch_size, 1, 1, kv_len),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                sub_tt = ttnn.subtract(ones_tt, raw_tt)
                ttnn.deallocate(ones_tt)
                ttnn.deallocate(raw_tt)
                clamped = ttnn.clamp(sub_tt, 0.0, 1.0)
                ttnn.deallocate(sub_tt)
                scaled = ttnn.multiply(clamped, PAD_MASK_VALUE)
                ttnn.deallocate(clamped)
                tt_additive = ttnn.repeat(scaled, ttnn.Shape((1, 1, q_len, 1)))
                ttnn.deallocate(scaled)
                if os.environ.get("TT_SYMBIOTE_DEBUG_ATTN_MASK"):
                    print(
                        f"[TTNNGR00TSelfAttention mask] encoder mask shape=({batch_size}, 1, {q_len}, {kv_len})",
                        flush=True,
                    )

        if is_causal:
            ones_qk = ttnn.ones(
                (q_len, kv_len),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tri = ttnn.tril(ones_qk, diagonal=0)
            upper_ones = ttnn.subtract(ones_qk, tri)
            causal_tt = ttnn.multiply(upper_ones, PAD_MASK_VALUE)
            ttnn.deallocate(ones_qk)
            ttnn.deallocate(tri)
            ttnn.deallocate(upper_ones)
            causal_tt = ttnn.unsqueeze(causal_tt, 0)
            causal_tt = ttnn.unsqueeze(causal_tt, 0)
            causal_tt = ttnn.repeat(causal_tt, ttnn.Shape((batch_size, 1, 1, 1)))
            if tt_additive is not None:
                combined = ttnn.add(tt_additive, causal_tt)
                ttnn.deallocate(tt_additive)
                ttnn.deallocate(causal_tt)
                tt_additive = combined
            else:
                tt_additive = causal_tt

        if q_pad > 0 or kv_pad > 0:
            if tt_additive is None:
                tt_additive = ttnn.zeros(
                    (batch_size, 1, q_len, kv_len),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            tt_additive = ttnn.pad(tt_additive, [[0, 0], [0, 0], [0, q_pad], [0, kv_pad]], value=PAD_MASK_VALUE)

        return tt_additive

    def _pcc(self, a, b):
        """Pearson correlation of two tensors (DIAG_ATTN_PCC)."""
        if not isinstance(a, torch.Tensor):
            a = a.to_torch if hasattr(a, "to_torch") else ttnn.to_torch(a)
        if not isinstance(b, torch.Tensor):
            b = b.to_torch if hasattr(b, "to_torch") else ttnn.to_torch(b)
        a, b = a.float().flatten(), b.float().flatten()
        if a.numel() != b.numel():
            return float("nan")
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    def _rms_norm_on_device(self, tt_tensor, tt_norm, device):
        """Apply Q/K RMSNorm on device."""
        t = tt_tensor.to_ttnn if hasattr(tt_tensor, "to_ttnn") else tt_tensor
        return tt_norm.forward(t)

    def _rope_torch_fallback(self, q_raw, k_raw, cos_torch, sin_torch, device):
        """RoPE on device with torch cos/sin."""
        q = q_raw.to_ttnn if hasattr(q_raw, "to_ttnn") else q_raw
        k = k_raw.to_ttnn if hasattr(k_raw, "to_ttnn") else k_raw
        cos_tt = ttnn.from_torch(
            cos_torch.to(torch.bfloat16).contiguous(), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        sin_tt = ttnn.from_torch(
            sin_torch.to(torch.bfloat16).contiguous(), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        return self.rope(q, k, cos_tt, sin_tt)

    def forward(self, hidden_states, *args, **kwargs):
        """Q/K/V proj, optional norm+RoPE, SDPA, O_proj."""
        if self.tt_q_proj is None:
            return hidden_states, None

        from models.experimental.tt_symbiote.core.utils import ensure_tile_layout

        h_raw = hidden_states.to_ttnn if hasattr(hidden_states, "to_ttnn") else hidden_states
        if isinstance(h_raw, ttnn.Tensor) and h_raw.layout != ttnn.TILE_LAYOUT:
            h_tile = ensure_tile_layout(h_raw)
            hidden_states = TorchTTNNTensor(h_tile) if hasattr(hidden_states, "to_ttnn") else h_tile

        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        attention_mask = kwargs.get("attention_mask", None)
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        if os.environ.get("TT_SYMBIOTE_DEBUG_ATTN_HEADS") and encoder_hidden_states is not None:
            _name = getattr(self, "module_name", "?")
            if "attn1" in str(_name):
                print(
                    f"[TTNNGR00TSelfAttention] {_name} num_heads={self.num_heads} num_kv_heads={self.num_kv_heads} GQA={self.num_kv_heads != self.num_heads}",
                    flush=True,
                )

        q_w = self.tt_q_proj(hidden_states)
        k_w = self.tt_k_proj(kv_src)
        v_w = self.tt_v_proj(kv_src)

        _env = os.environ.get("TT_SYMBIOTE_DIAG_ATTN_PCC")
        _tl = getattr(self, "_torch_layer", None)
        _enc = encoder_hidden_states is not None
        if _enc and _env:
            _name = getattr(self, "module_name", "?")
            if "action_head" in str(_name) and "attn1" in str(_name):
                print(f"[DIAG] {_name} _torch_layer={_tl is not None}", flush=True)
        if _env and _tl is not None and _enc:
            name = getattr(self, "module_name", "?")
            try:
                _dev = getattr(self, "device", None)
                if _dev is not None and hasattr(ttnn, "synchronize_device"):
                    try:
                        ttnn.synchronize_device(_dev)
                    except Exception:
                        pass

                def _to_torch(x):
                    if hasattr(x, "to_torch"):
                        v = getattr(x, "to_torch")
                        return v() if callable(v) else v
                    if hasattr(x, "elem") and x.elem is not None:
                        return x.elem
                    if hasattr(x, "storage"):
                        return ttnn.to_torch(x)
                    return x

                h = _to_torch(hidden_states)
                k = _to_torch(kv_src)
                if not (isinstance(h, torch.Tensor) and isinstance(k, torch.Tensor)):
                    print(f"[DIAG_ATTN_PCC] {name} skip: h={type(h).__name__} k={type(k).__name__}", flush=True)
                else:
                    with torch.no_grad():
                        tq = self._torch_layer.to_q(h)
                        tk = self._torch_layer.to_k(k)
                        tv = self._torch_layer.to_v(k)
                    pcc_q = self._pcc(q_w, tq)
                    pcc_k = self._pcc(k_w, tk)
                    pcc_v = self._pcc(v_w, tv)
                    print(
                        f"[DIAG_ATTN_PCC] {name} Q_proj={pcc_q:.4f} K_proj={pcc_k:.4f} V_proj={pcc_v:.4f}", flush=True
                    )
            except Exception as exc:
                print(f"[DIAG_ATTN_PCC] {name} ERROR: {exc}", flush=True)

        def prepare_heads_on_device(t, num_heads, apply_pad=True):
            """(B, seq, H) -> (B, num_heads, seq, d_head); optional pad to 32 for SDPA."""
            raw = t.to_ttnn if hasattr(t, "to_ttnn") else t
            if raw.layout != ttnn.TILE_LAYOUT:
                raw = ttnn.to_layout(raw, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            b, s, h = raw.shape
            d_head = h // num_heads

            t_reshaped = ttnn.reshape(raw, (b, s, num_heads, d_head))
            t_transposed = ttnn.transpose(t_reshaped, 1, 2)
            if t_transposed.layout != ttnn.TILE_LAYOUT:
                t_transposed = ttnn.to_layout(t_transposed, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            pad_s = (32 - (s % 32)) % 32
            pad_d = (32 - (d_head % 32)) % 32

            if apply_pad and (pad_s > 0 or pad_d > 0):
                padding_config = [[0, 0], [0, 0], [0, pad_s], [0, pad_d]]
                t_transposed = ttnn.pad(t_transposed, padding_config, value=0.0)

            return t_transposed, b, s, h, d_head, pad_s, pad_d

        is_self_attn = encoder_hidden_states is None
        if is_self_attn and (
            self.tt_q_norm is not None or self.tt_k_norm is not None or kwargs.get("position_embeddings")
        ):
            q_4d, b, q_len, h, d_head, q_pad_s, q_pad_d = prepare_heads_on_device(q_w, self.num_heads, apply_pad=False)
            k_4d, _, kv_len, _, _, kv_pad_s, kv_pad_d = prepare_heads_on_device(k_w, self.num_kv_heads, apply_pad=False)
            hw_dev = q_4d.device()
            if self.tt_q_norm is not None:
                q_4d = self._rms_norm_on_device(q_4d, self.tt_q_norm, hw_dev)
            if self.tt_k_norm is not None:
                k_4d = self._rms_norm_on_device(k_4d, self.tt_k_norm, hw_dev)
            pos_emb = kwargs.get("position_embeddings", None)
            if pos_emb is not None:
                cos, sin = pos_emb
                if os.environ.get("TT_SYMBIOTE_DIAG_LM_ATTN_STAGES"):
                    _n = getattr(self, "module_name", "?")
                    if "self_attn" in str(_n) and "language_model" in str(_n):
                        print(f"[DIAG_LM_ATTN] {_n} RoPE: position_embeddings provided, applying", flush=True)
                ct = (
                    cos
                    if isinstance(cos, torch.Tensor)
                    else (getattr(cos, "elem", None) if hasattr(cos, "elem") else None)
                )
                st = (
                    sin
                    if isinstance(sin, torch.Tensor)
                    else (getattr(sin, "elem", None) if hasattr(sin, "elem") else None)
                )
                if ct is not None and st is not None:
                    sq = q_4d.shape[2]
                    if isinstance(ct, ttnn.Tensor):
                        ct_t = ct
                    elif hasattr(ct, "to_ttnn"):
                        ct_t = ct.to_ttnn
                    else:
                        ct_t = ttnn.from_torch(
                            ct.to(torch.bfloat16).contiguous(),
                            device=hw_dev,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    if isinstance(st, ttnn.Tensor):
                        st_t = st
                    elif hasattr(st, "to_ttnn"):
                        st_t = st.to_ttnn
                    else:
                        st_t = ttnn.from_torch(
                            st.to(torch.bfloat16).contiguous(),
                            device=hw_dev,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    ct_t = ttnn.to_device(ct_t, hw_dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    st_t = ttnn.to_device(st_t, hw_dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    need_len = sq
                    ndim = len(ct_t.shape)
                    if ndim == 3:
                        ct_t = ttnn.unsqueeze(ct_t, 1)
                        st_t = ttnn.unsqueeze(st_t, 1)
                    ct_len = ct_t.shape[2]
                    if ct_len < need_len:
                        pad_len = need_len - ct_len
                        ct_t = ttnn.pad(
                            ct_t,
                            [[0, 0], [0, 0], [0, pad_len], [0, 0]],
                            value=1.0,
                        )
                        st_t = ttnn.pad(
                            st_t,
                            [[0, 0], [0, 0], [0, pad_len], [0, 0]],
                            value=0.0,
                        )
                    q_rot, k_rot = self.rope(
                        q_4d.to_ttnn if hasattr(q_4d, "to_ttnn") else q_4d,
                        k_4d.to_ttnn if hasattr(k_4d, "to_ttnn") else k_4d,
                        ct_t,
                        st_t,
                    )
                    q_4d = q_rot if isinstance(q_rot, TorchTTNNTensor) else TorchTTNNTensor(q_rot)
                    k_4d = k_rot if isinstance(k_rot, TorchTTNNTensor) else TorchTTNNTensor(k_rot)

            def _to_ttnn(t):
                return t.to_ttnn if hasattr(t, "to_ttnn") else t

            if q_pad_s > 0 or q_pad_d > 0:
                q_t = _to_ttnn(q_4d)
                q_raw = ttnn.pad(q_t, [[0, 0], [0, 0], [0, q_pad_s], [0, q_pad_d]], value=0.0)
            else:
                q_raw = _to_ttnn(q_4d)
            if kv_pad_s > 0 or kv_pad_d > 0:
                k_t = _to_ttnn(k_4d)
                k_raw = ttnn.pad(k_t, [[0, 0], [0, 0], [0, kv_pad_s], [0, kv_pad_d]], value=0.0)
            else:
                k_raw = _to_ttnn(k_4d)
            v_raw, _, _, _, _, _, _ = prepare_heads_on_device(v_w, self.num_kv_heads, apply_pad=True)
        else:
            q_raw, b, q_len, h, d_head, q_pad_s, q_pad_d = prepare_heads_on_device(q_w, self.num_heads, apply_pad=True)
            k_raw, _, kv_len, _, _, kv_pad_s, kv_pad_d = prepare_heads_on_device(k_w, self.num_kv_heads, apply_pad=True)
            v_raw, _, _, _, _, _, _ = prepare_heads_on_device(v_w, self.num_kv_heads, apply_pad=True)
            hw_dev = q_raw.device()

        if os.environ.get("TT_SYMBIOTE_DEBUG_ATTN_PAD"):
            _n = getattr(self, "module_name", "?")
            if "attn1" in str(_n) or "self_attn" in str(_n):
                print(
                    f"[prepare_heads] {_n} q_len={q_len} kv_len={kv_len} d_head={d_head} q_pad_s={q_pad_s} kv_pad_s={kv_pad_s}",
                    flush=True,
                )
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_t = k_raw.to_ttnn if hasattr(k_raw, "to_ttnn") else k_raw
            v_t = v_raw.to_ttnn if hasattr(v_raw, "to_ttnn") else v_raw
            if k_t.layout != ttnn.TILE_LAYOUT:
                k_t = ttnn.to_layout(k_t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if v_t.layout != ttnn.TILE_LAYOUT:
                v_t = ttnn.to_layout(v_t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k_raw = ttnn.repeat_interleave(k_t, n_rep, dim=1)
            v_raw = ttnn.repeat_interleave(v_t, n_rep, dim=1)

        def _tt(t):
            return t.to_ttnn if hasattr(t, "to_ttnn") else t

        _sdpa_dtypes = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)
        q_raw, k_raw, v_raw = _tt(q_raw), _tt(k_raw), _tt(v_raw)
        if q_raw.dtype not in _sdpa_dtypes:
            q_raw = ttnn.typecast(q_raw, ttnn.bfloat16)
        if k_raw.dtype not in _sdpa_dtypes:
            k_raw = ttnn.typecast(k_raw, ttnn.bfloat16)
        if v_raw.dtype not in _sdpa_dtypes:
            v_raw = ttnn.typecast(v_raw, ttnn.bfloat16)

        q_seq = q_raw.shape[2]
        kv_seq = k_raw.shape[2]
        use_causal = is_self_attn and getattr(self._torch_layer, "is_causal", False)
        tt_attn_mask = self._prepare_attention_mask_for_ttnn(
            attention_mask, b, q_seq, kv_seq, 0, 0, hw_dev, is_causal=use_causal
        )
        if tt_attn_mask is not None and (tt_attn_mask.shape[2] != q_seq or tt_attn_mask.shape[3] != kv_seq):
            tt_attn_mask = ttnn.slice(tt_attn_mask, [0, 0, 0, 0], [b, 1, q_seq, kv_seq])
        grid = hw_dev.compute_with_storage_grid_size()
        q_chunk = 32
        k_chunk = 32
        for c in (32, 64, 96, 128, 160, 192, 224, 256):
            if q_len % c == 0 and c <= 256:
                q_chunk = c
                break
        for c in (32, 64, 96, 128, 160, 192, 224, 256):
            if kv_len % c == 0 and c <= 256:
                k_chunk = c
                break
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        attn_out_raw = ttnn.transformer.scaled_dot_product_attention(
            q_raw,
            k_raw,
            v_raw,
            attn_mask=tt_attn_mask,
            is_causal=False,
            scale=float(d_head**-0.5),
            program_config=program_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        if q_pad_s > 0 or q_pad_d > 0:
            attn_out_raw = ttnn.slice(attn_out_raw, [0, 0, 0, 0], [b, self.num_heads, q_len, d_head])
        out_transposed = ttnn.transpose(attn_out_raw, 1, 2)
        merged_dev = ttnn.reshape(out_transposed, (b, q_len, h))

        if merged_dev.layout != ttnn.TILE_LAYOUT:
            merged_dev = ttnn.to_layout(merged_dev, ttnn.TILE_LAYOUT)

        if self.tt_o_proj is not None:
            tt_out = self.tt_o_proj(merged_dev)
            final_output = tt_out if isinstance(tt_out, TorchTTNNTensor) else TorchTTNNTensor(tt_out)
        else:
            final_output = TorchTTNNTensor(merged_dev)

        return final_output, None
