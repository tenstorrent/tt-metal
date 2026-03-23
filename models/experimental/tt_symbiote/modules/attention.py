# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention mechanism implementations for TTNN."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

try:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
except ImportError:
    sdpa_attention_forward = None  # type: ignore[misc, assignment]

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
    TTNNLinearIColShardedWRowSharded,
    TTNNLinearIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.rope import (
    TTNNRotaryPositionEmbedding,
    TTNNDistributedRotaryPositionEmbedding,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


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
        if sdpa_attention_forward is None:
            raise ImportError(
                "sdpa_attention_forward from transformers.integrations.sdpa_attention is required for TorchSDPAAttention."
            )
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
        bias1 = (
            fused_qkv.query.bias if fused_qkv.query.bias is not None else torch.zeros_like(fused_qkv.query.weight[:, 0])
        )
        bias2 = fused_qkv.key.bias if fused_qkv.key.bias is not None else torch.zeros_like(fused_qkv.key.weight[:, 0])
        bias3 = (
            fused_qkv.value.bias if fused_qkv.value.bias is not None else torch.zeros_like(fused_qkv.value.weight[:, 0])
        )

        qkv_bias = torch.cat([bias1, bias2, bias3])
        qkv_bias = torch.cat(
            [
                bias1,
                bias2,
                bias3,
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
        self.rope = TTNNRotaryPositionEmbedding()
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
        self.qkv_same_shape = True

    def init_fused_parameters(self, num_attention_heads, hidden_size: int):
        self.qkv_proj = TTNNFusedQKVSelfAttention.from_torch(
            PytorchFusedQKVSelfAttention(
                self.torch_layer.q_proj,
                self.torch_layer.k_proj,
                self.torch_layer.v_proj,
                num_attention_heads,
                hidden_size,
            ),
        )
        self.o_proj = TTNNLinear.from_torch(self.torch_layer.o_proj)

    def init_parameters(self):
        self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
        self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
        self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)
        self.o_proj = TTNNLinear.from_torch(self.torch_layer.o_proj)

    @classmethod
    def from_torch(cls, llama_attn: "LlamaAttention"):
        new_attn = cls()
        new_attn._fallback_torch_layer = llama_attn
        new_attn.num_key_value_groups = getattr(llama_attn, "num_key_value_groups", 1)
        # Fuse Q/K/V for self-attention (zero-pad K bias)
        new_attn.qkv_same_shape = (
            llama_attn.q_proj.weight.shape == llama_attn.k_proj.weight.shape
            and llama_attn.q_proj.weight.shape == llama_attn.v_proj.weight.shape
        )
        if new_attn.qkv_same_shape:
            new_attn.init_fused_parameters(llama_attn.config.num_attention_heads, llama_attn.config.hidden_size)
        else:
            new_attn.init_parameters()
        return new_attn

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,  # will become mandatory in v4.46
        **kwargs,
    ):
        past_key_values = kwargs.get("past_key_value", past_key_values) if past_key_values is None else past_key_values
        if self.qkv_same_shape:
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            input_shape = list(hidden_states.shape)[:-1]
            hidden_shape = (*input_shape, -1, self.torch_layer.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.torch_layer.rotary_emb(value_states.to_torch, TorchTTNNTensor(position_ids).to_torch)
        else:
            cos, sin = position_embeddings

        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.torch_layer.layer_idx, cache_kwargs
            )

        original_q_len = query_states.shape[2]
        kv_len = key_states.shape[2]

        if self.torch_layer.is_causal and original_q_len < kv_len:
            # Pad query: [B, H, q_len, D] -> [B, H, kv_len, D]
            pad_len = kv_len - original_q_len
            # Create zero padding on device
            pad_shape = (query_states.shape[0], query_states.shape[1], pad_len, query_states.shape[3])
            zero_pad = ttnn.zeros(
                pad_shape,
                device=hidden_states.device(),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=hidden_states.dtype,
            )
            query_states = ttnn.concat([zero_pad, query_states.to_ttnn], dim=2)

        attn_out = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            None,
            dropout=0.0,
            scaling=self.torch_layer.scaling,
            is_causal=self.torch_layer.is_causal,
            transpose_output=False,
        )
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out.to_ttnn)
        attn_out = ttnn.squeeze(attn_out, 1)
        # Slice output if query was padded
        if self.torch_layer.is_causal and original_q_len < kv_len:
            # Slice: [B, kv_len, D] -> [B, q_len, D]
            attn_out = attn_out[:, -original_q_len:, :]

        return self.o_proj(attn_out), None


class TTNNGlm4MoeLiteAttention(TTNNModule):
    """TTNN-accelerated Multi-Latent Attention for Glm4MoeLite.

    This implements the same MLA architecture as DeepSeek V3 but for Glm4MoeLite,
    with latent compression for memory-efficient KV cache.
    """

    def __init__(self):
        super().__init__()
        # Will be set in from_torch
        self.q_lora_rank = None
        self.kv_lora_rank = None
        self.qk_nope_head_dim = None
        self.qk_rope_head_dim = None
        self.qk_head_dim = None
        self.v_head_dim = None
        self.num_heads = None
        self.scaling = None
        self.is_causal = True

        # Submodules (set in from_torch)
        self.q_a_proj = None
        self.q_a_layernorm = None
        self.q_b_proj = None
        self.kv_a_proj_with_mqa = None
        self.kv_a_layernorm = None
        self.kv_b_proj = None
        self.o_proj = None
        self.rope = None
        self.sdpa = None

    @classmethod
    def from_torch(cls, torch_attn: "Glm4MoeLiteAttention"):
        """Create TTNNGlm4MoeLiteAttention from PyTorch Glm4MoeLiteAttention.

        Args:
            torch_attn: PyTorch Glm4MoeLiteAttention module

        Returns:
            TTNNGlm4MoeLiteAttention instance
        """
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        # Copy config parameters
        new_attn.q_lora_rank = torch_attn.q_lora_rank
        new_attn.kv_lora_rank = torch_attn.kv_lora_rank
        new_attn.qk_nope_head_dim = torch_attn.qk_nope_head_dim
        new_attn.qk_rope_head_dim = torch_attn.qk_rope_head_dim
        new_attn.qk_head_dim = torch_attn.qk_head_dim
        new_attn.v_head_dim = torch_attn.v_head_dim
        new_attn.num_heads = torch_attn.num_heads
        new_attn.scaling = torch_attn.scaling

        # Convert Q projection path (if using LoRA)
        if torch_attn.q_lora_rank is not None:
            new_attn.q_a_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_attn.q_a_proj)
            new_attn.q_a_layernorm = TTNNDistributedRMSNorm.from_torch(torch_attn.q_a_layernorm)
            new_attn.q_b_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_attn.q_b_proj)
        else:
            # Direct Q projection (no LoRA)
            new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(torch_attn.q_proj)

        # Convert KV projection path (always uses MQA + LoRA)
        new_attn.kv_a_proj_with_mqa = TTNNLinearIColShardedWRowSharded.from_torch(torch_attn.kv_a_proj_with_mqa)
        new_attn.kv_a_layernorm = TTNNDistributedRMSNorm.from_torch(torch_attn.kv_a_layernorm)
        new_attn.kv_b_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.kv_b_proj)

        # Convert output projection
        new_attn.o_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_attn.o_proj)

        # Initialize RoPE module
        new_attn.rope = TTNNDistributedRotaryPositionEmbedding()

        # Initialize SDPA module
        new_attn.sdpa = TTNNSDPAAttention()
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)
        new_attn.sdpa.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(new_attn.core_grid.x, new_attn.core_grid.y),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        new_attn.sdpa.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        return new_attn

    def _forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        past_key_values: Optional["Cache"],
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Forward pass for prefill mode (seq_len>1).

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_embeddings: (cos, sin) tensors for RoPE
            attention_mask: Optional attention mask
            past_key_values: KV cache
            cache_position: Cache position indices

        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # Ensure TILE layout
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Q projection path
        if self.q_lora_rank is not None:
            q_latent = self.q_a_proj(hidden_states)
            q_latent = self.q_a_layernorm(q_latent)
            q_states = self.q_b_proj(q_latent)
        else:
            q_states = self.q_proj(hidden_states)

        q_states = ttnn.experimental.all_gather_async(
            q_states.to_ttnn,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # Reshape Q
        q_states = ttnn.reshape(q_states, (batch_size, seq_length, self.num_heads, -1))
        q_states = ttnn.permute(q_states, (0, 2, 1, 3))

        # Split Q into nope and rope parts
        q_pass = ttnn.slice(q_states, (0, 0, 0, 0), (batch_size, self.num_heads, seq_length, self.qk_nope_head_dim))
        q_rot = ttnn.slice(
            q_states,
            (0, 0, 0, self.qk_nope_head_dim),
            (batch_size, self.num_heads, seq_length, self.qk_head_dim),
        )

        # KV projection path
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        compressed_kv = ttnn.experimental.all_gather_async(
            compressed_kv.to_ttnn,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # Split compressed KV
        k_pass = ttnn.slice(compressed_kv, (0, 0, 0), (batch_size, seq_length, self.kv_lora_rank))
        k_rot = ttnn.slice(
            compressed_kv,
            (0, 0, self.kv_lora_rank),
            (batch_size, seq_length, self.kv_lora_rank + self.qk_rope_head_dim),
        )

        # Process k_pass
        k_pass = ttnn.rms_norm(k_pass)
        k_pass = self.kv_b_proj(k_pass)
        k_pass = ttnn.experimental.all_gather_async(
            k_pass.to_ttnn,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        k_pass = ttnn.reshape(k_pass, (batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim))
        k_pass = ttnn.permute(k_pass, (0, 2, 1, 3))

        # Split into key and value
        key_nope = ttnn.slice(k_pass, (0, 0, 0, 0), (batch_size, self.num_heads, seq_length, self.qk_nope_head_dim))
        value_states = ttnn.slice(
            k_pass,
            (0, 0, 0, self.qk_nope_head_dim),
            (batch_size, self.num_heads, seq_length, self.qk_nope_head_dim + self.v_head_dim),
        )

        # Reshape k_rot for RoPE
        k_rot = ttnn.reshape(k_rot, (batch_size, 1, seq_length, self.qk_rope_head_dim))

        # Apply RoPE
        cos, sin = position_embeddings
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, 1)
        cos = ttnn.experimental.all_gather_async(
            cos,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, 1)
        sin = ttnn.experimental.all_gather_async(
            sin,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        q_rot, k_rot = self.rope(q_rot, k_rot, cos, sin)

        # Expand k_rot to match heads
        k_rot = ttnn.repeat(k_rot.to_ttnn, (1, self.num_heads, 1, 1))

        # Concatenate nope and rope parts
        query_states = ttnn.concat([q_pass, q_rot.to_ttnn], dim=-1)
        key_states = ttnn.concat([key_nope, k_rot], dim=-1)

        # Update KV cache if provided
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            torch_tensors = [TorchTTNNTensor(key_states), TorchTTNNTensor(value_states)]
            orig_shapes = [key_states.shape, value_states.shape]
            torch_tensors = [
                torch_tensor.to_torch[: orig_shape[0], : orig_shape[1], : orig_shape[2], : orig_shape[3]]
                for orig_shape, torch_tensor in zip(orig_shapes, torch_tensors)
            ]
            key_states, value_states = past_key_values.update(
                *torch_tensors,
                self._fallback_torch_layer.layer_idx,
                cache_kwargs,
            )
            key_states, value_states = [TorchTTNNTensor(key_states), TorchTTNNTensor(value_states)]
            key_states = ttnn.to_device(key_states.to_ttnn, self.device)
            value_states = ttnn.to_device(value_states.to_ttnn, self.device)
            key_states = ttnn.experimental.all_gather_async(
                key_states,
                dim=-1,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
            value_states = ttnn.experimental.all_gather_async(
                value_states,
                dim=-1,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Linear,
            )

        # Pad value_states if needed
        if self.qk_head_dim != self.v_head_dim:
            pad_size = self.qk_head_dim - self.v_head_dim
            value_states = ttnn.pad(value_states, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)

        # Scaled dot-product attention (causal for prefill)
        attn_output = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            transpose_output=True,
        ).to_ttnn

        # Remove padding
        if self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        # Reshape and output projection
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_heads * self.v_head_dim))
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Forward pass with automatic decode/prefill dispatch.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_embeddings: (cos, sin) tensors for RoPE
            attention_mask: Optional attention mask
            past_key_values: KV cache
            cache_position: Cache position indices
            **kwargs: Additional arguments

        Returns:
            Tuple of (output_tensor, attention_weights)
        """

        return self._forward_prefill(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
        )


def _gated_attention_rotate_half_ttnn(x):
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _gated_attention_apply_rotary_ttnn(q, k, cos, sin):
    cos = ttnn.unsqueeze(cos, 1)
    sin = ttnn.unsqueeze(sin, 1)
    rotary_dim = cos.shape[-1]
    full_dim = q.shape[-1]
    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]
    q_embed = ttnn.add(
        ttnn.multiply(q_rot, cos),
        ttnn.multiply(_gated_attention_rotate_half_ttnn(q_rot), sin),
    )
    k_embed = ttnn.add(
        ttnn.multiply(k_rot, cos),
        ttnn.multiply(_gated_attention_rotate_half_ttnn(k_rot), sin),
    )
    if rotary_dim < full_dim:
        q_pass = q[..., rotary_dim:]
        k_pass = k[..., rotary_dim:]
        q_embed = ttnn.concat([q_embed, q_pass], dim=-1)
        k_embed = ttnn.concat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def _gated_attention_rms_norm_zero_centered_ttnn(x, weight, eps=1e-6):
    x_sq = ttnn.multiply(x, x)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps))
    x_normed = ttnn.multiply(x, inv_rms)
    scale = ttnn.add(weight, 1.0)
    return ttnn.multiply(x_normed, scale)


def _gated_attention_sdpa_config(device, seq_len):
    grid_size = device.compute_with_storage_grid_size()
    q_chunk = 256 if seq_len >= 2048 else 64
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk,
        k_chunk_size=q_chunk,
        exp_approx_mode=False,
    )


def _gated_attention_ttnn_kv_to_torch(
    key_states,
    value_states,
    ref_torch: Optional[torch.Tensor],
    *,
    mesh_composer=None,
):
    """Host-side torch tensors for ``DynamicCache.update`` (device/dtype aligned to ``ref_torch`` when given).

    Mesh tensors require ``mesh_composer`` (e.g. ``ConcatMesh2dToTensor``) so shards are concatenated
    into one logical torch tensor before the HF cache API.
    """
    k_cast = ttnn.typecast(key_states, ttnn.bfloat16)
    v_cast = ttnn.typecast(value_states, ttnn.bfloat16)
    to_torch_kw: dict[str, Any] = {}
    if mesh_composer is not None:
        to_torch_kw["mesh_composer"] = mesh_composer
    k_t = ttnn.to_torch(k_cast, **to_torch_kw)
    v_t = ttnn.to_torch(v_cast, **to_torch_kw)
    if ref_torch is not None:
        k_t = k_t.to(device=ref_torch.device, dtype=ref_torch.dtype)
        v_t = v_t.to(device=ref_torch.device, dtype=ref_torch.dtype)
    return k_t, v_t


def gated_attention_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    device,
    norm_eps=1e-6,
    *,
    past_key_values=None,
    layer_idx: Optional[int] = None,
    cache_position=None,
    cos_torch: Optional[torch.Tensor] = None,
    sin_torch: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    mesh_composer=None,
    mesh_mapper=None,
):
    B = hidden_states.shape[0]
    T = hidden_states.shape[1]
    scaling = head_dim**-0.5
    qg = ttnn.linear(hidden_states, q_proj_weight)
    qg = ttnn.reshape(qg, [B, T, num_attention_heads, head_dim * 2])
    query_states, gate = ttnn.chunk(qg, 2, dim=-1)
    gate = ttnn.reshape(gate, [B, T, num_attention_heads * head_dim])
    query_states = _gated_attention_rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
    query_states = ttnn.transpose(query_states, 1, 2)
    key_states = ttnn.linear(hidden_states, k_proj_weight)
    key_states = ttnn.reshape(key_states, [B, T, num_key_value_heads, head_dim])
    key_states = _gated_attention_rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
    key_states = ttnn.transpose(key_states, 1, 2)
    value_states = ttnn.linear(hidden_states, v_proj_weight)
    value_states = ttnn.reshape(value_states, [B, T, num_key_value_heads, head_dim])
    value_states = ttnn.transpose(value_states, 1, 2)
    query_states, key_states = _gated_attention_apply_rotary_ttnn(query_states, key_states, cos, sin)

    original_q_len = T
    ref_torch = cos_torch if isinstance(cos_torch, torch.Tensor) else sin_torch
    if past_key_values is not None and layer_idx is not None:
        k_torch, v_torch = _gated_attention_ttnn_kv_to_torch(
            key_states, value_states, ref_torch, mesh_composer=mesh_composer
        )
        cache_kwargs: dict[str, Any] | None = None
        if sin_torch is not None or cos_torch is not None or cache_position is not None:
            cache_kwargs = {
                "sin": sin_torch,
                "cos": cos_torch,
                "cache_position": cache_position,
            }
        k_full, v_full = past_key_values.update(k_torch, v_torch, layer_idx, cache_kwargs)
        from_torch_kw: dict[str, Any] = {
            "dtype": ttnn.bfloat16,
            "layout": ttnn.TILE_LAYOUT,
            "device": device,
        }
        if mesh_mapper is not None:
            from_torch_kw["mesh_mapper"] = mesh_mapper
        key_states = ttnn.from_torch(k_full.contiguous().to(torch.bfloat16), **from_torch_kw)
        value_states = ttnn.from_torch(v_full.contiguous().to(torch.bfloat16), **from_torch_kw)
        if key_states.layout != ttnn.TILE_LAYOUT:
            key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if value_states.layout != ttnn.TILE_LAYOUT:
            value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    kv_len = key_states.shape[2]
    if is_causal and original_q_len < kv_len:
        pad_len = kv_len - original_q_len
        pad_shape = (query_states.shape[0], query_states.shape[1], pad_len, query_states.shape[3])
        zero_pad = ttnn.zeros(
            pad_shape,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=hidden_states.dtype,
        )
        query_states = ttnn.concat([zero_pad, query_states], dim=2)

    sdpa_seq = max(original_q_len, kv_len)
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        is_causal=is_causal,
        scale=scaling,
        program_config=_gated_attention_sdpa_config(device, sdpa_seq),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    if is_causal and original_q_len < kv_len:
        # [B, H, kv_len, head_dim] -> keep only positions for the current query span
        attn_output = ttnn.slice(
            attn_output,
            (0, 0, kv_len - original_q_len, 0),
            (B, num_attention_heads, kv_len, head_dim),
        )
    attn_output = ttnn.transpose(attn_output, 1, 2)
    attn_output = ttnn.reshape(attn_output, [B, original_q_len, num_attention_heads * head_dim])
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)
    attn_output = ttnn.linear(attn_output, o_proj_weight)
    return attn_output


class TTNNQwen3NextGatedAttention(TTNNModule):
    """Qwen3-Next gated attention on TTNN (RoPE → HF ``past_key_values.update`` → SDPA).

    K/V round-trip through Torch for the cache API on mesh; left-pad queries when ``q_len < kv_len``.
    ``attention_mask`` is not applied on this path.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn
        new_attn.layer_idx = getattr(torch_attn, "layer_idx", 0)
        new_attn.num_attention_heads = torch_attn.config.num_attention_heads
        new_attn.num_key_value_heads = torch_attn.config.num_key_value_heads
        new_attn.head_dim = torch_attn.head_dim
        new_attn.hidden_size = torch_attn.config.hidden_size
        new_attn.norm_eps = torch_attn.config.rms_norm_eps
        new_attn.is_causal = getattr(torch_attn, "is_causal", True)
        return new_attn

    def preprocess_weights_impl(self):
        t = self._fallback_torch_layer
        self.tt_q_proj = ttnn.from_torch(
            t.q_proj.weight.T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_k_proj = ttnn.from_torch(
            t.k_proj.weight.T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_v_proj = ttnn.from_torch(
            t.v_proj.weight.T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_o_proj = ttnn.from_torch(
            t.o_proj.weight.T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_q_norm = ttnn.from_torch(
            t.q_norm.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self.tt_k_norm = ttnn.from_torch(
            t.k_norm.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        self.tt_q_proj = ttnn.to_device(self.tt_q_proj, self.device)
        self.tt_k_proj = ttnn.to_device(self.tt_k_proj, self.device)
        self.tt_v_proj = ttnn.to_device(self.tt_v_proj, self.device)
        self.tt_o_proj = ttnn.to_device(self.tt_o_proj, self.device)
        self.tt_q_norm = ttnn.to_device(self.tt_q_norm, self.device)
        self.tt_k_norm = ttnn.to_device(self.tt_k_norm, self.device)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        past_key_values = kwargs.get("past_key_value", past_key_values) if past_key_values is None else past_key_values
        cos, sin = position_embeddings
        cos_torch = cos if isinstance(cos, torch.Tensor) else None
        sin_torch = sin if isinstance(sin, torch.Tensor) else None
        if isinstance(hidden_states, TorchTTNNTensor):
            hidden_states = hidden_states.to_ttnn
        if isinstance(cos, torch.Tensor):
            cos = ttnn.from_torch(cos.to(torch.bfloat16), device=self.device, layout=ttnn.TILE_LAYOUT)
        if isinstance(sin, torch.Tensor):
            sin = ttnn.from_torch(sin.to(torch.bfloat16), device=self.device, layout=ttnn.TILE_LAYOUT)
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # All-gather hidden_states when tensor parallel (sharded hidden dim)
        need_reduce_scatter = (
            self.device_state is not None
            and self.device.get_num_devices() > 1
            and hidden_states.shape[-1] != self.hidden_size
        )
        if need_reduce_scatter:
            hidden_states = ttnn.experimental.all_gather_async(
                hidden_states,
                dim=-1,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        mesh_composer = None
        mesh_mapper = None
        if (
            past_key_values is not None
            and self.device_state is not None
            and self.device.get_num_devices() > 1
            and self.device_state.tensor_config is not None
        ):
            mesh_composer = self.device_state.tensor_config.mesh_composer
            mesh_mapper = self.device_state.tensor_config.mesh_mapper
        out = gated_attention_forward_ttnn(
            hidden_states,
            self.tt_q_proj,
            self.tt_k_proj,
            self.tt_v_proj,
            self.tt_o_proj,
            self.tt_q_norm,
            self.tt_k_norm,
            cos,
            sin,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.device,
            norm_eps=self.norm_eps,
            past_key_values=past_key_values,
            layer_idx=self.layer_idx,
            cache_position=cache_position,
            cos_torch=cos_torch,
            sin_torch=sin_torch,
            is_causal=self.is_causal,
            mesh_composer=mesh_composer,
            mesh_mapper=mesh_mapper,
        )
        if need_reduce_scatter:
            # Reduce-scatter output to match sharded residual for residual add
            out = ttnn.reshape(out, (out.shape[0], 1, out.shape[1], out.shape[2]))
            out = ttnn.experimental.reduce_scatter_minimal_async(
                out,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            )
            out = ttnn.div(out, float(self.device.get_num_devices()))
            out = ttnn.squeeze(out, 1)
        return out, None
