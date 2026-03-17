# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention mechanism implementations for TTNN."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math
import torch
from torch.nn import functional as F
import ttnn

try:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
except ImportError:
    sdpa_attention_forward = None
    print("Could not import sdpa_attention_forward from transformers.integrations.sdpa_attention. ")

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding


def _scaled_dot_product_attention_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None,
    is_causal: bool | None,
) -> torch.Tensor:
    """Fallback when transformers.integrations.sdpa_attention is not available (e.g. DPL torch path).
    Apply scale once to match TTNN: use F.sdpa(..., scale=scale) when supported to avoid double-scaling.
    """
    scale = scaling if scaling is not None else (query.size(-1) ** -0.5)
    try:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal or False,
            scale=scale,
        )
    except TypeError:
        # Older PyTorch: no scale arg; use default 1/sqrt(d) via F.sdpa (no pre-multiply to avoid double-scaling)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal or False
        )


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
        if sdpa_attention_forward is not None:
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
            # transformers returns (B, S, H, D); normalize to (B, H, S, D) to match TTNN
            attn_output = attn_output.transpose(1, 2)
        else:
            # Fallback when transformers SDPA integration not available (e.g. DPL); already (B, H, S, D)
            attn_output = _scaled_dot_product_attention_fallback(query, key, value, attention_mask, scaling, is_causal)
        # TTNN returns (B, H, S, D) when transpose_output=False, (B, S, H, D) when True
        if transpose_output:
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
        new_attn._scaling = getattr(llama_attn, "scaling", None)
        if new_attn._scaling is None:
            head_dim = getattr(
                llama_attn, "head_dim", llama_attn.config.hidden_size // llama_attn.config.num_attention_heads
            )
            new_attn._scaling = head_dim**-0.5
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
        if attention_mask is not None:
            print(
                "Warning: attention_mask is not None, but TTNN LlamaAttention does not support it yet."
            )  # --- IGNORE ---
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
            print("Warning: position_embeddings is None, computing from position_ids.")  # --- IGNORE ---
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
            scaling=self._scaling,
            is_causal=self.torch_layer.is_causal,
            transpose_output=False,
        )
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out.to_ttnn)
        attn_out = ttnn.squeeze(attn_out, 1)
        # Slice output if query was padded
        if self.torch_layer.is_causal and original_q_len < kv_len:
            # Slice: [B, kv_len, D] -> [B, q_len, D]
            attn_out = attn_out[:, -original_q_len:, :]

        return self.o_proj(attn_out), None, past_key_values


def _get_rel_pos_sam(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Relative position table (q_size, k_size, head_dim). Matches working tt_sam / deepencoder."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos = rel_pos.float()
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(rel_pos.dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def _add_decomposed_rel_pos_sam(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """q: (B, n_heads, q_h*q_w, head_dim). Returns attn_bias (B, n_heads, S, S)."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _get_rel_pos_sam(q_h, k_h, rel_pos_h)
    Rw = _get_rel_pos_sam(q_w, k_w, rel_pos_w)
    B, n_heads, S, dim = q.shape
    r_q = q.reshape(B, n_heads, q_h, q_w, dim)
    rel_h = torch.einsum("bnhwc,hkc->bnhwk", r_q, Rh)
    rel_w = torch.einsum("bnhwc,wkc->bnhwk", r_q, Rw)
    rel_h = rel_h.reshape(B, n_heads, S, k_h, 1)
    rel_w = rel_w.reshape(B, n_heads, S, 1, k_w)
    return (rel_h + rel_w).reshape(B, n_heads, S, k_h * k_w)


def compute_sam_attn_bias(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    spatial_size: Tuple[int, int],
) -> torch.Tensor:
    """q: (B, n_heads, S, head_dim). Returns attn_bias (B, n_heads, S, S) for SDPA."""
    H, W = spatial_size
    dtype = q.dtype
    q = q.float()
    rel_pos_h = rel_pos_h.float()
    rel_pos_w = rel_pos_w.float()
    bias = _add_decomposed_rel_pos_sam(q, rel_pos_h, rel_pos_w, (H, W), (H, W))
    return bias.to(dtype)


def _window_partition_unpartition_sam(
    x_bhwc: ttnn.Tensor,
    device: ttnn.Device,
    H: int,
    W: int,
    window_size: int,
    C: int,
    run_attn_per_window,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Partition (B, H, W, C) into windows, run run_attn_per_window(win (1,ws,ws,C)) per window, unpartition."""
    B = x_bhwc.shape[0]
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    Hp, Wp = H + pad_h, W + pad_w

    if pad_h > 0 or pad_w > 0:
        x_t = ttnn.to_torch(x_bhwc)
        if x_t.device.type != "cpu":
            x_t = x_t.cpu()
        x_t = x_t.reshape(B, H, W, C)
        x_t = F.pad(x_t, (0, 0, 0, pad_w, 0, pad_h))
        x_bhwc = ttnn.from_torch(
            x_t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=memory_config,
        )
        x_t = None

    if pad_h > 0 or pad_w > 0:
        x_rm = x_bhwc
    else:
        x_rm = ttnn.to_layout(x_bhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
        if x_rm is not x_bhwc:
            ttnn.deallocate(x_bhwc)
    if tuple(x_rm.shape) != (B, Hp, Wp, C):
        x_rm = ttnn.reshape(x_rm, (B, Hp, Wp, C))
    nH, nW = Hp // window_size, Wp // window_size
    ws = window_size
    x_rm = ttnn.reshape(x_rm, (B, nH, ws, Wp, C))
    x_rm = ttnn.reshape(x_rm, (B, nH, ws, nW, ws, C))
    x_rm = ttnn.permute(x_rm, (0, 1, 3, 2, 4, 5))
    num_windows = nH * nW
    x_rm = ttnn.reshape(x_rm, (B, num_windows, ws * ws, C))

    out_list = []
    for b in range(B):
        for i in range(num_windows):
            win = ttnn.slice(x_rm, (b, i, 0, 0), (b + 1, i + 1, ws * ws, C), memory_config=memory_config)
            win = ttnn.reshape(win, (1, ws, ws, C))
            win = ttnn.to_layout(win, ttnn.TILE_LAYOUT, memory_config=memory_config)
            out_i = run_attn_per_window(win)
            ttnn.deallocate(win)
            out_i = out_i.ttnn_tensor if hasattr(out_i, "ttnn_tensor") else out_i
            out_i = ttnn.to_layout(out_i, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
            out_list.append(out_i)
    ttnn.deallocate(x_rm)

    out_cat = ttnn.concat(out_list, dim=0, memory_config=memory_config)
    for t in out_list:
        ttnn.deallocate(t)
    out_cat = ttnn.reshape(out_cat, (B, nH, nW, ws, ws, C))
    out_cat = ttnn.permute(out_cat, (0, 1, 3, 2, 4, 5))
    out_cat = ttnn.reshape(out_cat, (B, Hp, Wp, C))
    if Hp > H or Wp > W:
        out_cat = ttnn.slice(out_cat, (0, 0, 0, 0), (B, H, W, C), memory_config=memory_config)
    out_cat = ttnn.reshape(out_cat, (B, H * W, C))
    out_tt = ttnn.to_layout(out_cat, ttnn.TILE_LAYOUT, memory_config=memory_config)
    ttnn.deallocate(out_cat)
    out_tt = ttnn.reshape(out_tt, (B, H, W, C))
    return out_tt


class TTNNSAMAttention(TTNNModule):
    """TTNN version of SAM image encoder Attention (deepencoder.py).
    Single class for both global and windowed: when window_size > 0, partitions into windows,
    runs this attention per window, then unpartitions. Otherwise runs on full (B, H, W, C).
    Uses manual attention (matmul QK^T -> scale -> [optional +attn_bias] -> softmax -> matmul @ V).
    Optional relative position bias (use_rel_pos): computed on host from q, then added to scores.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, scale: float, window_size: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = scale
        self.window_size = window_size
        self._use_rel_pos = False
        self._rel_pos_h = None
        self._rel_pos_w = None
        self.sdpa = TTNNSDPAAttention()
        self.sdpa.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._attn_compute_config = self.sdpa.compute_kernel_config

    @classmethod
    def from_torch(cls, sam_attn: "torch.nn.Module", window_size: int = 0):
        """Build from SAM Attention (blocks[i].attn). Optionally set window_size for windowed path."""
        dim = sam_attn.qkv.in_features
        num_heads = sam_attn.num_heads
        head_dim = dim // num_heads
        scale = head_dim**-0.5
        new_attn = cls(dim=dim, num_heads=num_heads, head_dim=head_dim, scale=scale, window_size=window_size)
        new_attn._fallback_torch_layer = sam_attn
        new_attn.qkv = TTNNLinear.from_torch(sam_attn.qkv)
        new_attn.proj = TTNNLinear.from_torch(sam_attn.proj)
        if getattr(sam_attn, "use_rel_pos", False):
            new_attn._use_rel_pos = True
            new_attn._rel_pos_h = sam_attn.rel_pos_h.detach()
            new_attn._rel_pos_w = sam_attn.rel_pos_w.detach()
        return new_attn

    def _forward_core(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Core attention on a single tensor (B, H, W, C). Used as-is for global or per-window."""
        B, H, W, C = x.shape
        seq_len = H * W
        # (B, H, W, C) -> (B, H*W, C)
        x = ttnn.reshape(x, ttnn.Shape((B, seq_len, C)))
        # Add seq dim for linear: (B, 1, H*W, C)
        x = ttnn.unsqueeze(x, 1)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv_out = self.qkv(x)
        qkv_t = qkv_out.ttnn_tensor if hasattr(qkv_out, "ttnn_tensor") else qkv_out
        qkv_t = ttnn.to_memory_config(qkv_t, ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
        )
        ttnn.deallocate(qkv_t)
        # Optional relative position bias (on host, then add to scores)
        attn_bias_tt = None
        if self._use_rel_pos and self._rel_pos_h is not None and self._rel_pos_w is not None:
            q_t = ttnn.to_torch(q)
            if q_t.device.type != "cpu":
                q_t = q_t.cpu()
            attn_bias = compute_sam_attn_bias(q_t, self._rel_pos_h, self._rel_pos_w, (H, W))
            attn_bias_tt = ttnn.from_torch(
                attn_bias.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        # Manual SDPA: Q@K^T -> scale -> [+bias] -> softmax -> @V
        cfg = self._attn_compute_config
        qk = ttnn.matmul(
            q,
            k,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=cfg,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        qk = ttnn.multiply(qk, self.scaling, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if attn_bias_tt is not None:
            qk = ttnn.add(qk, attn_bias_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_bias_tt)
        try:
            attn_weights = ttnn.softmax(
                qk,
                dim=-1,
                compute_kernel_config=cfg,
                numeric_stable=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qk)
        except RuntimeError:
            qk_torch = ttnn.to_torch(qk)
            ttnn.deallocate(qk)
            attn_weights_torch = torch.nn.functional.softmax(qk_torch.float(), dim=-1).to(torch.bfloat16)
            attn_weights = ttnn.from_torch(
                attn_weights_torch,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attn_out = ttnn.matmul(
            attn_weights,
            v,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=cfg,
        )
        ttnn.deallocate(attn_weights)
        ttnn.deallocate(v)
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out)
        attn_out = ttnn.squeeze(attn_out, 1)
        # (B, H*W, C) -> (B, H, W, C)
        attn_out = ttnn.reshape(attn_out, ttnn.Shape((B, H, W, C)))
        out = self.proj(attn_out)
        return out.ttnn_tensor if hasattr(out, "ttnn_tensor") else out

    def forward(self, x):
        """x: (B, H, W, C). When window_size > 0, runs attention per window; else global."""
        if hasattr(x, "to_ttnn"):
            x = x.to_ttnn
        elif isinstance(x, torch.Tensor):
            x = ttnn.from_torch(x, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.window_size > 0:
            B, H, W, C = x.shape
            return _window_partition_unpartition_sam(
                x,
                device=self.device,
                H=H,
                W=W,
                window_size=self.window_size,
                C=C,
                run_attn_per_window=self._forward_core,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._forward_core(x)


class TTNNNoTPAttention(TTNNModule):
    """
    No Tensor Parallelism Attention using TTNN operations.

    Implements multi-head self-attention with QKV projection and scaled dot product attention.
    """

    def __init__(self, num_heads, n_local_heads, head_dim, max_seq_len, use_flash_attention):
        """
        Initialize attention layer.

        Args:
            cfg: Configuration dict with num_attention_heads, hidden_size, etc.
            weights: PyTorch weights dict (optional)
            device: TTNN device
        """
        super().__init__()

        self.num_heads = num_heads
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_flash_attention = use_flash_attention
        self.hidden_size = self.head_dim * self.num_heads
        self.torch_layer_cp = None

    @classmethod
    def from_torch(cls, NoTPAttention):
        """Create TTNN module from PyTorch equivalent."""
        new_Attn = cls(
            NoTPAttention.num_heads,
            NoTPAttention.n_local_heads,
            NoTPAttention.head_dim,
            NoTPAttention.max_seq_len,
            NoTPAttention.use_flash_attention,
        )
        new_Attn._fallback_torch_layer = NoTPAttention
        new_Attn.torch_layer_cp = NoTPAttention
        return new_Attn

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        # Load QKV projection weights
        qkv_weight = self.torch_layer_cp.qkv_proj.weight.data  # (hidden_size * 3, hidden_size)
        qkv_bias = None
        if self.torch_layer_cp.qkv_proj.bias is not None:
            qkv_bias = self.torch_layer_cp.qkv_proj.bias.data

        # Split into Q, K, V
        q_weight = qkv_weight[: self.hidden_size, :].T  # (hidden_size, hidden_size)
        k_weight = qkv_weight[self.hidden_size : 2 * self.hidden_size, :].T
        v_weight = qkv_weight[2 * self.hidden_size :, :].T

        # Convert to TTNN
        self.q_weight = ttnn.from_torch(
            q_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k_weight = ttnn.from_torch(
            k_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_weight = ttnn.from_torch(
            v_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if qkv_bias is not None:
            q_bias = qkv_bias[: self.hidden_size]
            k_bias = qkv_bias[self.hidden_size : 2 * self.hidden_size]
            v_bias = qkv_bias[2 * self.hidden_size :]

            self.q_bias = self.tensor_1d_to_2d_ttnn(q_bias)
            self.k_bias = self.tensor_1d_to_2d_ttnn(k_bias)
            self.v_bias = self.tensor_1d_to_2d_ttnn(v_bias)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        # Output projection
        out_weight = self.torch_layer_cp.out_proj.weight.data.T  # (hidden_size, hidden_size)
        self.out_weight = ttnn.from_torch(
            out_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        out_bias = self.torch_layer_cp.out_proj.bias.data
        if out_bias is not None:
            self.out_bias = self.tensor_1d_to_2d_ttnn(out_bias)
        else:
            self.out_bias = None

    def tensor_1d_to_2d_ttnn(self, tensor_1d: torch.Tensor, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        """
        Convert 1D PyTorch tensor to 2D TTNN tensor (1, N) for bias operations.

        Args:
            tensor_1d: 1D PyTorch tensor
            device: TTNN device
            dtype: TTNN data type

        Returns:
            2D TTNN tensor of shape (1, N)
        """
        tensor_2d = tensor_1d.unsqueeze(0)
        return ttnn.from_torch(
            tensor_2d,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.q_weight = ttnn.to_device(self.q_weight, self.device)
        self.k_weight = ttnn.to_device(self.k_weight, self.device)
        self.v_weight = ttnn.to_device(self.v_weight, self.device)
        self.out_weight = ttnn.to_device(self.out_weight, self.device)

        if self.q_bias is not None and self.k_bias is not None and self.v_bias is not None:
            self.q_bias = ttnn.to_device(self.q_bias, self.device)
            self.k_bias = ttnn.to_device(self.k_bias, self.device)
            self.v_bias = ttnn.to_device(self.v_bias, self.device)
        if self.out_bias is not None:
            self.out_bias = ttnn.to_device(self.out_bias, self.device)

    def deallocate_weights_impl(self):
        """Deallocate device memory."""

        ttnn.deallocate(self.q_weight)
        ttnn.deallocate(self.k_weight)
        ttnn.deallocate(self.v_weight)
        ttnn.deallocate(self.out_weight)
        if self.q_bias is not None and self.k_bias is not None and self.v_bias is not None:
            ttnn.deallocate(self.q_bias)
            ttnn.deallocate(self.k_bias)
            ttnn.deallocate(self.v_bias)
        if self.out_bias is not None:
            ttnn.deallocate(self.out_bias)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of attention.

        Args:
            x: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        if isinstance(x, TorchTTNNTensor):
            x = x.to_ttnn
            x = ttnn.to_device(x, device=self.device)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        bsz, seqlen, _ = x.shape

        # QKV projections
        query = ttnn.linear(
            x,
            self.q_weight,
            bias=self.q_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        key = ttnn.linear(
            x,
            self.k_weight,
            bias=self.k_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        value = ttnn.linear(
            x,
            self.v_weight,
            bias=self.v_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Reshape to (batch_size, seqlen, num_heads, head_dim)
        query = ttnn.reshape(query, (bsz, seqlen, self.num_heads, self.head_dim))
        key = ttnn.reshape(key, (bsz, seqlen, self.num_heads, self.head_dim))
        value = ttnn.reshape(value, (bsz, seqlen, self.num_heads, self.head_dim))

        # Permute to (batch_size, num_heads, seqlen, head_dim)
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # Scaled dot product attention
        scale = 1.0 / math.sqrt(self.head_dim)

        try:
            # Configure SDPA (chunk sizes must be multiples of TILE_SIZE=32)
            TILE_SIZE = 32
            chunk_size = max(128, ((seqlen + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE)

            device_grid = self.device.compute_with_storage_grid_size()
            grid_x = min(8, device_grid.x)
            grid_y = min(8, device_grid.y)

            sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid_x, grid_y),
                q_chunk_size=chunk_size,
                k_chunk_size=chunk_size,
                exp_approx_mode=False,
            )

            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            attention_output = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=scale,
                is_causal=False,
                program_config=sdpa_cfg,
                compute_kernel_config=compute_kernel_config,
            )

            # Permute back to (batch_size, seqlen, num_heads, head_dim)
            attention_output = ttnn.permute(attention_output, (0, 2, 1, 3))

            # Reshape to (batch_size, seqlen, hidden_size)
            attention_output = ttnn.reshape(attention_output, (bsz, seqlen, self.hidden_size))
        except RuntimeError:
            q_torch = ttnn.to_torch(query).float()
            k_torch = ttnn.to_torch(key).float()
            v_torch = ttnn.to_torch(value).float()
            attn_out_torch = torch.nn.functional.scaled_dot_product_attention(
                q_torch, k_torch, v_torch, scale=scale
            ).to(torch.bfloat16)
            attn_out_torch = attn_out_torch.permute(0, 2, 1, 3).reshape(bsz, seqlen, self.hidden_size)
            attention_output = ttnn.from_torch(
                attn_out_torch,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Output projection
        output = ttnn.linear(
            attention_output,
            self.out_weight,
            bias=self.out_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Cleanup
        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)
        ttnn.deallocate(attention_output)

        output = ttnn.to_torch(output)

        return output
