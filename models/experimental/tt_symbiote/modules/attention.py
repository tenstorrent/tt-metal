# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention mechanism implementations for TTNN."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.linear import TTNNLinear


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
        assert attention_mask is None, "TTNNSDPAAttention currently only supports attention_mask=None"
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
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scaling,
            program_config=self.program_config,
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
        new_fused_qkv = TTNNFusedQKVSelfAttention()
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
        new_self_attention = TTNNSelfAttention(
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
        new_self_attention = TTNNViTSelfAttention(
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
