# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Attention mechanism implementations for TTNN."""

from dataclasses import dataclass
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
    TTNNLinearIColShardedWRowSharded,
    TTNNLinearIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.rope import (
    TTNNRotaryPositionEmbedding,
    TTNNDistributedRotaryPositionEmbedding,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm

try:
    from transformers.cache_utils import Cache, CacheLayerMixin
except ImportError:
    Cache = object
    CacheLayerMixin = None


class _PagedCacheLayer(CacheLayerMixin if CacheLayerMixin is not None else object):
    """CacheLayerMixin stub so HF Cache.__init__ is satisfied."""

    def lazy_initialization(self, key_states, value_states):
        pass

    def update(self, key_states, value_states, cache_kwargs=None):
        return key_states, value_states

    def get_mask_sizes(self, cache_position):
        return 0, 0

    def get_seq_length(self):
        return 0

    def get_max_cache_shape(self):
        return 0


@dataclass
class PagedAttentionConfig:
    block_size: int = 64
    max_num_blocks: int = 2048
    batch_size: int = 1

    @property
    def max_seq_length(self) -> int:
        return self.max_num_blocks * self.block_size

    @property
    def blocks_per_sequence(self) -> int:
        return self.max_num_blocks // self.batch_size


class TTNNPagedAttentionKVCache(Cache):
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        config: PagedAttentionConfig,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if Cache is not object:
            super().__init__(layer_class_to_replicate=_PagedCacheLayer)
        else:
            super().__init__()

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.config = config
        self.dtype = dtype
        self._device = device
        self._seq_lengths: list[int] = [0] * num_layers
        self._seen_tokens = 0

        page_table = torch.arange(config.max_num_blocks, dtype=torch.int32)
        self.page_table = page_table.reshape(config.batch_size, config.blocks_per_sequence)

        self._tt_key_cache: list[Optional[ttnn.Tensor]] = [None] * num_layers
        self._tt_value_cache: list[Optional[ttnn.Tensor]] = [None] * num_layers
        self._tt_page_table: Optional[ttnn.Tensor] = None
        self._is_on_device = False

    def to_device(self, device) -> "TTNNPagedAttentionKVCache":
        if self._is_on_device and self._device == device:
            return self

        self._device = device
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None

        cache_shape = (
            self.config.max_num_blocks,
            self.num_kv_heads,
            self.config.block_size,
            self.head_dim,
        )

        for layer_idx in range(self.num_layers):
            self._tt_key_cache[layer_idx] = ttnn.zeros(
                cache_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._tt_value_cache[layer_idx] = ttnn.zeros(
                cache_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self._tt_page_table = ttnn.from_torch(
            self.page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._is_on_device = True
        return self

    def paged_fill_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        batch_idx: int = 0,
    ):
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")

        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        page_table = self._tt_page_table

        max_len = self.config.blocks_per_sequence * self.config.block_size
        seq_len = key_states.shape[2]
        if seq_len > max_len:
            key_states = key_states[:, :, :max_len, :]
            value_states = value_states[:, :, :max_len, :]
            seq_len = max_len

        ttnn.experimental.paged_fill_cache(k_cache, key_states, page_table, batch_idx=batch_idx)
        ttnn.experimental.paged_fill_cache(v_cache, value_states, page_table, batch_idx=batch_idx)

        self._seq_lengths[layer_idx] += seq_len
        if layer_idx == 0:
            self._seen_tokens += seq_len

    def paged_update_on_device(
        self,
        key_states: ttnn.Tensor,
        value_states: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
    ):
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")

        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        page_table = self._tt_page_table

        ttnn.experimental.paged_update_cache(
            k_cache,
            key_states,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            value_states,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )

        seq_len = key_states.shape[0]
        self._seq_lengths[layer_idx] += seq_len
        if layer_idx == 0:
            self._seen_tokens += seq_len

    def paged_sdpa_decode(
        self,
        query: ttnn.Tensor,
        layer_idx: int,
        current_pos: ttnn.Tensor,
        scale: float = 1.0,
        program_config=None,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        if not self._is_on_device:
            raise RuntimeError("KV cache not on device. Call to_device(device).")

        k_cache = self._tt_key_cache[layer_idx]
        v_cache = self._tt_value_cache[layer_idx]
        page_table = self._tt_page_table

        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            k_cache,
            v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(key_states, TorchTTNNTensor):
            key_states = key_states.to_torch
        if isinstance(value_states, TorchTTNNTensor):
            value_states = value_states.to_torch
        if isinstance(key_states, ttnn.Tensor):
            key_states = ttnn.to_torch(key_states)
        if isinstance(value_states, ttnn.Tensor):
            value_states = ttnn.to_torch(value_states)

        seq_len = key_states.shape[2]
        self._seq_lengths[layer_idx] += seq_len
        if layer_idx == 0:
            self._seen_tokens += seq_len

        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_lengths[layer_idx]

    def get_max_cache_shape(self) -> Optional[int]:
        return self.config.max_seq_length


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
        self._sdpa_available = True

    def _matmul_attention(self, query, key, value, is_causal, scaling, attention_mask, transpose_output):
        import math

        scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
        key_t = ttnn.permute(key, (0, 1, 3, 2))
        scores = ttnn.matmul(query, key_t)
        scores = ttnn.multiply(scores, scale)

        if is_causal:
            seq_len = query.shape[2]
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(torch.bfloat16)
            causal_mask = ttnn.from_torch(
                causal_mask.unsqueeze(0).unsqueeze(0),
                layout=ttnn.TILE_LAYOUT,
                device=query.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            scores = ttnn.add(scores, causal_mask)
        elif attention_mask is not None:
            scores = ttnn.add(scores, attention_mask)

        scores = ttnn.softmax(scores, dim=-1)
        attn_output = ttnn.matmul(scores, value)

        if transpose_output:
            attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        return attn_output

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

        if self._sdpa_available:
            try:
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
            except RuntimeError as e:
                print(
                    f"TTNNSDPAAttention: ttnn SDPA failed, falling back to matmul attention. "
                    f"Q={query.shape} K={key.shape} V={value.shape} is_causal={is_causal} "
                    f"Error: {e}"
                )
                self._sdpa_available = False

        return self._matmul_attention(query, key, value, is_causal, scaling, attention_mask, transpose_output)


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
        return new_self_attention

    def move_weights_to_device_impl(self):
        """Initialize SDPA config when device is available."""
        super().move_weights_to_device_impl()
        if self.sdpa.program_config is None:
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.sdpa.program_config = program_config
            self.sdpa.compute_kernel_config = compute_kernel_config

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

    def move_weights_to_device_impl(self):
        """Initialize SDPA config when device is available."""
        super().move_weights_to_device_impl()
        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
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
        self.qkv_same_shape = True

    def move_weights_to_device_impl(self):
        """Initialize SDPA config when device is available."""
        super().move_weights_to_device_impl()
        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

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

    Supports both standard DynamicCache and TTNNPagedAttentionKVCache
    for paged attention with on-device KV storage.
    """

    def __init__(self):
        super().__init__()
        self.q_lora_rank = None
        self.kv_lora_rank = None
        self.qk_nope_head_dim = None
        self.qk_rope_head_dim = None
        self.qk_head_dim = None
        self.v_head_dim = None
        self.num_heads = None
        self.scaling = None
        self.is_causal = True

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
    def from_torch(cls, torch_attn: "Glm4MoeLiteAttention", distributed: bool = True):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        new_attn.q_lora_rank = torch_attn.q_lora_rank
        new_attn.kv_lora_rank = torch_attn.kv_lora_rank
        new_attn.qk_nope_head_dim = torch_attn.qk_nope_head_dim
        new_attn.qk_rope_head_dim = torch_attn.qk_rope_head_dim
        new_attn.qk_head_dim = torch_attn.qk_head_dim
        new_attn.v_head_dim = torch_attn.v_head_dim
        new_attn.num_heads = torch_attn.num_heads
        new_attn.scaling = torch_attn.scaling

        from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm

        LinearCls = TTNNLinearIColShardedWRowSharded if distributed else TTNNLinear
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        NormCls = TTNNDistributedRMSNorm if distributed else TTNNRMSNorm

        if torch_attn.q_lora_rank is not None:
            new_attn.q_a_proj = LinearCls.from_torch(torch_attn.q_a_proj)
            new_attn.q_a_layernorm = NormCls.from_torch(torch_attn.q_a_layernorm)
            new_attn.q_b_proj = LinearCls.from_torch(torch_attn.q_b_proj)
        else:
            new_attn.q_proj = LinearCls.from_torch(torch_attn.q_proj)

        new_attn.kv_a_proj_with_mqa = LinearCls.from_torch(torch_attn.kv_a_proj_with_mqa)
        new_attn.kv_a_layernorm = NormCls.from_torch(torch_attn.kv_a_layernorm)
        new_attn.kv_b_proj = LinearClsOut.from_torch(torch_attn.kv_b_proj)

        new_attn.o_proj = LinearClsOut.from_torch(torch_attn.o_proj)

        new_attn.rope = TTNNDistributedRotaryPositionEmbedding() if distributed else TTNNRotaryPositionEmbedding()

        new_attn.sdpa = TTNNSDPAAttention()
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def move_weights_to_device_impl(self):
        """Initialize SDPA config when device is available."""
        super().move_weights_to_device_impl()
        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        kv_a_ln = self._fallback_torch_layer.kv_a_layernorm
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        self._kv_a_ln_weight = ttnn.from_torch(
            kv_a_ln.weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._kv_a_ln_eps = kv_a_ln.variance_epsilon

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        if not self._is_distributed:
            return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
        t = tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
        return ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    def _permute_cos_sin_to_meta_format(self, cos, sin):
        """Convert cos/sin from HF doubled-half format to Meta interleaved-pair format.

        HF format:  [c(f0), c(f1), ..., c(f_{d/2-1}), c(f0), c(f1), ..., c(f_{d/2-1})]
        Meta format: [c(f0), c(f0), c(f1), c(f1), ..., c(f_{d/2-1}), c(f_{d/2-1})]

        rotary_embedding_llama expects Meta format where each adjacent pair shares
        the same frequency, matching its interleaved rotation convention.
        """
        half_dim = self.qk_rope_head_dim // 2
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if self._is_distributed:
            mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
            cos_t = ttnn.to_torch(cos, mesh_composer=mesh_composer)[:1]
            sin_t = ttnn.to_torch(sin, mesh_composer=mesh_composer)[:1]
        else:
            cos_t = ttnn.to_torch(cos)
            sin_t = ttnn.to_torch(sin)

        cos_t = cos_t[..., :half_dim].repeat_interleave(2, dim=-1)
        sin_t = sin_t[..., :half_dim].repeat_interleave(2, dim=-1)

        cos = ttnn.from_torch(
            cos_t,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.from_torch(
            sin_t,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cos, sin

    def _project_qkv(self, hidden_states, batch_size, seq_length, position_embeddings):
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self.q_lora_rank is not None:
            q_latent = self.q_a_proj(hidden_states)
            q_latent = self.q_a_layernorm(q_latent)
            q_states = self.q_b_proj(q_latent)
        else:
            q_states = self.q_proj(hidden_states)

        q_states = self._maybe_all_gather(q_states)

        q_states = ttnn.reshape(q_states, (batch_size, seq_length, self.num_heads, -1))
        q_states = ttnn.permute(q_states, (0, 2, 1, 3))

        q_pass = ttnn.slice(q_states, (0, 0, 0, 0), (batch_size, self.num_heads, seq_length, self.qk_nope_head_dim))
        q_rot = ttnn.slice(
            q_states,
            (0, 0, 0, self.qk_nope_head_dim),
            (batch_size, self.num_heads, seq_length, self.qk_head_dim),
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = self._maybe_all_gather(compressed_kv)

        k_pass_flat = ttnn.slice(compressed_kv, (0, 0, 0), (batch_size, seq_length, self.kv_lora_rank))
        k_rot = ttnn.slice(
            compressed_kv,
            (0, 0, self.kv_lora_rank),
            (batch_size, seq_length, self.kv_lora_rank + self.qk_rope_head_dim),
        )

        if len(k_pass_flat.shape) == 3:
            k_pass_flat = ttnn.unsqueeze(k_pass_flat, 1)
        k_pass_flat = ttnn.rms_norm(k_pass_flat, weight=self._kv_a_ln_weight, epsilon=self._kv_a_ln_eps)
        k_pass_flat = ttnn.squeeze(k_pass_flat, 1)
        kv_full = self.kv_b_proj(k_pass_flat)
        kv_full = self._maybe_all_gather(kv_full)

        kv_full = ttnn.reshape(
            kv_full, (batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        )
        kv_full = ttnn.permute(kv_full, (0, 2, 1, 3))

        key_nope = ttnn.slice(kv_full, (0, 0, 0, 0), (batch_size, self.num_heads, seq_length, self.qk_nope_head_dim))
        value_states = ttnn.slice(
            kv_full,
            (0, 0, 0, self.qk_nope_head_dim),
            (batch_size, self.num_heads, seq_length, self.qk_nope_head_dim + self.v_head_dim),
        )

        k_rot = ttnn.reshape(k_rot, (batch_size, 1, seq_length, self.qk_rope_head_dim))

        cos, sin = position_embeddings
        if len(cos.shape) == 3:
            cos = ttnn.unsqueeze(cos, 1)
        if self._is_distributed:
            cos = self._maybe_all_gather(cos)
        if len(sin.shape) == 3:
            sin = ttnn.unsqueeze(sin, 1)
        if self._is_distributed:
            sin = self._maybe_all_gather(sin)

        cos, sin = self._permute_cos_sin_to_meta_format(cos, sin)

        q_rot, k_rot = self.rope(q_rot, k_rot, cos, sin)

        k_rot = ttnn.repeat(k_rot.to_ttnn, (1, self.num_heads, 1, 1))

        query_states = ttnn.concat([q_pass, q_rot.to_ttnn], dim=-1)
        key_states = ttnn.concat([key_nope, k_rot], dim=-1)

        return query_states, key_states, value_states, cos, sin

    def _forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        query_states, key_states, value_states, cos, sin = self._project_qkv(
            hidden_states, batch_size, seq_length, position_embeddings
        )

        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)

        if past_key_values is not None:
            layer_idx = self._fallback_torch_layer.layer_idx

            if use_paged:
                past_key_values.paged_fill_on_device(
                    key_states,
                    value_states,
                    layer_idx=layer_idx,
                    batch_idx=0,
                )
                # key_states / value_states are already all-gathered from
                # _project_qkv — no second all-gather needed.
            else:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                torch_tensors = [TorchTTNNTensor(key_states), TorchTTNNTensor(value_states)]
                orig_shapes = [key_states.shape, value_states.shape]

                torch_tensors = [
                    torch_tensor.to_torch[: orig_shape[0], : orig_shape[1], : orig_shape[2], : orig_shape[3]]
                    for orig_shape, torch_tensor in zip(orig_shapes, torch_tensors)
                ]

                key_states, value_states = past_key_values.update(
                    *torch_tensors,
                    layer_idx,
                    cache_kwargs,
                )
                key_states, value_states = [TorchTTNNTensor(key_states), TorchTTNNTensor(value_states)]
                key_states = ttnn.to_device(key_states.to_ttnn, self.device)
                value_states = ttnn.to_device(value_states.to_ttnn, self.device)
                key_states = self._maybe_all_gather(key_states)
                value_states = self._maybe_all_gather(value_states)

        if self.qk_head_dim != self.v_head_dim:
            pad_size = self.qk_head_dim - self.v_head_dim
            value_states = ttnn.pad(value_states, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)

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

        if self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_heads * self.v_head_dim))
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _to_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Convert a multi-device tensor to an explicitly replicated tensor.

        After all-gather the data is identical on every device but the mesh
        topology metadata differs from ReplicateTensorToMesh.  Paged-attention
        kernels require the replicated topology, so we round-trip through the
        host for decode tokens (tiny tensors, negligible overhead).
        """
        if self.device.get_num_devices() <= 1:
            return tensor
        t = tensor
        if isinstance(t, TorchTTNNTensor):
            t = t.to_ttnn
        orig_shape = list(t.shape)
        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)
        t_torch = t_torch[: orig_shape[0]]
        return ttnn.from_torch(
            t_torch,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=t.dtype,
            layout=t.layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forward_decode_paged(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        past_key_values: "TTNNPagedAttentionKVCache",
        cache_position: Optional[torch.LongTensor],
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        """Decode path using paged attention with on-device KV cache.

        TTNN paged kernels require tensors in [1, batch, heads, head_dim]
        layout (``S B H D``) whereas ``_project_qkv`` returns the standard
        [batch, heads, seq, head_dim] (``B H S D``).  This method handles
        the permute, L1 sharding required by ``paged_update_cache``, and
        the MLA-aware SDPA decode call.
        """
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        query_states, key_states, value_states, cos, sin = self._project_qkv(
            hidden_states, batch_size, seq_length, position_embeddings
        )
        # _project_qkv returns [B, H, S, D]:
        #   Q : [B, num_heads, 1, qk_head_dim]
        #   K : [B, num_heads, 1, qk_head_dim]
        #   V : [B, num_heads, 1, v_head_dim]

        layer_idx = self._fallback_torch_layer.layer_idx

        # --- resolve cache position to a 1-D torch int32 tensor [batch] ---
        if cache_position is None:
            cur_pos = past_key_values.get_seq_length(layer_idx)
            cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
        else:
            cp = cache_position
            if isinstance(cp, TorchTTNNTensor):
                cp = cp.to_torch
            if isinstance(cp, ttnn.Tensor):
                mesh_composer = None
                if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
                    mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
                cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)
            cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        # 1-D [batch_size] tensor for paged_update_cache & paged_sdpa_decode
        cur_pos_tt = ttnn.from_torch(
            cache_position_tensor,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- permute B H S D  →  S B H D  (the layout paged kernels expect) ---
        query_states = ttnn.permute(query_states, (2, 0, 1, 3))
        key_states = ttnn.permute(key_states, (2, 0, 1, 3))
        value_states = ttnn.permute(value_states, (2, 0, 1, 3))

        # --- pad V to qk_head_dim so K/V caches share the same last dim ---
        if self.qk_head_dim != self.v_head_dim:
            pad_size = self.qk_head_dim - self.v_head_dim
            value_states = ttnn.pad(value_states, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)

        # --- multi-device: convert all-gathered topology → replicated ---
        if self.device.get_num_devices() > 1:
            query_states = self._to_replicated(query_states)
            key_states = self._to_replicated(key_states)
            value_states = self._to_replicated(value_states)

        tile_size = 32
        shard_h = ((self.num_heads + tile_size - 1) // tile_size) * tile_size

        core_grid = ttnn.CoreGrid(y=1, x=batch_size)
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(shard_h, self.qk_head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        key_states = ttnn.to_memory_config(key_states, shard_cfg)
        value_states = ttnn.to_memory_config(value_states, shard_cfg)

        # --- update the on-device paged KV cache ---
        past_key_values.paged_update_on_device(
            key_states,
            value_states,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        # --- paged SDPA decode (Q stays in DRAM) ---
        attn_output = past_key_values.paged_sdpa_decode(
            query_states,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.program_config,
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )
        # attn_output: [1, B, H, qk_head_dim]

        # --- convert back to [B, S, H*D_v] for the output projection ---
        attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, qk_head_dim]
        if self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_heads * self.v_head_dim))
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        seq_length = hidden_states.shape[1]
        use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)

        if use_paged and seq_length == 1:
            return self._forward_decode_paged(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
            )

        return self._forward_prefill(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
        )
