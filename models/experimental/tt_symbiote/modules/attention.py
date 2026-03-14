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
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object

try:
    from transformers.cache_utils import CacheLayerMixin
except ImportError:
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

        past_key_values._seq_lengths[layer_idx] += seq_length
        if layer_idx == 0:
            past_key_values._seen_tokens += seq_length

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
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        is_causal=True,
        scale=scaling,
        program_config=_gated_attention_sdpa_config(device, T),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    attn_output = ttnn.transpose(attn_output, 1, 2)
    attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim])
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)
    attn_output = ttnn.linear(attn_output, o_proj_weight)
    return attn_output


class TTNNQwen3NextGatedAttention(TTNNModule):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn
        new_attn.num_attention_heads = torch_attn.config.num_attention_heads
        new_attn.num_key_value_heads = torch_attn.config.num_key_value_heads
        new_attn.head_dim = torch_attn.head_dim
        new_attn.hidden_size = torch_attn.config.hidden_size
        new_attn.norm_eps = torch_attn.config.rms_norm_eps
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
        cos, sin = position_embeddings
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


class TTNNBailingMoEAttention(TTNNModule):
    """TTNN Attention for BailingMoeV2 (Ling-mini-2.0 model).

    Supports both standard DynamicCache and TTNNPagedAttentionKVCache
    for paged attention with on-device KV storage.
    """

    def __init__(self):
        super().__init__()
        self.num_heads = None
        self.num_kv_heads = None
        self.head_dim = None
        self.hidden_size = None
        self.use_qk_norm = False
        self.partial_rotary_factor = 1.0
        self.is_causal = True
        self.scaling = None

        self.query_key_value = None
        self.dense = None
        self.query_layernorm = None
        self.key_layernorm = None
        self.rope = None
        self.sdpa = None

        # Separate Q, K, V projections for distributed mode when num_kv_heads < num_devices
        # In this case, Q is sharded (num_heads >= num_devices) but K/V must be replicated
        self._use_separate_qkv = False
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager."""
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
        t = tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
        gathered = ttnn.experimental.all_gather_async(
            t,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        # Ensure output is BFLOAT16 for compatibility with downstream ops (e.g., RoPE)
        if gathered.dtype != ttnn.bfloat16:
            gathered = ttnn.typecast(gathered, ttnn.bfloat16)
        return gathered

    def _to_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Convert a multi-device tensor to an explicitly replicated tensor.

        After all-gather the data is identical on every device but the mesh
        topology metadata differs from ReplicateTensorToMesh. Paged-attention
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

    @classmethod
    def from_torch(cls, torch_attn, distributed: bool = True):
        """Create TTNNBailingMoEAttention from BailingMoeV2Attention/SdpaAttention.

        Args:
            torch_attn: PyTorch BailingMoeV2 attention module
            distributed: Whether to use distributed linear/norm modules for mesh devices.
                         Defaults to True for multi-device compatibility.

        Note:
            When distributed=True and the model has fewer KV heads than devices (e.g.,
            Ling-mini-2.0 with 4 KV heads on 8 devices), this method automatically
            splits the fused QKV projection into separate Q, K, V projections:
            - Q projection uses TTNNLinearIColShardedWRowSharded (sharded, since num_heads >= num_devices)
            - K/V projections use TTNNLinearIReplicatedWColSharded (replicated input, col-sharded output)
            This allows running on more devices than KV heads.
        """
        from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm

        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn

        # Extract attention configuration
        config = torch_attn.config
        new_attn.num_heads = config.num_attention_heads
        new_attn.num_kv_heads = config.num_key_value_heads
        new_attn.head_dim = config.hidden_size // config.num_attention_heads
        new_attn.hidden_size = config.hidden_size
        new_attn.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        new_attn.use_qk_norm = getattr(config, "use_qk_norm", False)
        new_attn.scaling = new_attn.head_dim**-0.5

        # Select linear class based on distributed mode
        LinearCls = TTNNLinearIColShardedWRowSharded if distributed else TTNNLinear
        LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
        NormCls = TTNNDistributedRMSNorm if distributed else TTNNRMSNorm

        if distributed:
            # In distributed mode, we need separate Q, K, V projections to handle
            # the case where num_kv_heads < num_devices. This allows:
            # - Q projection to be sharded (num_heads >= num_devices typically)
            # - K/V projections to be replicated (num_kv_heads < num_devices)
            new_attn._use_separate_qkv = True

            # Split the fused query_key_value weight into separate Q, K, V weights
            qkv_weight = torch_attn.query_key_value.weight  # [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]
            q_size = new_attn.num_heads * new_attn.head_dim  # e.g., 16 * 128 = 2048
            kv_size = new_attn.num_kv_heads * new_attn.head_dim  # e.g., 4 * 128 = 512

            q_weight = qkv_weight[:q_size, :]
            k_weight = qkv_weight[q_size : q_size + kv_size, :]
            v_weight = qkv_weight[q_size + kv_size :, :]

            # Handle bias if present
            q_bias = k_bias = v_bias = None
            if torch_attn.query_key_value.bias is not None:
                qkv_bias = torch_attn.query_key_value.bias
                q_bias = qkv_bias[:q_size]
                k_bias = qkv_bias[q_size : q_size + kv_size]
                v_bias = qkv_bias[q_size + kv_size :]

            # Create temporary torch.nn.Linear modules for from_torch
            import torch.nn as nn

            q_linear = nn.Linear(new_attn.hidden_size, q_size, bias=q_bias is not None)
            q_linear.weight.data = q_weight
            if q_bias is not None:
                q_linear.bias.data = q_bias

            k_linear = nn.Linear(new_attn.hidden_size, kv_size, bias=k_bias is not None)
            k_linear.weight.data = k_weight
            if k_bias is not None:
                k_linear.bias.data = k_bias

            v_linear = nn.Linear(new_attn.hidden_size, kv_size, bias=v_bias is not None)
            v_linear.weight.data = v_weight
            if v_bias is not None:
                v_linear.bias.data = v_bias

            # Q projection: sharded input, row-sharded output (num_heads >= num_devices)
            new_attn.q_proj = LinearCls.from_torch(q_linear)

            # K/V projections: replicated input, col-sharded output (num_kv_heads < num_devices)
            # Using TTNNLinearIReplicatedWColSharded allows K/V to work even when
            # num_kv_heads < num_devices because the input is replicated
            new_attn.k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)
            new_attn.v_proj = TTNNLinearIReplicatedWColSharded.from_torch(v_linear)

            # No fused QKV in distributed mode with separate projections
            new_attn.query_key_value = None
        else:
            # Non-distributed mode: use fused query_key_value projection
            new_attn._use_separate_qkv = False
            new_attn.query_key_value = LinearCls.from_torch(torch_attn.query_key_value)

        # Create dense (output) projection
        new_attn.dense = LinearClsOut.from_torch(torch_attn.dense)

        # Create QK normalization layers if enabled
        # QK norms operate on head_dim (128), not hidden_size (2048), so always use
        # non-distributed version. Distributed norm tries to shard head_dim // 32 = 4
        # chunks across 8 devices, which fails.
        if new_attn.use_qk_norm:
            new_attn.query_layernorm = TTNNRMSNorm.from_torch(torch_attn.query_layernorm)
            new_attn.key_layernorm = TTNNRMSNorm.from_torch(torch_attn.key_layernorm)

        # Create RoPE and SDPA modules
        # When partial_rotary_factor < 1.0, use non-distributed RoPE which handles
        # partial rotary correctly. TTNNDistributedRotaryPositionEmbedding's underlying
        # rotary_embedding_llama kernel requires cos.shape[-1] == head_dim.
        # This follows the same pattern as TTNNQwen3FullAttention.
        uses_partial_rotary = new_attn.partial_rotary_factor < 1.0
        if uses_partial_rotary:
            new_attn.rope = TTNNRotaryPositionEmbedding()
        else:
            new_attn.rope = TTNNDistributedRotaryPositionEmbedding() if distributed else TTNNRotaryPositionEmbedding()
        new_attn.sdpa = TTNNSDPAAttention()
        new_attn.core_grid = ttnn.CoreGrid(y=8, x=8)

        return new_attn

    def preprocess_weights_impl(self):
        """Preprocess weights for TTNN operations.

        Note: Base class handles calling preprocess_weights() on all child modules.
        """
        super().preprocess_weights_impl()

    def move_weights_to_device_impl(self):
        """Move weights to device and initialize SDPA config.

        Note: Base class handles calling move_weights_to_device() on all child modules.
        """
        super().move_weights_to_device_impl()

        # Initialize SDPA config when device is available
        if self.sdpa.program_config is None:
            self.sdpa.program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=256,
                k_chunk_size=256,
                exp_approx_mode=False,
            )
            self.sdpa.decode_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
            )
            self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def _split_qkv(self, qkv: ttnn.Tensor, batch_size: int, seq_length: int):
        """Split fused QKV tensor into separate Q, K, V tensors.

        Args:
            qkv: Fused QKV tensor of shape [batch, seq, (num_heads + 2*num_kv_heads) * head_dim]
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Tuple of (query_states, key_states, value_states)
        """
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        # Split along last dimension
        query_states = ttnn.slice(qkv, (0, 0, 0), (batch_size, seq_length, q_size))
        key_states = ttnn.slice(qkv, (0, 0, q_size), (batch_size, seq_length, q_size + kv_size))
        value_states = ttnn.slice(qkv, (0, 0, q_size + kv_size), (batch_size, seq_length, q_size + 2 * kv_size))

        # Reshape to [batch, seq, num_heads, head_dim]
        query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
        key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
        value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))

        # Transpose to [batch, heads, seq, head_dim]
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        return query_states, key_states, value_states

    def _apply_qk_norm(self, query_states: ttnn.Tensor, key_states: ttnn.Tensor):
        """Apply QK normalization if enabled.

        Args:
            query_states: Query tensor [batch, heads, seq, head_dim]
            key_states: Key tensor [batch, heads, seq, head_dim]

        Returns:
            Tuple of (normalized_query, normalized_key)
        """
        if not self.use_qk_norm:
            return query_states, key_states

        # Reshape for normalization: [batch, heads, seq, head_dim] -> [batch*heads*seq, head_dim]
        batch_size, num_heads, seq_length, head_dim = query_states.shape
        batch_kv, num_kv_heads, seq_length_k, head_dim_k = key_states.shape

        # Apply normalization
        q_reshaped = ttnn.reshape(query_states, (batch_size * num_heads * seq_length, head_dim))
        k_reshaped = ttnn.reshape(key_states, (batch_kv * num_kv_heads * seq_length_k, head_dim_k))

        q_normed = self.query_layernorm(q_reshaped)
        k_normed = self.key_layernorm(k_reshaped)

        # Unwrap TorchTTNNTensor if needed
        if hasattr(q_normed, "to_ttnn"):
            q_normed = q_normed.to_ttnn
        if hasattr(k_normed, "to_ttnn"):
            k_normed = k_normed.to_ttnn

        # Ensure BFLOAT16 dtype for compatibility with downstream RoPE ops
        if q_normed.dtype != ttnn.bfloat16:
            q_normed = ttnn.typecast(q_normed, ttnn.bfloat16)
        if k_normed.dtype != ttnn.bfloat16:
            k_normed = ttnn.typecast(k_normed, ttnn.bfloat16)

        # Reshape back
        query_states = ttnn.reshape(q_normed, (batch_size, num_heads, seq_length, head_dim))
        key_states = ttnn.reshape(k_normed, (batch_kv, num_kv_heads, seq_length_k, head_dim_k))

        return query_states, key_states

    def _apply_partial_rope(
        self,
        query_states: ttnn.Tensor,
        key_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ):
        """Apply partial RoPE based on partial_rotary_factor.

        Args:
            query_states: Query tensor [batch, heads, seq, head_dim]
            key_states: Key tensor [batch, heads, seq, head_dim]
            cos: Cosine position embeddings
            sin: Sine position embeddings

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        # The RoPE module handles partial rotary embedding internally based on cos/sin dimensions
        # cos/sin should already be sized according to partial_rotary_factor
        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        # Handle TorchTTNNTensor wrapping
        if hasattr(query_states, "to_ttnn"):
            query_states = query_states.to_ttnn
        if hasattr(key_states, "to_ttnn"):
            key_states = key_states.to_ttnn

        return query_states, key_states

    def _forward_prefill(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
    ) -> tuple:
        """Forward pass for prefill phase."""
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # Ensure proper layout
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._use_separate_qkv:
            # Separate Q, K, V projections path (for distributed mode with num_kv_heads < num_devices)
            # Q projection can use sharded input (TTNNLinearIColShardedWRowSharded)
            query_states = self.q_proj(hidden_states)

            # K/V projections need replicated (all-gathered) input since they use
            # TTNNLinearIReplicatedWColSharded which expects full tensor width
            if self.device.get_num_devices() > 1:
                hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
            else:
                hidden_states_replicated = hidden_states

            key_states = self.k_proj(hidden_states_replicated)
            value_states = self.v_proj(hidden_states_replicated)

            # Deallocate the gathered tensor if we created one
            if self.device.get_num_devices() > 1:
                ttnn.deallocate(hidden_states_replicated)

            # All-gather projection outputs for reshape (distributed mode produces sharded outputs)
            # _maybe_all_gather also handles TorchTTNNTensor unwrapping
            query_states = self._maybe_all_gather(query_states)
            key_states = self._maybe_all_gather(key_states)
            value_states = self._maybe_all_gather(value_states)

            # Reshape to [batch, seq, num_heads, head_dim]
            query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
            key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
            value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))

            # Transpose to [batch, heads, seq, head_dim]
            query_states = ttnn.permute(query_states, (0, 2, 1, 3))
            key_states = ttnn.permute(key_states, (0, 2, 1, 3))
            value_states = ttnn.permute(value_states, (0, 2, 1, 3))
        else:
            # Fused QKV path (non-distributed or compatible distributed mode)
            qkv = self.query_key_value(hidden_states)
            if hasattr(qkv, "to_ttnn"):
                qkv = qkv.to_ttnn

            # Split into Q, K, V
            query_states, key_states, value_states = self._split_qkv(qkv, batch_size, seq_length)

        # Apply QK normalization if enabled
        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        # Apply RoPE
        cos, sin = position_embeddings

        # Handle position embeddings - they should be REPLICATED across devices, not sharded
        # The framework default shards inputs, but cos/sin must be identical on all devices
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def _ensure_replicated_tensor(t, name):
            """Convert tensor to TTNN with proper replication for multi-device.

            The framework default shards inputs, but position embeddings (cos/sin)
            must be identical on all devices, so we need to gather and re-replicate.
            """
            num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1

            # If it's already an TTNN tensor with wrong sharding, we need to convert back and re-convert
            if isinstance(t, ttnn.Tensor):
                # Check if tensor appears to be sharded (last dim smaller than expected)
                t_shape = list(t.shape)
                # Position embeddings should have shape like [1, 1, seq_len, rotary_dim] or [batch, seq_len, rotary_dim]
                # If last dim is divided by num_devices, it's been sharded
                if num_devices > 1 and t_shape[-1] < 32:  # rotary_dim is typically >= 32
                    # Tensor was sharded, need to convert back and re-convert with replication
                    # Use mesh_composer to gather the tensor first
                    torch_t = ttnn.to_torch(
                        t,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(self.device, self.device.shape, (0, -1)),
                    )
                    mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
                    return ttnn.from_torch(
                        torch_t.to(torch.bfloat16),
                        device=self.device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper,
                    )
                return t

            # If it's a TorchTTNNTensor, extract the original torch tensor and re-convert
            if isinstance(t, TorchTTNNTensor):
                if t.elem is not None:
                    torch_t = t.elem
                else:
                    torch_t = ttnn.to_torch(t.ttnn_tensor if t.ttnn_tensor is not None else t.to_ttnn)
                mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if num_devices > 1 else None
                return ttnn.from_torch(
                    torch_t.to(torch.bfloat16),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=mesh_mapper,
                )

            elif isinstance(t, torch.Tensor):
                mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if num_devices > 1 else None
                return ttnn.from_torch(
                    t.to(torch.bfloat16),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=mesh_mapper,
                )
            return t

        cos = _ensure_replicated_tensor(cos, "cos")
        sin = _ensure_replicated_tensor(sin, "sin")

        # Ensure query/key states are BFLOAT16 for RoPE compatibility
        if query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)

        # Handle KV cache
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

        # Apply SDPA
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
        )

        if hasattr(attn_output, "to_ttnn"):
            attn_output = attn_output.to_ttnn

        # Reshape and project output
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_heads * self.head_dim))
        attn_output = self.dense(attn_output)

        # Return format matches HuggingFace: (attn_output, attn_weights, past_key_values)
        return attn_output, None, past_key_values

    def _forward_decode_paged(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor],
        past_key_values: "TTNNPagedAttentionKVCache",
        cache_position: Optional[torch.LongTensor],
    ) -> tuple:
        """Decode path using paged attention with on-device KV cache."""
        batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

        # Ensure proper layout
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._use_separate_qkv:
            # Separate Q, K, V projections path (for distributed mode with num_kv_heads < num_devices)
            # Q projection can use sharded input (TTNNLinearIColShardedWRowSharded)
            query_states = self.q_proj(hidden_states)

            # K/V projections need replicated (all-gathered) input since they use
            # TTNNLinearIReplicatedWColSharded which expects full tensor width
            if self.device.get_num_devices() > 1:
                hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
            else:
                hidden_states_replicated = hidden_states

            key_states = self.k_proj(hidden_states_replicated)
            value_states = self.v_proj(hidden_states_replicated)

            # Deallocate the gathered tensor if we created one
            if self.device.get_num_devices() > 1:
                ttnn.deallocate(hidden_states_replicated)

            # All-gather projection outputs for reshape (distributed mode produces sharded outputs)
            # _maybe_all_gather also handles TorchTTNNTensor unwrapping
            query_states = self._maybe_all_gather(query_states)
            key_states = self._maybe_all_gather(key_states)
            value_states = self._maybe_all_gather(value_states)

            # Reshape to [batch, seq, num_heads, head_dim]
            query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
            key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
            value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))

            # Transpose to [batch, heads, seq, head_dim]
            query_states = ttnn.permute(query_states, (0, 2, 1, 3))
            key_states = ttnn.permute(key_states, (0, 2, 1, 3))
            value_states = ttnn.permute(value_states, (0, 2, 1, 3))
        else:
            # Fused QKV path (non-distributed or compatible distributed mode)
            qkv = self.query_key_value(hidden_states)
            if hasattr(qkv, "to_ttnn"):
                qkv = qkv.to_ttnn

            # Split into Q, K, V
            query_states, key_states, value_states = self._split_qkv(qkv, batch_size, seq_length)

        # Apply QK normalization if enabled
        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        # Apply RoPE
        cos, sin = position_embeddings
        if isinstance(cos, torch.Tensor):
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            cos = ttnn.from_torch(
                cos.to(torch.bfloat16),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mesh_mapper,
            )
        if isinstance(sin, torch.Tensor):
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            sin = ttnn.from_torch(
                sin.to(torch.bfloat16),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mesh_mapper,
            )

        # Ensure query/key states are BFLOAT16 for RoPE compatibility
        if query_states.dtype != ttnn.bfloat16:
            query_states = ttnn.typecast(query_states, ttnn.bfloat16)
        if key_states.dtype != ttnn.bfloat16:
            key_states = ttnn.typecast(key_states, ttnn.bfloat16)

        query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)

        layer_idx = self._fallback_torch_layer.layer_idx

        # Resolve cache position
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
        cur_pos_tt = ttnn.from_torch(
            cache_position_tensor,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Permute B H S D -> S B H D for paged kernels
        query_states = ttnn.permute(query_states, (2, 0, 1, 3))
        key_states = ttnn.permute(key_states, (2, 0, 1, 3))
        value_states = ttnn.permute(value_states, (2, 0, 1, 3))

        # Multi-device: convert all-gathered topology -> replicated for paged kernels
        if self.device.get_num_devices() > 1:
            query_states = self._to_replicated(query_states)
            key_states = self._to_replicated(key_states)
            value_states = self._to_replicated(value_states)

        # Update paged KV cache
        tile_size = 32
        shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size

        core_grid = ttnn.CoreGrid(y=1, x=batch_size)
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(shard_h, self.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        key_states = ttnn.to_memory_config(key_states, shard_cfg)
        value_states = ttnn.to_memory_config(value_states, shard_cfg)

        past_key_values.paged_update_on_device(
            key_states,
            value_states,
            layer_idx=layer_idx,
            current_pos=cur_pos_tt,
        )
        ttnn.deallocate(key_states)
        ttnn.deallocate(value_states)

        past_key_values._seq_lengths[layer_idx] += seq_length
        if layer_idx == 0:
            past_key_values._seen_tokens += seq_length

        # Paged SDPA decode
        # Use the same cur_pos_tt for both paged_update_on_device and paged_sdpa_decode
        # This matches the Qwen implementation semantics
        attn_output = past_key_values.paged_sdpa_decode(
            query_states,
            layer_idx,
            current_pos=cur_pos_tt,
            scale=self.scaling,
            program_config=self.sdpa.decode_program_config,  # Use decode config (q_chunk_size=0, k_chunk_size=0)
            compute_kernel_config=self.sdpa.compute_kernel_config,
        )

        # Convert back to [B, S, H*D] for output projection
        attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, head_dim]
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, self.num_heads * self.head_dim))
        attn_output = self.dense(attn_output)

        # Return format matches HuggingFace: (attn_output, attn_weights, past_key_values)
        return attn_output, None, past_key_values

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple:
        """Forward pass through BailingMoE attention.

        Args:
            hidden_states: Input tensor [batch, seq, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask
            past_key_values: KV cache (TTNNPagedAttentionKVCache or DynamicCache)
            cache_position: Position in cache for decode
            position_ids: Position IDs (unused, for compatibility)
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, None)
        """
        # Handle TorchTTNNTensor input
        if hasattr(hidden_states, "to_ttnn"):
            hidden_states = hidden_states.to_ttnn

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
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        is_causal=True,
        scale=scaling,
        program_config=_gated_attention_sdpa_config(device, T),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    attn_output = ttnn.transpose(attn_output, 1, 2)
    attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim])
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)
    attn_output = ttnn.linear(attn_output, o_proj_weight)
    return attn_output


class TTNNQwen3NextGatedAttention(TTNNModule):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_torch(cls, torch_attn):
        new_attn = cls()
        new_attn._fallback_torch_layer = torch_attn
        new_attn.num_attention_heads = torch_attn.config.num_attention_heads
        new_attn.num_key_value_heads = torch_attn.config.num_key_value_heads
        new_attn.head_dim = torch_attn.head_dim
        new_attn.norm_eps = torch_attn.config.rms_norm_eps
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

    def forward(self, hidden_states, position_embeddings):
        cos, sin = position_embeddings
        if isinstance(hidden_states, TorchTTNNTensor):
            hidden_states = hidden_states.to_ttnn
        if isinstance(cos, torch.Tensor):
            cos = ttnn.from_torch(cos.to(torch.bfloat16), device=self.device, layout=ttnn.TILE_LAYOUT)
        if isinstance(sin, torch.Tensor):
            sin = ttnn.from_torch(sin.to(torch.bfloat16), device=self.device, layout=ttnn.TILE_LAYOUT)
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return gated_attention_forward_ttnn(
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
        )
