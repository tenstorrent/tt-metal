# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full TP=4 TTNN autoregressive model for google/gemma-4-26B-A4B-it.

This wrapper intentionally stacks :class:`MultichipDecoder` without changing
its replicated-BF16 inter-layer residual contract.  Embedding weights are
hidden-dimension sharded and gathered once at model entry.  The tied LM head
is vocabulary sharded and leaves its logits sharded for split sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    FULL_KIND,
    HIDDEN_SIZE,
    SLIDING_KIND,
    _layer_kind,
    _text_config,
    _validate_text_config,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import TP_SIZE, MultichipDecoder
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allgather

MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_MAX_CONTEXT = 262_144
SLIDING_CACHE_TOKENS = 1_024
DECODE_SLOT_COUNT = 32


def _require_tp4(mesh_device: Any) -> None:
    if not isinstance(mesh_device, ttnn.MeshDevice) or tuple(mesh_device.shape) != (1, TP_SIZE):
        shape = tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else None
        raise ValueError(f"Gemma4FullModel requires the optimized 1x{TP_SIZE} mesh, got {shape}")


def _find_key(state_dict: dict[str, torch.Tensor], *keys: str) -> str:
    for key in keys:
        if key in state_dict:
            return key
    raise KeyError(f"none of the required checkpoint tensors exists: {keys}")


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class PagedCacheSpec:
    layer_idx: int
    layer_type: str
    block_size: int
    local_kv_heads: int
    head_dim: int
    capacity_tokens_per_slot: int
    cache_dtype: Any = ttnn.bfloat16

    @property
    def blocks_per_slot(self) -> int:
        return _round_up(self.capacity_tokens_per_slot, self.block_size) // self.block_size


@dataclass
class FullModelState:
    """Explicit cache/page-table/slot state shared by prefill and decode."""

    kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]]
    page_tables: list[ttnn.Tensor]
    cache_specs: list[PagedCacheSpec]
    max_batch_size: int
    slot_context_lengths: list[int]
    prompt_lens: list[int]
    positions: torch.Tensor
    active_mask: torch.Tensor


class Gemma4FullModel:
    """All-layer TTNN model preserving the optimized multichip decoder policy."""

    tp_size = TP_SIZE
    supports_on_device_sampling = True
    residual_layout = "replicated BF16 tile-layout DRAM [1,1,M,2816]"

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        hf_config: Any,
        state_dict: dict[str, torch.Tensor],
        max_seq_len: int = DEFAULT_MAX_CONTEXT,
        max_batch_size: int = 1,
        num_layers: int | None = None,
        layer_indices: Sequence[int] | None = None,
        tensor_cache_path: str | Path | None = None,
        create_kv_cache: bool = True,
    ) -> None:
        _require_tp4(mesh_device)
        generation_eos = getattr(hf_config, "eos_token_id", None)
        self.eos_token_ids = tuple(
            int(token_id)
            for token_id in (generation_eos if isinstance(generation_eos, (list, tuple)) else [generation_eos])
            if token_id is not None
        )
        text_config = _text_config(hf_config)
        _validate_text_config(text_config)
        if max_seq_len < 1 or max_seq_len > text_config.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {text_config.max_position_embeddings}], got {max_seq_len}")
        if max_batch_size < 1 or max_batch_size > 32:
            raise ValueError(f"max_batch_size must be in [1, 32], got {max_batch_size}")

        self.mesh_device = mesh_device
        self.hf_config = text_config
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        if layer_indices is not None and num_layers is not None:
            raise ValueError("pass either num_layers or layer_indices, not both")
        self.layer_indices = (
            list(layer_indices)
            if layer_indices is not None
            else list(range(num_layers or text_config.num_hidden_layers))
        )
        self.num_layers = len(self.layer_indices)
        self.vocab_size = text_config.vocab_size
        self.embed_scale = text_config.hidden_size**0.5
        self.final_logit_softcapping = text_config.final_logit_softcapping
        self.tensor_cache_path = Path(tensor_cache_path) if tensor_cache_path is not None else None
        self.mesh_config = MeshConfig((1, TP_SIZE), decode=ModeConfig(tp=TP_SIZE))
        self.ccl_manager = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
        self._replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        embed_key = _find_key(
            state_dict,
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        )
        norm_key = _find_key(
            state_dict,
            "model.language_model.norm.weight",
            "language_model.norm.weight",
            "model.norm.weight",
        )
        embed_weight = state_dict[embed_key]
        if tuple(embed_weight.shape) != (self.vocab_size, HIDDEN_SIZE):
            raise ValueError(f"unexpected embedding shape {tuple(embed_weight.shape)}")

        cache_root = self.tensor_cache_path / "full_model" if self.tensor_cache_path is not None else None
        common = dict(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        embed_cache = str(cache_root / "embedding") if cache_root is not None else None
        lm_cache = str(cache_root / "lm_head") if cache_root is not None else None
        norm_cache = str(cache_root / "final_norm") if cache_root is not None else None
        self.embedding_weight = ttnn.as_tensor(
            embed_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            cache_file_name=embed_cache,
            **common,
        )
        self.lm_head_weight = ttnn.as_tensor(
            embed_weight.transpose(0, 1).contiguous().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            cache_file_name=lm_cache,
            **common,
        )
        self.final_norm_weight = ttnn.as_tensor(
            state_dict[norm_key].reshape(1, 1, 1, HIDDEN_SIZE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate,
            cache_file_name=norm_cache,
            **common,
        )

        self.layers = []
        persistent_all_reduce_resources = None
        for layer_idx in self.layer_indices:
            layer = MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=text_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                tensor_cache_path=self.tensor_cache_path,
                persistent_all_reduce_resources=persistent_all_reduce_resources,
            )
            self.layers.append(layer)
            if layer.persistent_all_reduce_resources is not None:
                persistent_all_reduce_resources = layer.persistent_all_reduce_resources
        self.rope_caches = self._create_rope_caches(text_config, max_seq_len)
        self.cache_specs = self._make_cache_specs()
        self.state = self.allocate_state(max_batch_size=max_batch_size) if create_kv_cache else None

    def _create_rope_caches(self, config: Any, max_seq_len: int) -> dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        rope = Gemma4TextRotaryEmbedding(config)
        positions = torch.arange(max_seq_len).unsqueeze(0)
        # RotaryEmbedding only uses shape/device metadata from x.
        dummy = torch.empty(1, max_seq_len, config.hidden_size)
        caches = {}
        for layer_type in sorted({config.layer_types[i] for i in self.layer_indices}):
            cos, sin = rope(dummy, positions, layer_type=layer_type)
            caches[layer_type] = tuple(
                ttnn.as_tensor(
                    value.squeeze(0),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=self._replicate,
                )
                for value in (cos, sin)
            )
        return caches

    def _make_cache_specs(self) -> list[PagedCacheSpec]:
        specs = []
        for state_idx, layer_idx in enumerate(self.layer_indices):
            layer_type = self.hf_config.layer_types[layer_idx]
            kind = _layer_kind(layer_type)
            specs.append(
                PagedCacheSpec(
                    layer_idx=layer_idx,
                    layer_type=layer_type,
                    block_size=kind.block_size,
                    local_kv_heads=1 if kind is FULL_KIND else 2,
                    head_dim=kind.head_dim,
                    capacity_tokens_per_slot=self.max_seq_len if kind is FULL_KIND else SLIDING_CACHE_TOKENS,
                )
            )
        return specs

    def allocate_state(
        self,
        *,
        max_batch_size: int | None = None,
        slot_context_lengths: Sequence[int] | None = None,
    ) -> FullModelState:
        batch = max_batch_size or self.max_batch_size
        if batch < 1 or batch > self.max_batch_size:
            raise ValueError(f"state batch must be in [1, {self.max_batch_size}], got {batch}")
        if slot_context_lengths is None:
            base, remainder = divmod(self.max_seq_len, batch)
            slot_context_lengths = [base + (row < remainder) for row in range(batch)]
        else:
            slot_context_lengths = [int(value) for value in slot_context_lengths]
            if len(slot_context_lengths) != batch:
                raise ValueError("slot_context_lengths must contain one value per active row")
            if any(value < 1 for value in slot_context_lengths):
                raise ValueError("slot context capacities must be positive")
            if sum(slot_context_lengths) > self.max_seq_len:
                raise ValueError("aggregate slot capacity exceeds the full-context KV budget")
        kv_cache = []
        page_tables = []
        for spec in self.cache_specs:
            cache_lengths = (
                [spec.capacity_tokens_per_slot] * batch
                if spec.layer_type == "sliding_attention"
                else [int(value) for value in slot_context_lengths]
            )
            row_blocks = [_round_up(value, spec.block_size) // spec.block_size for value in cache_lengths]
            blocks = sum(row_blocks)
            shape = (blocks, spec.local_kv_heads, spec.block_size, spec.head_dim)
            cache_pair = tuple(
                ttnn.zeros(
                    shape,
                    dtype=spec.cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
            table = torch.zeros((DECODE_SLOT_COUNT, spec.blocks_per_slot), dtype=torch.int32)
            next_block = 0
            for row, row_block_count in enumerate(row_blocks):
                table[row, :row_block_count] = torch.arange(next_block, next_block + row_block_count, dtype=torch.int32)
                next_block += row_block_count
            page_table = ttnn.from_torch(
                table,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._replicate,
            )
            kv_cache.append(cache_pair)
            page_tables.append(page_table)
        return FullModelState(
            kv_cache=kv_cache,
            page_tables=page_tables,
            cache_specs=self.cache_specs,
            max_batch_size=batch,
            slot_context_lengths=list(slot_context_lengths) + [0] * (DECODE_SLOT_COUNT - batch),
            prompt_lens=[0] * DECODE_SLOT_COUNT,
            positions=torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32),
            active_mask=torch.zeros(DECODE_SLOT_COUNT, dtype=torch.bool),
        )

    def embed_tokens(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        if len(tokens.shape) == 4:
            tokens = ttnn.reshape(tokens, (tokens.shape[-2], tokens.shape[-1]))
        hidden = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        hidden = ttnn.mul(hidden, self.embed_scale)
        hidden = ttnn.unsqueeze_to_4D(hidden) if len(hidden.shape) == 3 else hidden
        hidden = ccl_allgather(hidden, self.mesh_config, self.ccl_manager, dim=3)
        return ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)

    def _rope_rows(
        self, layer_type: str, positions: ttnn.Tensor, *, decode: bool = False
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cos_cache, sin_cache = self.rope_caches[layer_type]
        rows = []
        for table in (cos_cache, sin_cache):
            value = ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT)
            if decode:
                batch = positions.shape[0]
                shape = (
                    (1, batch, 1, value.shape[-1])
                    if _layer_kind(layer_type) is SLIDING_KIND
                    else (
                        1,
                        1,
                        batch,
                        value.shape[-1],
                    )
                )
                value = ttnn.reshape(value, shape)
            elif len(value.shape) == 3:
                value = ttnn.unsqueeze_to_4D(value)
            rows.append(value)
        return tuple(rows)

    def _terminal(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        hidden = ttnn.rms_norm(
            hidden,
            weight=self.final_norm_weight,
            epsilon=self.hf_config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.linear(hidden, self.lm_head_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.final_logit_softcapping:
            logits = ttnn.mul(logits, 1.0 / self.final_logit_softcapping)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, self.final_logit_softcapping)
        return logits

    def prefill_forward(
        self,
        tokens: ttnn.Tensor,
        *,
        state: FullModelState,
        prompt_lens: Sequence[int],
        position_ids: ttnn.Tensor,
        user_id: int = 0,
        return_all_logits: bool = False,
    ) -> ttnn.Tensor:
        """Device-only prefill over explicit externally owned state.

        ``tokens`` and ``position_ids`` are already padded physical tensors;
        ``prompt_lens`` remains the logical contract and selects final rows.
        """
        hidden = self.embed_tokens(tokens)
        for state_idx, (layer_idx, layer) in enumerate(zip(self.layer_indices, self.layers)):
            layer_type = self.hf_config.layer_types[layer_idx]
            cos, sin = self._rope_rows(layer_type, position_ids)
            hidden = layer.prefill_forward(
                hidden,
                position_cos=cos,
                position_sin=sin,
                page_table=state.page_tables[state_idx],
                kv_cache=state.kv_cache[state_idx],
                user_id=user_id,
                cache_position_modulo=(SLIDING_CACHE_TOKENS if _layer_kind(layer_type) is SLIDING_KIND else None),
            )
        if not return_all_logits:
            if len(set(prompt_lens)) != 1:
                raise NotImplementedError("mixed-length terminal prefill slicing is owned by the generator")
            last = int(prompt_lens[0]) - 1
            hidden = ttnn.slice(hidden, (0, 0, last, 0), (hidden.shape[0], 1, last + 1, HIDDEN_SIZE))
        return self._terminal(hidden)

    def decode_forward(
        self,
        tokens: ttnn.Tensor,
        *,
        state: FullModelState,
        current_pos: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        batch_size: int | None = None,
    ) -> ttnn.Tensor:
        """Trace-safe token-to-sharded-logits decode over stable device state."""
        hidden = self.embed_tokens(tokens)
        logical_batch = int(batch_size or hidden.shape[-2])
        if logical_batch < 1 or logical_batch > hidden.shape[-2]:
            raise ValueError(f"decode batch_size must be in [1, {hidden.shape[-2]}]")
        if logical_batch != hidden.shape[-2]:
            hidden = ttnn.slice(hidden, (0, 0, 0, 0), (1, 1, logical_batch, HIDDEN_SIZE))
        logical_current_pos = current_pos
        logical_position_ids = position_ids
        if current_pos.shape[0] != logical_batch:
            logical_current_pos = ttnn.slice(current_pos, [0], [logical_batch])
            logical_position_ids = ttnn.slice(position_ids, [0], [logical_batch])
        for state_idx, (layer_idx, layer) in enumerate(zip(self.layer_indices, self.layers)):
            layer_type = self.hf_config.layer_types[layer_idx]
            cos, sin = self._rope_rows(layer_type, logical_position_ids, decode=True)
            page_table = state.page_tables[state_idx]
            if page_table.shape[0] != logical_batch:
                page_table = ttnn.slice(page_table, [0, 0], [logical_batch, page_table.shape[1]])
            hidden = layer.decode_forward(
                hidden,
                position_cos=cos,
                position_sin=sin,
                current_pos=logical_current_pos,
                page_table=page_table,
                kv_cache=state.kv_cache[state_idx],
                cache_position_modulo=(SLIDING_CACHE_TOKENS if _layer_kind(layer_type) is SLIDING_KIND else None),
            )
        return self._terminal(hidden)

    @staticmethod
    def sampling_args(mesh_device: ttnn.MeshDevice, *, max_batch_size: int) -> Any:
        """Minimal TTTv1 sampler args retained for the sampler comparison harness."""
        return SimpleNamespace(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_top_k=32,
            vocab_size=262_144,
            padded_vocab_size=262_144,
            num_devices=TP_SIZE,
            sub_core_grids=None,
            sampling_core_grid=None,
            users_row_sharded=False,
            is_galaxy=False,
            is_llama_vision=lambda: False,
        )


__all__ = ["DECODE_SLOT_COUNT", "FullModelState", "Gemma4FullModel", "PagedCacheSpec"]
