# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full TTNN autoregressive model for ``Qwen/Qwen3.6-35B-A3B``.

The model wraps the optimized 2x2 multichip decoder stack and owns the text
embedding, final language RMSNorm, vocab-sharded LM head, rotary tables, and
full-stack cache/state contract used by readiness generators.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

import ttnn
from models.common.sampling import SamplingGenerator
from models.common.utility_functions import nearest_32

from .functional_decoder import (
    HIDDEN_SIZE,
    MODEL_ID,
    QwenFullAttentionCache,
    QwenLinearAttentionState,
    _rms_norm,
    _text_config,
)
from .multichip_decoder import MultichipDecoder, _validate_target_mesh
from .optimized_decoder import DEFAULT_OPTIMIZED_POLICY, OptimizedDecoderPolicy
from .precision_config import (
    compute_fidelity_name,
    compute_kernel_config_from_fidelity,
    dtype_from_name,
    dtype_to_name,
    group_dtype_name,
    layer_exception,
    normalize_precision_config,
)

BLOCK_SIZE = 32
DEFAULT_PREFILL_CHUNK_SIZE = 64
DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_MAX_TOP_K = 32


@dataclass(frozen=True)
class QwenFullModelCache:
    """Full-model cache/state owned by the caller or generator."""

    full_attention: dict[int, QwenFullAttentionCache]
    linear_attention: dict[int, QwenLinearAttentionState]
    page_table: ttnn.Tensor
    page_table_host: torch.Tensor
    max_batch_size: int
    max_seq_len: int
    block_size: int


class _SafetensorCheckpoint:
    """Small streaming loader for the local HF sharded checkpoint."""

    def __init__(self, model_id: str = MODEL_ID, *, local_files_only: bool = True):
        index_path = hf_hub_download(model_id, "model.safetensors.index.json", local_files_only=local_files_only)
        self.index_path = Path(index_path)
        with self.index_path.open(encoding="utf-8") as f:
            data = json.load(f)
        self.weight_map: dict[str, str] = data["weight_map"]
        self.snapshot_dir = self.index_path.parent

    def load_tensor(self, key: str) -> torch.Tensor:
        shard_name = self.weight_map[key]
        with safe_open(self.snapshot_dir / shard_name, framework="pt", device="cpu") as shard:
            return shard.get_tensor(key)

    def load_prefix(self, prefix: str, *, strip_prefix: bool = True) -> dict[str, torch.Tensor]:
        keys = [key for key in self.weight_map if key.startswith(prefix)]
        if not keys:
            raise KeyError(f"no tensors found under checkpoint prefix {prefix!r}")
        out: dict[str, torch.Tensor] = {}
        for shard_name in sorted({self.weight_map[key] for key in keys}):
            shard_keys = [key for key in keys if self.weight_map[key] == shard_name]
            with safe_open(self.snapshot_dir / shard_name, framework="pt", device="cpu") as shard:
                for key in shard_keys:
                    out[key.removeprefix(prefix) if strip_prefix else key] = shard.get_tensor(key)
        return out


def _replicate_mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _flat_vocab_mapper(mesh_device):
    return ttnn.ShardTensorToMesh(mesh_device, dim=3)


def _tt_bf16(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def _tt_int(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.int32) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(mesh_device),
    )


def _shape(tensor: ttnn.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _slice(tensor: ttnn.Tensor, starts: tuple[int, ...], ends: tuple[int, ...]) -> ttnn.Tensor:
    return ttnn.slice(tensor, starts, ends, (1,) * len(starts))


def _as_text_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


def _group_dtype(config: dict[str, Any], group: str, *, layer_type: str | None = None):
    return dtype_from_name(group_dtype_name(config, group, layer_type=layer_type))


def _decoder_policy_for_layer(config: dict[str, Any], *, layer_idx: int, layer_type: str) -> OptimizedDecoderPolicy:
    exception = layer_exception(config, layer_idx)
    group_overrides = exception.get("weight_groups", {})
    compute_overrides = exception.get("compute_fidelities", {})

    def dtype_for(group: str):
        if group in group_overrides:
            value = (
                group_overrides[group]["dtype"] if isinstance(group_overrides[group], dict) else group_overrides[group]
            )
            if isinstance(value, dict):
                value = value[layer_type]
            return dtype_from_name(str(value))
        return _group_dtype(config, group, layer_type=layer_type)

    def fidelity_for(group: str) -> str:
        return str(compute_overrides.get(group, compute_fidelity_name(config, group)))

    return replace(
        DEFAULT_OPTIMIZED_POLICY,
        attention_weight_dtype=dtype_for("attention"),
        linear_attention_weight_dtype=dtype_for("linear_attention"),
        shared_moe_weight_dtype=dtype_for("shared_moe"),
        routed_moe_weight_dtype=dtype_for("routed_moe"),
        attention_compute_fidelity=fidelity_for("attention"),
        linear_attention_compute_fidelity=fidelity_for("linear_attention"),
        router_compute_fidelity=fidelity_for("router"),
        shared_moe_compute_fidelity=fidelity_for("shared_moe"),
        routed_moe_compute_fidelity=fidelity_for("routed_moe"),
        lm_head_compute_fidelity=fidelity_for("lm_head"),
        ccl_dtype=str(config.get("ccl_dtype", "bf16")),
    )


def _vocab_size(hf_config) -> int:
    return int(_as_text_config(hf_config).vocab_size)


def _layer_state_from_dict(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    prefixes = (
        f"model.language_model.layers.{layer_idx}.",
        f"model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
    )
    for prefix in prefixes:
        out = {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}
        if out:
            return out
    return {key: value for key, value in state_dict.items() if not key.startswith("model.language_model.")}


def _make_page_table(batch: int, max_seq_len: int, block_size: int) -> torch.Tensor:
    blocks_per_user = math.ceil(max_seq_len / block_size)
    return torch.arange(batch * blocks_per_user, dtype=torch.int32).reshape(batch, blocks_per_user)


def _is_ttnn_tensor(value: Any) -> bool:
    return isinstance(value, ttnn.Tensor)


def _copy_bf16_to_existing(src: ttnn.Tensor, dst: ttnn.Tensor) -> ttnn.Tensor:
    """Trace-safe BF16 buffer update used for recurrent linear-attention state."""

    return ttnn.add(src, 0.0, output_tensor=dst, memory_config=dst.memory_config())


class QwenFullModel:
    """Serving-ready full autoregressive path for the 2x2 Qwen multichip stack."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        layers: list[MultichipDecoder],
        embedding_weight: ttnn.Tensor,
        final_norm_weight: ttnn.Tensor,
        lm_head_weight: ttnn.Tensor,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_seq_len: int | None = None,
        block_size: int = BLOCK_SIZE,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        sampling: SamplingGenerator | None = None,
        cos_matrix: ttnn.Tensor | None = None,
        sin_matrix: ttnn.Tensor | None = None,
        precision_config: dict[str, Any] | None = None,
        precision_config_source: str = "<built-in-default>",
    ):
        _validate_target_mesh(mesh_device, MultichipDecoder.mesh_plan)
        if prefill_chunk_size % block_size != 0:
            raise ValueError("prefill_chunk_size must be a multiple of block_size for paged-cache chunk fill")

        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.text_config = _as_text_config(hf_config)
        self.cfg = _text_config(self.text_config)
        self.layers = layers
        self.embedding_weight = embedding_weight
        self.final_norm_weight = final_norm_weight
        self.lm_head_weight = lm_head_weight
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len if max_seq_len is not None else self.cfg.max_position_embeddings)
        self.block_size = int(block_size)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.vocab_size = _vocab_size(self.text_config)
        self.padded_vocab_size = self.vocab_size
        self.sampling = sampling
        self.cos_matrix = cos_matrix
        self.sin_matrix = sin_matrix
        self.precision_config = normalize_precision_config(precision_config)
        self.precision_config_source = precision_config_source
        self.kv_cache_dtype = dtype_from_name(str(self.precision_config["kv_cache_dtype"]))
        self.linear_state_dtype = dtype_from_name(str(self.precision_config["linear_state_dtype"]))
        self.activation_dtype = dtype_from_name(str(self.precision_config["activation_dtype"]))
        self.residual_dtype = dtype_from_name(str(self.precision_config["residual_dtype"]))
        self.logits_dtype = dtype_from_name(str(self.precision_config["logits_dtype"]))
        self.sampling_dtype = dtype_from_name(str(self.precision_config["sampling_dtype"]))
        self.lm_head_compute_kernel_config = compute_kernel_config_from_fidelity(
            compute_fidelity_name(self.precision_config, "lm_head")
        )
        self.full_attention_layers = [
            idx for idx, layer_type in enumerate(self.cfg.layer_types) if layer_type == "full_attention"
        ]
        self.linear_attention_layers = [
            idx for idx, layer_type in enumerate(self.cfg.layer_types) if layer_type == "linear_attention"
        ]

    @classmethod
    def from_hf(
        cls,
        *,
        mesh_device,
        model_id: str = MODEL_ID,
        local_files_only: bool = True,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_seq_len: int | None = None,
        block_size: int = BLOCK_SIZE,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        lm_head_dtype=None,
        enable_on_device_sampling: bool = True,
        load_rope_tables: bool = True,
        precision_config: dict[str, Any] | None = None,
        precision_config_source: str = "<built-in-default>",
    ) -> "QwenFullModel":
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
        text_config = _as_text_config(hf_config)
        checkpoint = _SafetensorCheckpoint(model_id, local_files_only=local_files_only)
        return cls.from_checkpoint(
            mesh_device=mesh_device,
            hf_config=text_config,
            checkpoint=checkpoint,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            lm_head_dtype=lm_head_dtype,
            enable_on_device_sampling=enable_on_device_sampling,
            load_rope_tables=load_rope_tables,
            precision_config=precision_config,
            precision_config_source=precision_config_source,
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        mesh_device,
        hf_config,
        checkpoint: _SafetensorCheckpoint,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_seq_len: int | None = None,
        block_size: int = BLOCK_SIZE,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        lm_head_dtype=None,
        enable_on_device_sampling: bool = True,
        load_rope_tables: bool = True,
        precision_config: dict[str, Any] | None = None,
        precision_config_source: str = "<built-in-default>",
    ) -> "QwenFullModel":
        text_config = _as_text_config(hf_config)
        cfg = _text_config(text_config)
        if cfg.hidden_size != HIDDEN_SIZE:
            raise ValueError(f"{MODEL_ID} full model expects hidden_size={HIDDEN_SIZE}, got {cfg.hidden_size}")
        precision_config = normalize_precision_config(precision_config)
        if lm_head_dtype is None:
            lm_head_dtype = _group_dtype(precision_config, "lm_head")

        layers: list[MultichipDecoder] = []
        for layer_idx in range(cfg.num_hidden_layers):
            prefix = f"model.language_model.layers.{layer_idx}."
            state = checkpoint.load_prefix(prefix)
            layers.append(
                MultichipDecoder.from_state_dict(
                    state,
                    hf_config=text_config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    policy=_decoder_policy_for_layer(
                        precision_config,
                        layer_idx=layer_idx,
                        layer_type=cfg.layer_types[layer_idx],
                    ),
                )
            )
            del state

        embedding = checkpoint.load_tensor("model.language_model.embed_tokens.weight").unsqueeze(0).unsqueeze(0)
        embedding_weight = ttnn.as_tensor(
            embedding.contiguous(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(mesh_device),
        )
        del embedding

        norm = checkpoint.load_tensor("model.language_model.norm.weight")
        final_norm_weight = ttnn.as_tensor(
            (1.0 + norm).reshape(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(mesh_device),
        )
        del norm

        lm_head = checkpoint.load_tensor("lm_head.weight").transpose(0, 1).contiguous().unsqueeze(0).unsqueeze(0)
        lm_head_weight = ttnn.as_tensor(
            lm_head,
            dtype=lm_head_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_flat_vocab_mapper(mesh_device),
        )
        del lm_head

        sampling = (
            SamplingGenerator(
                args=_make_sampling_args(text_config, mesh_device, max_batch_size), mesh_device=mesh_device, tt_ccl=None
            )
            if enable_on_device_sampling
            else None
        )
        cos_matrix = sin_matrix = None
        resolved_max_seq_len = int(max_seq_len if max_seq_len is not None else cfg.max_position_embeddings)
        if load_rope_tables:
            cos_matrix, sin_matrix = _build_rope_lookup_tables(text_config, mesh_device, resolved_max_seq_len)

        return cls(
            mesh_device=mesh_device,
            hf_config=text_config,
            layers=layers,
            embedding_weight=embedding_weight,
            final_norm_weight=final_norm_weight,
            lm_head_weight=lm_head_weight,
            max_batch_size=max_batch_size,
            max_seq_len=resolved_max_seq_len,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            sampling=sampling,
            cos_matrix=cos_matrix,
            sin_matrix=sin_matrix,
            precision_config=precision_config,
            precision_config_source=precision_config_source,
        )

    @classmethod
    def from_state_dict(
        cls,
        *,
        mesh_device,
        hf_config,
        state_dict: dict[str, torch.Tensor],
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_seq_len: int | None = None,
        block_size: int = BLOCK_SIZE,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        lm_head_dtype=None,
        enable_on_device_sampling: bool = True,
        load_rope_tables: bool = False,
        precision_config: dict[str, Any] | None = None,
        precision_config_source: str = "<built-in-default>",
    ) -> "QwenFullModel":
        text_config = _as_text_config(hf_config)
        cfg = _text_config(text_config)
        precision_config = normalize_precision_config(precision_config)
        if lm_head_dtype is None:
            lm_head_dtype = _group_dtype(precision_config, "lm_head")
        layers = [
            MultichipDecoder.from_state_dict(
                _layer_state_from_dict(state_dict, layer_idx),
                hf_config=text_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                policy=_decoder_policy_for_layer(
                    precision_config,
                    layer_idx=layer_idx,
                    layer_type=cfg.layer_types[layer_idx],
                ),
            )
            for layer_idx in range(cfg.num_hidden_layers)
        ]
        embed_key = "model.language_model.embed_tokens.weight"
        norm_key = "model.language_model.norm.weight"
        lm_key = "lm_head.weight"
        embedding_weight = ttnn.as_tensor(
            state_dict[embed_key].unsqueeze(0).unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(mesh_device),
        )
        final_norm_weight = ttnn.as_tensor(
            (1.0 + state_dict[norm_key]).reshape(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(mesh_device),
        )
        lm_head_weight = ttnn.as_tensor(
            state_dict[lm_key].transpose(0, 1).contiguous().unsqueeze(0).unsqueeze(0),
            dtype=lm_head_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_flat_vocab_mapper(mesh_device),
        )
        sampling = (
            SamplingGenerator(
                args=_make_sampling_args(text_config, mesh_device, max_batch_size), mesh_device=mesh_device, tt_ccl=None
            )
            if enable_on_device_sampling
            else None
        )
        resolved_max_seq_len = int(max_seq_len if max_seq_len is not None else cfg.max_position_embeddings)
        cos_matrix = sin_matrix = None
        if load_rope_tables:
            cos_matrix, sin_matrix = _build_rope_lookup_tables(text_config, mesh_device, resolved_max_seq_len)
        return cls(
            mesh_device=mesh_device,
            hf_config=text_config,
            layers=layers,
            embedding_weight=embedding_weight,
            final_norm_weight=final_norm_weight,
            lm_head_weight=lm_head_weight,
            max_batch_size=max_batch_size,
            max_seq_len=resolved_max_seq_len,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            sampling=sampling,
            cos_matrix=cos_matrix,
            sin_matrix=sin_matrix,
            precision_config=precision_config,
            precision_config_source=precision_config_source,
        )

    def allocate_cache(
        self,
        *,
        max_batch_size: int | None = None,
        max_seq_len: int | None = None,
        page_table: torch.Tensor | ttnn.Tensor | None = None,
    ) -> QwenFullModelCache:
        batch = int(max_batch_size if max_batch_size is not None else self.max_batch_size)
        seq = int(max_seq_len if max_seq_len is not None else self.max_seq_len)
        if batch > self.max_batch_size:
            raise ValueError(f"requested batch {batch} exceeds model max_batch_size={self.max_batch_size}")
        if seq > self.max_seq_len:
            raise ValueError(f"requested context {seq} exceeds model max_seq_len={self.max_seq_len}")

        full_attention = {
            idx: MultichipDecoder.allocate_full_attention_cache(
                hf_config=self.text_config,
                mesh_device=self.mesh_device,
                max_batch_size=batch,
                max_seq_len=seq,
                block_size=self.block_size,
                dtype=self.kv_cache_dtype,
            )
            for idx in self.full_attention_layers
        }
        linear_attention = {
            idx: MultichipDecoder.allocate_linear_attention_state(
                hf_config=self.text_config,
                mesh_device=self.mesh_device,
                batch_size=batch,
                dtype=self.linear_state_dtype,
            )
            for idx in self.linear_attention_layers
        }
        if page_table is None:
            page_table_host = _make_page_table(batch, seq, self.block_size)
            page_table_tt = _tt_int(page_table_host, self.mesh_device)
        elif _is_ttnn_tensor(page_table):
            page_table_host = _make_page_table(batch, seq, self.block_size)
            page_table_tt = page_table
        else:
            page_table_host = page_table.to(torch.int32).contiguous()
            page_table_tt = _tt_int(page_table_host, self.mesh_device)
        return QwenFullModelCache(
            full_attention=full_attention,
            linear_attention=linear_attention,
            page_table=page_table_tt,
            page_table_host=page_table_host,
            max_batch_size=batch,
            max_seq_len=seq,
            block_size=self.block_size,
        )

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor | None,
        kv_cache: QwenFullModelCache | None,
        prompt_lens: list[int],
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        """Run full-stack prefill and return host logits for readiness checks."""

        if tokens.ndim != 2:
            raise ValueError(f"prefill tokens must be [batch, seq], got shape {tuple(tokens.shape)}")
        batch, padded_len = tuple(tokens.shape)
        if batch != len(prompt_lens):
            raise ValueError("prompt_lens length must match token batch")
        cache = kv_cache or self.allocate_cache(
            max_batch_size=batch, max_seq_len=max(max(prompt_lens), 1), page_table=page_table
        )
        cache = self._cache_with_page_table(cache, page_table)
        out_rows: list[torch.Tensor] = []
        max_out_len = max(prompt_lens) if return_all_logits else 1
        for user_id, prompt_len in enumerate(prompt_lens):
            if prompt_len <= 0:
                out_rows.append(torch.zeros((max_out_len, self.vocab_size), dtype=torch.float32))
                continue
            if prompt_len > padded_len:
                raise ValueError(f"prompt_lens[{user_id}]={prompt_len} exceeds padded token length {padded_len}")
            row_logits = self.prefill_user(
                tokens[user_id : user_id + 1, :prompt_len],
                cache=cache,
                user_id=user_id,
                return_all_logits=return_all_logits,
            )
            if return_all_logits and row_logits.shape[0] < max_out_len:
                pad = torch.zeros((max_out_len - row_logits.shape[0], row_logits.shape[1]), dtype=row_logits.dtype)
                row_logits = torch.cat([row_logits, pad], dim=0)
            out_rows.append(row_logits)
        return torch.stack(out_rows, dim=0)

    def prefill_user(
        self,
        tokens: torch.Tensor,
        *,
        cache: QwenFullModelCache,
        user_id: int,
        page_table_user_id: int | None = None,
        return_all_logits: bool,
        return_tt_logits: bool = False,
    ) -> torch.Tensor | ttnn.Tensor:
        if return_all_logits and return_tt_logits:
            raise ValueError("return_all_logits and return_tt_logits are mutually exclusive")
        prompt_len = int(tokens.shape[1])
        if prompt_len > cache.max_seq_len:
            raise ValueError(f"prompt length {prompt_len} exceeds cache max_seq_len={cache.max_seq_len}")
        logits_chunks: list[torch.Tensor] = []
        last_logits: torch.Tensor | None = None
        last_tt_logits: ttnn.Tensor | None = None
        page_table_user_id = user_id if page_table_user_id is None else int(page_table_user_id)
        for start in range(0, prompt_len, self.prefill_chunk_size):
            end = min(start + self.prefill_chunk_size, prompt_len)
            chunk_tokens = tokens[:, start:end]
            hidden = self.embed_tokens(self._tokens_to_tt_prefill(chunk_tokens))
            for layer_idx, layer in enumerate(self.layers):
                if layer.layer_type == "full_attention":
                    position_embeddings = self._prefill_position_embeddings(start, end - start)
                    user_page_table = self._page_table_for_user(cache, page_table_user_id)
                    result = layer.prefill_forward(
                        hidden,
                        position_embeddings=position_embeddings,
                        page_table=user_page_table,
                        kv_cache=cache.full_attention[layer_idx],
                        user_id=0,
                        chunk_page_table=self._chunk_page_table(cache, page_table_user_id, start, end),
                        chunk_start_idx=start if start > 0 else None,
                    )
                else:
                    linear_state = self._linear_state_for_user(
                        cache.linear_attention[layer_idx], user_id, cache.max_batch_size
                    )
                    result = layer.prefill_forward(hidden, linear_state=linear_state)
                    self._commit_linear_state_user(
                        cache.linear_attention[layer_idx],
                        result.linear_state,
                        user_id=user_id,
                        batch_size=cache.max_batch_size,
                    )
                hidden = result.hidden_states
            logits = self.apply_lm_head(hidden)
            if return_tt_logits:
                last_tt_logits = self._last_token_logits(logits, end - start)
                continue
            host = self.logits_to_torch(logits, batch=1, seq_len=end - start)[0]
            if return_all_logits:
                logits_chunks.append(host[: end - start])
            else:
                last_logits = host[end - start - 1 : end - start]
        if return_tt_logits:
            if last_tt_logits is None:
                raise RuntimeError("prefill produced no TT logits")
            return last_tt_logits
        if return_all_logits:
            return torch.cat(logits_chunks, dim=0)
        if last_logits is None:
            raise RuntimeError("prefill produced no logits")
        return last_logits

    def decode_forward(
        self,
        tokens: torch.Tensor | ttnn.Tensor,
        start_pos: torch.Tensor | ttnn.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor | None = None,
        kv_cache: QwenFullModelCache | None = None,
        return_tt_logits: bool = False,
        sample_on_device: bool = False,
        tt_out_tok: ttnn.Tensor | None = None,
    ) -> torch.Tensor | ttnn.Tensor:
        cache = kv_cache or self.allocate_cache(max_batch_size=_infer_batch(tokens), page_table=page_table)
        cache = self._cache_with_page_table(cache, page_table)
        tt_tokens = tokens if _is_ttnn_tensor(tokens) else self._tokens_to_tt_decode(tokens)
        tt_pos = start_pos if _is_ttnn_tensor(start_pos) else self._positions_to_tt(start_pos)
        logits = self.decode_logits_tt(tt_tokens, tt_pos, cache=cache)
        if sample_on_device:
            if self.sampling is None:
                raise RuntimeError("on-device sampling requested but sampling is disabled")
            if _shape(logits)[-2] < nearest_32(_shape(logits)[-2]):
                logits = ttnn.pad(
                    logits,
                    padding=[(0, 0), (0, 0), (0, nearest_32(_shape(logits)[-2]) - _shape(logits)[-2]), (0, 0)],
                    value=0.0,
                )
            sampled = self.sampling.sample(logits, enable_trace=False, tt_out_tok=tt_out_tok)
            return sampled[0] if isinstance(sampled, tuple) else sampled
        if return_tt_logits:
            return logits
        batch = _shape(logits)[-2]
        return self.decode_logits_to_torch(logits, batch=batch)

    def decode_logits_tt(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        cache: QwenFullModelCache,
    ) -> ttnn.Tensor:
        hidden = self.embed_tokens(tokens)
        position_embeddings = self.decode_position_embeddings(current_pos)
        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_type == "full_attention":
                result = layer.decode_forward(
                    hidden,
                    current_pos=current_pos,
                    position_embeddings=position_embeddings,
                    page_table=cache.page_table,
                    kv_cache=cache.full_attention[layer_idx],
                )
            else:
                result = layer.decode_forward(
                    hidden, current_pos=current_pos, linear_state=cache.linear_attention[layer_idx]
                )
                self._commit_linear_state(cache.linear_attention[layer_idx], result.linear_state)
            hidden = result.hidden_states
        return self.apply_lm_head(hidden)

    def embed_tokens(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        hidden = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(_shape(hidden)) == 3:
            hidden = ttnn.unsqueeze_to_4D(hidden)
        return hidden

    def apply_lm_head(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = _rms_norm(hidden_states, self.final_norm_weight, self.cfg.rms_norm_eps)
        return ttnn.linear(
            hidden_states,
            self.lm_head_weight,
            dtype=self.logits_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.lm_head_compute_kernel_config,
        )

    def _last_token_logits(self, logits: ttnn.Tensor, chunk_len: int) -> ttnn.Tensor:
        shape = _shape(logits)
        return ttnn.slice(logits, (0, 0, chunk_len - 1, 0), (shape[0], shape[1], chunk_len, shape[3]))

    def logits_to_torch(self, logits: ttnn.Tensor, *, batch: int, seq_len: int) -> torch.Tensor:
        full = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3)).float()
        full = full.reshape(1, batch, seq_len, -1)
        return full[0, :, :, : self.vocab_size]

    def decode_logits_to_torch(self, logits: ttnn.Tensor, *, batch: int) -> torch.Tensor:
        return self.logits_to_torch(logits, batch=batch, seq_len=1)[:, 0, :]

    def _tokens_to_tt_prefill(self, tokens: torch.Tensor) -> ttnn.Tensor:
        tokens = tokens.to(torch.uint32).contiguous().reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.mesh_device),
        )

    def _tokens_to_tt_decode(self, tokens: torch.Tensor) -> ttnn.Tensor:
        if tokens.ndim == 2:
            if tokens.shape[1] != 1:
                raise ValueError(f"decode tokens must have one column, got shape {tuple(tokens.shape)}")
            tokens = tokens[:, 0]
        tokens = tokens.to(torch.uint32).contiguous().reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.mesh_device),
        )

    def _positions_to_tt(self, positions: torch.Tensor) -> ttnn.Tensor:
        return _tt_int(positions.to(torch.int32).contiguous(), self.mesh_device)

    def _prefill_position_embeddings(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        pos = torch.arange(start, start + length, dtype=torch.long).reshape(1, length)
        cos, sin = _rope_cpu(self.text_config, pos)
        return _tt_bf16(cos.unsqueeze(1), self.mesh_device), _tt_bf16(sin.unsqueeze(1), self.mesh_device)

    def decode_position_embeddings(self, current_pos: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if self.cos_matrix is None or self.sin_matrix is None:
            raise RuntimeError("decode_position_embeddings requires load_rope_tables=True")
        if len(_shape(current_pos)) == 1:
            rot_idxs = ttnn.reshape(current_pos, (1, _shape(current_pos)[0]))
        else:
            rot_idxs = current_pos
        rot_idxs = ttnn.typecast(rot_idxs, ttnn.uint32)
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)
        return cos, sin

    def _chunk_page_table(self, cache: QwenFullModelCache, user_id: int, start: int, end: int) -> ttnn.Tensor:
        start_block = start // self.block_size
        end_block = math.ceil(end / self.block_size)
        if cache.max_batch_size == 1:
            return _slice(cache.page_table, (0, start_block), (1, end_block))
        return _slice(cache.page_table, (user_id, start_block), (user_id + 1, end_block))

    def _page_table_for_user(self, cache: QwenFullModelCache, user_id: int) -> ttnn.Tensor:
        if cache.max_batch_size == 1:
            return cache.page_table
        return _slice(cache.page_table, (user_id, 0), (user_id + 1, _shape(cache.page_table)[1]))

    def _cache_with_page_table(
        self,
        cache: QwenFullModelCache,
        page_table: torch.Tensor | ttnn.Tensor | None,
    ) -> QwenFullModelCache:
        if page_table is None or page_table is cache.page_table:
            return cache
        if _is_ttnn_tensor(page_table):
            return replace(cache, page_table=page_table)
        page_table_host = page_table.to(torch.int32).contiguous()
        return replace(cache, page_table=_tt_int(page_table_host, self.mesh_device), page_table_host=page_table_host)

    def _linear_state_for_user(
        self,
        state: QwenLinearAttentionState,
        user_id: int,
        batch_size: int,
    ) -> QwenLinearAttentionState:
        if batch_size == 1:
            return state
        if user_id < 0 or user_id >= batch_size:
            raise ValueError(f"user_id {user_id} is outside cache batch size {batch_size}")
        conv_state = tuple(
            _slice(tap, (0, 0, user_id, 0), (1, 1, user_id + 1, _shape(tap)[3])) for tap in state.conv_state
        )
        recurrent_shape = _shape(state.recurrent_state)
        heads_per_user = recurrent_shape[1] // batch_size
        head_start = user_id * heads_per_user
        head_end = head_start + heads_per_user
        recurrent_state = _slice(
            state.recurrent_state,
            (0, head_start, 0, 0),
            (recurrent_shape[0], head_end, recurrent_shape[2], recurrent_shape[3]),
        )
        return QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)

    def _commit_linear_state_user(
        self,
        dst: QwenLinearAttentionState,
        src: QwenLinearAttentionState | None,
        *,
        user_id: int,
        batch_size: int,
    ) -> QwenLinearAttentionState:
        if batch_size == 1:
            return self._commit_linear_state(dst, src)
        if src is None:
            raise RuntimeError("linear-attention layer did not return updated state")
        if user_id < 0 or user_id >= batch_size:
            raise ValueError(f"user_id {user_id} is outside cache batch size {batch_size}")
        for src_t, dst_t in zip(src.conv_state, dst.conv_state, strict=True):
            self._copy_user_slice(src_t, dst_t, dim=2, start=user_id, end=user_id + 1)
        recurrent_shape = _shape(dst.recurrent_state)
        heads_per_user = recurrent_shape[1] // batch_size
        head_start = user_id * heads_per_user
        self._copy_user_slice(
            src.recurrent_state, dst.recurrent_state, dim=1, start=head_start, end=head_start + heads_per_user
        )
        return dst

    def _copy_user_slice(self, src: ttnn.Tensor, dst: ttnn.Tensor, *, dim: int, start: int, end: int) -> ttnn.Tensor:
        dst_shape = _shape(dst)
        parts: list[ttnn.Tensor] = []
        if start > 0:
            before_start = [0] * len(dst_shape)
            before_end = list(dst_shape)
            before_end[dim] = start
            parts.append(_slice(dst, tuple(before_start), tuple(before_end)))
        parts.append(src)
        if end < dst_shape[dim]:
            after_start = [0] * len(dst_shape)
            after_end = list(dst_shape)
            after_start[dim] = end
            parts.append(_slice(dst, tuple(after_start), tuple(after_end)))
        merged = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return _copy_bf16_to_existing(merged, dst)

    def _commit_linear_state(
        self,
        dst: QwenLinearAttentionState,
        src: QwenLinearAttentionState | None,
    ) -> QwenLinearAttentionState:
        if src is None:
            raise RuntimeError("linear-attention layer did not return updated state")
        for src_t, dst_t in zip(src.conv_state, dst.conv_state, strict=True):
            _copy_bf16_to_existing(src_t, dst_t)
        _copy_bf16_to_existing(src.recurrent_state, dst.recurrent_state)
        return dst

    def remap_linear_attention_state(
        self,
        cache: QwenFullModelCache,
        slot_remap: torch.Tensor | list[int] | tuple[int, ...] | None,
    ) -> None:
        """Move per-user linear-attention state after vLLM condenses slots.

        ``slot_remap[i] = j`` means decode row ``i`` should read state that
        previously lived in slot ``j``. Full-attention KV state is page-table
        addressed, so only the recurrent/conv linear-attention state moves.
        """

        if slot_remap is None or cache.max_batch_size <= 1:
            return
        if isinstance(slot_remap, torch.Tensor):
            remap = [int(v) for v in slot_remap.reshape(-1).tolist()]
        else:
            remap = [int(v) for v in slot_remap]
        if len(remap) < cache.max_batch_size:
            remap.extend(range(len(remap), cache.max_batch_size))
        remap = remap[: cache.max_batch_size]
        if all(src == dst for dst, src in enumerate(remap)):
            return
        if any(src < 0 or src >= cache.max_batch_size for src in remap):
            raise ValueError(f"slot_remap contains out-of-range entries for batch {cache.max_batch_size}: {remap}")

        for state in cache.linear_attention.values():
            for tap in state.conv_state:
                tap_shape = _shape(tap)
                pieces = [_slice(tap, (0, 0, src, 0), (1, 1, src + 1, tap_shape[3])) for src in remap]
                merged = (
                    pieces[0]
                    if len(pieces) == 1
                    else ttnn.concat(
                        pieces,
                        dim=2,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                _copy_bf16_to_existing(merged, tap)

            recurrent_shape = _shape(state.recurrent_state)
            heads_per_user = recurrent_shape[1] // cache.max_batch_size
            pieces = []
            for src in remap:
                start = src * heads_per_user
                pieces.append(
                    _slice(
                        state.recurrent_state,
                        (0, start, 0, 0),
                        (recurrent_shape[0], start + heads_per_user, recurrent_shape[2], recurrent_shape[3]),
                    )
                )
            merged = (
                pieces[0]
                if len(pieces) == 1
                else ttnn.concat(
                    pieces,
                    dim=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            _copy_bf16_to_existing(merged, state.recurrent_state)

    def reset_linear_attention_state(
        self,
        cache: QwenFullModelCache,
        slots: Iterable[int],
    ) -> None:
        """Clear persistent linear-attention state for newly assigned serving slots."""

        reset_slots = sorted({int(slot) for slot in slots})
        if not reset_slots or cache.max_batch_size <= 0:
            return
        if any(slot < 0 or slot >= cache.max_batch_size for slot in reset_slots):
            raise ValueError(f"linear-attention reset slots exceed cache batch {cache.max_batch_size}: {reset_slots}")
        reset_set = set(reset_slots)
        reset_all = len(reset_set) == cache.max_batch_size

        for state in cache.linear_attention.values():
            for tap in state.conv_state:
                if reset_all:
                    _copy_bf16_to_existing(ttnn.zeros_like(tap), tap)
                    continue
                tap_shape = _shape(tap)
                pieces = []
                for slot in range(cache.max_batch_size):
                    piece = _slice(tap, (0, 0, slot, 0), (1, 1, slot + 1, tap_shape[3]))
                    pieces.append(ttnn.zeros_like(piece) if slot in reset_set else piece)
                merged = (
                    pieces[0]
                    if len(pieces) == 1
                    else ttnn.concat(
                        pieces,
                        dim=2,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                _copy_bf16_to_existing(merged, tap)

            if reset_all:
                _copy_bf16_to_existing(ttnn.zeros_like(state.recurrent_state), state.recurrent_state)
                continue
            recurrent_shape = _shape(state.recurrent_state)
            heads_per_user = recurrent_shape[1] // cache.max_batch_size
            pieces = []
            for slot in range(cache.max_batch_size):
                start = slot * heads_per_user
                piece = _slice(
                    state.recurrent_state,
                    (0, start, 0, 0),
                    (recurrent_shape[0], start + heads_per_user, recurrent_shape[2], recurrent_shape[3]),
                )
                pieces.append(ttnn.zeros_like(piece) if slot in reset_set else piece)
            merged = (
                pieces[0]
                if len(pieces) == 1
                else ttnn.concat(
                    pieces,
                    dim=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            _copy_bf16_to_existing(merged, state.recurrent_state)

    def describe_precision_policy(self) -> dict[str, Any]:
        """Return observed runtime precision knobs from constructed TTNN modules."""

        def kernel_name(value) -> str:
            if value is None:
                return "default"
            return str(value.math_fidelity).split(".")[-1]

        first_linear = next((layer for layer in self.layers if layer.layer_type == "linear_attention"), None)
        first_full = next((layer for layer in self.layers if layer.layer_type == "full_attention"), None)
        observed: dict[str, Any] = {
            "config_id": self.precision_config.get("config_id"),
            "source": self.precision_config_source,
            "activation_dtype": str(self.precision_config["activation_dtype"]),
            "residual_dtype": str(self.precision_config["residual_dtype"]),
            "ccl_dtype": str(self.precision_config["ccl_dtype"]),
            "kv_cache_dtype": dtype_to_name(self.kv_cache_dtype),
            "linear_state_dtype": dtype_to_name(self.linear_state_dtype),
            "logits_dtype": dtype_to_name(self.logits_dtype),
            "sampling_dtype": dtype_to_name(self.sampling_dtype),
            "lm_head_weight_dtype": dtype_to_name(self.lm_head_weight.dtype),
            "lm_head_compute_fidelity": kernel_name(self.lm_head_compute_kernel_config),
            "weight_groups": copy.deepcopy(self.precision_config["weight_groups"]),
            "layer_exceptions": copy.deepcopy(self.precision_config.get("layer_exceptions", {})),
            "compute_fidelities": copy.deepcopy(self.precision_config.get("compute_fidelities", {})),
        }
        if first_full is not None:
            mixer = first_full.token_mixer
            observed["first_full_attention_layer"] = {
                "layer_idx": first_full.layer_idx,
                "qkgv_weight_dtype": dtype_to_name(mixer.qkgv_proj.dtype),
                "o_proj_weight_dtype": dtype_to_name(mixer.o_proj.dtype),
                "compute_fidelity": kernel_name(mixer.compute_kernel_config),
                "ccl_dtype": mixer.plan.ccl_dtype,
                "routed_moe_weight_dtype": dtype_to_name(first_full.mlp.routed_gate_up.dtype),
                "routed_moe_compute_fidelity": kernel_name(first_full.mlp.routed_compute_kernel_config),
                "shared_moe_weight_dtype": dtype_to_name(first_full.mlp.shared_gate_up.dtype),
                "shared_moe_compute_fidelity": kernel_name(first_full.mlp.shared_compute_kernel_config),
            }
        if first_linear is not None:
            mixer = first_linear.token_mixer
            observed["first_linear_attention_layer"] = {
                "layer_idx": first_linear.layer_idx,
                "in_proj_weight_dtype": dtype_to_name(mixer.in_proj_qkv_zba.dtype),
                "out_proj_weight_dtype": dtype_to_name(mixer.out_proj.dtype),
                "compute_fidelity": kernel_name(mixer.projection_compute_kernel_config),
                "ccl_dtype": mixer.plan.ccl_dtype,
                "routed_moe_weight_dtype": dtype_to_name(first_linear.mlp.routed_gate_up.dtype),
                "routed_moe_compute_fidelity": kernel_name(first_linear.mlp.routed_compute_kernel_config),
                "shared_moe_weight_dtype": dtype_to_name(first_linear.mlp.shared_gate_up.dtype),
                "shared_moe_compute_fidelity": kernel_name(first_linear.mlp.shared_compute_kernel_config),
            }
        return observed


def _infer_batch(tokens: torch.Tensor | ttnn.Tensor) -> int:
    if _is_ttnn_tensor(tokens):
        shape = _shape(tokens)
        return int(shape[-1] if len(shape) >= 4 else shape[-1])
    return int(tokens.shape[0])


def _make_sampling_args(text_config, mesh_device, max_batch_size: int):
    args = SimpleNamespace()
    args.vocab_size = int(text_config.vocab_size)
    args.padded_vocab_size = int(text_config.vocab_size)
    args.cluster_shape = (1, mesh_device.get_num_devices())
    args.sampling_all_gather_axis = 1
    args.num_devices = mesh_device.get_num_devices()
    args.max_batch_size = int(max_batch_size)
    args.max_top_k = DEFAULT_MAX_TOP_K
    args.pad_logits_to_power_of_2 = True
    args.use_composite_topk_all_gather = True
    args.model_config = {}
    args.sampling_dp = 1
    args.use_topk_logprobs = False
    return args


def _rope_cpu(text_config, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = Qwen3_5MoeTextRotaryEmbedding(text_config)
    marker = torch.empty((), dtype=torch.bfloat16)
    return rotary(marker, position_ids)


def _build_rope_lookup_tables(text_config, mesh_device, max_seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    positions = torch.arange(max_seq_len, dtype=torch.long).reshape(1, max_seq_len)
    cos, sin = _rope_cpu(text_config, positions)
    cos = cos[0].contiguous()
    sin = sin[0].contiguous()
    cos_tt = ttnn.from_torch(
        cos,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(mesh_device),
    )
    return cos_tt, sin_tt


def iter_runtime_fallback_audit() -> Iterable[str]:
    yield "decoder_stack=MultichipDecoder"
    yield "mesh=(2,2); tensor_parallel_axis=columns; expert_parallel_axis=rows"
    yield "decoder_boundary=replicated_bf16_dram_interleaved"
    yield "lm_head=vocab_sharded_flat_4way"
    yield "sampling=common_sampling_generator_flat_4way_topk1_composite_gather"
    yield "host_sampling_compatibility=explicit_generator_mode_only"
    yield "no_single_chip_or_host_decoder_fallback"


__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_PREFILL_CHUNK_SIZE",
    "MODEL_ID",
    "QwenFullModel",
    "QwenFullModelCache",
    "iter_runtime_fallback_audit",
]
