# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Common-prefill adapter for Gemma 4 31B mixed global/sliding KV migration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams


def migration_runtime_paged_caches(num_layers: int) -> list:
    """Per-layer cache placeholders: CP ring attention owns compute history."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    return [None] * num_layers


class Gemma4PrefillConfig:
    """Static dimensions consumed by the common runner and producer."""

    EMB_SIZE = 5376
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    NUM_LAYERS = 60
    NUM_KEY_VALUE_HEADS = 16
    HEAD_DIM = 256
    ROTARY_DIM = 256
    GLOBAL_NUM_KEY_VALUE_HEADS = 4
    GLOBAL_HEAD_DIM = 512
    GLOBAL_ROTARY_DIM = 128
    GLOBAL_MIGRATION_ROW_DIM = 640
    GLOBAL_LAYERS = tuple(range(5, 60, 6))
    SLIDING_LAYERS = tuple(i for i in range(60) if i % 6 != 5)
    SLIDING_NUM_KEY_VALUE_HEADS = 16
    SLIDING_HEAD_DIM = 256


@dataclass
class Gemma4KvCaches(KvCaches):
    """Native paged caches plus the compact decode-row migration cache."""

    paged: list
    migration: object
    sliding_migration: object


class Gemma4PrefillAdapter(PrefillModelAdapter):
    """Gemma 4 31B adapter, initially restricted to one CP8/TP4 rank."""

    name = "gemma4_31b"
    model_config = Gemma4PrefillConfig
    hf_model_default = "models/demos/gemma4/configs/gemma-4-31B-it"
    ttnn_cache_default = ""
    prefill_trace_default = ""
    default_gate_mode = "DEVICE_FP32"
    supports_dflash = False
    supports_multi_turn = False
    pipeline_activation_emb_tp_sharded = True
    h2d_input_shard_dim = 1
    tokenizer_trust_remote_code = True
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"
    hf_repo_id = "google/gemma-4-31b-it"
    env_var = "GEMMA4_MODEL_PATH"
    num_layers_to_download = 60

    @staticmethod
    def _model_path() -> str:
        return os.environ.get("PREFILL_HF_MODEL") or os.environ.get(
            "GEMMA4_MODEL_PATH", Gemma4PrefillAdapter.hf_model_default
        )

    def load_hf_config(self):
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        model_path = self._model_path()
        raw = Gemma4ModelArgs.load_hf_config(model_path)
        args = Gemma4ModelArgs.from_hf_config(raw)
        args._hf_text_config = getattr(raw, "text_config", raw)
        args.model_cache_path = Gemma4ModelArgs.resolve_model_cache_path(model_path)
        if args.num_hidden_layers != self.model_config.NUM_LAYERS:
            raise ValueError(
                f"gemma4_31b expects {self.model_config.NUM_LAYERS} layers, got {args.num_hidden_layers} "
                f"from {model_path}"
            )
        return args

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        del mesh_shape
        import ttnn
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        model_path = self._model_path()
        args = Gemma4ModelArgs()
        args.model_cache_path = Gemma4ModelArgs.resolve_model_cache_path(model_path)
        return args.weight_cache_path(ttnn.bfloat16)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        import ttnn
        from models.demos.gemma4.tt.attention.global_migration import (
            allocate_global_migration_cache,
            global_layer_indices,
        )
        from models.demos.gemma4.tt.attention.sliding_migration import (
            allocate_sliding_migration_cache,
            sliding_layer_indices,
        )

        if not (params.is_first_rank and params.is_last_rank and params.first_layer_idx == 0):
            raise NotImplementedError("initial Gemma 4 prefill migration support is single-rank only")
        if params.num_layers != self.model_config.NUM_LAYERS:
            raise ValueError(f"Gemma 4 requires all 60 layers on the single rank, got {params.num_layers}")
        if params.mesh_shape != (8, 4) or params.sp_axis != 0 or params.tp_axis != 1:
            raise ValueError(
                f"Gemma 4 migration requires mesh=(8,4), sp_axis=0, tp_axis=1; got "
                f"{params.mesh_shape}, {params.sp_axis}, {params.tp_axis}"
            )

        if params.max_seq_len % (ttnn.TILE_SIZE * params.sp_factor):
            raise ValueError(
                f"max_seq_len={params.max_seq_len} must divide into " f"{ttnn.TILE_SIZE}-token chunks across CP8"
            )
        # This cache crosses a raw-byte migration boundary, so its dtype is a
        # protocol constant rather than a model-path-dependent precision-table
        # lookup (HF snapshot directory names are not stable identifiers).
        cache_dtype = ttnn.bfloat8_b
        # CP ring attention owns the full-width compute history. The migration
        # runtime therefore has no paged K/V tensors for either layer family.
        paged = migration_runtime_paged_caches(params.num_layers)

        globals_ = global_layer_indices(hf_config.layer_types)
        if globals_ != self.model_config.GLOBAL_LAYERS:
            raise ValueError(f"unexpected Gemma 4 global layer pattern: {globals_}")
        sliding = sliding_layer_indices(hf_config.layer_types)
        if sliding != self.model_config.SLIDING_LAYERS:
            raise ValueError(f"unexpected Gemma 4 sliding layer pattern: {sliding}")
        sliding_cache_len = int(os.environ.get("PREFILL_GEMMA4_SLIDING_CACHE_LEN", str(params.max_seq_len)))
        if not 0 < sliding_cache_len <= params.max_seq_len:
            raise ValueError(
                f"PREFILL_GEMMA4_SLIDING_CACHE_LEN must be in (0, {params.max_seq_len}], " f"got {sliding_cache_len}"
            )
        if sliding_cache_len % params.chunk_size:
            raise ValueError(
                f"PREFILL_GEMMA4_SLIDING_CACHE_LEN={sliding_cache_len} must be divisible by "
                f"PREFILL_CHUNK_SIZE={params.chunk_size}"
            )
        migration = allocate_global_migration_cache(
            mesh_device,
            num_users=params.num_users,
            num_layers=len(globals_),
            max_seq_len=params.max_seq_len,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            dtype=cache_dtype,
        )
        try:
            sliding_migration = allocate_sliding_migration_cache(
                mesh_device,
                num_users=params.num_users,
                num_layers=len(sliding),
                max_seq_len=sliding_cache_len,
                sp_axis=params.sp_axis,
                tp_axis=params.tp_axis,
                dtype=cache_dtype,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Failed to allocate Gemma 4 full-history sliding migration K/V. "
                f"Current profile: users={params.num_users}, layers={len(sliding)}, "
                f"length={sliding_cache_len}, CP={params.sp_factor}. Lower "
                "PREFILL_GEMMA4_SLIDING_CACHE_LEN for bring-up."
            ) from error
        return Gemma4KvCaches(
            paged=paged,
            migration=migration,
            sliding_migration=sliding_migration,
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.gemma4.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            weight_cache_path=params.weight_cache_path,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
            use_trace=params.use_trace,
            sliding_cache_len=int(os.environ.get("PREFILL_GEMMA4_SLIDING_CACHE_LEN", str(params.max_seq_len))),
        )
        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            model_path=self._model_path(),
            config=config,
        )

    @property
    def reference_model_cls(self):
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration
