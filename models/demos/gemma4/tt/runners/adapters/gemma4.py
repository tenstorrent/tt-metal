# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Common-prefill adapter for Gemma 4 31B global KV migration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams


class Gemma4PrefillConfig:
    """Static dimensions consumed by the common runner and producer."""

    EMB_SIZE = 5376
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    NUM_LAYERS = 60
    NUM_KEY_VALUE_HEADS = 4
    HEAD_DIM = 512
    ROTARY_DIM = 128
    MIGRATION_ROW_DIM = 640
    GLOBAL_LAYERS = tuple(range(5, 60, 6))


@dataclass
class Gemma4KvCaches(KvCaches):
    """Native paged caches plus the compact decode-row migration cache."""

    paged: list
    migration: object


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
        from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
        from models.demos.gemma4.tt.attention.global_migration import (
            allocate_global_migration_cache,
            global_layer_indices,
        )
        from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
        from models.tt_transformers.tt.common import PagedAttentionConfig

        if not (params.is_first_rank and params.is_last_rank and params.first_layer_idx == 0):
            raise NotImplementedError("initial Gemma 4 prefill migration support is single-rank only")
        if params.num_layers != self.model_config.NUM_LAYERS:
            raise ValueError(f"Gemma 4 requires all 60 layers on the single rank, got {params.num_layers}")
        if params.mesh_shape != (8, 4) or params.sp_axis != 0 or params.tp_axis != 1:
            raise ValueError(
                f"Gemma 4 migration requires mesh=(8,4), sp_axis=0, tp_axis=1; got "
                f"{params.mesh_shape}, {params.sp_axis}, {params.tp_axis}"
            )

        block_size = 64
        if params.max_seq_len % (block_size * params.sp_factor):
            raise ValueError(f"max_seq_len={params.max_seq_len} must divide into {block_size}-token blocks across CP8")
        # This cache crosses a raw-byte migration boundary, so its dtype is a
        # protocol constant rather than a model-path-dependent precision-table
        # lookup (HF snapshot directory names are not stable identifiers).
        cache_dtype = ttnn.bfloat8_b

        max_blocks = params.num_users * params.max_seq_len // block_size
        paged_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_blocks)
        paged = []
        for layer_idx in range(params.num_layers):
            config = Gemma4AttentionConfig(hf_config, layer_idx)
            if config.is_sliding:
                config.cache_position_modulo = config.sliding_window
                block_override = params.num_users * config.sliding_window // block_size
            else:
                block_override = max_blocks // params.sp_factor
            paged.append(
                init_kv_cache(
                    mesh_device,
                    config,
                    max_batch_size=params.num_users,
                    max_seq_len=params.max_seq_len,
                    paged_attention_config=paged_config,
                    cache_dtype=cache_dtype,
                    max_num_blocks_override=block_override,
                )
            )

        globals_ = global_layer_indices(hf_config.layer_types)
        if globals_ != self.model_config.GLOBAL_LAYERS:
            raise ValueError(f"unexpected Gemma 4 global layer pattern: {globals_}")
        migration = allocate_global_migration_cache(
            mesh_device,
            num_users=params.num_users,
            num_layers=len(globals_),
            max_seq_len=params.max_seq_len,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            dtype=cache_dtype,
        )
        return Gemma4KvCaches(paged=paged, migration=migration)

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
