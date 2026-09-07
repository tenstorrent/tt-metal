# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 31B adapter for the model-agnostic prefill runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import ttnn
from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams

os.environ.setdefault("PREFILL_NUM_LAYERS", "60")


class Gemma4PrefillConfig:
    EMB_SIZE = 5376
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    NUM_LAYERS = 60
    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 16
    NUM_GLOBAL_KEY_VALUE_HEADS = 4
    HEAD_DIM = 256
    GLOBAL_HEAD_DIM = 512
    VOCAB_SIZE = 262144


def _model_path() -> str:
    return os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL") or Gemma4PrefillAdapter.hf_model_default


class Gemma4PrefillAdapter(PrefillModelAdapter):
    """Context-parallel Gemma 4 prefill with durable migration-ready KV.

    Pipeline-parallel capable: the engine hands each rank its own layer window and
    ``is_first_rank`` / ``is_last_rank``, and everything downstream is keyed by the
    GLOBAL layer index. The canonical shapes are a single (8,4) rank (CP=8 x TP=4) and
    four (8,1) ranks (CP=8 x TP=1), one per column of a Blackhole galaxy.
    """

    name = "gemma4_31b"
    model_config = Gemma4PrefillConfig
    hf_model_default = "google/gemma-4-31B-it"
    ttnn_cache_default = ""
    prefill_trace_default = ""
    default_gate_mode = "DEVICE_FP32"
    pipeline_activation_emb_tp_sharded = False

    def load_hf_config(self):
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        return Gemma4ModelArgs.load_hf_config(_model_path())

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        hf_config = Gemma4ModelArgs.load_hf_config(_model_path())
        args = Gemma4ModelArgs.from_hf_config(hf_config)
        args.model_cache_path = args.resolve_model_cache_path(_model_path())
        args.cluster_shape = mesh_shape
        return args.weight_cache_path(ttnn.bfloat16)

    @staticmethod
    def _mesh_config(params: PrefillRunParams):
        from models.demos.gemma4.config import MeshConfig, ModeConfig

        return MeshConfig(
            params.mesh_shape,
            decode=ModeConfig(tp=params.tp_factor),
            prefill=ModeConfig(tp=params.tp_factor, sp=params.sp_factor),
        )

    @staticmethod
    def _validate(params: PrefillRunParams) -> None:
        cp = params.sp_factor
        tp = params.tp_factor
        if cp <= 1:
            raise NotImplementedError(f"Gemma 4 common prefill requires context parallelism, got sp={cp}")
        # ring_joint asserts halo_tokens <= N_local_q, and the 1024-token sliding window
        # rounds up to a 32-tile halo, so a CP rank's Q slab must cover a whole window.
        if params.chunk_size % (cp * 1024):
            raise ValueError(
                f"chunk_size={params.chunk_size} must be a multiple of 1024*cp={1024 * cp}: every CP "
                f"rank needs a Q slab at least one sliding window wide"
            )
        if params.max_seq_len % params.chunk_size:
            raise ValueError(f"max_seq_len={params.max_seq_len} must be divisible by chunk={params.chunk_size}")
        for heads, what in (
            (Gemma4PrefillConfig.NUM_ATTENTION_HEADS, "query"),
            (Gemma4PrefillConfig.NUM_KEY_VALUE_HEADS, "sliding KV"),
            (Gemma4PrefillConfig.NUM_GLOBAL_KEY_VALUE_HEADS, "global KV"),
        ):
            if heads % tp:
                raise ValueError(f"TP={tp} does not divide Gemma 4's {heads} {what} heads")

    @staticmethod
    def _model_args(hf_config, params: PrefillRunParams):
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        args = Gemma4ModelArgs.from_hf_config(hf_config)
        args.max_seq_len = params.max_seq_len
        return args

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.gemma4.tt.runners.kv_caches import allocate_ring_kv_caches

        self._validate(params)
        return allocate_ring_kv_caches(
            mesh_device,
            self._model_args(hf_config, params),
            self._mesh_config(params),
            num_users=params.num_users,
            max_seq_len=params.max_seq_len,
            num_layers=params.num_layers,
            first_layer_idx=params.first_layer_idx,
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.gemma4.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        self._validate(params)
        config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            weight_cache_path=params.weight_cache_path,
            first_layer_idx=params.first_layer_idx,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            use_trace=params.use_trace,
        )
        return TtPrefillRuntime(mesh_device=mesh_device, model_path=_model_path(), config=config)
