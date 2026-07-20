# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``MiniMaxM3PrefillAdapter`` — the MiniMax-M3 plug-in for the model-agnostic prefill runner.

The prefill runner (``models/demos/common/prefill/runners/prefill_runner.py``) is model-agnostic: it
owns rank topology, the H2D socket, and the request serving loop, and drives a model through a
``PrefillModelAdapter`` (see ``models/demos/common/prefill/adapter.py``). This adapter is pure glue:
it says where M3's config / weights / trace live and how to build its runtime; all operational behavior
(running a chunk, the KV layout, PCC) lives on ``TtPrefillRuntime`` (``tt/tt_prefill_runtime.py``).

M3 subclasses ``PrefillModelAdapter`` directly (not the DeepSeek MLA base): it is a different
architecture (GQA + block-sparse MSA) with a REGULAR TP-head-sharded triple KV cache (K / V / index_k),
not the DeepSeek merged/replicated kvpe cache. Only single-rank prefill is wired — no pipeline (D2D).
KV-chunk-table migration IS wired: the multi-tensor cache is described by a multi-config table (one
config per (tensor, head-shard); see ``tt/runners/kv_chunk_table.py``).

Import-safety: the heavy stack (TtPrefillRuntime / Model / transformers AutoConfig / weight loading) is
imported lazily inside the methods that need it, so ``import ...adapters.minimax_m3`` stays cheap enough
for the H2D producer (which reads only path/trace attributes).

Env the operator sets (mirrors the rest of the M3 ecosystem):
  HF_MODEL / PREFILL_HF_MODEL   real MiniMax-M3 checkpoint dir (VL-wrapped config + bf16 safetensors)
  TT_CACHE_PATH                 tilized weight-cache root (defaults to the checkpoint dir)
  PREFILL_TRACE_DIR             golden trace dir (metadata.json + kv_cache/) for KV-PCC validation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams

# The common runner reads PREFILL_NUM_LAYERS with a hardcoded default of 61 (DeepSeek's layer count).
# M3 has 60 decoder layers. This adapter module is imported (via get_adapter) BEFORE the runner reads
# PREFILL_NUM_LAYERS, so a setdefault here gives M3 the right default without touching the common runner.
# An explicit PREFILL_NUM_LAYERS still wins (e.g. a partial-model bring-up run).
os.environ.setdefault("PREFILL_NUM_LAYERS", "60")


class MiniMaxM3Config:
    """Static model-dimension constants the common runner reads. The runner uses only
    ``FABRIC_PAYLOAD_SIZE`` (the fabric router's max packet payload, mirrored from the embedding dim as
    in the DeepSeek config); the rest document M3's dimensions for readers."""

    EMB_SIZE = 6144  # hidden_size
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload (mirrors DeepSeekV3Config convention)
    NUM_LAYERS = 60
    NUM_ATTENTION_HEADS = 64
    NUM_KEY_VALUE_HEADS = 4
    HEAD_DIM = 128
    ROTARY_DIM = 64
    VOCAB_SIZE = 200064


def _model_path() -> str:
    """Resolve the MiniMax-M3 checkpoint dir: PREFILL_HF_MODEL > HF_MODEL > hf_model_default. This is the
    dir AutoConfig / the weight loader read, and (absent TT_CACHE_PATH) the tilized-cache root."""
    return os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL") or MiniMaxM3PrefillAdapter.hf_model_default


class MiniMaxM3PrefillAdapter(PrefillModelAdapter):
    """MiniMax-M3 prefill adapter (GQA + MSA over TtPrefillRuntime, single-rank)."""

    # --- identity & runner defaults ---
    name = "minimax_m3"
    model_config = MiniMaxM3Config
    # AutoConfig can load M3 only from the full VL checkpoint (it carries the remote config code); the
    # repo config-only dir does not register model_type=minimax_m3. Operators set HF_MODEL/PREFILL_HF_MODEL.
    hf_model_default = "/mnt/models/MiniMaxAI/MiniMax-M3-ref/"
    ttnn_cache_default = ""  # M3 caches under the checkpoint dir (or TT_CACHE_PATH); no separate root
    default_gate_mode = "DEVICE_FP32"  # unused by M3 (kept for runner contract parity)
    prefill_trace_default = "/data/philei/models/minimax-m3-prefill-cache/golden/longbook_10240"

    l1_small_size = 0

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        """Load M3's HF config from the checkpoint and unwrap the text backbone (the published config is
        VL-wrapped, model_type=minimax_m3_vl; the LM dims live under ``.text_config``). The runner sets
        ``max_seq_len`` on the result afterward."""
        from transformers import AutoConfig

        model_path = _model_path()
        logger.info(f"Loading HF config for model={self.name!r} from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return getattr(cfg, "text_config", cfg)

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """The tilized ``.tensorbin`` cache dir, mirroring ``ModelArgs.weight_cache_path(bfloat8_b)`` (the
        layout the cache-populate run wrote): ``{TT_CACHE_PATH or checkpoint_dir}/tensor_cache_bfp8_{MeshShape}``.
        The model reads its per-tensor caches from here (on a hit the bf16 source is never touched)."""
        cache_dir = os.environ.get("TT_CACHE_PATH")
        cache_dir = Path(cache_dir) if cache_dir else Path(_model_path())
        sp, tp = mesh_shape
        path = cache_dir / f"tensor_cache_bfp8_{ttnn.MeshShape(sp, tp)}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # KV cache + runtime build
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Allocate M3's regular TP-head-sharded triple KV cache (K / V / index_k), one shared cache of
        num_users * num_layers user-major slots. The engine owns the returned object and passes it into
        every runtime call."""
        from models.demos.minimax_m3.tt.attention import allocate_kv_caches

        return allocate_kv_caches(
            mesh_device,
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            sp_axis=params.sp_axis,
            num_users=params.num_users,
            head_dim=hf_config.head_dim,
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Construct the M3 model and return the runtime. Weights load from the tilized cache when it is
        complete (empty state_dict, ~869GB bf16 source never read — DeepSeek's state_dict={} trick);
        otherwise the bf16 source is read (slow, first-run / cache-populate)."""
        from models.demos.minimax_m3.tt.model_config import ModelArgs
        from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
        from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

        # EXPERT_DTYPE selects the MoE routed-expert weight dtype (bf4 default here / bf8). The tilized
        # cache holds both, so either stays on the fast cache path. Same knob name as the M3 generate +
        # PCC harnesses. It feeds TtPrefillRuntimeConfig.expert_weight_dtype below.
        expert_dtype = ttnn.bfloat4_b if os.environ.get("EXPERT_DTYPE", "bf4") == "bf4" else ttnn.bfloat8_b
        logger.info(f"[minimax_m3] expert_dtype={expert_dtype} (EXPERT_DTYPE={os.environ.get('EXPERT_DTYPE', 'bf4')})")
        cache_path = params.weight_cache_path
        force_load = os.environ.get("M3_FORCE_LOAD_WEIGHTS") == "1"
        cache_only = not force_load and (
            os.environ.get("M3_WEIGHTS_FROM_CACHE") == "1"
            or weight_cache_is_complete(cache_path, hf_config, params.num_layers, expert_dtype)
        )
        if cache_only:
            logger.info(f"[minimax_m3] tilized weight cache complete at {cache_path}; loading from cache")
            state_dict = {}
        else:
            model_path = _model_path()
            logger.warning(
                f"[minimax_m3] weight cache incomplete at {cache_path}; reading bf16 source from {model_path} "
                f"(slow — populate the cache once to skip this)."
            )
            # Read only the shards for the layers this run builds.
            os.environ.setdefault("M3_LOAD_NLAYERS", str(params.num_layers))
            state_dict = ModelArgs.load_state_dict(model_path)

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            expert_weight_dtype=expert_dtype,
            weight_cache_path=cache_path,
            first_layer_idx=params.first_layer_idx,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
        )
        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            config=runtime_config,
        )
