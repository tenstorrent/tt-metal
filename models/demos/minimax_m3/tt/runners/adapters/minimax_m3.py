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
not the DeepSeek merged/replicated kvpe cache. Single-rank AND pipeline-parallel (multi-galaxy D2D)
prefill are wired: the runtime slices the model by rank (first_layer_idx / is_first_rank / is_last_rank),
and the D2D activation ships emb-replicated across TP (pipeline_activation_emb_tp_sharded=False) to match
M3's SP residual layout. KV-chunk-table migration is wired for BOTH: the multi-tensor cache is described
by a multi-config table (one config per (tensor, head-shard); see ``tt/runners/kv_chunk_table.py``), and
with a pipeline the gathered stage layouts merge every stage's layers into one table at global layer
indices (one layout per cache — k, v, index_k — via ``kv_migration_stages``).

A multi-turn continuation may resume mid-chunk at any 32-token boundary (the KV-cache writer's tile grid),
as the producer's ``PREFILL_PRODUCER_MULTI_TURN_PROB`` mode does.

Import-safety: the heavy stack (TtPrefillRuntime / Model / transformers AutoConfig / weight loading) is
imported lazily inside the methods that need it, so ``import ...adapters.minimax_m3`` stays cheap enough
for the H2D producer (which reads only path/trace attributes).

Env the operator sets (mirrors the rest of the M3 ecosystem):
  HF_MODEL / PREFILL_HF_MODEL   MiniMax-M3-architecture checkpoint dir (VL-wrapped config + bf16 or MXFP8/NVFP4 safetensors);
                                default: the selected variant's checkpoint dir (utils/model_variant.py)
  M3_MODEL_VARIANT              MiniMax-M3 (default) or MiniMax-M3.1 when no checkpoint is given; a given checkpoint's
                                config decides the variant itself
  TT_CACHE_PATH                 tilized weight-cache root (default: the variant's weka cache root for its default
                                checkpoint when present, else the checkpoint dir)
  PREFILL_TRACE_DIR             golden trace dir (metadata.json + kv_cache/) for KV-PCC validation
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams
from models.demos.minimax_m3.utils.model_variant import (
    DEFAULT_VARIANT,
    VARIANTS,
    checkpoint_dir_env,
    read_checkpoint_text_config,
    select_variant,
)

# Golden trace the runner and the block tests default to, under the variant's golden dir.
DEFAULT_TRACE = "longbook_10240"


def model_dims(text_config: dict) -> SimpleNamespace:
    """Model-dimension constants the common runner reads, from the run's text config: ``NUM_LAYERS`` (its
    layer-count default, overridable with PREFILL_NUM_LAYERS for a partial-model bring-up) and
    ``FABRIC_PAYLOAD_SIZE`` (the fabric router's max packet payload, mirrored from the embedding dim as in the
    DeepSeek config); the rest describe the attention / KV layout."""
    return SimpleNamespace(
        EMB_SIZE=text_config["hidden_size"],
        FABRIC_PAYLOAD_SIZE=text_config["hidden_size"],
        NUM_LAYERS=text_config["num_hidden_layers"],
        NUM_ATTENTION_HEADS=text_config["num_attention_heads"],
        NUM_KEY_VALUE_HEADS=text_config["num_key_value_heads"],
        HEAD_DIM=text_config["head_dim"],
        ROTARY_DIM=text_config["rotary_dim"],
        VOCAB_SIZE=text_config["vocab_size"],
    )


def _model_path() -> str:
    """Resolve the MiniMax-M3 checkpoint dir: PREFILL_HF_MODEL > HF_MODEL > the selected variant's default dir. This
    is the dir AutoConfig / the weight loader read, and (absent TT_CACHE_PATH) the legacy tilized-cache root; see
    model_config.default_weight_cache_root."""
    return checkpoint_dir_env() or select_variant().weights_dir


class MiniMaxM3PrefillAdapter(PrefillModelAdapter):
    """MiniMax-M3 prefill adapter (GQA + MSA over TtPrefillRuntime). The model variant (and with it the default
    checkpoint, golden trace and dimension constants) is resolved when the runner instantiates the adapter."""

    # --- identity & runner defaults ---
    name = "minimax_m3"
    # Class-level values are the default variant's; __init__ re-targets them to the selected variant.
    # AutoConfig can load M3 only from the full VL checkpoint (it carries the remote config code); the repo
    # config-only dir does not register model_type=minimax_m3.
    model_config = model_dims(VARIANTS[DEFAULT_VARIANT].config)
    hf_model_default = VARIANTS[DEFAULT_VARIANT].weights_dir + "/"
    ttnn_cache_default = (
        ""  # unused: see model_config.default_weight_cache_root (TT_CACHE_PATH > variant weka root > ckpt)
    )
    default_gate_mode = "DEVICE_FP32"  # unused by M3 (kept for runner contract parity)
    prefill_trace_default = f"{VARIANTS[DEFAULT_VARIANT].golden_dir}/{DEFAULT_TRACE}"

    def __init__(self):
        super().__init__()
        checkpoint = checkpoint_dir_env()
        self.variant = select_variant(checkpoint)
        self.hf_model_default = self.variant.weights_dir + "/"
        self.prefill_trace_default = f"{self.variant.golden_dir}/{DEFAULT_TRACE}"
        self.test_prefill_trace_default = self.prefill_trace_default
        text_config = read_checkpoint_text_config(checkpoint or self.variant.weights_dir) or self.variant.config
        self.model_config = model_dims(text_config)
        logger.info(f"[minimax_m3] model variant {self.variant.name}")

    # high_bw_all_gather (MSA cache read) parks its semaphores in L1_SMALL; 0 makes it fall back to general
    # L1 with a warning and an L1-fragmentation risk. Same value as tt/ccl.py L1_SMALL_SIZE (kept literal so
    # importing the adapter stays cheap).
    l1_small_size = 1152

    # The D2D hidden state ships in the residual stream's layer-boundary layout (see tt/residual.py).
    @property
    def pipeline_activation_emb_tp_sharded(self):
        from models.demos.minimax_m3.tt.residual import use_sharded_residual

        return use_sharded_residual()

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        """Load M3's HF config from the checkpoint and unwrap the text backbone (the published config is
        VL-wrapped, model_type=minimax_m3_vl; the LM dims live under ``.text_config``). The runner sets
        ``max_seq_len`` on the result afterward."""
        from transformers import AutoConfig

        from models.demos.minimax_m3.utils.checkpoint import text_config_of

        model_path = _model_path()
        logger.info(f"Loading HF config for model={self.name!r} from {model_path}")
        return text_config_of(AutoConfig.from_pretrained(model_path, trust_remote_code=True))

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """The tilized ``.tensorbin`` cache dir, mirroring ``ModelArgs.weight_cache_path(bfloat8_b)`` (the
        layout the cache-populate run wrote): ``{default_weight_cache_root()}/tensor_cache_bfp8_{MeshShape}``.
        The model reads its per-tensor caches from here (on a hit the bf16 source is never touched)."""
        from models.demos.minimax_m3.tt.model_config import default_weight_cache_root

        cache_dir = default_weight_cache_root(_model_path())
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
            bank_aligned=True,
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
            or weight_cache_is_complete(
                cache_path,
                hf_config,
                params.num_layers,
                expert_dtype,
                first_layer_idx=params.first_layer_idx,
                is_first_rank=params.is_first_rank,
                is_last_rank=params.is_last_rank,
            )
        )
        if cache_only:
            logger.info(f"[minimax_m3] tilized weight cache complete at {cache_path}; loading from cache")
            state_dict = {}
        else:
            model_path = _model_path()
            logger.warning(
                f"[minimax_m3] weight cache incomplete at {cache_path}; reading source weights from {model_path} "
                f"(slow — populate the cache once to skip this)."
            )
            # Read only the shards for the GLOBAL layer slice this rank builds
            # ([first_layer_idx, first_layer_idx+num_layers)) so a pipeline rank with first_layer_idx>0
            # loads its own layers, not layers 0..num_layers-1. Assign unconditionally: params is the
            # authoritative resolved slice, so a stale value inherited from the environment must not win.
            os.environ["M3_LOAD_LAYER_START"] = str(params.first_layer_idx)
            os.environ["M3_LOAD_NLAYERS"] = str(params.num_layers)
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
            pipeline_activation_emb_tp_sharded=self.pipeline_activation_emb_tp_sharded,
            use_trace=params.use_trace,
            overlap_shared_expert=params.overlap_shared_expert_with_dispatch,
        )
        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            config=runtime_config,
        )
