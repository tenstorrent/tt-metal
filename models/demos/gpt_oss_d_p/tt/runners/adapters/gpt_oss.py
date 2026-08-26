# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``GptOssPrefillAdapter`` — the common/prefill engine <-> GPT-OSS boundary.

GPT-OSS is GQA (not MLA), so this subclasses ``PrefillModelAdapter`` DIRECTLY rather than the
DeepSeek-family ``MLAPrefillAdapter``. It is pure glue: it says where the config / weights / golden
trace live, allocates the GQA K/V cache, and builds the ``TtPrefillRuntime``. All operational
behavior (running a chunk, reading/writing the cache, PCC) lives on the runtime; the engine drives it
and owns all comms + the cache lifetime.

Import-light: the heavy model stack (TtPrefillRuntime, transformers AutoConfig, the KV cache) is
imported lazily inside the methods so ``import ...adapters.gpt_oss`` stays cheap for the H2D producers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig


@dataclass
class GptOssKvCaches(KvCaches):
    """Concrete KvCaches for GPT-OSS: a list holding the single GptOssKVCache (k+v struct),
    indexable so the runtime's _resolve_kv pulls the cache back via [0] (engine treats it opaquely)."""

    caches: list

    def __getitem__(self, idx):
        return self.caches[idx]


class GptOssPrefillAdapter(PrefillModelAdapter):
    """GPT-OSS-120B prefill adapter (GQA + attention sinks + sliding/full alternation + EP MoE)."""

    # --- identity & runner defaults ---
    name = "gpt_oss_d_p"
    model_config = GptOss120BConfig
    hf_model_default = "models/demos/gpt_oss/configs/gpt-oss-120b"  # config.json dir; PREFILL_HF_MODEL overrides
    ttnn_cache_default = ""  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides (empty => no cache)
    prefill_trace_default = ""  # golden trace dir (token_ids + KV); PREFILL_TRACE_DIR overrides
    default_gate_mode = "DEVICE_FP32"

    # --- test metadata ---
    hf_repo_id = "openai/gpt-oss-120b"
    env_var = "GPT_OSS_HF_MODEL"
    num_layers_to_download = 36
    tokenizer_trust_remote_code = True
    needs_flat_config_dir = False
    prefill_trace_layout = "single_file"

    # ------------------------------------------------------------------
    # HF config
    # ------------------------------------------------------------------
    def load_hf_config(self):
        from transformers import AutoConfig

        model_path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
        logger.info(f"Loading HF config for model={self.name!r} from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Defensive unwrap (gpt-oss is not a VL wrapper, but keep the pattern for variants).
        return getattr(cfg, "text_config", cfg)

    # ------------------------------------------------------------------
    # Weight cache
    # ------------------------------------------------------------------
    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        """Mirror the pytest weight_cache_path layout:
        $PREFILL_TTNN_CACHE / {name}_{arch}_{N}dev / {sp}x{tp}. None if explicitly empty.

        Use ``sp * tp`` for N — not ``ttnn.get_num_devices()``. The runner calls this from
        ``_print_config`` before ``open_mesh_device``; with co-located migration workers
        ``GetNumAvailableDevices`` can throw ``unordered_map::at`` and abort Gate 2.
        """
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        arch = "bh" if is_blackhole() else "wh"
        sp, tp = mesh_shape
        num_devices = int(sp) * int(tp)
        path = Path(env_cache) / f"{self.name}_{arch}_{num_devices}dev" / f"{sp}x{tp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # KV cache + runtime build
    # ------------------------------------------------------------------
    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams) -> KvCaches:
        """Allocate the GQA K/V cache (one shared cache of num_users * num_layers user-major slots;
        K/V heads on the TP cols, sequence SP-sharded block-cyclic). Dense GQA has no secondary cache,
        so the returned KvCaches holds just the one GptOssKVCache at index 0. The engine owns it and
        passes it into every runtime call.

        Index 0 is a ``GptOssKVCache`` (``.k`` / ``.v`` tensors). Migration uses the multi-config
        table in ``tt/runners/kv_chunk_table.py`` (config 0..N-1 = k heads, N..2N-1 = v heads).
        """
        from models.demos.gpt_oss_d_p.tt.attention import allocate_kv_cache

        return GptOssKvCaches(
            [
                allocate_kv_cache(
                    mesh_device,
                    num_layers=params.num_layers,
                    max_seq_len=params.max_seq_len,
                    sp_axis=params.sp_axis,
                    num_users=params.num_users,
                    head_dim=hf_config.head_dim,
                )
            ]
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        """Build the GPT-OSS model + runtime for this rank. The runtime is stateless w.r.t. the KV
        cache (owns_kv_cache=False): the engine allocated it via allocate_kv_cache and passes it into
        each call."""
        import ttnn
        from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
        from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            weight_cache_path=params.weight_cache_path,
            owns_kv_cache=False,  # engine owns the cache (from allocate_kv_cache); passed into every call
            # PREFILL_TOPOLOGY=linear runs pods without torus wraparound (same knob as the harness).
            topology=(
                ttnn.Topology.Linear if os.getenv("PREFILL_TOPOLOGY", "ring") == "linear" else ttnn.Topology.Ring
            ),
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            first_layer_idx=params.first_layer_idx,
        )

        if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1":
            state_dict = {}
        else:
            model_args = ModelArgs(mesh_device=mesh_device)
            logger.info("Loading real bf16 gpt-oss weights (slow: safetensors read)...")
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            config=runtime_config,
        )

    @property
    def reference_model_cls(self):
        from transformers import GptOssForCausalLM

        return GptOssForCausalLM
