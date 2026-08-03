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

import ttnn
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
        $PREFILL_TTNN_CACHE / {name}_{arch}_{N}dev / {sp}x{tp}. None if explicitly empty."""
        env_cache = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not env_cache:
            return None
        arch = "bh" if is_blackhole() else "wh"
        num_devices = ttnn.get_num_devices()
        sp, tp = mesh_shape
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

        NOTE: index 0 is a ``GptOssKVCache`` dataclass (holding the k + v tensors), not a single raw
        tensor as the DeepSeek MLA KVPE cache is. The standalone/PCC path treats kv_caches[0] opaquely
        and works. The migration path (build_kv_chunk_table) assumes a single primary tensor and is NOT
        wired for GPT-OSS yet (TODO: split into KvCaches([k, v]) + a GQA table builder if migration is
        needed)."""
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
        each call.

        Weight loading mirrors ``minimax_m3.build_runtime``: on a complete tilized cache pass an empty
        state_dict (fast path — ~869GB bf16 source never read); otherwise fall back to reading the
        bf16 source via ``ModelArgs.load_state_dict`` (slow first-run / cache-populate path). The
        cache-completeness check also verifies the routed-expert bias sidecar, without which
        ``MLP.__init__`` raises mid-build on a cache-only run.
        """
        from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
        from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
        from models.demos.gpt_oss_d_p.tt.runners.adapters.weight_cache import weight_cache_is_complete

        expert_dtype = ttnn.bfloat8_b if os.environ.get("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        cache_path = params.weight_cache_path
        force_load = os.environ.get("GPT_OSS_FORCE_LOAD_WEIGHTS") == "1"
        cache_only = not force_load and (
            os.environ.get("GPT_OSS_WEIGHTS_FROM_CACHE") == "1"
            or weight_cache_is_complete(cache_path, hf_config, params.num_layers, expert_dtype)
        )
        if cache_only:
            logger.info(f"[gpt_oss_d_p] tilized weight cache complete at {cache_path}; loading from cache")
            state_dict = {}
        else:
            model_path = os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default
            logger.warning(
                f"[gpt_oss_d_p] weight cache incomplete at {cache_path}; reading bf16 source from {model_path} "
                f"(slow — populate the cache once to skip this)."
            )
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
            owns_kv_cache=False,  # engine owns the cache (from allocate_kv_cache); passed into every call
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
        )
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
