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
from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig


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

        return KvCaches(
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
        )
        return TtPrefillRuntime(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            config=runtime_config,
        )

    @property
    def reference_model_cls(self):
        from transformers import GptOssForCausalLM

        return GptOssForCausalLM
