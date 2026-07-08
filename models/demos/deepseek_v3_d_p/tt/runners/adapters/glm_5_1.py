# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GLM-5.1 prefill adapter.

Same MLA + MoE family as DeepSeek-V3 / Kimi, so it subclasses ``MLAPrefillAdapter`` and
inherits the serving path (``build_runtime`` / ``allocate_kv_cache`` / ``weight_cache_path``).
GLM diverges in two ways the adapter has to account for:

  * The DSA (sparse) attention indexer. This is resolved from the config at model-build time
    (``resolve_has_indexer`` reads the ``index_*`` attrs), so no adapter wiring is needed beyond
    handing the runtime a config that carries them — ``glm_hf_config`` does.
  * ``model_type`` ``glm_moe_dsa`` is not registered with transformers, so ``AutoConfig`` (the
    base ``load_hf_config``) cannot load it; the config is hand-built via ``glm_hf_config``.

Like Kimi it has a single expert group with a device gate, so the MoE routing all-gather's
semaphores go to L1_SMALL (needs the L1_SMALL carve-out at mesh-open).
"""

from __future__ import annotations

import os
from typing import Callable

from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config, glm_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class GLM51Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "glm_5_1"
    model_config = GLM51Config
    # Config is hand-built (see load_hf_config); this stays empty so an accidental AutoConfig read
    # yields an obvious empty path. Real weights come from the TTNN cache.
    hf_model_default = ""
    ttnn_cache_default = "/data/nmilicevic/glm_tp4_cache"
    default_gate_mode = "DEVICE_FP32"  # GLM: single expert group
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/structured_traces/glm_51_code_debug_55k_vllm"

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL.
    l1_small_size = 512
    routing_use_l1_small_for_semaphores = True

    def load_hf_config(self):
        """GLM's ``glm_moe_dsa`` isn't AutoConfig-loadable, so return the hand-built HF-attribute
        config (carries the dims + the DSA ``index_*`` attrs the sparse path resolves against). The
        runner overwrites ``max_seq_len`` after; seed the builder with it so rope config is consistent."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        return glm_hf_config(max_seq=max_seq)

    # --- test metadata (HF download coordinates + PCC thresholds + golden trace) ---
    hf_repo_id = "zai-org/GLM-5.1-FP8"
    env_var = "GLM51_HF_MODEL"
    mla_ref_cache_env = "GLM51_MLA_REF_CACHE"
    ref_cache_env = "TT_GLM51_PREFILL_HOST_REF_CACHE"
    ttnn_cache_env = "TT_GLM51_PREFILL_TTNN_CACHE"
    supports_pretrained = True
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    prefill_trace_layout = "chunked_group_a_v1"
    test_prefill_trace_default = (
        "/mnt/models/deepseek-prefill-cache/golden/structured_traces/glm_51_code_debug_55k_vllm"
    )

    @property
    def config_builder(self) -> Callable:
        """The sparse-MLA reference tests resolve GLM's config through this (conftest
        _resolve_config_only); serving goes through load_hf_config. Both hand off to glm_hf_config."""
        return glm_hf_config
