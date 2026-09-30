# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GLM-5.2 prefill adapter — kept runnable through the runner, but no longer tested or in CI.

GLM-5.3 replaced GLM-5.2 as the tested GLM checkpoint. The two are architecturally identical — their
``config.json`` files differ only in ``transformers_version`` — so this subclasses ``GLM53Adapter`` and
overrides just the identity, the checkpoint / cache / golden-trace paths, and the env-var names. Model
config (``GLM53Config`` / ``glm_5_3_hf_config``), KV-cache allocation and the pipeline split points are
all inherited.

Select it with ``PREFILL_MODEL=glm_5_2`` or the ``glm52*.json`` manifests. The weight-cache dir is
``{ttnn_cache}/glm_5_2_bh_32dev/{sp}x{tp}`` (prefix from ``name``), so it never collides with a 5.3 cache.
"""

from __future__ import annotations

from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter


class GLM52Adapter(GLM53Adapter):
    # --- identity & runner defaults ---
    name = "glm_5_2"
    hf_model_default = "/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8"
    ttnn_cache_default = "/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache"
    # Must carry the DSA indexer-K cache (dsa/indexer_k_layer_*), or the indexer-K PCC checks silently skip.
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k"

    # --- test metadata (HF download coordinates + golden trace) ---
    hf_repo_id = "zai-org/GLM-5.2-FP8"
    env_var = "GLM52_HF_MODEL"
    mla_ref_cache_env = "GLM52_MLA_REF_CACHE"
    ref_cache_env = "TT_GLM52_PREFILL_HOST_REF_CACHE"
    ttnn_cache_env = "TT_GLM52_PREFILL_TTNN_CACHE"
    test_prefill_trace_default = prefill_trace_default
