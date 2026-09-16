# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GLM-5.3 prefill adapter.

Architecturally IDENTICAL to GLM-5.2 — only the checkpoint differs — so it subclasses
``GLM52Adapter`` and overrides just the identity, the download coordinates, and the default
cache/trace paths. Same pattern as ``kimi_k2_7.py`` on top of ``kimi_k2_6.py``.

"Identical" here is a measured claim, not an assumption: diffing ``zai-org/GLM-5.3/config.json``
against ``zai-org/GLM-5.2-FP8/config.json`` leaves ONE differing key, ``transformers_version``
(5.12.0 -> 5.15.0). Everything the device model reads matches — 78 layers, 256 routed experts in 1
group, hidden 6144, head_dim 192, qk_nope/rope 192/64, v_head 256, kv_lora 512, q_lora 2048,
``index_head_dim`` 128 / ``index_n_heads`` 32 / ``index_topk`` 2048, the same 21-full/57-shared
``indexer_types`` map, rope_theta 8e6, vocab 154880. So ``model_config`` (GLM52Config) and
``config_builder`` (glm_5_2_hf_config) are inherited unchanged rather than duplicated; a real
divergence in a later GLM must add its own config instead of editing this docstring.

Corroboration that this is the accepted shape: the blaze serving stack's own GLM-5.3 artifacts
(``/mnt/models/blaze/zai-org/blaze-glm53-cache``) are keyed ``GLM5_2_BLAZE_CONFIG_*``, i.e. it also
drives 5.3 through the 5.2 config.

Weight-cache dir is ``{ttnn_cache_env}/glm_5_3_bh_32dev/{sp}x{tp}`` — the ``glm_5_3`` prefix comes
from ``name``, so a 5.3 cache can never be mistaken for (or clash with) a 5.2 one, even when both
live under the same root.
"""

from __future__ import annotations

from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_2 import GLM52Adapter


class GLM53Adapter(GLM52Adapter):
    # --- identity & runner defaults ---
    name = "glm_5_3"
    # The FP8 checkpoint (`zai-org/GLM-5.3`), in HF hub-cache layout — `get_or_download_model`
    # accepts either a flat checkout or a hub root via `_resolve_hf_snapshot_dir`.
    # FP8 and not the BF16 release on purpose: FP8-dequantized weights are not bit-equal to native
    # BF16, and the only 5.3 capture that exists (glm-traces/glm53-1M-last5k) names `zai-org/GLM-5.3`
    # as its source model in its own README. That matters beyond serving, because
    # `generate_prompt_trace.py` falls back to `hf_model_default` — a BF16 default would silently
    # GENERATE goldens from the other flavor.
    # These two live on /data rather than the shared /mnt/models because /mnt/models sits at 99%
    # (1.5 TB free) and could not hold the 700 GB checkpoint plus the 401 GB cache. Restage both on
    # /mnt/models and repoint these when there is room; `PREFILL_HF_MODEL` / `GLM53_HF_MODEL` and
    # `PREFILL_TTNN_CACHE` / `TT_GLM53_PREFILL_TTNN_CACHE` override them meanwhile.
    hf_model_default = "/data/ppopovic/hf/models--zai-org--GLM-5.3"
    ttnn_cache_default = "/data/ppopovic/glm53/ttnn_cache"
    # Empty ON PURPOSE: there is no usable GLM-5.3 golden trace yet. The one capture that exists
    # (glm-traces/glm53-1M-last5k) carries no `dsa/indexer_k_layer_*`, so the indexer-K PCC checks
    # would silently no-op on it, and it is a 2^20-position capture needing max_seq_len=1048576.
    # An empty default keeps the runner honest rather than pointing it at a trace that cannot
    # validate this model. The RUNNER only logs this value, so runner-only runs are unaffected; the
    # PRODUCER resolves it (_resolve_slot_prompts -> resolve_trace_dir) and will raise unless
    # PREFILL_TRACE_DIR or PREFILL_PRODUCER_SLOT_TRACES is set explicitly.
    prefill_trace_default = ""

    # --- test metadata (HF download coordinates) ---
    # The bare repo IS the FP8 one (quantization_config: fmt e4m3, weight_block_size [128,128]),
    # mirroring how GLM-5.2 names `zai-org/GLM-5.2-FP8`. 141 safetensors, ~704 GiB.
    # NOT `GLM-5.3-Flash`: that is a different model (320B multimodal, hybrid sparse+LINEAR
    # attention) with no TT implementation for its linear-attention layers.
    hf_repo_id = "zai-org/GLM-5.3"
    env_var = "GLM53_HF_MODEL"
    mla_ref_cache_env = "GLM53_MLA_REF_CACHE"
    ref_cache_env = "TT_GLM53_PREFILL_HOST_REF_CACHE"
    ttnn_cache_env = "TT_GLM53_PREFILL_TTNN_CACHE"
    # Full depth, so the fixture's shard-selective download covers a 78-layer cache build instead of
    # the base class's 24-layer test slice. `download_model_weights` unions the shards for layers
    # 0..77 with the embed_tokens shard (which also carries `lm_head.weight`) and the `model.norm`
    # shard: 140 of 141. The one it leaves out holds only `model.layers.78.*`, the MTP layer
    # (`num_nextn_predict_layers: 1`), which prefill never builds.
    num_layers_to_download = 78

    # Also empty: with no trace configured, the chunked tests' _resolve_trace_dir returns a
    # "<unset PREFILL_TRACE_DIR for glm_5_3>" sentinel and the row SKIPS cleanly, instead of
    # running against a trace that cannot check what it appears to check.
    # For context on why glm53-1M-last5k is not wired in here: the chunked test picks indexer-K
    # layers via
    #   [i for i in range(num_layers) if (trace_dir / "dsa" / f"indexer_k_layer_{i}").exists()]
    # so a trace without those files makes the check a silent no-op rather than a failure.
    # glm53-1M-last5k ships 21 `dsa_topk_indices_layer_*` and ZERO `indexer_k_layer_*`, so wiring
    # it in here would buy a green row that checks less than it claims. It is also a 2^20-position
    # capture (rows [1043456, 1048576)), so it needs a max_seq_len=1048576 run — not the 55k config
    # the GLM-5.2 CI legs use. Point PREFILL_TRACE_DIR at it explicitly if you want it anyway.
    test_prefill_trace_default = ""
