# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K3.1 prefill adapter: Kimi-K3 with its 24 full-attention layers turned into DSA layers.

K3.1 is selected through K3's own stack (``TtKimiK3Runtime`` / ``TtKimiK3Transformer`` /
``TtKimiK3Block``, KDA, AttnRes, latent MoE), so it subclasses ``KimiK3Adapter`` and inherits its
runtime build, layer schedule, pipeline split points and migration-table layout; a K3 fix therefore
reaches K3.1 too. The model difference is carried by ``KimiK31Config`` / ``kimi_k3_1_hf_config`` (the
``index_*`` DSA fields), not by code here.

K3's block drives the DSA indexer K3.1's config makes ``ttMLA`` build (the transformer threads the
sparse kvpe format and the index cache), and K3's shared ``allocate_kv_cache`` gives K3.1 its
scaled-FP8 kvpe and bf8 indexer key cache.

What this class overrides is identity and artifacts: K3.1's weights are all retrained, so nothing
recorded against the K3 checkpoint (TTNN cache, golden traces, DFlash drafter, MLA reference) is valid
for it. Every DSA layer owns its indexer, so no top-k selection crosses a pipeline split and
``passes_indexer_indices`` keeps the base default (False).

Select it with ``PREFILL_MODEL=kimi_k3_1``.
"""

from __future__ import annotations

from typing import Callable

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter


class KimiK31Adapter(KimiK3Adapter):
    # --- identity & runner defaults ---
    name = "kimi_k3_1"
    model_config = KimiK31Config
    # Only printed by the runner: the config is hand-built (``config_builder``).
    hf_model_default = "/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized"
    ttnn_cache_default = "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3.1-Cache/Kimi-K3.1-Cache-prefill"
    # Miguel's 8192-token vLLM trace of K3.1 (kimi_k31_pro_10k_v2: all 93 decoder outputs, KV and indexer
    # keys of all 24 DSA layers), rewritten into K3's per-stream layout by a one-shot converter outside the repo
    # (the golden's metadata.json `converted_from` names its source and version).
    # Under scratch/ because stable/, where K3's golden sits, is writable by storage-wg only. `_conv2`: converter
    # version 2, whose decoder_output is K3's running sum (the version-1 directory holds the tracer's FFN output).
    prefill_trace_default = "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K3.1-Cache/golden/kimi_k31_pro_10k_v2_conv2"
    # Explicit: __init_subclass__ fills it only when None, and K3's is already set, so it would be inherited.
    test_prefill_trace_default = prefill_trace_default
    # The trace holds no KDA state snapshots.
    kda_golden_default = None
    mla_trace_defaults = ()

    # Kimi-K3-DSpark was trained against K3, not K3.1.
    supports_dflash = False
    # The DSA layers' sparse-MLA CCLs need more than K3's L1_SMALL (see KimiK31Config.L1_SMALL_SIZE).
    l1_small_size = KimiK31Config.L1_SMALL_SIZE
    dflash_model_default = ""
    dflash_golden_default = ""

    # --- test metadata ---
    hf_repo_id = "moonshotai/Kimi-K3.1-Pro"
    env_var = "KIMI_K31_HF_MODEL"
    mla_ref_cache_env = "KIMI_K31_MLA_REF_CACHE"
    ref_cache_env = "TT_KIMI_K31_PREFILL_HOST_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_K31_PREFILL_TTNN_CACHE"
    # Real-weight tests are not wired yet. None, not K3's: conftest's pretrained fixtures would otherwise
    # read K3's repo-local checkpoint, or download the whole multi-TB K3.1 one.
    supports_pretrained = False
    pretrained_mla_layer = None
    default_local_path = None

    @property
    def default_sparse_kv_cache_format(self):
        """Scaled-FP8 kvpe, as GLM-5.3: the DSA layers read it with sparse_sdpa, which cannot read bf8."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

        return MlaKvCacheFormat.SCALED_FP8

    @property
    def config_builder(self) -> Callable:
        from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import kimi_k3_1_hf_config

        return kimi_k3_1_hf_config

    @property
    def reference_attention_cls(self):
        """None: K3's MLA reference has no indexer, and K3.1 is scored against vLLM traces, not a CPU reference."""
        return None
