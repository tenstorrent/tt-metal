# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mistral Small 4 119B prefill adapter.

Dense MLA + MoE, so it subclasses ``MLAPrefillAdapter`` and inherits the whole serving path
(``build_runtime`` / ``allocate_kv_cache`` / ``weight_cache_path``). Mistral diverges from the
DeepSeek/Kimi residents in three ways that the adapter has to account for:

  * **Router.** ``n_group = topk_group = 1``, softmax top-4, **no** ``e_score_correction_bias``.
    The grouped-topk kernel only implements ``sigmoid`` / ``sqrtsoftplus`` affinities
    (``moe_grouped_topk.cpp:17-23`` TT_THROWs otherwise), so the sigmoid device gate is not usable.
    Mistral's rule — softmax over all experts, take top-4, renormalize (``norm_topk_prob: true``) —
    is mathematically identical to GPT-OSS routing (top-k on raw logits, then softmax over just the
    selected k), because softmax is monotonic so the selected indices agree and the renormalization
    IS the softmax over the selected subset. Hence ``default_gate_mode = "GPT_DEVICE"``.
  * **Config.** ``model_type "mistral4"`` sits under a ``Mistral3ForConditionalGeneration``
    multimodal wrapper, and transformers 5.x renamed the rope block to ``rope_parameters``. The
    config is hand-built via ``mistral4_hf_config``.
  * **Weights.** The checkpoint stores experts as ONE stacked fp8 tensor per projection with gate
    and up fused (``mlp.experts.gate_up_proj`` ``[128, 4096, 4096]``), not per-expert
    ``mlp.experts.{i}.*``; and fp8 is PER-TENSOR (rank-0 ``*_scale_inv``), not [128,128] block.
    Neither is handled by the shared loader yet — see ``supports_pretrained`` below.

Single expert group with a device gate, like Kimi and GLM, so the MoE routing all-gather's
semaphores go to L1_SMALL (needs the L1_SMALL carve-out at mesh-open).
"""

from __future__ import annotations

import os
from typing import Callable

from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config, mistral4_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class MistralSmall4Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "mistral_small4"
    model_config = MistralSmall4Config
    # Config is hand-built (see load_hf_config); this stays empty so an accidental AutoConfig read
    # yields an obvious empty path.
    hf_model_default = ""
    # No shared /mnt cache exists for this model yet. Left empty on purpose: MLAPrefillAdapter's
    # weight_cache_path returns None for an empty value rather than inventing a path under a
    # read-only mount. Set PREFILL_TTNN_CACHE (serving) / TT_MISTRAL4_PREFILL_TTNN_CACHE (tests)
    # to somewhere writable in $HOME.
    ttnn_cache_default = ""
    # GPT-OSS-equivalent routing — see the module docstring. NOT DEVICE_FP32: that would apply a
    # sigmoid router affinity and silently produce wrong weights.
    default_gate_mode = "GPT_DEVICE"
    # No golden trace captured yet.
    prefill_trace_default = ""

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL.
    l1_small_size = 768
    routing_use_l1_small_for_semaphores = True

    def load_hf_config(self):
        """Return the hand-built HF-attribute config. The runner overwrites ``max_seq_len`` after;
        seed the builder with it so the rope config is consistent."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        return mistral4_hf_config(max_seq=max_seq)

    # --- test metadata (HF download coordinates + PCC thresholds) ---
    hf_repo_id = "mistralai/Mistral-Small-4-119B-2603"
    env_var = "MISTRAL4_HF_MODEL"
    # No in-tree or shared checkpoint location yet, so resolution goes through MISTRAL4_HF_MODEL
    # (deliberately not a personal path — this branch is shared).
    default_local_path = None
    shared_path = None
    num_layers_to_download = 36
    ref_cache_env = "TT_MISTRAL4_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "MISTRAL4_MLA_REF_CACHE"
    ttnn_cache_env = "TT_MISTRAL4_PREFILL_TTNN_CACHE"
    # ⚠ False until the two weight-format gaps above are closed. The pretrained fixtures would
    # otherwise fail deep inside the dequantizer with a shape error; `supports_pretrained = False`
    # makes run_model() skip with a clear message instead (test_mla.py:189-190). Flip this in the
    # same change that lands the Mistral state-dict converter.
    supports_pretrained = False
    # Provisional, matching the other MLA+MoE residents. Re-baseline once real PCC is measured.
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971

    @property
    def config_builder(self) -> Callable:
        """The reference/config-only test path resolves Mistral's config through this
        (conftest ``_resolve_config_only``); serving goes through ``load_hf_config``. Both hand off
        to ``mistral4_hf_config``."""
        return mistral4_hf_config
