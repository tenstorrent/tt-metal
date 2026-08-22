# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mistral-Small-4-119B prefill adapter.

Dense MLA + MoE, the same family as DeepSeek-V3 / Kimi, so it subclasses ``MLAPrefillAdapter`` and
inherits the serving path. Its config is hand-built by ``mistral4_hf_config`` rather than read via
``AutoConfig``: the checkpoint's config uses transformers 5.x field names and does not express the
softmax-scale convention Mistral actually uses (see ``reference/mistral_small_4_config.py``).

Like Kimi it has a single expert group (``n_group = 1``) with a device gate, so the MoE routing
all-gather's semaphores go to L1_SMALL.
"""

from __future__ import annotations

import os
from typing import Callable

from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config, mistral4_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class MistralSmall4Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "mistral_small_4"
    model_config = MistralSmall4Config
    # Config is hand-built (see load_hf_config); point PREFILL_HF_MODEL / MISTRAL4_HF_MODEL at the
    # checkpoint for weights.
    hf_model_default = ""
    # No shared prefill TTNN cache exists for Mistral yet; set PREFILL_TTNN_CACHE (serving) or
    # TT_MISTRAL4_PREFILL_TTNN_CACHE (tests) to a writable dir.
    ttnn_cache_default = ""
    default_gate_mode = "DEVICE_FP32"  # single expert group (n_group = 1)
    prefill_trace_default = None  # no golden trace recorded yet; pass one via PREFILL_TRACE_DIR

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL. Routing
    # consumes 512 B; leave 256 B for MLA high-bandwidth-gather semaphores.
    l1_small_size = 768
    routing_use_l1_small_for_semaphores = True

    def load_hf_config(self):
        """Return the hand-built HF-attribute config. The runner overwrites ``max_seq_len`` after;
        seed the builder with it so the rope tables are consistent."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        return mistral4_hf_config(max_seq=max_seq)

    @property
    def config_builder(self) -> Callable:
        """Tests resolve the config through this; serving goes through ``load_hf_config``. Both hand
        off to ``mistral4_hf_config``."""
        return mistral4_hf_config

    # --- test metadata (HF download coordinates + PCC thresholds) ---
    hf_repo_id = "mistralai/Mistral-Small-4-119B-2603"
    env_var = "MISTRAL4_HF_MODEL"
    ref_cache_env = "TT_MISTRAL4_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "MISTRAL4_MLA_REF_CACHE"
    ttnn_cache_env = "TT_MISTRAL4_PREFILL_TTNN_CACHE"
    # Stock fast tokenizer + a natively-registered model_type, so neither trust_remote_code nor the
    # flat-config copy is needed.
    tokenizer_trust_remote_code = False
    needs_flat_config_dir = False
    # The checkpoint is fp8 with a per-tensor static scheme (weight_block_size null, a scalar
    # weight_scale_inv) rather than the [128,128] block scheme, so it dequantizes through
    # test_utils.is_per_tensor_fp8.
    supports_pretrained = True
    # AutoConfig can load this checkpoint but is the wrong source: quantization_config lives on the
    # outer (multimodal) config, so the unwrap to text_config drops it and the loader would refuse
    # the fp8 tensors. mistral4_hf_config also fixes the softmax-scale convention (see that module).
    config_builder_overrides_checkpoint = True
    # Routed experts are stacked: `mlp.experts.gate_up_proj` is one [128, 4096, 4096] fp8 tensor, not
    # per-expert gate_proj/up_proj. The pretrained fixture therefore loads attention only.
    packed_expert_checkpoint = True
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    prefill_trace_layout = "single_file"

    # --- CPU reference ---------------------------------------------------------------------------
    # No reference_* classes are wired: transformers' Mistral4Attention needs the precomputed
    # `position_embeddings` that run_reference_mla does not pass, and Mistral's plain-softmax router
    # with stacked experts does not match the state dict run_reference_moe expects. Leaving these
    # None makes those comparisons skip rather than error; MLA is still checked against MLAReference.
