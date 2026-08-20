# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mistral-Small-4-119B prefill adapter.

Dense MLA + MoE, the same family as DeepSeek-V3 / Kimi, so it subclasses ``MLAPrefillAdapter`` and
inherits the serving path. It diverges in one way the adapter accounts for: its HF config speaks
transformers 5.x ``rope_parameters`` (with ``rope_theta`` nested inside), while ttMLA reads a
top-level ``rope_theta`` plus a DeepSeek-shaped ``rope_scaling``. ``AutoConfig`` can load
``model_type: mistral4``, it just returns the wrong shape, so the config is hand-built via
``mistral4_hf_config`` — which also reconciles the softmax-scale convention. See that module's
docstring; the mscale detail is what makes a green PCC test misleading rather than failing.

Like Kimi it has a single expert group (``n_group = 1``) with a device gate, so the MoE routing
all-gather's semaphores go to L1_SMALL.
"""

from __future__ import annotations

import os
from typing import Callable

from models.demos.deepseek_v3_d_p.reference.mistral_small_4_119b_config import (
    Mistral4Small119BConfig,
    mistral4_hf_config,
)
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class MistralSmall4119BAdapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "mistral_small_4_119b"
    model_config = Mistral4Small119BConfig
    # Config is hand-built (see load_hf_config); this stays empty so an accidental AutoConfig read
    # yields an obvious empty path. Point PREFILL_HF_MODEL / MISTRAL4_HF_MODEL at the checkpoint.
    hf_model_default = ""
    # No shared prefill TTNN cache exists for Mistral yet. Empty => weight_cache_path returns None.
    # Set PREFILL_TTNN_CACHE (serving) or TT_MISTRAL4_PREFILL_TTNN_CACHE (tests) to a writable dir.
    ttnn_cache_default = ""
    default_gate_mode = "DEVICE_FP32"  # single expert group (n_group = 1)
    # No golden trace recorded yet. Set explicitly (the base declares it annotation-only) so
    # `.prefill_trace_default` is always readable; generate one and pass it via PREFILL_TRACE_DIR.
    prefill_trace_default = None

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
        off to ``mistral4_hf_config``, so device and host agree."""
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
    # test_utils.is_per_tensor_fp8. Only the MLA weights are exercised end-to-end so far.
    supports_pretrained = True
    # AutoConfig can load this checkpoint, but it is the wrong source: rope_scaling carries
    # mscale = 1.0 with factor = 128, which both the device path and the reference turn into a 2.2x
    # softmax correction Mistral does not apply (and applying it to both sides keeps PCC green while
    # the model is wrong); and quantization_config lives on the multimodal outer config, so the
    # unwrap to text_config drops it and the loader would refuse the fp8 tensors.
    config_builder_overrides_checkpoint = True
    # Routed experts are stacked: `mlp.experts.gate_up_proj` is one [128, 4096, 4096] fp8 tensor, not
    # per-expert gate_proj/up_proj. The pretrained fixture therefore loads attention only.
    packed_expert_checkpoint = True
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    prefill_trace_layout = "single_file"

    # No reference_* classes wired. transformers' Mistral4Attention is the correct upstream reference,
    # but its forward takes a precomputed `position_embeddings` and no `past_key_value`/`use_cache`,
    # so run_reference_mla cannot call it as-is; and the DeepSeek MoE reference expects a
    # `gate.e_score_correction_bias` that Mistral's plain softmax router does not have. Leaving these
    # None makes those comparisons skip rather than error; the device path is still checked against
    # MLAReference in test_mla.py.
