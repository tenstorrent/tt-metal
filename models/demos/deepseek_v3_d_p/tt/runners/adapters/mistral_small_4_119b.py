# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Mistral-Small-4-119B prefill adapter.

Dense MLA + MoE, the same family as DeepSeek-V3 / Kimi, so it subclasses ``MLAPrefillAdapter`` and
inherits the serving path (``build_runtime`` / ``allocate_kv_cache`` / ``weight_cache_path``).
Mistral diverges in one way the adapter has to account for:

  * Its HF config speaks transformers 5.x ``rope_parameters`` (with ``rope_theta`` nested inside),
    while ttMLA reads a top-level ``rope_theta`` plus a DeepSeek-shaped ``rope_scaling``. ``AutoConfig``
    CAN load ``model_type: mistral4`` (transformers 5.12 ships ``Mistral4Config`` natively) — it just
    returns the wrong SHAPE. So the config is hand-built via ``mistral4_hf_config``, which also
    reconciles the softmax-scale convention. See that module's docstring; the mscale detail is the
    one that makes a green PCC test misleading rather than failing.

Like Kimi it has a single expert group (``n_group = 1``) with a device gate, so the MoE routing
all-gather's semaphores go to L1_SMALL (needs the L1_SMALL carve-out at mesh-open).
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
    # Set PREFILL_TTNN_CACHE (serving) or TT_MISTRAL4_PREFILL_TTNN_CACHE (tests) to a WRITABLE dir of
    # your own; the shared /mnt/models tree is read-only and a cache write there dies with errno=13.
    ttnn_cache_default = ""
    default_gate_mode = "DEVICE_FP32"  # Mistral: single expert group (n_group = 1)
    # No golden trace recorded for Mistral yet. Set explicitly (the base declares it annotation-only)
    # so `.prefill_trace_default` is always readable; generate one with generate_prompt_trace.py and
    # pass it via PREFILL_TRACE_DIR.
    prefill_trace_default = None

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL.
    # Routing consumes 512 B; leave 256 B for MLA high-bandwidth-gather semaphores.
    l1_small_size = 768
    routing_use_l1_small_for_semaphores = True

    def load_hf_config(self):
        """Return the hand-built HF-attribute config. ``AutoConfig`` would load ``mistral4`` fine but
        hand back ``rope_parameters`` instead of the ``rope_scaling`` / top-level ``rope_theta`` that
        ttMLA subscripts. The runner overwrites ``max_seq_len`` after; seed the builder with it so the
        rope tables are consistent."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        return mistral4_hf_config(max_seq=max_seq)

    @property
    def config_builder(self) -> Callable:
        """Tests resolve the config through this (conftest ``_resolve_config_only``); serving goes
        through ``load_hf_config``. Both hand off to ``mistral4_hf_config``, so device and host agree."""
        return mistral4_hf_config

    # --- test metadata (HF download coordinates + PCC thresholds) ---
    hf_repo_id = "mistralai/Mistral-Small-4-119B-2603"
    env_var = "MISTRAL4_HF_MODEL"
    ref_cache_env = "TT_MISTRAL4_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "MISTRAL4_MLA_REF_CACHE"
    ttnn_cache_env = "TT_MISTRAL4_PREFILL_TTNN_CACHE"
    # Stock fast tokenizer + a natively-registered model_type, so neither the trust_remote_code import
    # path nor the flat-config copy is needed.
    tokenizer_trust_remote_code = False
    needs_flat_config_dir = False
    # The checkpoint is fp8 with a per-tensor static scheme (weight_block_size null, a scalar
    # weight_scale_inv) rather than the [128,128] block scheme, so it dequantizes through
    # test_utils.is_per_tensor_fp8 / _dequantize_per_tensor_fp8_state_dict. That path also covers the
    # stacked experts (gate_up_proj [n_experts, ...] with a [n_experts,1,1] scale) by broadcasting,
    # but only the MLA weights are exercised end-to-end so far -- the MoE side still needs the packed
    # gate_up_proj split into gate/up before the FFN reference can consume it.
    supports_pretrained = True
    # AutoConfig CAN load this checkpoint (mistral4 is registered), but two things make it the wrong
    # source for the TT stack, so the pretrained path is routed through mistral4_hf_config as well:
    #   1. rope_scaling carries mscale=1.0 / mscale_all_dim=1.0 with factor=128. ttMLA and the vendored
    #      DeepSeek reference both turn those into a YaRN softmax correction (2.206x here) that
    #      Mistral4Attention does not apply -- and they apply it to BOTH sides, so PCC stays green while
    #      the model is wrong. mistral4_hf_config zeroes them; see its docstring.
    #   2. quantization_config lives on the multimodal OUTER config and is dropped by the unwrap to
    #      text_config, so the loader would see an unquantized config and refuse the fp8 tensors.
    # (rope_theta is likewise only reachable nested inside rope_parameters, not as a top-level field.)
    config_builder_overrides_checkpoint = True
    # Routed experts are stacked: `mlp.experts.gate_up_proj` is one [128, 4096, 4096] fp8 tensor with a
    # [128,1,1] scale, not per-expert gate_proj/up_proj. The pretrained fixture therefore loads
    # attention only, which is what the MLA tests need; the MoE tests wait on a gate/up split.
    packed_expert_checkpoint = True
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    # generate_prompt_trace.py writes one safetensors per layer.
    prefill_trace_layout = "single_file"

    # No reference_* classes wired. transformers' Mistral4Attention is the correct upstream reference,
    # but its forward takes a precomputed `position_embeddings` tuple and no `past_key_value`/`use_cache`,
    # so reference_runners.run_reference_mla cannot call it as-is; and the DeepSeek MoE reference expects
    # a `gate.e_score_correction_bias` that Mistral4TopkRouter (plain softmax router) does not have.
    # Leaving these None makes those comparisons skip rather than error. The device path is still checked
    # against MLAReference in test_mla.py; wiring a Mistral-shaped shim is the natural follow-up.
