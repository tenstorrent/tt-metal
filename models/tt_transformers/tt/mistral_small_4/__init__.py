# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mistral Small 4 (HF ``mistral4``) — ttnn building blocks, developed incrementally.

Some parity tests import **HF Mistral4** (``transformers.models.mistral4``); use current
``transformers`` where needed (see ``tests/mistral_small_4/requirements.txt``).

End-to-end **text** inference (prefill logits, KV steps, greedy) is exposed as
:class:`Mistral4HybridTextBackbone` in :mod:`text_backbone` (wrapping :mod:`model_backbone` /
:mod:`causal_lm_prefill` / :mod:`causal_lm_incremental`).
"""

from models.tt_transformers.tt.mistral_small_4.attention_device import attention_forward_device_sdpa_bf16
from models.tt_transformers.tt.mistral_small_4.attention_full import (
    attention_forward_hybrid_bf16,
    attention_forward_reference_torch,
)
from models.tt_transformers.tt.mistral_small_4.attention_slice import (
    attention_kv_after_kv_b_bottleneck_bf16,
    attention_kv_b_and_k_rot_from_compressed_bf16,
    attention_q_after_q_bottleneck_bf16,
)
from models.tt_transformers.tt.mistral_small_4.decoder_layer import (
    decoder_layer_dense_forward_reference_torch,
    decoder_layer_dense_hybrid_forward_bf16,
    decoder_layer_moe_forward_reference_torch,
    decoder_layer_moe_hybrid_forward_bf16,
)
from models.tt_transformers.tt.mistral_small_4.causal_lm_incremental import (
    causal_lm_greedy_generate_hf_reference_torch,
    causal_lm_greedy_generate_hybrid_bf16,
    causal_lm_incremental_step_logits_hybrid_bf16,
)
from models.tt_transformers.tt.mistral_small_4.causal_lm_prefill import (
    causal_lm_prefill_logits_hf_reference_torch,
    causal_lm_prefill_logits_hybrid_bf16,
    causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16,
    causal_lm_prefill_logits_reference_torch,
)
from models.tt_transformers.tt.mistral_small_4.causal_mask import (
    mistral4_causal_attention_mask,
    mistral4_prefill_causal_attention_mask,
)
from models.tt_transformers.tt.mistral_small_4.dense_mlp import dense_mlp_bf16
from models.tt_transformers.tt.mistral_small_4.embedding import (
    embedding_lookup_bf16,
    embedding_lookup_reference_torch,
)
from models.tt_transformers.tt.mistral_small_4.model_backbone import (
    language_model_backbone_forward_reference_torch,
    language_model_backbone_hybrid_forward_bf16,
    language_model_backbone_hybrid_forward_incremental_bf16,
    language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16,
    language_model_backbone_last_hidden_hf_reference_torch,
)
from models.tt_transformers.tt.mistral_small_4.router import router_logits_bf16
from models.tt_transformers.tt.mistral_small_4.routing import (
    route_tokens_from_probs_torch,
    route_tokens_to_experts_reference_torch,
    router_softmax_then_route_bf16,
)
from models.tt_transformers.tt.mistral_small_4.lm_head import lm_head_logits_bf16, lm_head_logits_reference_torch
from models.tt_transformers.tt.mistral_small_4.moe_naive import mistral4_naive_moe_routed_bf16
from models.tt_transformers.tt.mistral_small_4.linear import (
    linear_bf16_no_bias,
    linear_bf16_no_bias_device,
    linear_bf16_no_bias_reference_torch,
    to_tt_x_bsh_flat,
    tt_flat_to_torch_bsh,
)
from models.tt_transformers.tt.mistral_small_4.rms_norm import rms_norm_bf16
from models.tt_transformers.tt.mistral_small_4.text_backbone import (
    Mistral4HybridTextBackbone,
    load_mistral4_for_causal_lm_bf16,
)

__all__ = [
    "causal_lm_greedy_generate_hf_reference_torch",
    "causal_lm_greedy_generate_hybrid_bf16",
    "causal_lm_incremental_step_logits_hybrid_bf16",
    "causal_lm_prefill_logits_hf_reference_torch",
    "causal_lm_prefill_logits_hybrid_bf16",
    "causal_lm_prefill_logits_hybrid_with_hf_prefill_causal_mask_bf16",
    "causal_lm_prefill_logits_reference_torch",
    "mistral4_causal_attention_mask",
    "mistral4_prefill_causal_attention_mask",
    "attention_forward_device_sdpa_bf16",
    "attention_forward_hybrid_bf16",
    "attention_forward_reference_torch",
    "attention_kv_after_kv_b_bottleneck_bf16",
    "attention_kv_b_and_k_rot_from_compressed_bf16",
    "attention_q_after_q_bottleneck_bf16",
    "decoder_layer_dense_forward_reference_torch",
    "decoder_layer_dense_hybrid_forward_bf16",
    "decoder_layer_moe_forward_reference_torch",
    "decoder_layer_moe_hybrid_forward_bf16",
    "mistral4_naive_moe_routed_bf16",
    "dense_mlp_bf16",
    "embedding_lookup_bf16",
    "embedding_lookup_reference_torch",
    "language_model_backbone_forward_reference_torch",
    "language_model_backbone_hybrid_forward_bf16",
    "language_model_backbone_hybrid_forward_incremental_bf16",
    "language_model_backbone_hybrid_with_hf_prefill_causal_mask_bf16",
    "language_model_backbone_last_hidden_hf_reference_torch",
    "lm_head_logits_bf16",
    "lm_head_logits_reference_torch",
    "route_tokens_from_probs_torch",
    "route_tokens_to_experts_reference_torch",
    "router_logits_bf16",
    "router_softmax_then_route_bf16",
    "linear_bf16_no_bias",
    "linear_bf16_no_bias_device",
    "linear_bf16_no_bias_reference_torch",
    "rms_norm_bf16",
    "to_tt_x_bsh_flat",
    "tt_flat_to_torch_bsh",
    "Mistral4HybridTextBackbone",
    "load_mistral4_for_causal_lm_bf16",
]
