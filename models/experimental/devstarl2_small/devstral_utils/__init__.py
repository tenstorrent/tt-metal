# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Devstral-2 / Ministral3 TT demos (mesh, prefill padding, LM head, FP8 shim)."""

from models.experimental.devstarl2_small.devstral_utils.multimodal_demo_helpers import (
    DEFAULT_MODEL_ID,
    apply_devstral_hf_trust_patches,
    apply_fp8_dequantize_compat,
    cpu_lm_head_logits_last_token,
    demo_lm_head_max_columns_per_device,
    devstral_supports_on_device_sampling,
    eos_token_ids,
    host_input_ids_to_tt_replicated,
    open_devstral_demo_mesh,
    pad_input_ids_and_positions_for_tt_prefill,
    squeeze_tt_hidden_to_bsh,
    text_model_root,
    tt_append_uint32_token,
    tt_forward_prefill_from_device_ids,
    tt_forward_prefill_from_ids,
    tt_lm_head_logits_block,
    tt_lm_head_logits_last_token,
    tt_prefill_hidden_states_from_ids,
    tt_prefill_target_seqlen,
    tt_replicated_ids_to_torch_long,
    tt_sampling_output_token_id,
)

__all__ = (
    "DEFAULT_MODEL_ID",
    "apply_devstral_hf_trust_patches",
    "apply_fp8_dequantize_compat",
    "cpu_lm_head_logits_last_token",
    "demo_lm_head_max_columns_per_device",
    "devstral_supports_on_device_sampling",
    "eos_token_ids",
    "host_input_ids_to_tt_replicated",
    "open_devstral_demo_mesh",
    "pad_input_ids_and_positions_for_tt_prefill",
    "squeeze_tt_hidden_to_bsh",
    "text_model_root",
    "tt_append_uint32_token",
    "tt_forward_prefill_from_device_ids",
    "tt_forward_prefill_from_ids",
    "tt_lm_head_logits_block",
    "tt_lm_head_logits_last_token",
    "tt_prefill_hidden_states_from_ids",
    "tt_prefill_target_seqlen",
    "tt_replicated_ids_to_torch_long",
    "tt_sampling_output_token_id",
)
