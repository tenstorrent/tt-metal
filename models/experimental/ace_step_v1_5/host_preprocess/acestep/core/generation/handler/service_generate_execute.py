# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Execution helpers for service generation preprocessing."""

from typing import Any, Dict, Tuple


class ServiceGenerateExecuteMixin:
    """Unpack helpers for normalized service-generation requests."""

    def _unpack_service_processed_data(self, processed_data: Tuple[Any, ...]) -> Dict[str, Any]:
        """Convert batch preprocessing tuple into a keyed payload."""
        (
            keys,
            text_inputs,
            src_latents,
            target_latents,
            text_hidden_states,
            text_attention_mask,
            lyric_hidden_states,
            lyric_attention_mask,
            _audio_attention_mask,
            refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask,
            chunk_mask,
            spans,
            is_covers,
            _audio_codes,
            lyric_token_idss,
            precomputed_lm_hints_25Hz,
            non_cover_text_hidden_states,
            non_cover_text_attention_masks,
            repaint_mask,
        ) = processed_data
        return {
            "keys": keys,
            "text_inputs": text_inputs,
            "src_latents": src_latents,
            "target_latents": target_latents,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "lyric_hidden_states": lyric_hidden_states,
            "lyric_attention_mask": lyric_attention_mask,
            "refer_audio_acoustic_hidden_states_packed": refer_audio_acoustic_hidden_states_packed,
            "refer_audio_order_mask": refer_audio_order_mask,
            "chunk_mask": chunk_mask,
            "spans": spans,
            "is_covers": is_covers,
            "lyric_token_idss": lyric_token_idss,
            "precomputed_lm_hints_25Hz": precomputed_lm_hints_25Hz,
            "non_cover_text_hidden_states": non_cover_text_hidden_states,
            "non_cover_text_attention_masks": non_cover_text_attention_masks,
            "repaint_mask": repaint_mask,
        }
