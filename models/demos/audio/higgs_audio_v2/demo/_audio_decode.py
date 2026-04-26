# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def initialize_delay_pattern_state(audio_sequence: torch.Tensor, config) -> tuple[int, int | None]:
    num_delay = config.audio_num_codebooks - int((audio_sequence[:, -1] == config.audio_stream_bos_id).sum().item())
    num_remaining_delays = None
    all_eos_indices = (audio_sequence[:, -1] == config.audio_stream_eos_id).nonzero()
    if torch.numel(all_eos_indices) > 0:
        last_eos_idx = int(all_eos_indices[-1].item())
        num_remaining_delays = config.audio_num_codebooks - last_eos_idx - 1
    return num_delay, num_remaining_delays


def apply_delay_pattern_to_greedy_audio_tokens(
    audio_logits: torch.Tensor,
    config,
    num_delay: int,
    num_remaining_delays: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int | None, bool]:
    next_audio_tokens = torch.argmax(audio_logits, dim=-1)
    return apply_delay_pattern_to_selected_audio_tokens(next_audio_tokens, config, num_delay, num_remaining_delays)


def apply_delay_pattern_to_selected_audio_tokens(
    next_audio_tokens: torch.Tensor,
    config,
    num_delay: int,
    num_remaining_delays: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int | None, bool]:
    next_audio_tokens = next_audio_tokens.clone()
    active_mask = torch.ones_like(next_audio_tokens, dtype=torch.bool)
    finished = False

    if not config.use_delay_pattern:
        return next_audio_tokens, active_mask, num_delay, num_remaining_delays, finished

    if num_delay + 1 < next_audio_tokens.shape[0]:
        active_mask[(num_delay + 1) :] = False
        next_audio_tokens[(num_delay + 1) :] = config.audio_stream_bos_id
        num_delay += 1

    if num_remaining_delays is not None:
        eos_prefix = config.audio_num_codebooks - num_remaining_delays
        active_mask[:eos_prefix] = False
        next_audio_tokens[:eos_prefix] = config.audio_stream_eos_id
        num_remaining_delays -= 1
    else:
        all_eos_indices = (next_audio_tokens == config.audio_stream_eos_id).nonzero()
        if torch.numel(all_eos_indices) > 0:
            last_eos_idx = int(all_eos_indices[-1].item())
            if last_eos_idx > 0:
                active_mask[:last_eos_idx] = False
                next_audio_tokens[:last_eos_idx] = config.audio_stream_eos_id
            num_remaining_delays = config.audio_num_codebooks - last_eos_idx - 1

    if num_remaining_delays is not None and num_remaining_delays <= 0:
        finished = True
        num_delay = 0
        num_remaining_delays = None

    return next_audio_tokens, active_mask, num_delay, num_remaining_delays, finished
