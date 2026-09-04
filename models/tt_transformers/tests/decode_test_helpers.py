# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


def decode_step_state(start_pos, iteration, prompt_length, max_seq_len):
    """Return the next position/input and number of cache entries written so far."""
    current_position = start_pos + iteration
    if current_position >= max_seq_len:
        raise ValueError(
            f"decode position {current_position} exceeds the configured maximum sequence length {max_seq_len}"
        )

    next_position = start_pos + iteration + 1
    next_token_index = iteration + 1
    if next_token_index >= prompt_length:
        next_token_index = None
    num_written = iteration + 1
    return next_position, next_token_index, num_written


def teacher_forced_decode_token(*, reference_token=None, device_token=None):
    """Choose one sampled token for every model participating in a comparison."""
    if reference_token is not None:
        return reference_token
    if device_token is not None:
        return device_token
    raise ValueError("a reference or device token is required to continue decoding")
