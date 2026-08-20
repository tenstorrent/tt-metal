# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


def decode_step_state(start_pos, iteration, prompt_length, max_seq_len):
    """Return the next position/input and number of cache entries written so far."""
    next_position = start_pos + iteration + 1
    next_token_index = iteration + 1
    if next_token_index >= prompt_length:
        next_token_index = None
    num_written = min(iteration + 1, max_seq_len - start_pos)
    return next_position, next_token_index, num_written
