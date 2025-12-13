# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch

MAX_SEQ_LEN = 4 * 1024


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def get_decode_mask(pos_idx, sliding_window, max_seq_len=MAX_SEQ_LEN):
    """Function to create a decoding mask for the attention mechanism."""
    mask = torch.triu(torch.full((1, 1, max_seq_len, max_seq_len), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(torch.full((1, 1, max_seq_len, max_seq_len), -float("inf")), diagonal=-sliding_window)
    mask = mask[:, :, pos_idx : pos_idx + 1, :]
    return mask
