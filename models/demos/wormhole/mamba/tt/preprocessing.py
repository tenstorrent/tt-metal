# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from typing import Optional, Tuple


def split_sequence_length(x: ttnn.Tensor, batch: int = 0, chunk_size: int = 32):
    """
    Generator function to yield chunks of a tensor of shape (1, 1, B, L) into (1, 1, 1, chunk_size).

    Parameters:
    tensor (ttnn.Tensor): The input tensor of shape (1, 1, B, L).
    batch (int): The batch dimension to select. Default is 0.
    chunk_size (int): The size of each chunk along the third dimension. Default is 32.

    Yields:
    ttnn.Tensor: Chunks of the input tensor of shape (1, 1, 1, chunk_size).
    """

    assert x.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected input to be row-major layout (was {x.layout})"
    assert len(x.shape) == 4, f"Expected input to be rank 4 (was {x.shape})"
    assert x.shape[3] % 32 == 0, "Sequence length size must be multiple of 32"

    assert chunk_size % 32 == 0, "Chunk size must be multiple of 32"

    _, _, B, L = x.shape

    assert batch < B, f"Expected batch index (was {batch}) to be less than the size of batch dimension (was {B})"

    for i in range(0, L, chunk_size):
        slice_start = (0, 0, batch, i)
        slice_end = (1, 1, batch + 1, i + chunk_size)
        yield ttnn.slice(x, slice_start, slice_end)


def select_chunk_size(sequence_length: int, max_chunk_size: int):
    for chunk_size in range(max_chunk_size, 0, -32):
        if sequence_length > chunk_size:
            return chunk_size
    return 0  # If no valid chunk size is found


def split_input_into_prefill_and_decode_segments(
    x, chunk_size: int = 32
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    assert len(x.shape) == 2, f"Expected input sequence to be rank 2 (shape was {x.shape})"
    _, sequence_length = x.shape
    if sequence_length <= chunk_size:
        return (None, x)
    else:
        num_chunks_for_prefill = sequence_length // chunk_size
        if sequence_length % chunk_size == 0:
            num_chunks_for_prefill -= 1
        segments = x.split(num_chunks_for_prefill * chunk_size, dim=-1)
        assert len(segments) == 2, f"Expected input to be split into two segments (was {len(segments)}"
        return (segments[0], segments[1])
