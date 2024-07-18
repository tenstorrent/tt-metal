# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def split_sequence_length(x, batch: int = 0, chunk_size: int = 32):
    """
    Generator function to yield chunks of a tensor of shape (1, 1, B, L) into (1, 1, 1, chunk_size).

    Parameters:
    tensor (torch.Tensor): The input tensor of shape (1, 1, B, L).
    batch (int): The batch dimension to select. Default is 0.
    chunk_size (int): The size of each chunk along the third dimension. Default is 32.

    Yields:
    torch.Tensor: Chunks of the input tensor of shape (1, 1, 1, chunk_size).
    """

    assert x.layout == ttnn.ROW_MAJOR_LAYOUT, f"Expected input to be row-major layout (was {x.layout})"
    assert len(x.shape) == 4, f"Expected input to be rank 4 (was {x.shape})"
    assert x.shape[3] % 32 == 0, "Sequence length size must be multiple of 32"

    assert chunk_size % 32 == 0, "Chunk size must be multiple of 32"

    _, _, B, L = x.shape

    assert batch < B, f"Expected batch index (was {batch}) to be less than the size of batch dimension (was {B})"

    for i in range(0, L, chunk_size):
        slice_start = (0, 0, batch, i)
        slice_end = (0, 0, batch, i + chunk_size - 1)
        yield ttnn.slice(x, ttnn.Shape(slice_start), ttnn.Shape(slice_end))
