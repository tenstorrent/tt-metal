# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stored value of one Qwen3.5 media item: n_tokens bf16 embeddings of width hidden (4096 for the 9B), with no header.

Written by the media module's encode and read by the prefill runner, which knows n_tokens from the media ref. Row i is
the i-th image token's embedding, in the order the <|image_pad|> run appears in the prompt (the HF processor's 2x2
merge-block order).

NOTE (Qwen3.5 vs MiniMax-M3): prefill also needs the image grid, not just n_tokens: M-RoPE gives each image token a
(t, row, col) position and the text after an image resumes at max(grid_h, grid_w) / 2 past it, so a 32 x 16 grid and
a 16 x 32 grid (same n_tokens) need different positions, and decode needs the request's rope delta. The grid has to
reach prefill and decode through the media ref, or come back as a header here.
"""

import numpy as np

_BF16_BYTES = 2


def stored_nbytes(n_tokens: int, hidden: int) -> int:
    return n_tokens * hidden * _BF16_BYTES


def embeddings(buf: memoryview, hidden: int) -> np.ndarray:
    """[n_tokens, hidden] view of the value as bf16 bit patterns (uint16). Row i is the i-th image token's."""
    if not len(buf) or len(buf) % stored_nbytes(1, hidden):
        raise ValueError(f"stored value of {len(buf)} B is not whole bf16 rows of width {hidden}")
    return np.frombuffer(buf, dtype=np.uint16).reshape(-1, hidden)
