# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stored value of one media item: n_tokens bf16 embeddings of width hidden, with no header.

Written by the media module's encode and read by the prefill runtime, which knows n_tokens from the media ref.
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
