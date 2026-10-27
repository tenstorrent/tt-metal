# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.
"""

PREFILL_SP_FACTOR = 8  # Production configuration on a single galaxy stage
PREFILL_CHUNK_TOKENS = 5 * 1024  # Production configuration across prefill models


def prefill_chunk_tokens_per_chip(sp_factor: int = PREFILL_SP_FACTOR, chunk_tokens: int = PREFILL_CHUNK_TOKENS) -> int:
    """Rows each chip holds when a ``chunk_tokens`` chunk is split over ``sp_factor`` SP chips."""
    assert chunk_tokens % sp_factor == 0, f"chunk {chunk_tokens} does not split evenly over SP={sp_factor}"
    return chunk_tokens // sp_factor


PREFILL_CHUNK_TOKENS_PER_CHIP = prefill_chunk_tokens_per_chip()  # 640: the 8x4 stage (SP=8)
# 1280: a 4x4 stage (SP=4) at the same chunk. At SP=4, PREFILL_CHUNK_TOKENS_PER_CHIP is instead the
# per-chip share of a 2560-token chunk, the other chunk size the 4x4 plan considers.
SP4_CHUNK_TOKENS_PER_CHIP = prefill_chunk_tokens_per_chip(4)
