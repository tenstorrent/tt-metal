# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free: the standalone generator's KV cache is always a whole number of SDPA K chunks."""
from __future__ import annotations

import pytest

from models.demos.laguna.tt.generator import BLOCK_SIZE, KV_SEQ_ALIGN, kv_cache_seq_len


def test_alignment_covers_block_and_every_k_chunk():
    for k_chunk in (BLOCK_SIZE, 64, 128):
        assert KV_SEQ_ALIGN % k_chunk == 0


@pytest.mark.parametrize("needed", [1, 31, 64, 129, 336, 4097, 131071])
def test_cache_len_is_aligned_and_sufficient(needed):
    got = kv_cache_seq_len(needed, 131072)
    assert got % KV_SEQ_ALIGN == 0
    assert needed <= got < needed + KV_SEQ_ALIGN


def test_s_regression_request_no_longer_ends_mid_chunk():
    # Laguna-S AIME24 teacher forcing: 235 prompt + 100 generated + 1 = 336 positions. The old
    # BLOCK_SIZE-only rounding gave 352 slots (5.5 chunks of 64) and garbage from position 320.
    assert kv_cache_seq_len(336, 131072) == 384


def test_capped_at_max_and_rejects_unaligned_max():
    assert kv_cache_seq_len(200000, 131072) == 131072
    with pytest.raises(ValueError, match="multiple of 128"):
        kv_cache_seq_len(10, 131072 - 32)
