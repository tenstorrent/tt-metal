# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: dense_cache_read_ok picks the dense SP cache-read path only for caches ring_joint can read
(whole Q-sized slabs per device, at least two, bf8 to match dense_sp_attention's gather buffers)."""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.minimax_m3.tt.attention.dense_sp import dense_cache_read_ok

SP, CHUNK = 8, 5120
SEQ_LEN = CHUNK // SP  # per-device Q rows


def _cache(max_seq_len, dtype=ttnn.bfloat8_b):
    return SimpleNamespace(max_seq_len=max_seq_len, k=SimpleNamespace(dtype=dtype))


@pytest.mark.parametrize(
    "max_seq_len,dtype,expected",
    [
        (2 * CHUNK, ttnn.bfloat8_b, True),
        (10 * CHUNK, ttnn.bfloat8_b, True),
        (CHUNK, ttnn.bfloat8_b, False),  # capacity == chunk: Q.seq == K.seq per device
        (CHUNK + CHUNK // 2, ttnn.bfloat8_b, False),  # not a whole number of Q slabs
        (2 * CHUNK, ttnn.bfloat16, False),  # gather buffers are bf8
    ],
)
def test_dense_cache_read_ok(max_seq_len, dtype, expected):
    assert dense_cache_read_ok(_cache(max_seq_len, dtype), SEQ_LEN, SP) is expected


def test_dense_cache_read_ok_without_cache():
    assert dense_cache_read_ok(None, SEQ_LEN, SP) is False
