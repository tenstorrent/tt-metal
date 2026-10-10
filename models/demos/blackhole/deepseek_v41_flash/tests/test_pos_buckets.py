# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the position buckets of the verify traces: the bucket list (DSV41_POS_BUCKETS) and the scored key length of an indexer per bucket (no device)."""

import pytest

from models.demos.blackhole.deepseek_v41_flash.tt import indexer as ix
from models.demos.blackhole.deepseek_v41_flash.tt import spec_model as sm


def test_pos_bucket_list(monkeypatch):
    monkeypatch.delenv("DSV41_POS_BUCKETS", raising=False)
    default = sm.pos_buckets(131200)
    monkeypatch.setenv("DSV41_POS_BUCKETS", "0")
    assert sm.pos_buckets(131200) == []
    monkeypatch.setenv("DSV41_POS_BUCKETS", "32768,2048,8192,2048,200000")
    assert sm.pos_buckets(131200) == [2048, 8192, 32768]
    assert sm.pos_buckets(4096) == [
        2048
    ]  # buckets at or above the built context are dropped (the full slab covers them)
    assert sm.pos_buckets(2048) == []
    assert default == sorted(default)


class _Stub(ix.DSV41DecodeIndexer):
    def __init__(self, ratio, n_alloc, backend="matmul", topk=512):
        self.ratio, self.n_alloc, self.backend, self.topk, self.nb = ratio, n_alloc, backend, topk, None
        self.made = []

    def _arange_for(self, n):
        self.made.append(n)


@pytest.mark.parametrize("ratio,ctx", [(2, 131072 + 64), (1, 131072 + 64)])
def test_scored_length_covers_the_bucket(ratio, ctx):
    n_alloc = -(-(ctx // ratio + 32) // 32) * 32
    for P in (2048, 8192, 32768):
        s = _Stub(ratio, n_alloc)
        n = s.set_pos_bucket(P)
        assert (
            n % 32 == 0 and n >= P // ratio + 32 and n >= 544
        )  # every entry of a position < P plus the group in progress, top-512 fits
        assert n < n_alloc and s.n_score == n and s.made[-1] == n
    s = _Stub(ratio, n_alloc)
    assert s.set_pos_bucket(None) == n_alloc and s.nb is None
    assert s.set_pos_bucket(10**9) == n_alloc and s.nb is None  # a bucket past the slab = the slab
    f = _Stub(ratio, n_alloc, backend="fused")
    assert f.set_pos_bucket(2048) == n_alloc and f.nb is None  # the fused backend scores the whole slab
