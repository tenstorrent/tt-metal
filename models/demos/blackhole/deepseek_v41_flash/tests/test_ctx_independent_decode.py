# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests: the decode cost of a step follows the POSITIONS, not the built context (no device).

A 1M-context build decoded a 64k prompt at 64.8 ms/token against 47.0 ms on a 65k build (B=4, plain): the fused indexer backend scores the whole key slab at every step (524,288 entries per index
layer at 1M), and the position buckets (which bound the scored length) existed only up to 32768 and only for the matmul backend. These tests pin the two rules that fix it: the backend of a long
build is the bucketed matmul one, and the bucket ladder reaches the built context, so the scored key length stays within 2x of the positions at every context size.
"""

import pytest

from models.demos.blackhole.deepseek_v41_flash.tt import indexer as ix

RATIOS = (1, 2)  # compress ratios of the indexer layers (20 layers of ratio 1, 18 of ratio 2)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DSV41_POS_BUCKETS", "DSV41_IDX_BACKEND", "DSV41_IDX_BUCKETED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DSV41_IDX_SWITCH", "262144")  # the vLLM adapter default


class _Stub(ix.DSV41DecodeIndexer):
    def __init__(self, ratio, n_alloc, backend="matmul", topk=512):
        self.ratio, self.n_alloc, self.backend, self.topk, self.nb = ratio, n_alloc, backend, topk, None

    def _arange_for(self, n):
        pass


def _n_alloc(ctx, ratio):
    return -(-(ctx // ratio + 32) // 32) * 32


def _scored(max_ctx, pos, ratio):
    """key entries one index layer scores at decode position ``pos`` (+ the step's own row) on a build of ``max_ctx`` tokens"""
    n_alloc = _n_alloc(max_ctx, ratio)
    backend = ix.default_backend(max_ctx // ratio)
    buckets = ix.pos_buckets(max_ctx + 64)
    P = next((b for b in buckets if pos + 1 <= b), None)
    return _Stub(ratio, n_alloc, backend).set_pos_bucket(P)


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("max_ctx", [65536, 131072, 262144, 524288, 1048576])
def test_long_builds_use_the_bucketed_matmul_backend(max_ctx, ratio):
    assert ix.default_backend(max_ctx // ratio) == "matmul"


def test_backend_overrides(monkeypatch):
    monkeypatch.setenv("DSV41_IDX_BUCKETED", "0")
    assert ix.default_backend(524288) == "fused" and ix.default_backend(1048576) == "fused"  # the old switch-only rule
    assert ix.default_backend(131072) == "matmul"
    monkeypatch.setenv("DSV41_IDX_BACKEND", "fused")
    assert ix.default_backend(1000) == "fused"
    monkeypatch.delenv("DSV41_IDX_BUCKETED")
    assert ix.default_backend(ix.BUCKETED_MATMUL_MAX_ENTRIES + 1) == "fused"  # past the 1M build: no bucketed matmul


@pytest.mark.parametrize("max_ctx", [65536, 131072, 262144, 524288, 1048576])
def test_bucket_ladder_reaches_the_built_context(max_ctx):
    b = ix.pos_buckets(max_ctx + 64)
    assert b == sorted(b) and b[:4] == [512, 2048, 8192, 32768][: len(b[:4])]
    assert 2 * b[-1] >= max_ctx  # nothing above the last bucket is more than 2x its predecessor
    ladder = [x for x in b if x > 32768]
    assert all(y == 2 * x for x, y in zip(ladder, ladder[1:]))


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("pos", [33000, 40000, 65000, 70000, 100000, 131000, 200000, 262000, 400000, 524000])
def test_scored_length_follows_the_positions_not_the_build(pos, ratio):
    """at any position the 1M build scores at most 2x (+1 tile group) what a build just big enough for it scores"""
    small_ctx = 1 << max(16, (pos + 128).bit_length())  # the smallest power-of-two build holding the position
    big, small = _scored(1048576, pos, ratio), _scored(small_ctx, pos, ratio)
    assert big <= 2 * small + 64, (pos, big, small)
    assert big <= 2 * (pos // ratio) + 128, (pos, big)  # and within 2x of the entries the position can hold


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("pos", [600, 5000, 20000, 32000])
def test_short_positions_identical_on_every_build(pos, ratio):
    assert _scored(65536, pos, ratio) == _scored(262144, pos, ratio) == _scored(1048576, pos, ratio)


def test_factor_4_ladder_and_trace_budget():
    """the speculative runner captures 3 traces per bucket: its default ladder (factor 4) keeps the bucket count at 1M within what fits the 650 MB production trace region (5 sets, the previous default +2;
    the factor-2 ladder, 8 sets, overflowed it by 9.5 MB)"""
    b = ix.pos_buckets(1048576 + 64, 4)
    assert b == [512, 2048, 8192, 32768, 131072, 524288]
    assert len(b) + 1 <= 7  # + the full-slab set
    assert len(ix.pos_buckets(1048576 + 64, 2)) + 1 == 9
    assert ix.pos_buckets(65600, 4) == [512, 2048, 8192, 32768]  # a 65k build: unchanged from before the ladder


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("pos", [33000, 70000, 140000, 300000, 520000])
def test_spec_ladder_scored_length_within_4x(pos, ratio):
    n_alloc = _n_alloc(1048576, ratio)
    buckets = ix.pos_buckets(1048576 + 64, 4)
    P = next((b for b in buckets if pos + 1 <= b), None)
    n = _Stub(ratio, n_alloc).set_pos_bucket(P)
    assert n <= 4 * (pos // ratio) + 128
