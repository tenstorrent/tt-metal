# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Tests for the (H, W) LRU cache in `Learnable2DInterpPosEmb`.

The interpolation depends only on the output grid shape, not on image
content, so the same (H, W) input must yield byte-identical output. The
cache implementation:
  - on a hit, returns the cached fp32 tensor (no F.interpolate call),
  - on a miss, computes once, inserts into the LRU,
  - evicts the least-recently-used entry past the configured capacity.

Tests use a small synthetic posemb weight so they run host-only without
loading HF. Correctness is checked against the uncached path
(`_interp_one_uncached`) for byte-identical equality.

Run:
    pytest models/demos/deepseek_v3/tests/moonvit/test_pos_emb_cache.py -v
"""
from __future__ import annotations

import torch

from models.demos.deepseek_v3.tt.moonvit.pos_emb import Learnable2DInterpPosEmb


def _make_posemb(h_base: int = 16, w_base: int = 16, d: int = 32, seed: int = 0, cache_size=64):
    """Synthesize a small Learnable2DInterpPosEmb for host-only tests."""
    torch.manual_seed(seed)
    weight = torch.randn(h_base, w_base, d, dtype=torch.float32)
    return Learnable2DInterpPosEmb(weight=weight, cache_size=cache_size)


@torch.no_grad()
def test_cache_hit_returns_identical_to_miss():
    """Cache hit must produce a byte-identical result to the uncached compute."""
    pe = _make_posemb()
    h, w = 8, 12  # not equal to base; must go through F.interpolate path

    # First call: miss, computes and caches.
    out_miss = pe._interp_one(h, w).clone()
    assert pe.cache_misses == 1
    assert pe.cache_hits == 0

    # Second call: hit, must equal the cached value.
    out_hit = pe._interp_one(h, w)
    assert pe.cache_hits == 1
    assert pe.cache_misses == 1
    assert torch.equal(out_miss, out_hit), "cache hit diverged from cached value"

    # And byte-identical to the uncached computation as well.
    out_uncached = pe._interp_one_uncached(h, w)
    assert torch.equal(out_miss, out_uncached), "cached value diverged from uncached compute"


@torch.no_grad()
def test_cache_fast_path_for_base_shape():
    """Matching the base shape uses the fast path; cache still tracks it as hit/miss."""
    pe = _make_posemb(h_base=16, w_base=16)

    out1 = pe._interp_one(16, 16)
    # The base-shape fast path doesn't call F.interpolate but still goes through
    # the cache wrapper, so it counts as a miss the first time.
    assert pe.cache_misses == 1
    assert pe.cache_hits == 0

    out2 = pe._interp_one(16, 16)
    assert pe.cache_hits == 1
    assert torch.equal(out1, out2)


@torch.no_grad()
def test_cache_lru_eviction():
    """Past capacity, the LRU evicts oldest unused entries."""
    pe = _make_posemb(cache_size=3)

    # Insert 3 entries.
    pe._interp_one(4, 4)
    pe._interp_one(5, 5)
    pe._interp_one(6, 6)
    assert len(pe._cache) == 3

    # Touch (4,4) to bump it to most-recent.
    pe._interp_one(4, 4)
    # Insert (7, 7): should evict (5, 5) (the LRU entry).
    pe._interp_one(7, 7)
    assert len(pe._cache) == 3
    assert (5, 5) not in pe._cache, "expected (5, 5) to be evicted"
    assert (4, 4) in pe._cache
    assert (6, 6) in pe._cache
    assert (7, 7) in pe._cache


@torch.no_grad()
def test_prewarm():
    """Prewarm populates the cache without counting as a miss/hit."""
    pe = _make_posemb()
    shapes = [(4, 4), (8, 8), (12, 12), (16, 16)]

    pe.prewarm(shapes)
    assert pe.cache_misses == 0
    assert pe.cache_hits == 0
    assert all(s in pe._cache for s in shapes)

    # Subsequent calls hit.
    pe._interp_one(8, 8)
    pe._interp_one(12, 12)
    assert pe.cache_hits == 2
    assert pe.cache_misses == 0


@torch.no_grad()
def test_compute_uses_cache_across_mixed_grid_hws():
    """`compute(grid_hws)` benefits from cache when the same (H, W) recurs."""
    pe = _make_posemb()
    # Two-image batch where both images share the same grid shape.
    grid_tensor = torch.tensor([[8, 8], [8, 8]], dtype=torch.long)

    out = pe.compute(grid_tensor, dtype=torch.float32)
    # First image is a miss; second image is a hit.
    assert pe.cache_misses == 1
    assert pe.cache_hits == 1
    # Output is the concatenation of both interpolated grids.
    expected_l = 2 * 8 * 8
    assert out.shape == (expected_l, pe.dim)
    # The two halves must be identical because they came from the same (h, w).
    halves = out.view(2, 8 * 8, pe.dim)
    assert torch.equal(halves[0], halves[1])


@torch.no_grad()
def test_cache_disabled():
    """cache_size=None disables caching; counters stay zero, no eviction."""
    pe = _make_posemb(cache_size=None)
    pe._interp_one(4, 4)
    pe._interp_one(4, 4)
    pe._interp_one(5, 5)
    assert pe.cache_hits == 0
    assert pe.cache_misses == 0
    assert len(pe._cache) == 0


@torch.no_grad()
def test_clear_cache():
    pe = _make_posemb()
    pe._interp_one(4, 4)
    pe._interp_one(4, 4)
    assert pe.cache_hits == 1
    assert pe.cache_misses == 1
    assert len(pe._cache) == 1

    pe.clear_cache()
    assert pe.cache_hits == 0
    assert pe.cache_misses == 0
    assert len(pe._cache) == 0

    # Next call should miss again.
    pe._interp_one(4, 4)
    assert pe.cache_misses == 1
