# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for the low-priority eviction of pure stride snapshots in GdnPrefixStateCache and for the spec plan
that marks them (QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO).

Run: pytest models/demos/blackhole/qwen36/tests/test_prefix_cache_priority_host.py -q
"""
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_prefix_fast_host import _cache, fake_ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

__all__ = ["fake_ttnn"]  # fixture re-export


@pytest.fixture(autouse=True)
def _low_prio_on(monkeypatch):
    monkeypatch.delenv("QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO", raising=False)


def _toks(n=4096):
    return torch.arange(n, dtype=torch.int32) % 991


def _key(c, t, pos):
    return (pos, c._digest(t.numpy(), pos))


def _fill(c, t, positions, low=()):
    for p in positions:
        c.save(t, p, low_prio=p in low)


def test_low_keys_are_evicted_before_older_normal_keys(fake_ttnn):
    c, t = _cache(3), _toks()
    _fill(c, t, [128, 256, 384], low={256, 384})
    c.save(t, 512)  # evicts the LRU low key (256), not the older normal 128
    assert [k[0] for k in c._lru] == [128, 384, 512]
    c.save(t, 640)
    assert [k[0] for k in c._lru] == [128, 512, 640]
    assert c.stats["evictions"] == 2 and c.stats["evictions_low"] == 2
    c.save(t, 768)  # no low key left: plain LRU
    assert [k[0] for k in c._lru] == [512, 640, 768]
    assert c.stats["evictions"] == 3 and c.stats["evictions_low"] == 2
    assert "evictions_low=2" in c.summary() and "low=0" in c.summary()


def test_only_normal_keys_stay_lru(fake_ttnn):
    c, t = _cache(3), _toks()
    _fill(c, t, [128, 256, 384])
    c.lookup(t, 128)  # refresh 128
    c.save(t, 512)
    assert sorted(k[0] for k in c._lru) == [128, 384, 512]
    assert c.stats.get("evictions_low", 0) == 0


def test_lookup_hit_upgrades_a_low_key(fake_ttnn):
    c, t = _cache(3), _toks()
    _fill(c, t, [128, 256, 384], low={128, 256})
    assert c.lookup(t, 256)[0] == 256  # hit: 256 becomes normal and most recent
    assert _key(c, t, 256) not in c._low() and _key(c, t, 128) in c._low()
    c.save(t, 512)
    assert sorted(k[0] for k in c._lru) == [256, 384, 512]  # 128 (still low) went first
    c.save(t, 640)  # no low key left: LRU = 384 (256 was refreshed by the hit)
    assert sorted(k[0] for k in c._lru) == [256, 512, 640]


def test_low_save_never_downgrades_but_normal_save_upgrades(fake_ttnn):
    c, t = _cache(2), _toks()
    c.save(t, 128)
    c.save(t, 128, low_prio=True)  # existing normal key stays normal
    assert not c._low()
    c.save(t, 256, low_prio=True)
    assert len(c._low()) == 1
    c.save(t, 256)  # normal save of an existing low key upgrades it
    assert not c._low()
    c.save(t, 384, low_prio=True)
    c.save(t, 512)  # the low 384 is the victim and leaves the low set
    assert [k[0] for k in c._lru] == [256, 512] and not c._low()


def test_pinned_keys_are_never_evicted(fake_ttnn):
    c, t = _cache(3), _toks()
    _fill(c, t, [128, 256, 384], low={128, 256})
    c._pins().update({0: _key(c, t, 128), 1: _key(c, t, 384)})  # a pinned low key and a pinned normal key
    c.save(t, 512)
    assert sorted(k[0] for k in c._lru) == [128, 384, 512]  # only the unpinned low 256 could go
    c.save(t, 640)  # the unpinned normal 512 is next
    assert sorted(k[0] for k in c._lru) == [128, 384, 640]
    c._pins()[2] = _key(c, t, 640)
    assert c.save(t, 768) is False and c.stats["skipped_saves"] == 1


def test_env_off_is_the_old_lru_order(fake_ttnn, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO", "0")
    c, t = _cache(3), _toks()
    _fill(c, t, [128, 256, 384], low={256, 384})
    c.save(t, 512)
    assert [k[0] for k in c._lru] == [256, 384, 512]  # the oldest (128) went, low marks ignored
    c.save(t, 640)
    assert [k[0] for k in c._lru] == [384, 512, 640]


def _plan(L, start, B=8, **kw):
    m = types.SimpleNamespace(
        _prefix_cache=_cache(4), args=types.SimpleNamespace(max_batch_size=B), _chunked_chunk_size=2048
    )
    return Qwen36Model.plan_spec_prefix(m, torch.arange(L), L, start, **kw)


def test_plan_marks_exactly_the_pure_stride_points():
    s, _, pts = _plan(200000, 0)  # strides + the end-of-prompt point
    assert s == 0 and pts[-1] == 199936 and 199936 not in pts.low
    assert pts.low == set(range(8192, 199998, 8192)) and set(pts) == pts.low | {199936}
    # vLLM hit at 40960: learn points 40832 / 40960 (40960 is also a stride point -> not pure) + end point
    s, _, pts = _plan(50000, 40960)
    assert s == 0 and {40832, 40960, 49920} <= set(pts)
    assert pts.low == {8192 * k for k in range(1, 7)} - {40960}
    # non-final chunk: no end point; forced points are never low (8192 is also a stride point)
    s, _, pts = _plan(30000, 0, force_points=[8192, 12288], nonfinal=True)
    assert list(pts) == [8192, 12288, 16384, 24576] and pts.low == {16384, 24576}
    # short prompt: end point / last full chunk only, nothing low
    assert _plan(5000, 0)[2].low == set()
