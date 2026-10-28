# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for the GDN prefix snapshot policy knobs of GdnPrefixStateCache:

  * the host-RAM spill tier (QWEN36_PREFIX_HOST_SLOTS): spill on eviction, host hit, restore through the staging slot,
    host LRU, min-gain / min-pos / low-priority filters, one tier per key;
  * value-aware eviction (QWEN36_PREFIX_CACHE_VALUE_EVICT): end-of-prompt and multi-hit snapshots are kept longer;
  * the true resume metrics (record_resume / summary) and the end-of-prompt marks of the spec plan.

Device tensors are faked as cells holding a value; ttnn.copy / from_device / copy_host_to_device_tensor move the value,
so "restored state == saved state" is checked end to end through the cache bookkeeping.

Run: pytest models/demos/blackhole/qwen36/tests/test_prefix_cache_host_tier_host.py -q
"""
import types
from collections import OrderedDict

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.prefix_cache import GdnPrefixStateCache, HostSnap
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

ENV = (
    "QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO",
    "QWEN36_PREFIX_CACHE_VALUE_EVICT",
    "QWEN36_PREFIX_CACHE_PROTECT_HITS",
    "QWEN36_PREFIX_CACHE_PROTECT_FRAC",
    "QWEN36_PREFIX_HOST_SLOTS",
    "QWEN36_PREFIX_HOST_SPILL_MIN_POS",
    "QWEN36_PREFIX_HOST_MIN_GAIN",
    "QWEN36_PREFIX_HOST_SPILL_LOW",
)


class Cell:
    """A fake device tensor: one value (the 'bytes')."""

    def __init__(self, v=None):
        self.v = v

    def __repr__(self):
        return f"Cell({self.v!r})"


@pytest.fixture
def dev(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    rec = {"copy": [], "from_device": [], "h2d": []}

    def _copy(src, dst):
        rec["copy"].append((src, dst))
        dst.v = src.v

    def _from_device(t, *a, **k):
        rec["from_device"].append(t)
        return ("host", t.v)

    def _h2d(h, d, *a, **k):
        rec["h2d"].append((h, d))
        assert h[0] == "host"
        d.v = h[1]

    monkeypatch.setattr(ttnn, "copy", _copy)
    monkeypatch.setattr(ttnn, "from_device", _from_device)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", _h2d)
    return rec


def _cache(num_slots=2, host_slots=0, layers=2):
    """A GdnPrefixStateCache without the device allocation (same recipe as test_spec_prefix_fast_host._cache) whose
    bound state, slots and staging slot are Cells."""
    c = GdnPrefixStateCache.__new__(GdnPrefixStateCache)
    c.model, c.name, c.num_slots, c.debug = None, "test", num_slots, False
    c._dns = [types.SimpleNamespace(rec_state=Cell(), conv_carry=Cell()) for _ in range(layers)]
    c._slots = [[(Cell(), Cell()) for _ in range(layers)] for _ in range(num_slots)]
    c._free = list(range(num_slots))
    c._lru, c._by_pos, c._positions = OrderedDict(), {}, []
    c.stats = dict(lookups=0, hits=0, tokens_skipped=0, saves=0, evictions=0)
    c.host_slots = host_slots
    c._host = OrderedDict()
    c._staging = [(Cell(), Cell()) for _ in range(layers)] if host_slots > 0 else None
    return c


def _toks(n=40000):
    return torch.arange(n, dtype=torch.int32) % 991


def _bind(c, tag):
    """Make the bound GDN state the (distinct per layer) state `tag`."""
    for i, dn in enumerate(c._dns):
        dn.rec_state.v, dn.conv_carry.v = (tag, "rec", i), (tag, "carry", i)


def _bound(c):
    return [(dn.rec_state.v, dn.conv_carry.v) for dn in c._dns]


def _state(tag, layers=2):
    return [((tag, "rec", i), (tag, "carry", i)) for i in range(layers)]


def _save(c, t, pos, **kw):
    _bind(c, f"S{pos}")
    return c.save(t, pos, **kw)


def _resume(c, t, max_pos):
    """lookup + restore into a junk bound state; -> (pos, handle, bound state after the restore)."""
    _bind(c, "junk")
    pos, h = c.lookup(t, max_pos)
    if h is not None:
        c.restore(h)
    return pos, h, _bound(c)


# --------------------------------------------------------------------------- #
# host spill tier
# --------------------------------------------------------------------------- #
def test_host_tier_off_by_default_drops_the_victim(dev):
    c, t = _cache(1), _toks()
    _save(c, t, 4096)
    _save(c, t, 8192)  # evicts 4096: no host tier, gone
    assert dev["from_device"] == [] and c._host == OrderedDict()
    assert c.lookup(t, 4096 + 128) == (0, None)
    assert "host:" not in c.summary()


def test_spill_then_host_hit_restores_the_exact_state(dev):
    c, t = _cache(1, host_slots=4), _toks()
    _save(c, t, 4096)
    _save(c, t, 8192)  # evicts 4096 -> host
    assert len(dev["from_device"]) == 2 * 2  # rec + carry of each GDN layer, read before the slot was overwritten
    assert list(c._host) == [(4096, c._digest(t.numpy(), 4096))]
    assert c.stats["spills"] == 1 and c.stats["evictions"] == 1
    # the device slot now holds 8192: resuming <= 8000 must come from host, with the 4096 state bit for bit
    pos, h, bound = _resume(c, t, 8000)
    assert pos == 4096 and isinstance(h, HostSnap)
    assert bound == _state("S4096")
    assert len(dev["h2d"]) == 4 and all(d in sum(map(list, c._staging), []) for _, d in dev["h2d"])
    assert c.stats["host_hits"] == 1 and c.stats["host_restores"] == 1 and c.stats["hits"] == 1
    # device hit is still the device slot
    pos, h, bound = _resume(c, t, 8192)
    assert pos == 8192 and h == 0 and bound == _state("S8192")
    s = c.summary()
    assert "host: entries=1/4 spills=1" in s and "host_hits=1" in s


def test_host_restore_matches_a_device_restore(dev):
    """The same snapshot restored from the device slot and (after a spill) from host leaves the same bound state."""
    c, t = _cache(1, host_slots=2), _toks()
    _save(c, t, 4096)
    _, _, from_device = _resume(c, t, 4096)
    _save(c, t, 8192)
    _, h, from_host = _resume(c, t, 4096)
    assert isinstance(h, HostSnap) and from_host == from_device == _state("S4096")


def test_host_lru_capacity_drops_the_oldest_from_the_index(dev):
    c, t = _cache(1, host_slots=1), _toks()
    for p in (4096, 8192, 12288):
        _save(c, t, p)  # spills 4096, then 8192 (4096 leaves host)
    assert [k[0] for k in c._host] == [8192] and c.stats["host_evictions"] == 1
    assert 4096 not in c._positions and c._positions == [8192, 12288]
    assert c.lookup(t, 4096 + 1000) == (0, None)


def test_min_gain_prefers_a_close_device_snapshot(dev, monkeypatch):
    c, t = _cache(2, host_slots=4), _toks()
    _save(c, t, 2560)
    _save(c, t, 30000)
    _save(c, t, 2048)  # evicts 2560 (LRU) -> host; device: 30000, 2048
    assert [k[0] for k in c._host] == [2560]
    assert c.lookup(t, 3000)[0] == 2048  # host 2560 is only 512 tokens later: the device snapshot wins
    monkeypatch.setenv("QWEN36_PREFIX_HOST_MIN_GAIN", "256")
    pos, h = c.lookup(t, 3000)
    assert pos == 2560 and isinstance(h, HostSnap)


def test_host_only_candidate_below_min_gain_is_a_miss(dev, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_HOST_SPILL_MIN_POS", "0")
    c, t = _cache(1, host_slots=4), _toks()
    _save(c, t, 512)
    _save(c, t, 30000)
    assert [k[0] for k in c._host] == [512]
    assert c.lookup(t, 1000) == (0, None)  # 512 < 1024 min gain over "from 0"


def test_spill_filters_short_and_low_priority(dev, monkeypatch):
    c, t = _cache(1, host_slots=4), _toks()
    _save(c, t, 1024)  # < 2048: recompute is cheaper than a spill
    _save(c, t, 16384, low_prio=True)  # evicts 1024: not spilled
    _save(c, t, 24576)  # evicts the low 16384: not spilled
    assert c._host == OrderedDict() and dev["from_device"] == []
    assert c._positions == [24576]
    monkeypatch.setenv("QWEN36_PREFIX_HOST_SPILL_LOW", "1")
    _save(c, t, 32768, low_prio=True)  # evicts the normal 24576 -> host
    _save(c, t, 36864)  # evicts the low 32768 -> host (SPILL_LOW=1)
    assert [k[0] for k in c._host] == [24576, 32768]


def test_resave_of_a_host_key_moves_it_back_to_device(dev):
    c, t = _cache(1, host_slots=4), _toks()
    _save(c, t, 4096)
    _save(c, t, 8192)  # 4096 -> host
    _save(c, t, 4096)  # same state recomputed: device again (8192 -> host), never in both tiers
    k4096 = (4096, c._digest(t.numpy(), 4096))
    assert k4096 in c._lru and k4096 not in c._host
    assert [k[0] for k in c._host] == [8192]
    assert c._positions == [4096, 8192]


def test_pinned_and_skipped_saves_never_spill(dev):
    c, t = _cache(1, host_slots=4), _toks()
    _save(c, t, 4096)
    c._pins()[0] = (4096, c._digest(t.numpy(), 4096))
    assert _save(c, t, 8192) is False and c.stats["skipped_saves"] == 1
    assert dev["from_device"] == [] and c._host == OrderedDict()


def test_failed_spill_only_loses_the_snapshot(dev, monkeypatch):
    c, t = _cache(1, host_slots=4), _toks()
    _save(c, t, 4096)

    def _boom(*a, **k):
        raise RuntimeError("read failed")

    monkeypatch.setattr(ttnn, "from_device", _boom)
    assert _save(c, t, 8192) is True and c.stats["spill_errors"] == 1
    assert c._host == OrderedDict() and c._positions == [8192]
    assert _resume(c, t, 8192)[2] == _state("S8192")


# --------------------------------------------------------------------------- #
# value-aware eviction
# --------------------------------------------------------------------------- #
def test_value_evict_off_ignores_protect(dev):
    c, t = _cache(2), _toks()
    _save(c, t, 4096, protect=True)
    _save(c, t, 8192)
    _save(c, t, 12288)  # plain LRU: the protected 4096 goes
    assert sorted(k[0] for k in c._lru) == [8192, 12288]


def test_value_evict_keeps_end_of_prompt_checkpoints(dev, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_VALUE_EVICT", "1")
    c, t = _cache(4), _toks()
    _save(c, t, 4096, protect=True)
    _save(c, t, 8192)
    _save(c, t, 12288)
    _save(c, t, 16384)
    _save(c, t, 20480)  # 4096 (end) is protected: the LRU normal 8192 goes
    assert sorted(k[0] for k in c._lru) == [4096, 12288, 16384, 20480]
    assert "protected=1" in c.summary()


def test_value_evict_keeps_multi_hit_prefixes(dev, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_VALUE_EVICT", "1")
    c, t = _cache(3), _toks()
    _save(c, t, 4096)
    _save(c, t, 8192)
    c.lookup(t, 4096)
    c.lookup(t, 4096)  # 2 hits: shared prefix
    c.lookup(t, 8192)  # 1 hit, most recent
    _save(c, t, 12288)
    _save(c, t, 16384)  # evicts 8192 (1 hit, LRU among unprotected) although 4096 is older
    assert sorted(k[0] for k in c._lru) == [4096, 12288, 16384]


def test_value_evict_low_stride_still_goes_first(dev, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_VALUE_EVICT", "1")
    c, t = _cache(3), _toks()
    _save(c, t, 4096, protect=True)
    _save(c, t, 8192, low_prio=True)
    _save(c, t, 12288)
    _save(c, t, 16384)
    assert sorted(k[0] for k in c._lru) == [4096, 12288, 16384]


def test_value_evict_protected_share_is_capped(dev, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_VALUE_EVICT", "1")
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_PROTECT_FRAC", "0.5")
    c, t = _cache(4), _toks()
    for p in (4096, 8192, 12288):
        _save(c, t, p, protect=True)  # 3 protected > cap 2: the oldest (4096) counts as normal
    _save(c, t, 16384)
    _save(c, t, 20480)
    assert sorted(k[0] for k in c._lru) == [8192, 12288, 16384, 20480]
    _save(c, t, 24576)  # protected 8192 / 12288 within the cap: the normal 16384 goes
    assert sorted(k[0] for k in c._lru) == [8192, 12288, 20480, 24576]


def test_value_evict_spills_protected_victims_to_host(dev, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_CACHE_VALUE_EVICT", "1")
    c, t = _cache(1, host_slots=2), _toks()
    _save(c, t, 4096, protect=True)
    _save(c, t, 8192)  # the only slot: the protected entry is still evicted, and spilled
    assert [k[0] for k in c._host] == [4096] and not c._end() - set(c._lru)


# --------------------------------------------------------------------------- #
# plan end marks, true resume metrics
# --------------------------------------------------------------------------- #
def _plan(L, start, **kw):
    m = types.SimpleNamespace(
        _prefix_cache=_cache(4), args=types.SimpleNamespace(max_batch_size=8), _chunked_chunk_size=2048
    )
    return Qwen36Model.plan_spec_prefix(m, torch.arange(L), L, start, **kw)


def test_plan_marks_the_end_of_prompt_point(dev):
    assert _plan(200000, 0)[2].end == {199936}
    assert _plan(5000, 0)[2].end == {4096}  # short prompt: the last full chunk stands in for the end
    pts = _plan(30000, 0, force_points=[12288])[2]
    assert pts.end == {29952} and 12288 in pts
    assert _plan(30000, 0, force_points=[8192, 12288], nonfinal=True)[2].end == set()


def test_plan_end_point_follows_min_len(dev, monkeypatch):
    monkeypatch.setenv("QWEN36_PREFIX_SNAPSHOT_MIN_LEN", "512")
    s, _, pts = _plan(1000, 0)
    assert list(pts) == [896] and pts.end == {896}


def test_record_resume_accounts_true_positions(dev):
    c = _cache(2, host_slots=2)
    c.record_resume(0, 0, 0, None)  # no vLLM hit: nothing
    c.record_resume(4352, 4224, 4096, 0)  # device, 128 below the cap
    c.record_resume(9000, 8960, 8192, HostSnap((8192, b""), []))  # host
    c.record_resume(3000, 2944, 0, None)  # vLLM hit, nothing usable: full recompute
    st = c.stats
    assert st["resume_prefills"] == 3 and st["vllm_cached_tokens"] == 4352 + 9000 + 3000
    assert st["gdn_resumed_tokens"] == 4096 + 8192
    assert st["hidden_recompute_tokens"] == 256 + 808 + 3000
    assert st["below_cap_tokens"] == 128 + 768 + 2944
    assert (st["resume_device"], st["resume_host"], st["resume_none"]) == (1, 1, 1)
    s = c.summary()
    assert "hidden_recompute_tok=4064" in s and "from_host=1" in s


def test_note_resume_fallback_feeds_record_resume(dev):
    c = _cache(2)
    eng = types.SimpleNamespace(B=8, _chunk_cache=lambda: c)
    MTPSpecEngine._note_resume_fallback(eng, [1] * 5000, 4352, 5000, (4096, (c, 0), []), True)
    MTPSpecEngine._note_resume_fallback(eng, [1] * 5000, 4352, 5000, None, True)
    assert c.stats["resume_device"] == 1 and c.stats["resume_none"] == 1
    assert c.stats["hidden_recompute_tokens"] == 256 + 4352
    assert eng.resume_fallbacks == 2  # both resumed below the cap 4224
