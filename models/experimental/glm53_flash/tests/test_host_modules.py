# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host tests for the generator's planning helpers (``tt/generator.py``) and the prefix-cache bookkeeping
(``tt/prefix_cache.py``). No device needed."""

from __future__ import annotations

import pytest

from models.experimental.glm53_flash.tt.generator import (
    MODEL_PREFILL_ALIGN,
    PREFILL_CHUNK,
    SLICE_SPLIT_FRONT,
    ChunkPlan,
    blocks_per_seq,
    decode_widths,
    plan_prefill,
    row_align,
    slice_boundary,
    slice_plan,
    slice_safe_bucket,
)
from models.experimental.glm53_flash.tt.prefix_cache import (
    BOUNDARY,
    PAGE,
    PrefillCosts,
    PrefixStore,
    SnapshotBytes,
    prefix_hashes,
    prefix_key,
    signature,
    ttft_ladder,
)

# =============================================================================================
# generator planning
# =============================================================================================


def test_decode_widths():
    assert decode_widths(32) == (1, 2, 4, 8, 16, 32)
    assert decode_widths(5) == (1, 2, 4, 5)
    assert decode_widths(1) == (1,)
    assert decode_widths(12, bucketed=False) == (12,)
    assert decode_widths(10, bucketed=[8, 3, 3, 100, 0]) == (3, 8, 10)


@pytest.mark.parametrize(
    "env,want",
    [
        ({}, MODEL_PREFILL_ALIGN),
        ({"GLM53_PF_ROW_ALIGN": "0"}, MODEL_PREFILL_ALIGN),
        ({"GLM53_PF_ROW_ALIGN": "32"}, MODEL_PREFILL_ALIGN),
        ({"GLM53_PF_ROW_ALIGN": "256"}, 256),
        ({"GLM53_PF_ROW_ALIGN": "bucket"}, 2048),
        ({"GLM53_PF_ROW_ALIGN": "Bucket", "GLM53_PREFILL_BUCKETS": "128, 4096,512"}, 4096),
    ],
)
def test_row_align(env, want):
    assert row_align(env) == want


def test_blocks_per_seq():
    # the padded tail of the last chunk may run align - 1 rows past max_seq_len; rows are multiples of 8 entries
    assert blocks_per_seq(128, align=64) == 8
    assert blocks_per_seq(64 * 9, align=64) == 16
    assert blocks_per_seq(65536, align=64) == 1032
    assert blocks_per_seq(65536, align=2048) == 1056
    assert blocks_per_seq(65536, block_size=32, align=64) == 2056


@pytest.mark.parametrize("seq_len,start", [(1, 0), (5, 0), (64, 0), (2048, 0), (2049, 0), (5000, 4096)])
def test_plan_prefill_covers_the_prompt(seq_len, start):
    plans = plan_prefill(seq_len, start_pos=start)
    assert sum(p.n_valid for p in plans) == seq_len
    assert all(p.physical % MODEL_PREFILL_ALIGN == 0 and p.n_valid <= p.physical <= PREFILL_CHUNK for p in plans)
    assert all(p.n_valid == PREFILL_CHUNK for p in plans[:-1])
    for a, b in zip(plans, plans[1:]):
        assert b.offset == a.offset + a.n_valid and b.start == a.start + a.n_valid
    assert plans[0] == ChunkPlan(start=start, offset=0, n_valid=min(seq_len, PREFILL_CHUNK), physical=plans[0].physical)


def test_plan_prefill_rejects_empty(expect_error):
    with expect_error(ValueError, ""):
        plan_prefill(0)


def test_slice_boundary():
    assert not slice_boundary(0, 32)  # row 0 is never a boundary
    assert slice_boundary(32, 32) and slice_boundary(96, 32)
    assert not slice_boundary(33, 32)
    assert not slice_boundary(64, 0)  # replicated bucket


# The SP geometry of the docstrings: 8 chips, R = bucket / 8 rows per chip.
CHIPS = 8


def make_choose(buckets):
    def natural(n):
        return min(b for b in buckets if b >= n)

    def choose(start, n):
        nat = natural(n)
        return nat, slice_safe_bucket(n, nat, buckets, rows=lambda c: c // CHIPS, runs=lambda c: True)

    return choose


def test_slice_safe_bucket():
    rows = lambda c: c // CHIPS
    assert slice_safe_bucket(90, 256, [256, 512], rows, lambda c: True) == 256  # 89 % 32 != 0
    assert slice_safe_bucket(97, 256, [256, 512, 2048], rows, lambda c: True) == 512  # 96 % 32 == 0, 96 % 64 != 0
    assert slice_safe_bucket(97, 256, [256, 512, 2048], rows, lambda c: c != 512) == 2048  # 512 not captured
    assert slice_safe_bucket(97, 256, [256], rows, lambda c: True) is None


@pytest.mark.parametrize(
    "buckets,n,want",
    [
        ((256, 512, 2048), 90, ([90], "")),
        ((256, 512, 2048), 97, ([97], "larger")),
        ((256, 512, 2048), 65, ([65], "larger")),  # 512 is unsafe too; the split is infeasible
        ((256, 512, 2048), 193, ([193], "larger")),  # 2,048 rows beat the 256 + 2,048 of the split
        ((256, 512, 2048, 8192), 513, ([SLICE_SPLIT_FRONT, 513 - SLICE_SPLIT_FRONT], "split")),  # vs 8,192 rows
        ((256,), 97, ([97], "unavoidable")),
    ],
)
def test_slice_plan(buckets, n, want):
    assert slice_plan(0, n, SLICE_SPLIT_FRONT, make_choose(buckets)) == want


# =============================================================================================
# prefix cache: keys
# =============================================================================================

SIG = signature(model="glm53", tp=1)


def tokens(n, seed=0):
    return [(seed * 7919 + 31 * i) % 154856 for i in range(n)]


def test_signature():
    assert signature(a=1, b=2) == signature(b=2, a=1)
    assert signature(a=1, b=2) != signature(a=1, b=3)
    assert len(SIG) == 16


def test_prefix_hashes_chain(expect_error):
    t = tokens(3 * BOUNDARY + 50)
    hs = prefix_hashes(t, SIG)
    assert len(hs) == 3  # complete blocks only
    assert prefix_hashes(t[: 2 * BOUNDARY], SIG) == hs[:2]
    other = list(t)
    other[5] += 1  # a change in block 0 changes every later key
    assert all(a != b for a, b in zip(prefix_hashes(other, SIG), hs))
    assert prefix_hashes(t, signature(model="other")) != hs
    assert prefix_key(t, 2 * BOUNDARY, SIG) == hs[1]
    for bad in (0, BOUNDARY - 1, 4 * BOUNDARY):
        with expect_error(ValueError, ""):
            prefix_key(t, bad, SIG)


# =============================================================================================
# prefix cache: store
# =============================================================================================


def pages_for(n_tokens, first=0):
    return list(range(first, first + -(-n_tokens // PAGE)))


def make_store(**kw):
    kw.setdefault("max_batch", 4)
    kw.setdefault("device_entries", 2)
    return PrefixStore(sig=SIG, entry_bytes=1000, **kw)


def register_device(store, t, n, first_page=0, kind="P"):
    ent = store.reserve_device()
    assert ent is not None
    return store.register(store.key(t, n), n_tokens=n, pages=pages_for(n, first_page), kind=kind, device_entry=ent)


def test_boundaries():
    s = make_store()
    assert s.boundaries(300) == [128, 256]
    assert s.boundaries(256) == [128]  # the last token must be prefilled for logits
    assert s.last_boundary(256) == 128 and s.last_boundary(257) == 256 and s.last_boundary(1) == 0


def test_longest_cached_and_attach(expect_error):
    s = make_store()
    t = tokens(600)
    assert s.longest_cached(t) == (0, None)
    e1 = register_device(s, t, 128)
    e2 = register_device(s, t, 384)
    assert s.longest_cached(t) == (384, e2)
    assert s.longest_cached(t, max_tokens=383) == (128, e1)
    assert s.longest_cached(t[:384]) == (128, e1)  # len - 1 caps the hit
    diverged = t[:200] + tokens(400, seed=9)
    assert s.longest_cached(diverged) == (128, e1)  # partial match on the first block

    pos, e = s.attach(e2.key, slot=1)
    assert (pos, e.readers, e.hits) == (384, 1, 1)
    assert s.stats["tokens_saved"] == 384
    with expect_error(KeyError, ""):
        s.attach("missing", slot=2)
    assert s.stats["misses"] == 1
    s.attach(e1.key, slot=1)  # re-attaching a slot detaches the old one
    assert (e2.readers, e1.readers) == (0, 1)
    s.release_slot(1)
    assert e1.readers == 0 and s.attached == {}


def test_register_validation(expect_error):
    s = make_store()
    t = tokens(300)
    ent = s.reserve_device()
    with expect_error(ValueError, "multiple"):
        s.register("k", n_tokens=100, pages=[0, 1], kind="P", device_entry=ent)
    with expect_error(ValueError, "pages"):
        s.register("k", n_tokens=256, pages=[0, 1, 2], kind="P", device_entry=ent)
    with expect_error(ValueError, "one tier"):
        s.register("k", n_tokens=128, pages=[0, 1], kind="P", device_entry=ent, host_blob=object())
    with expect_error(ValueError, "lives"):
        s.register("k", n_tokens=128, pages=[0, 1], kind="P")
    with expect_error(ValueError, "not reserved"):
        s.register("k", n_tokens=128, pages=[0, 1], kind="P", device_entry=7)
    e = s.register(s.key(t, 128), n_tokens=128, pages=[0, 1, 2, 3], kind="P", device_entry=ent)
    assert e.pages == (0, 1)  # trimmed to the prefix's pages


def test_pins_follow_entries():
    s = make_store()
    t = tokens(300)
    e1 = register_device(s, t, 128, first_page=10)
    e2 = register_device(s, t, 256, first_page=10)  # shares pages 10, 11
    assert all(s.pinned(p) for p in (10, 11, 12, 13)) and not s.pinned(14)
    s.evict(e2.key)
    assert s.pinned(10) and s.pinned(11) and not s.pinned(12)
    s.evict(e1.key)
    assert s.pins == {}


def test_device_tier_lru_eviction_spares_readers():
    s = make_store(device_entries=2)
    t = tokens(600)
    e1 = register_device(s, t, 128)
    e2 = register_device(s, t, 256)
    s.attach(e1.key, slot=0)  # e1 has a reader; e2 is the only evictable entry
    ent = s.reserve_device()
    assert ent == e2.device_entry and not s.has(e2.key) and s.has(e1.key)
    s.unreserve(ent)
    e3 = register_device(s, t, 384)
    s.attach(e3.key, slot=1)
    assert s.reserve_device() is None  # every device entry has a reader
    assert s.stats["evicted_device"] == 1


def test_spill_to_host_tier_on_pressure(expect_error):
    s = make_store(device_entries=1, host_bytes=2500)
    s.spill_fn = lambda e: ("blob:" + e.key[:4], 1000)
    t = tokens(600)
    e1 = register_device(s, t, 128)
    e2 = register_device(s, t, 256)  # the device is full: e1 is spilled, not dropped
    assert s.entries[e1.key].tier == "host" and s.entries[e1.key].host_blob.startswith("blob:")
    assert s.host_used == 1000 and e2.tier == "device"
    assert s.longest_cached(t) == (256, e2)
    s.evict(e2.key)
    assert s.longest_cached(t)[1].key == e1.key
    s.promote(e1.key, s.reserve_device())
    assert s.entries[e1.key].tier == "device" and s.host_used == 0
    with expect_error(ValueError, ""):
        s.promote(e1.key, 0)


def test_spill_refused_or_failing_drops_the_victim():
    for fn, stat in ((lambda e: None, "spill_refused"), (lambda e: 1 / 0, "spill_failed")):
        s = make_store(device_entries=1, host_bytes=5000)
        s.spill_fn = fn
        t = tokens(300)
        e1 = register_device(s, t, 128)
        register_device(s, t, 256)
        assert not s.has(e1.key) and s.stats[stat] == 1


def test_host_tier_budget(expect_error):
    s = make_store(device_entries=0, host_bytes=2000)
    t = tokens(600)
    keys = [s.key(t, n) for n in (128, 256, 384)]
    for k, n in zip(keys, (128, 256, 384)):
        s.register(k, n_tokens=n, pages=pages_for(n), kind="P", host_blob=b"x")
    assert not s.has(keys[0]) and s.has(keys[1]) and s.has(keys[2]) and s.host_used == 2000
    s.attach(keys[1], 0)
    s.attach(keys[2], 1)
    with expect_error(ValueError, "readers"):
        s.register(s.key(t, 512 - 128), n_tokens=384, pages=pages_for(384), kind="P", host_blob=b"y", bytes=1500)
    with expect_error(ValueError, "no host tier"):
        make_store(host_bytes=0).register("k", n_tokens=128, pages=[0, 1], kind="P", host_blob=b"x")


def test_spill_validation(expect_error):
    s = make_store(device_entries=1, host_bytes=500)
    t = tokens(300)
    e = register_device(s, t, 128)
    with expect_error(ValueError, "exceeds"):
        s.spill(e.key, b"x")  # entry_bytes 1000 > host tier
    s.attach(e.key, 0)
    with expect_error(ValueError, "readers"):
        s.spill(e.key, b"x", bytes=100)
    s.detach(0)
    s.spill(e.key, b"x", bytes=100)
    with expect_error(ValueError, "not on the device"):
        s.spill(e.key, b"x", bytes=100)
    assert s.spill_candidates() == []


def test_trailing_generation_snapshot_lifecycle(expect_error):
    s = make_store(device_entries=2)
    t = tokens(400)
    with expect_error(ValueError, ""):
        s.trailing_entry(0, 100)
    ent = s.trailing_entry(0, 128)
    assert s.trailing_entry(0, 256) == ent and s.trailing_position(0) == 256  # overwritten in place
    s.move_slot(0, 3)  # compaction moves it with the request
    assert s.trailing_position(3) == 256 and s.trailing_position(0) == 0
    e = s.commit_trailing(3, t, pages_for(400))
    assert e.kind == "G" and e.n_tokens == 256 and e.device_entry == ent and s.longest_cached(t) == (256, e)
    assert s.commit_trailing(3, t, []) is None

    # a trailing snapshot past the request's known tokens is dropped, freeing its entry
    ent2 = s.trailing_entry(1, 384)
    assert s.commit_trailing(1, t[:300], pages_for(300)) is None and ent2 in s.free_device

    s.trailing_entry(2, 128)
    s.release_slot(2)  # not committed: dropped
    assert s.trailing == {}


def test_invalidate():
    s = make_store()
    t = tokens(300)
    register_device(s, t, 128)
    register_device(s, t, 256)
    assert s.invalidate(sig="new") == 2
    assert s.entries == {} and s.pins == {} and s.sig == "new" and len(s.free_device) == 2
    assert s.longest_cached(t) == (0, None)
    r = s.report()
    assert r["entries"] == 0 and r["invalidated"] == 2 and r["registered"] == 2


def test_snapshot_bytes():
    b = SnapshotBytes()
    assert b.kda_total == 34 * (64 * 128 * 128 * 4 + 3 * 24576 * 2)
    assert b.kda_per_chip == -(-b.kda_total // 8)
    assert b.dsa_per_chip(4096, sparse=False) == 0
    assert b.dsa_per_chip(4096, sparse=True) == 11 * (32 * (32 * 128 * 2) + 8 * 2048)
    assert b.entry_per_chip(4096, sparse=True) == b.kda_per_chip + b.dsa_per_chip(4096, True)
    assert b.pages_per_chip(65) == 2 * 11 * 64 * 512 * 2


def test_prefill_costs_and_ladder(expect_error):
    c = PrefillCosts()
    assert c.replay_ms(2048) == 441.0
    assert c.replay_ms(100) == 202.0 and c.replay_ms(300) == 259.0
    assert c.replay_ms(2048 * 16) == 16 * 441.0 and c.chunks(2049) == 2
    assert c.ttft_ms(128) == 202.0 + 22.0
    assert c.ttft_ms(2048, cached=1920) == 202.0 + 22.0 + 1.5
    assert c.ttft_ms(2048, cached=1920, attach="host") == 202.0 + 22.0 + 450.0
    with expect_error(ValueError, ""):
        c.ttft_ms(128, cached=128)
    rows = ttft_ladder()
    assert [r["context"] for r in rows] == [128, 2048, 32768]
    for r in rows:
        assert r["cached_prefix_suffix_ms"] <= r["cold_ms"]
        assert r["cached_prefix"] % BOUNDARY == 0 and r["resume_suffix"] > 0
