# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the prefix-cache bookkeeping: CPU mirrors of the ring and block-table logic (no device)."""

import random

import torch

from models.demos.muse_glimmer_30b.tt import prefix_cache as pc

BLOCK, WINDOW, RING_BLOCKS = 64, 2048, 36  # the real model's numbers: ceil((2048 + 256) / 64) = 36


def _ring_fill(cache, row, values, modulo_blocks=RING_BLOCKS):
    """Mirror of paged_fill_cache with cache_position_modulo: local row i lands at table[(i % modulo) // block]."""
    for i, v in enumerate(values):
        idx = i % (modulo_blocks * BLOCK)
        cache[int(row[idx // BLOCK]), idx % BLOCK] = v


def _gather(cache, segs):
    return torch.cat([cache[b : b + n].reshape(-1) for b, n in segs])


def _prefill_into_ring(cache, ring, start, n_real):
    """Fill positions [start, start + n_real) as the model does: padded to 256 rows, junk (-1) past the real rows."""
    fill_len = -(-n_real // 256) * 256
    vals = torch.full((fill_len,), -1, dtype=torch.int64)
    vals[:n_real] = torch.arange(start, start + n_real)
    row = pc.shifted_ring_row(ring, start, RING_BLOCKS, BLOCK, RING_BLOCKS)
    _ring_fill(cache, row, vals)


def _copy_ring(cache, src, dst):
    cache[dst * RING_BLOCKS : (dst + 1) * RING_BLOCKS] = cache[src * RING_BLOCKS : (src + 1) * RING_BLOCKS].clone()


def test_chain_hashes_are_prefix_stable_and_content_sensitive():
    a = torch.arange(1000)
    h = pc.chain_hashes(a)
    assert len(h) == 3  # 768 tokens hashed, the 232-token tail is not a boundary
    assert pc.chain_hashes(a[:600]) == h[:2]
    b = a.clone()
    b[300] += 1
    hb = pc.chain_hashes(b)
    assert hb[0] == h[0] and hb[1] != h[1] and hb[2] != h[2]
    assert pc.chain_hashes(a, salt=b"x")[0] != h[0]
    assert pc.chain_hashes(a[:255]) == []


def test_plan_resumes_at_largest_snapshot_boundary_below_cached():
    reg = pc.RingSnapshotRegistry(first=33, capacity=3)
    prompt = torch.arange(5000)
    ring, key = reg.register(prompt)
    assert ring == 33 and reg.entries[key][1] == 4864  # floor(5000 / 256) * 256
    nxt = torch.cat([prompt, torch.arange(10_000, 13_000)])
    plan = reg.plan(nxt, cached_tokens=4992)  # vLLM caches whole 64-token blocks: floor(5000 / 64) * 64
    assert (plan.start, plan.snapshot_ring, plan.cached_tokens) == (4864, 33, 4992)
    assert reg.hits == 1


def test_plan_without_snapshot_recomputes_from_zero():
    reg = pc.RingSnapshotRegistry(first=33, capacity=2)
    plan = reg.plan(torch.arange(5000), cached_tokens=2048)
    assert (plan.start, plan.snapshot_ring) == (0, None)
    assert reg.misses == 1 and reg.stale_cached_tokens == 2048
    assert reg.plan(torch.arange(5000), cached_tokens=0).start == 0 and reg.misses == 1


def test_plan_never_resumes_past_cached_tokens():
    reg = pc.RingSnapshotRegistry(first=0, capacity=2)
    reg.register(torch.arange(1024))  # boundary 1024
    p = torch.arange(1024 + 700)
    assert reg.plan(p, cached_tokens=1024).start == 1024
    assert reg.plan(p, cached_tokens=960).start == 0  # snapshot is past what vLLM has cached
    assert reg.plan(torch.arange(1024), cached_tokens=1024).start == 0  # at least one token must be computed


def test_diverging_prompt_hits_the_shared_boundary_only():
    reg = pc.RingSnapshotRegistry(first=0, capacity=4)
    a = torch.arange(3000)
    reg.register(a)
    b = a.clone()
    b[2800:] += 7  # shares the first 2800 tokens; the snapshot boundary is 2816, past the divergence
    assert reg.plan(b, cached_tokens=2816).start == 0


def test_registry_lru_eviction_and_ring_reuse():
    reg = pc.RingSnapshotRegistry(first=10, capacity=2)
    ps = [torch.arange(i * 10_000, i * 10_000 + 1024) for i in range(3)]
    r0, _ = reg.register(ps[0])
    r1, _ = reg.register(ps[1])
    assert (r0, r1) == (10, 11)
    reg.plan(torch.cat([ps[0], torch.arange(50)]), cached_tokens=1024)  # touch ps[0]: ps[1] is now the LRU
    r2, _ = reg.register(ps[2])
    assert r2 == r1  # evicted ring is reused
    assert reg.plan(torch.cat([ps[1], torch.arange(50)]), cached_tokens=1024).start == 0
    assert reg.plan(torch.cat([ps[0], torch.arange(50)]), cached_tokens=1024).start == 1024


def test_register_refreshes_same_prefix_in_place_and_skips_short_or_disabled():
    reg = pc.RingSnapshotRegistry(first=5, capacity=2)
    a = torch.arange(2000)
    r, k = reg.register(a)
    assert reg.register(torch.cat([a, torch.arange(100)[:0]])) == (r, k)
    assert len(reg.entries) == 1
    assert reg.register(torch.arange(255)) is None
    assert pc.RingSnapshotRegistry(first=0, capacity=0).register(a) is None


def test_shifted_page_row_and_cached_block_redirect():
    row = torch.arange(100, 100 + 20, dtype=torch.int32)  # request owns blocks 100..119
    s = pc.shifted_page_row(row, 256, BLOCK, 18)
    assert s.tolist() == list(range(104, 120)) + [0, 0]
    f = pc.fill_row_without_cached(row, 256, 448, BLOCK, 18)  # blocks 4,5,6 (rows 256..447) are vLLM-cached
    assert f.tolist() == [0, 0, 0] + list(range(107, 120)) + [0, 0]
    assert pc.fill_row_without_cached(row, 256, 256, BLOCK, 18)[0] == 104  # nothing cached past the resume point
    assert pc.fill_row_without_cached(row, 256, 100, BLOCK, 18)[0] == 104  # cached length below the start: untouched


def test_history_segments_wrap_and_short_prefix():
    assert pc.history_segments(256, WINDOW, BLOCK, RING_BLOCKS, ring=2) == [(72, 4)]  # only 4 blocks of history
    assert pc.history_segments(2048, WINDOW, BLOCK, RING_BLOCKS, ring=0) == [(0, 32)]
    # start / 64 = 40: history blocks 8..39 -> ring blocks 8..35 then 0..3
    assert pc.history_segments(2560, WINDOW, BLOCK, RING_BLOCKS, ring=1) == [(36 + 8, 28), (36, 4)]
    for start in range(256, 20_000, 256):
        segs = pc.history_segments(start, WINDOW, BLOCK, RING_BLOCKS, ring=3)
        assert len(segs) <= 2 and sum(n for _, n in segs) == min(WINDOW, start) // BLOCK


def test_resume_bucket_caps_to_rope_tables(expect_error):
    buckets = [128, 256, 512, 1024, 2048]
    assert pc.resume_bucket(300, buckets, start=1024, max_seq_len=131072) == 512
    assert pc.resume_bucket(300, buckets, start=131072 - 256, max_seq_len=131072) == 256
    with expect_error(ValueError, "exceeds the largest bucket"):
        pc.resume_bucket(3000, buckets, start=0, max_seq_len=131072)


def test_ring_snapshot_restore_resume_reproduces_the_window_over_many_turns():
    """The claim behind the design: a ring copied at the end of a prefill holds [s - 2048, s) at the resume position
    s = floor(P / 256) * 256, for any P, and stays right after the resumed prefill and another snapshot."""
    rng = random.Random(7)
    n_rings = 6
    cache = torch.full((n_rings * RING_BLOCKS, BLOCK), -2, dtype=torch.int64)
    live, retained = 0, 4
    # turn 0: from scratch
    P = rng.randrange(300, 9000)
    _prefill_into_ring(cache, live, 0, P)
    for _ in range(40):
        s = P // 256 * 256
        _copy_ring(cache, live, retained)  # snapshot at the end of the prefill
        # the live ring is then overwritten by decode of up to ~200 tokens (positions P ..)
        for pos in range(P, P + rng.randrange(0, 200)):
            cache[live * RING_BLOCKS + (pos // BLOCK) % RING_BLOCKS, pos % BLOCK] = pos
        # next turn resumes from the snapshot into a different live ring
        live = 1 - live
        _copy_ring(cache, retained, live)
        segs = pc.history_segments(s, WINDOW, BLOCK, RING_BLOCKS, ring=live)
        hist = _gather(cache, segs)
        assert hist.tolist() == list(range(s - min(WINDOW, s), s)), (P, s)
        m = rng.randrange(1, 12_000)
        _prefill_into_ring(cache, live, s, m)
        P = s + m
        # the new ring must hold the window of the new boundary
        s2 = P // 256 * 256
        got = _gather(cache, pc.history_segments(s2, WINDOW, BLOCK, RING_BLOCKS, ring=live))
        assert got.tolist() == list(range(s2 - min(WINDOW, s2), s2))


def _sliding_attention(q, k, v, window):
    s = q.shape[0]
    scores = q @ k.T / q.shape[-1] ** 0.5
    i, j = torch.arange(s)[:, None], torch.arange(k.shape[0])[None, :]
    scores = scores.masked_fill(~((j <= i) & (j > i - window)), float("-inf"))
    return torch.softmax(scores, -1) @ v


def test_history_plus_zero_q_padding_equals_full_sliding_attention():
    """The sliding layers' resume SDPA: q padded with `hist` dummy rows over concat(hist K/V, new K/V), causal
    sliding window, first `hist` output rows dropped == the rows of an uncached full prefill."""
    torch.manual_seed(0)
    window, d, total = 16, 8, 96
    q, k, v = (torch.randn(total, d, dtype=torch.float64) for _ in range(3))
    full = _sliding_attention(q, k, v, window)
    for s in (16, 32, 48, 80):
        hist = min(window, s)
        qe = torch.cat([torch.zeros(hist, d, dtype=torch.float64), q[s:]])
        ke, ve = k[s - hist :], v[s - hist :]
        out = _sliding_attention(qe, ke, ve, window)[hist:]
        assert torch.allclose(out, full[s:], atol=1e-12), s
