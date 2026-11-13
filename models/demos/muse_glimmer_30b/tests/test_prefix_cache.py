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


# ---- adapter level: fakes stand in for ttnn, the model and the device tensors --------------------------------------

import importlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from models.demos.muse_glimmer_30b.tt import generator_vllm_base as gv  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.kv_caches = []

    def prefill_forward_logits_tt(
        self, tokens, user_id, pt, pad_len=None, tail=256, sliding_pt=None, resume=None, fill_len=None
    ):
        self.calls.append(
            dict(tokens=tokens, pt=pt, pad_len=pad_len, tail=tail, sliding_pt=sliding_pt, resume=resume, fill_len=fill_len)
        )
        return "logits"

    def logits_to_torch(self, logits, n):
        return torch.zeros(n, 4)


def _adapter(monkeypatch, snapshots=3):
    monkeypatch.setenv("MUSE_PREFIX_CACHE", "1")
    monkeypatch.setenv("MUSE_PREFIX_SNAPSHOTS", str(snapshots))
    monkeypatch.setattr(gv.ttnn, "deallocate", lambda t: None)
    a = gv.MuseGlimmerForConditionalGeneration()
    a.args = SimpleNamespace(
        sliding_kv_blocks=RING_BLOCKS,
        max_prefill_len=131072,
        max_seq_len=131072,
        block_size=BLOCK,
        sliding_window=WINDOW,
        n_layers=52,
    )
    a.n_blocks = 131072 // BLOCK
    a.buckets = gv.prefill_buckets(131072)
    a.model = _FakeModel()
    a._prefix = pc.RingSnapshotRegistry(first=gv.DECODE_B_PAD + 1, capacity=snapshots)
    a._page_table_tt = lambda row, width: ("tt", row.clone(), width)
    a.copies = []
    a._copy_ring = lambda src, dst: a.copies.append((src, dst))
    return a


def _run(a, prompt, cached, row0=100, slot=0):
    pt = (torch.arange(len(prompt) // BLOCK + 64) + row0).to(torch.int32)[None]
    a.prefill_forward(prompt[None], pt, None, [len(prompt)], start_pos=[cached], empty_slots=[slot])
    return a.model.calls[-1]


def test_capability_follows_env(monkeypatch):
    try:
        monkeypatch.setenv("MUSE_PREFIX_CACHE", "1")
        importlib.reload(gv)
        assert gv.MuseGlimmerForConditionalGeneration.model_capabilities["supports_prefix_caching"] is True
        assert gv.MuseGlimmerForConditionalGeneration.model_capabilities["supports_chunked_prefill"] is False
        monkeypatch.delenv("MUSE_PREFIX_CACHE")
        importlib.reload(gv)
        assert gv.MuseGlimmerForConditionalGeneration.model_capabilities["supports_prefix_caching"] is False
    finally:
        monkeypatch.undo()
        importlib.reload(gv)


def test_first_prompt_prefills_whole_and_snapshots_its_ring(monkeypatch):
    a = _adapter(monkeypatch)
    prompt = torch.arange(5000)
    call = _run(a, prompt, cached=0, slot=3)
    assert call["resume"] is None and call["tokens"].shape[0] == 5000
    assert call["fill_len"] == 5120  # the ring takes the prompt rounded to 256, not the coarse bucket
    live = a._row_ring[3]
    assert a.copies == [(live, gv.DECODE_B_PAD + 1)]  # snapshot of the slot's ring into the first retained ring


def test_second_turn_resumes_from_snapshot_with_shifted_tables(monkeypatch):
    a = _adapter(monkeypatch)
    prompt = torch.arange(5000)
    _run(a, prompt, cached=0, slot=3)
    a.copies.clear()
    nxt = torch.cat([prompt, torch.arange(70_000, 73_000)])  # 8000 tokens, shares the first 5000
    call = _run(a, nxt, cached=4992, slot=5)  # vLLM caches 78 whole blocks of the shared prefix
    r = call["resume"]
    assert r is not None and r.start == 4864  # floor(5000 / 256) * 256
    live = a._row_ring[5]
    assert a.copies[0] == (gv.DECODE_B_PAD + 1, live)  # restore the retained ring into the slot's live ring first
    assert call["tokens"].tolist() == nxt[4864:].tolist()  # 3136 rows
    assert call["fill_len"] == 3328
    assert call["pad_len"] >= 3136 and call["pad_len"] in a.buckets
    # full layers: the fill table starts at block 4864 / 64 = 76; blocks 76..77 are vLLM-cached -> null block 0
    kind, fill_row, width = r.full_fill_pt
    assert width == -(-call["pad_len"] // BLOCK)
    assert fill_row[:2].tolist() == [0, 0] and fill_row[2].item() == 100 + 78
    # read table is the request's real, unshifted table at full width
    assert call["pt"][1][0].item() == 100 and call["pt"][2] == a.n_blocks
    # sliding layers: the ring table is shifted by 76 blocks: chunk block 0 -> ring block 76 % 36 = 4
    ring_tt = call["sliding_pt"][0]
    assert ring_tt[1][:3].tolist() == [live * RING_BLOCKS + 4, live * RING_BLOCKS + 5, live * RING_BLOCKS + 6]
    # the 8000-token prompt has its own boundary (7936): a new retained ring; the old entry stays until LRU evicts it
    assert a.copies[1] == (live, gv.DECODE_B_PAD + 2)


def test_hit_without_snapshot_recomputes_from_zero_and_counts_a_miss(monkeypatch):
    a = _adapter(monkeypatch)
    call = _run(a, torch.arange(3000), cached=1024)
    assert call["resume"] is None and call["tokens"].shape[0] == 3000
    assert a._prefix.misses == 1


def test_prefix_cache_off_keeps_the_start_zero_contract(monkeypatch, expect_error):
    monkeypatch.delenv("MUSE_PREFIX_CACHE", raising=False)
    a = gv.MuseGlimmerForConditionalGeneration()
    with expect_error(ValueError, "position 0"):
        a.prefill_forward(torch.zeros(1, 8, dtype=torch.int64), torch.zeros(1, 1), None, [8], start_pos=[256])


def test_resumed_bucket_stays_inside_rope_tables(monkeypatch):
    a = _adapter(monkeypatch)
    prompt = torch.arange(130_000)
    _run(a, prompt, cached=0)
    nxt = torch.cat([prompt, torch.arange(1000)])
    call = _run(a, nxt, cached=129_984)
    assert call["resume"].start + call["pad_len"] <= 131072
