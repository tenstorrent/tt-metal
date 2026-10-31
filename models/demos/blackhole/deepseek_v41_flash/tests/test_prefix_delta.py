# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free test of tt/prefix_cache.DeltaStore (block / delta host snapshots, DSV41_APC_DELTA=1).

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_prefix_delta.py

The fake model state is a pure function of the token prefix, like the real one: tail tensor "keys" (ratio 2, 2 column shards, rows [1,1,n,4]), a replicated tail tensor "kvt" (ratio 2) and pool
latents of source 0 (ratio 2, rows of 5): row j depends on the tokens [0, (j + 1) * 2). The "halo" (full kind) depends on the last 4 tokens before s.
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt.prefix_cache import DeltaStore, PrefixSnapshot, PrefixStore

CHUNK, RATIO, COLS = 8, 2, 2


def _row(toks, j, salt):
    h = float(toks[: (j + 1) * RATIO].sum()) % 997.0
    return torch.tensor([h, j + salt, salt, 1.0], dtype=torch.bfloat16)


def truth(toks, s):
    """full-form snapshot at s of the fake state"""
    n = s // RATIO // COLS
    keys = [torch.stack([_row(toks, j * COLS + c, 10 + c) for j in range(n)]).reshape(1, 1, n, 4) for c in range(COLS)]
    kvt = torch.stack([_row(toks, j, 20) for j in range(s // RATIO)]).reshape(1, 1, s // RATIO, 4)
    pool = torch.stack(
        [torch.cat([_row(toks, j, 30)] * 1 + [torch.zeros(1, dtype=torch.bfloat16)]) for j in range(s // RATIO)]
    )
    halo = toks[s - 4 : s].float().reshape(1, 1, 4, 1).expand(1, 1, 4, 3).to(torch.bfloat16).clone()
    return keys, kvt, pool, halo


def snap_at(toks, s, s_lo=0, full=True):
    keys, kvt, pool, halo = truth(toks, s)
    lo_n = s_lo // RATIO // COLS
    sn = PrefixSnapshot(s=s, toks=toks[:s].clone(), s_lo=s_lo, full=full)
    sn.carried["halo"] = [halo]
    sn.carried["keys"] = [k[:, :, lo_n:].clone() for k in keys]
    sn.carried["kvt"] = [kvt[:, :, s_lo // RATIO :].clone()] * COLS
    sn.pool[0] = pool[s_lo // RATIO :].clone()
    sn.tails = ("keys", "kvt")
    return sn


def toks_of(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 1000, (n,), generator=g, dtype=torch.int64)


def put_delta(st, toks, s, **kw):
    keys = st.block_keys(toks, s)
    s_lo = st.have_prefix(keys) * CHUNK
    return st.put(snap_at(toks, s, s_lo, **kw)), s_lo


def check_equal(st, rec, toks, s):
    full = st.assemble(rec)
    keys, kvt, pool, halo = truth(toks, s)
    assert full.s == s and torch.equal(full.toks, toks[:s])
    assert torch.equal(full.carried["halo"][0], halo)
    assert len(full.carried["keys"]) == COLS
    for c in range(COLS):
        assert torch.equal(full.carried["keys"][c], keys[c])
    assert torch.equal(full.carried["kvt"][0], kvt)
    assert torch.equal(full.pool[0], pool)


def test_delta_chain_assembles_to_the_whole_snapshot():
    t = toks_of(200)
    st = DeltaStore(CHUNK, 10**9)
    for s in (16, 48, 96, 64):  # (an out-of-order one too: its blocks are all held already)
        ok, s_lo = put_delta(st, t, s)
        assert ok
    # the second put (48) started at 16, the third (96) at 48, the last (64) read nothing new
    for s in (16, 48, 96, 64):
        rec = st.lookup(t, s, s + CHUNK, margin=CHUNK)
        assert rec is not None and rec.s == s
        check_equal(st, rec, t, s)


def test_delta_bytes_are_far_below_whole_snapshots():
    t = toks_of(400)
    d, w = DeltaStore(CHUNK, 10**9), PrefixStore(CHUNK, 10**9)
    for s in range(16, 321, 16):
        put_delta(d, t, s)
        w.put(snap_at(t, s))
    # halo (24 bytes) per record + each block once, against the whole prefix per snapshot
    assert d.bytes < w.bytes / 8
    assert len(d.blocks) == 320 // CHUNK


def test_branching_prompts_share_the_common_blocks():
    a = toks_of(120, 1)
    b = a.clone()
    b[70:] = toks_of(50, 2)  # diverges inside block 8 (tokens 64..71)
    st = DeltaStore(CHUNK, 10**9)
    put_delta(st, a, 112)
    n0 = len(st.blocks)
    ok, s_lo = put_delta(st, b, 112)
    assert ok and s_lo == 64  # blocks 0..7 shared
    assert len(st.blocks) == n0 + (112 - 64) // CHUNK
    check_equal(st, st.lookup(a, 112, 112 + CHUNK, margin=CHUNK), a, 112)
    check_equal(st, st.lookup(b, 112, 112 + CHUNK, margin=CHUNK), b, 112)
    assert st.lookup(b, 112, 112 + CHUNK, margin=CHUNK).s == 112


def test_eviction_frees_a_block_with_its_last_record_and_keeps_shared_ones():
    t = toks_of(200, 3)
    st = DeltaStore(CHUNK, 10**9)
    put_delta(st, t, 32)
    put_delta(st, t, 64)
    n_blocks = len(st.blocks)
    own = [r.nbytes for r in st.snaps.values()]
    # evict the older record (32): the blocks of [0, 32) are still referenced by the 64 record
    k32 = next(iter(st.snaps))
    st.drop(k32)
    assert len(st.blocks) == n_blocks
    k64 = next(iter(st.snaps))
    st.drop(k64)
    assert not st.blocks and st.bytes == 0 and own


def test_budget_evicts_lru_records_and_accounts_blocks():
    t = toks_of(300, 4)
    probe = DeltaStore(CHUNK, 10**9)
    put_delta(probe, t, 128)
    halo = probe.snaps[next(iter(probe.snaps))].nbytes
    st = DeltaStore(CHUNK, probe.bytes + int(2.5 * halo))
    for s in (16, 32, 48, 64, 80, 96, 112, 128):
        put_delta(st, t, s)
        assert st.bytes <= st.budget
    assert st.evictions > 0
    # accounting is exact: bytes = halos + live blocks
    assert st.bytes == sum(r.nbytes for r in st.snaps.values()) + sum(b.nbytes for b in st.blocks.values())
    assert all(b.refs > 0 for b in st.blocks.values())
    top = max(r.s for r in st.snaps.values())
    assert top == 128
    check_equal(st, st.lookup(t, 128, 128 + CHUNK, margin=CHUNK), t, 128)


def test_a_record_whose_block_vanished_before_finish_is_refused():
    t = toks_of(100, 5)
    st = DeltaStore(CHUNK, 10**9)
    put_delta(st, t, 32)
    keys = st.block_keys(t, 64)
    s_lo = st.have_prefix(keys) * CHUNK  # 32: the read at ``begin`` skipped the first 4 blocks
    st.clear()  # (evicted before ``finish``)
    assert not st.put(snap_at(t, 64, s_lo)) and st.lost == 1 and not st.blocks


def test_full_replaces_partial_and_partial_never_replaces_full():
    t = toks_of(100, 6)
    st = DeltaStore(CHUNK, 10**9)
    assert put_delta(st, t, 32, full=False)[0]
    assert st.has(t, 32, full=False) and not st.has(t, 32, full=True)
    assert put_delta(st, t, 32, full=True)[0]
    assert st.has(t, 32, full=True) and len(st.snaps) == 1
    assert not put_delta(st, t, 32, full=False)[0]
    assert all(b.refs == 1 for b in st.blocks.values())
    check_equal(st, st.lookup(t, 32, 32 + CHUNK, margin=CHUNK), t, 32)


def test_halo_only_record_when_every_block_is_cached():
    t = toks_of(100, 7)
    st = DeltaStore(CHUNK, 10**9)
    put_delta(st, t, 64)
    b0 = st.bytes
    ok, s_lo = put_delta(st, t, 40)
    assert ok and s_lo == 40  # nothing to read, only the halo of a finer point
    assert st.bytes == b0 + st.lookup(t, 40, 40 + CHUNK, margin=CHUNK).nbytes
    check_equal(st, st.lookup(t, 40, 40 + CHUNK, margin=CHUNK), t, 40)


def test_prefix_cache_in_delta_mode_reads_only_new_rows_and_restores_the_whole_state(monkeypatch):
    """PrefixCache with DSV41_APC_DELTA=1 over a fake device: ``begin_snapshot`` asks for rows from the end of the cached run, ``restore_group`` hands ``restore_users`` the full-form state"""
    import types

    from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC

    monkeypatch.setenv("DSV41_APC_DELTA", "1")
    monkeypatch.setenv("DSV41_APC_MARGIN", str(CHUNK))
    monkeypatch.delenv("DSV41_APC_GRID", raising=False)
    monkeypatch.delenv("DSV41_APC_FILE", raising=False)
    reads, restored = [], []

    def begin(model, b, s, toks, s_lo=0):
        reads.append((s, s_lo))
        return types.SimpleNamespace(s=s, toks=torch.as_tensor(toks)[:s].clone(), s_lo=s_lo)

    def finish(model, p):
        sn = snap_at(p.toks, p.s, p.s_lo)
        return sn

    monkeypatch.setattr(PC, "snapshot_begin", begin)
    monkeypatch.setattr(PC, "snapshot_finish", finish)
    monkeypatch.setattr(PC, "delta_unsupported", lambda m, c: "")
    monkeypatch.setattr(PC, "restore_users", lambda m, snap, bs: restored.append(snap) or 0.0)
    cache = PC.PrefixCache(types.SimpleNamespace(), CHUNK, 10**9, "all", log=lambda *_: None)
    assert isinstance(cache.store, DeltaStore)
    t = toks_of(160, 9)
    for B in range(
        CHUNK, 80 + 1, CHUNK
    ):  # turn 1: 80 tokens, a snapshot at every boundary, pending ones finish together
        cache.begin_snapshot(0, B, t[:80])
    cache.finish_pending()
    assert [r[1] for r in reads] == [0] + [
        CHUNK * i for i in range(1, 10)
    ]  # each one starts where the previous (still pending) one ended
    # turn 2: the prompt grows to 160: the lookup takes the deepest snapshot <= 160 - margin, snapshots continue from there
    snap = cache.lookup(0, t, 160)
    assert snap is not None and snap.s == 80
    cache.restore_group(snap, [0])
    check_equal_full(restored[-1], t, 80)
    reads.clear()
    for B in range(88, 160 + 1, CHUNK):
        cache.begin_snapshot(0, B, t)
    cache.finish_pending()
    assert reads[0] == (88, 80) and all(lo == B - CHUNK for B, lo in reads)
    snap = cache.lookup(0, t, 160)
    cache.restore_group(snap, [0])
    check_equal_full(restored[-1], t, snap.s)


def check_equal_full(full, toks, s):
    keys, kvt, pool, halo = truth(toks, s)
    assert full.s == s and torch.equal(full.carried["halo"][0], halo)
    for c in range(COLS):
        assert torch.equal(full.carried["keys"][c], keys[c])
    assert torch.equal(full.carried["kvt"][0], kvt) and torch.equal(full.pool[0], pool)


def test_decode_output_snapshot_flow(monkeypatch):
    """PrefixCache.decode_poll / decode_finish (DSV41_APC_DECODE=1): one halo capture per crossed chunk multiple beyond the prompt's own, the record at the latest capture, delta read from the end of the
    cached prompt blocks, a following prompt (prompt + answer + more) hits it"""
    import types

    from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC

    monkeypatch.setenv("DSV41_APC_DELTA", "1")
    monkeypatch.setenv("DSV41_APC_DECODE", "1")
    monkeypatch.setenv("DSV41_APC_MARGIN", str(CHUNK))
    monkeypatch.delenv("DSV41_APC_GRID", raising=False)
    monkeypatch.delenv("DSV41_APC_FILE", raising=False)
    caps, reads, restored = [], [], []

    def begin(model, b, s, toks, s_lo=0):
        return types.SimpleNamespace(s=s, toks=torch.as_tensor(toks)[:s].clone(), s_lo=s_lo)

    monkeypatch.setattr(PC, "snapshot_begin", begin)
    monkeypatch.setattr(PC, "snapshot_finish", lambda model, p: snap_at(p.toks, p.s, p.s_lo))
    monkeypatch.setattr(PC, "delta_unsupported", lambda m, c: "")
    monkeypatch.setattr(PC, "decode_unsupported", lambda m: "")
    monkeypatch.setattr(PC, "restore_users", lambda m, snap, bs: restored.append(snap) or 0.0)
    monkeypatch.setattr(PC, "decode_halo_begin", lambda m, b, s: caps.append(s) or (s, "cap"))

    def dsnap(model, b, s, toks, s_lo, halo):
        reads.append((s, s_lo))
        assert halo == (s, "cap")
        return snap_at(torch.as_tensor(toks).long().reshape(-1), s, s_lo)

    monkeypatch.setattr(PC, "decode_snapshot", dsnap)
    cache = PC.PrefixCache(types.SimpleNamespace(), CHUNK, 10**9, "all", log=lambda *_: None)
    t = toks_of(400, 21)
    prompt = t[:76]
    for B in range(CHUNK, 72 + 1, CHUNK):  # the prompt's own tip snapshot at 72
        cache.begin_snapshot(0, B, prompt)
    cache.finish_pending()
    assert cache.store.has(prompt, 72)
    for n in range(77, 200):  # decode: 77 .. 199 tokens hold KV
        cache.decode_poll(0, n)
    assert caps == [80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192]
    cache.decode_finish(0, t[:199])
    assert reads == [(192, 72)]  # the answer's blocks only: the prompt's blocks up to 72 are cached
    rec = cache.lookup(0, t, 300)
    assert rec.s == 192
    cache.restore_group(rec, [0])
    check_equal_full(restored[-1], t, 192)
    # a user that never crossed a new multiple stores nothing; a stale state (n goes down) restarts
    cache.decode_poll(1, 100)
    cache.decode_poll(1, 101)
    cache.decode_finish(1, t[:101])
    assert reads == [(192, 72)]
    cache.decode_poll(2, 150)
    cache.decode_poll(2, 90)  # new request on the same user
    assert cache.dec[2]["s0"] == 88 and cache.dec[2]["cap"] is None
