# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tt/hy4_blocks_store.py (shadow block store, brief B45_3 S1).
Run: python3 -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/hy4_preview/tests/test_hy4_blocks_cpu.py -q"""
import importlib.util
import os
import random

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hy4_blocks_store_ut", os.path.join(_HERE, "../tt/hy4_blocks_store.py"))
BS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BS)
C = 512


def toks(seed, n):
    return np.random.default_rng(seed).integers(1, 120000, size=n, dtype=np.int64)


def row_local_hit(resident, t, A=512):
    """hy4_serve_engine.hit_len semantics"""
    if resident is None or len(resident) == 0:
        return 0
    n = min(len(resident), len(t))
    ne = (resident[:n] != t[:n]).nonzero()[0]
    m = int(ne[0]) if len(ne) else n
    return (min(m, len(t) - 1) // A) * A


def test_digest_chain_matches_dsv4():
    import importlib.util as iu

    p = os.path.join(_HERE, "../../deepseek_v41_flash/tt/apc_paged.py")
    s = iu.spec_from_file_location("ap_ut", p)
    m = iu.module_from_spec(s)
    s.loader.exec_module(m)
    t = toks(1, 5000)
    assert BS.digests_of(t) == m.chain_digests(t, chunk=512)
    assert BS.digests_of(t, b"x") == m.chain_digests(t, chunk=512, salt=b"x")


@pytest.mark.parametrize("seed", range(20))
def test_zero_shadow_reproduces_row_local(seed):
    rnd = random.Random(seed)
    st = BS.ShadowStore(rows=1, shadow_pages=0)
    base = toks(seed, 9000)
    for _ in range(10):
        res = base[: rnd.randrange(0, 9000) // C * C] if rnd.random() < 0.8 else None
        n = rnd.randrange(1, 9000)
        t = base[:n].copy()
        if rnd.random() < 0.5 and n > 10:
            t[rnd.randrange(0, n)] += 1
        p = st.plan_hit(0, t, res)
        assert p.start == row_local_hit(res, t)
        assert p.copy_in == [] and st.plan_out(0, res) == []


def test_publish_then_shadow_hit_from_other_chain():
    st = BS.ShadowStore(rows=1, shadow_pages=64)
    sysp = toks(1, 4 * C)
    a = np.concatenate([sysp, toks(2, 6 * C)])
    b = np.concatenate([sysp, toks(3, 6 * C)])
    out = st.plan_out(0, a, session="A")
    assert [i for i, _ in out] == list(range(10))
    # chain B is resident, A conversation returns: shadow holds A, active holds B (shares the 4 system blocks)
    q = np.concatenate([a, toks(4, 700)])
    p = st.plan_hit(0, q, resident=b)
    assert p.start == 10 * C and p.active_blocks == 4 and p.shadow_blocks == 6
    assert [d for _, d in p.copy_in] == list(range(4, 10))  # active page index
    st.unpin(p)
    st.check()


def test_shared_prefix_stored_once():
    st = BS.ShadowStore(rows=1, shadow_pages=64)
    sysp = toks(1, 4 * C)
    a = np.concatenate([sysp, toks(2, 2 * C)])
    b = np.concatenate([sysp, toks(3, 2 * C)])
    assert len(st.plan_out(0, a)) == 6
    assert len(st.plan_out(0, b)) == 2  # 4 shared blocks not copied again
    assert st.used(0) == 8
    st.check()


def test_leaf_first_lru_and_tip_protect():
    st = BS.ShadowStore(rows=1, shadow_pages=8, tip_sessions=1)
    a, b, c = toks(1, 4 * C), toks(2, 4 * C), toks(3, 3 * C)
    st.plan_out(0, a, session="A")
    st.plan_out(0, b, session="B")  # full: 8 blocks, tip = B
    out = st.plan_out(
        0, c, session="C"
    )  # needs 3: evicts leaves of A (LRU, A not tip any more? tip keeps only the newest session B)
    assert len(out) == 3
    # A lost its leaf blocks first (leaf-first: depth 4,3,2), B intact
    assert len(st.shadow_chain(0, BS.digests_of(b))) == 4
    assert len(st.shadow_chain(0, BS.digests_of(a))) == 1
    st.check()


def test_tip_is_never_evicted_even_if_oldest():
    st = BS.ShadowStore(rows=1, shadow_pages=4, tip_sessions=1)
    a = toks(1, 4 * C)
    st.plan_out(0, a, session="A")
    out = st.plan_out(0, toks(2, 2 * C), session=None)  # A is the tip: nothing evictable
    assert out == [] and st.stats["skipped"] == 2
    assert len(st.shadow_chain(0, BS.digests_of(a))) == 4


def test_pins_block_eviction():
    st = BS.ShadowStore(rows=1, shadow_pages=4)
    a = toks(1, 4 * C)
    st.plan_out(0, a)
    p = st.plan_hit(0, np.concatenate([a, toks(9, 10)]), resident=None)
    assert p.start == 4 * C and len(p.copy_in) == 4
    assert st.plan_out(0, toks(2, 2 * C)) == []  # every block pinned by the in-flight copy
    st.unpin(p)
    assert len(st.plan_out(0, toks(2, 2 * C))) == 2
    st.check()


def test_hit_margin_never_whole_prompt():
    st = BS.ShadowStore(rows=1, shadow_pages=16)
    a = toks(1, 4 * C)
    st.plan_out(0, a)
    p = st.plan_hit(
        0, a.copy(), resident=None
    )  # identical prompt of exactly 4 blocks: hit at most 3 blocks (>= 1 token to prefill)
    assert p.start == 3 * C
    p0 = st.plan_hit(0, a[:C].copy(), resident=None)
    assert p0.start == 0


def test_keep_from_and_rootless():
    st = BS.ShadowStore(rows=1, shadow_pages=16)
    a = toks(1, 4 * C)
    # blocks 0,1 stay active (not published), 2,3 would be rootless: nothing published
    assert st.plan_out(0, a, keep_from=2) == []
    st.check()


def test_rows_independent():
    st = BS.ShadowStore(rows=2, shadow_pages=8)
    a = toks(1, 3 * C)
    st.plan_out(0, a)
    assert st.plan_hit(1, np.concatenate([a, toks(2, 5)]), None).start == 0
    assert st.plan_hit(0, np.concatenate([a, toks(2, 5)]), None).start == 3 * C


@pytest.mark.parametrize("seed", range(10))
def test_random_ops_invariants_and_more_than_4_conversations(seed):
    rnd = random.Random(seed)
    st = BS.ShadowStore(rows=1, shadow_pages=120, tip_sessions=2)
    sysp = toks(100, 6 * C)
    convs = [np.concatenate([sysp, toks(200 + i, rnd.randrange(3, 14) * C)]) for i in range(8)]
    res = None
    for step in range(60):
        i = rnd.randrange(8)
        convs[i] = np.concatenate([convs[i], toks(1000 + step, rnd.randrange(1, 3) * C)])
        q = np.concatenate([convs[i], toks(5000 + step, 37)])
        p = st.plan_hit(0, q, res)
        assert p.start % C == 0 and p.start <= len(q) - 1
        assert p.start >= row_local_hit(res, q)  # never worse than the retained chain
        st.unpin(p)
        st.plan_out(0, res if res is not None else None, session=None) if res is not None else None
        res = q[: len(q) // C * C]
        st.check()
    # 8 conversations of <= 14+ blocks (+shared 6) do not all fit in 120 pages, but several survive
    held = sum(1 for c in convs if len(st.shadow_chain(0, BS.digests_of(c))) > 6)
    assert held >= 1


def test_b8_user_slot_bases():
    """HY4_B8_ON32: 2 users per row, user slot u owns pages [u*ppu,(u+1)*ppu), shadow after both: page ids are absolute"""
    ppu = 1000
    st = BS.ShadowStore(rows=1, shadow_pages=8, shadow_base=2 * ppu)
    a = toks(1, 3 * C)
    out = st.plan_out(0, a, active_base=ppu)  # user slot 1
    assert [o[0] for o in out] == [ppu, ppu + 1, ppu + 2] and all(2 * ppu <= o[1] < 2 * ppu + 8 for o in out)
    p = st.plan_hit(0, np.concatenate([a, toks(2, 5)]), None, active_base=0)  # served into user slot 0
    assert [d for _, d in p.copy_in] == [0, 1, 2] and all(s >= 2 * ppu for s, _ in p.copy_in)
