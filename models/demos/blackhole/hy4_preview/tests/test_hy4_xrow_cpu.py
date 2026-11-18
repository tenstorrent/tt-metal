# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tt/hy4_xrow.py (B45x): chunk / page index math, CPU reference of the row-to-row copy (also by emulating the 3-step device pipeline), the row-choice policy (plan_copies), flags.
Run (worktree root):  python3 -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/hy4_preview/tests/test_hy4_xrow_cpu.py -q"""
import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hy4_xrow_under_test", os.path.join(_HERE, "../tt/hy4_xrow.py"))
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)


def _raises_value_error(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def toks(seed, n):
    return np.random.RandomState(seed).randint(1000, 100000, n).astype(np.int32)


def test_blocks_of():
    assert X.blocks_of(512) == 1 and X.blocks_of(65536) == 128 and X.blocks_of(291840) == 570
    for bad in (0, -512, 1, 511, 513, 1000, 65537):
        assert _raises_value_error(X.blocks_of, bad)


@pytest.mark.parametrize("npool,cp", [(1, 128), (128, 128), (129, 128), (570, 128), (570, 256), (7, 3)])
def test_chunk_plan_covers(npool, cp):
    ch = X.chunk_plan(npool, cp)
    assert ch[0][0] == 0 and sum(n for _, n in ch) == npool
    for (p0, n), (q0, _) in zip(ch, ch[1:]):
        assert p0 + n == q0
    assert all(0 < n <= cp for _, n in ch)


def _pools(rows, cols, pages, units, w=5, seed=0):
    rs = np.random.RandomState(seed)
    return [[rs.randint(0, 255, (pages * units, w)).astype(np.uint8) for _ in range(cols)] for _ in range(rows)]


@pytest.mark.parametrize("units", [64, 8])
@pytest.mark.parametrize("src,dst", [(0, 1), (3, 0), (2, 1), (1, 3)])
@pytest.mark.parametrize("h", [512, 1536, 5120])
def test_ref_copy_semantics(units, src, dst, h):
    pools = _pools(4, 8, 12, units)
    out = X.ref_copy(pools, src, dst, h, units)
    n = h // 512 * units
    for r in range(4):
        for c in range(8):
            if r == dst:
                assert (out[r][c][:n] == pools[src][c][:n]).all()
                assert (out[r][c][n:] == pools[dst][c][n:]).all()  # beyond the copy: untouched
            else:
                assert (out[r][c] == pools[r][c]).all()  # other rows incl. the source untouched
    assert (pools[dst][0] != out[dst][0]).any()  # input not modified in place


def test_ref_copy_rejects():
    p = _pools(2, 2, 4, 8)
    assert _raises_value_error(X.ref_copy, p, 0, 0, 512, 8)
    assert _raises_value_error(X.ref_copy, p, 0, 1, 700, 8)


def _emulate(pools, src, dst, h, units, cp):
    """the device pipeline on numpy: stage-in -> all_gather over rows -> stage-out with off / skip (page indices as the kernel computes them)."""
    rows, cols = len(pools), len(pools[0])
    out = [[x.copy() for x in row] for row in pools]
    for p0, n in X.chunk_plan(X.blocks_of(h), cp):
        s_in, s_out, d0, npg = X.page_ranges(p0, n, units, cp, src)
        for c in range(cols):
            stage = []
            for r in range(rows):
                st = np.zeros((cp * units, pools[0][0].shape[1]), np.uint8)
                st[:npg] = out[r][c][s_in : s_in + npg]
                stage.append(st)
            full = np.concatenate(stage, 0)  # all_gather dim 0
            for r in range(rows):
                if r == dst:
                    out[r][c][d0 : d0 + npg] = full[s_out : s_out + npg]
    return out


@pytest.mark.parametrize("units,cp", [(64, 128), (64, 3), (8, 256), (8, 4)])
@pytest.mark.parametrize("src,dst", [(0, 1), (3, 2), (2, 0)])
def test_pipeline_matches_reference(units, cp, src, dst):
    pools = _pools(4, 2, 17, units)
    h = 512 * 11
    a = _emulate(pools, src, dst, h, units, cp)
    b = X.ref_copy(pools, src, dst, h, units)
    assert all((a[r][c] == b[r][c]).all() for r in range(4) for c in range(2))


def test_partial_last_chunk_never_touches_beyond():
    pools = _pools(4, 1, 9, 8)
    a = _emulate(pools, 1, 2, 512 * 5, 8, 4)  # chunks 4 + 1
    assert (a[2][0][5 * 8 :] == pools[2][0][5 * 8 :]).all()


# ---- row choice policy
def _plan(rets, live, prompts, cur=None, taken=(), **kw):
    cur = cur or list(range(len(prompts)))
    return X.plan_copies(rets, live, set(taken), prompts, cur, **kw)


def test_copy_when_source_busy():
    base = toks(1, 20000)
    rets = [base[:16384].copy(), None, None, None]
    live = [True, False, False, False]
    p = np.concatenate([base[:16384], toks(2, 3000)])
    rows, cp = _plan(rets, live, [p], cur=[1])
    assert rows[0] != 0 and cp == {0: (0, 16384)}


def test_own_match_beats_copy():
    base = toks(1, 20000)
    rets = [None, base[:16384].copy(), base[:16384].copy(), None]
    live = [False] * 4
    p = np.concatenate([base[:16384], toks(2, 3000)])
    rows, cp = _plan(rets, live, [p])
    assert rows[0] in (1, 2) and cp == {}


def test_copy_from_free_row_when_longer_than_own():
    base = toks(1, 40000)
    rets = [base[:32768].copy(), base[:2048].copy(), None, None]
    live = [False] * 4
    p = np.concatenate([base[:32768], toks(2, 3000)])
    # row 0 would be chosen by itself (longest own match): no copy
    rows, cp = _plan(rets, live, [p])
    assert rows[0] == 0 and cp == {}
    # row 0 busy: best free is row 1 (2048), copy from 0 gains 30720 >= min
    live[0] = True
    rows, cp = _plan(rets, live, [p], cur=[1])
    assert rows[0] == 1 and cp == {0: (0, 32768)}


def test_no_free_row_no_copy():
    base = toks(1, 20000)
    rets = [base[:16384].copy()] * 4
    live = [True] * 4
    p = np.concatenate([base[:16384], toks(2, 3000)])
    rows, cp = _plan(rets, live, [p], cur=[2])
    assert rows == [2] and cp == {}


def test_below_min_hit_no_copy():
    base = toks(1, 5000)
    rets = [base[:512].copy(), None, None, None]
    live = [True, False, False, False]
    p = np.concatenate([base[:512], toks(2, 3000)])
    rows, cp = _plan(rets, live, [p], cur=[1])
    assert cp == {}


def test_cap_limits_h():
    base = toks(1, 40000)
    rets = [base[:32768].copy(), None, None, None]
    live = [True, False, False, False]
    p = np.concatenate([base[:32768], toks(2, 3000)])
    rows, cp = _plan(rets, live, [p], cur=[1], cap=16384)
    assert cp == {0: (0, 16384)}


def test_source_not_destination_and_distinct_dsts():
    base = toks(1, 40000)
    rets = [base[:16384].copy(), None, None, None]
    live = [True, False, False, False]
    ps = [np.concatenate([base[:16384], toks(10 + i, 2000)]) for i in range(2)]
    rows, cp = _plan(rets, live, ps, cur=[1, 2])
    assert len(set(rows)) == 2 and 0 not in rows
    assert all(s == 0 for s, _ in cp.values()) and len(cp) == 2


def test_dst_never_a_source_of_another_copy():
    a, b = toks(1, 20000), toks(2, 20000)
    rets = [a[:8192].copy(), b[:8192].copy(), None, None]
    live = [True, True, False, False]
    ps = [np.concatenate([a[:8192], toks(5, 2000)]), np.concatenate([b[:8192], toks(6, 2000)])]
    rows, cp = _plan(rets, live, ps, cur=[2, 3])
    srcs = {s for s, _ in cp.values()}
    assert srcs.isdisjoint(set(rows[i] for i in cp))


def test_last_token_never_cached():
    base = toks(1, 16384)
    rets = [base.copy(), None, None, None]
    live = [True, False, False, False]
    rows, cp = _plan(rets, live, [base.copy()], cur=[1])  # prompt == retained chain: one token must stay to prefill
    assert cp[0][1] == 16384 - 512


def test_flags(monkeypatch):
    monkeypatch.delenv("HY4_XROW", raising=False)
    assert not X.enabled()
    monkeypatch.setenv("HY4_XROW", "1")
    assert X.enabled()
    monkeypatch.setenv("HY4_XROW_FILE", "/nonexistent/x")
    assert X.enabled()


def test_cost_model_matches_kv_design():
    assert abs(X.copy_cost_s(65536) - 0.55) < 1e-6 and X.copy_cost_s(1024) < X.cold_cost_s(1024)
