# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the row-local prefix cache bookkeeping (Hy4Engine.match_len / hit_len / assign_rows / park_pos); no device."""
import numpy as np

from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import ROWS, Hy4Engine


def eng(pf=512):
    e = object.__new__(Hy4Engine)
    e.xrow, e.xrow_on, e.xrow_plan = None, False, {}
    e.pfx, e.pf_rows, e.pfx_redo = True, pf, True
    e.ret, e.ret_t, e.live = [None] * ROWS, [0.0] * ROWS, [False] * ROWS
    return e


def toks(n, seed):
    return np.random.default_rng(seed).integers(1000, 100000, n).astype(np.int32)


def test_hit_aligned_and_leaves_a_token():
    e = eng()
    a = toks(5000, 1)
    e.ret[2] = a[:4608].copy()
    assert e.match_len(2, a) == 4608
    assert e.hit_len(2, a[:4608]) == 4096  # at least one token left, chunk aligned
    b = a.copy()
    b[1500] += 1
    assert e.match_len(2, b) == 1500 and e.hit_len(2, b) == 1024
    assert e.hit_len(0, a) == 0


def test_assign_prefers_matching_free_row_never_live():
    e = eng()
    a, b = toks(3000, 1), toks(3000, 2)
    e.ret[1], e.ret[3] = a[:2560].copy(), b[:2560].copy()
    assert e.assign_rows([a, b], [0, 1]) == [1, 3]
    e.live[1] = True
    r = e.assign_rows([a], [0])
    assert r[0] != 1


def test_assign_no_match_spares_retained_rows():
    e = eng()
    e.ret[0], e.ret[1], e.ret[2] = toks(1024, 5), toks(1024, 6), toks(1024, 7)
    e.ret_t[:] = [3.0, 1.0, 2.0, 0.0]
    assert e.assign_rows([toks(900, 9)], [0]) == [3]  # the empty row
    e.ret[3] = toks(512, 8)
    e.ret_t[3] = 5.0
    assert e.assign_rows([toks(900, 9)], [0]) == [1]  # least recently used


def test_two_requests_same_prefix_one_row_each():
    e = eng()
    a = toks(3000, 1)
    e.ret[1] = a[:2560].copy()
    out = e.assign_rows([a, a], [0, 1])
    assert len(set(out)) == 2 and 1 in out


def test_park_pos():
    e = eng()
    assert e.park_pos(0) == 0
    e.ret[0] = toks(1024, 1)
    assert e.park_pos(0) == 1024
    e.pfx = False
    assert e.park_pos(0) == 0
