# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the host-side spec-step fixes (QWEN36_SPEC_HOSTFIX): new vs old code paths. No device.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_hostfix_host.py -q
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tt.model import _spec_kv_write_conflicts_old, spec_kv_write_conflicts


# ---- R5: spec_kv_write_conflicts ------------------------------------------------------------------------
def _call(fn, *args):
    try:
        return ("ok", fn(*args))
    except IndexError as e:
        return ("err", str(e))


@pytest.mark.parametrize("seed", range(40))
def test_conflicts_random_match_old(seed):
    g = torch.Generator().manual_seed(seed)
    B, T, bs, nb = 8, 4, 64, int(torch.randint(2, 40, (1,), generator=g))
    # few distinct blocks and close positions force same-tile collisions; zeros give null blocks
    pt = torch.randint(0, 4, (B, nb), generator=g, dtype=torch.int32)
    pos = torch.randint(-3, nb * bs - T, (B,), generator=g).tolist()
    if seed % 2:
        pos = [int(torch.randint(0, 2 * bs, (1,), generator=g)) if p >= 0 else p for p in pos]
    for pt_in in (pt, pt.to(torch.int64)):
        assert _call(spec_kv_write_conflicts, pt_in, pos, T, bs) == _call(
            _spec_kv_write_conflicts_old, pt_in, pos, T, bs
        )


def test_conflicts_edge_cases(monkeypatch):
    bs, T = 64, 4
    pt = torch.zeros(8, 6, dtype=torch.int32)
    assert spec_kv_write_conflicts(pt, [-1] * 8, T, bs) == []  # all rows inactive
    assert spec_kv_write_conflicts(pt, [5] * 8, T, bs) == []  # all null blocks
    pt[:, 0] = 7  # all rows live on the same block and tile: every call collides
    pos = [3, 3, 40, 40, 70, -1, 0, 31]
    got = spec_kv_write_conflicts(pt, pos, T, bs)
    assert got == _spec_kv_write_conflicts_old(pt, pos, T, bs) and got
    # duplicate slots (identical rows and positions)
    assert spec_kv_write_conflicts(pt, [10, 10], T, bs) == _spec_kv_write_conflicts_old(pt, [10, 10], T, bs)
    # position past the table: same IndexError text; the first offender in (j, u) order is reported
    pos = [0, 6 * bs - 2, 6 * bs - 1, 0, 0, 0, 0, 0]
    a, b = _call(spec_kv_write_conflicts, pt, pos, T, bs), _call(_spec_kv_write_conflicts_old, pt, pos, T, bs)
    assert a == b and a[0] == "err"
    # list input, ragged rows, tensor positions, fewer positions than rows, empty positions
    rows = [[1, 2, 3], [1, 2], [4]]
    assert spec_kv_write_conflicts(rows, [0, 0, 0], T, bs) == _spec_kv_write_conflicts_old(rows, [0, 0, 0], T, bs)
    assert spec_kv_write_conflicts(pt, torch.tensor([0, 5, 9]), T, bs) == _spec_kv_write_conflicts_old(
        pt, [0, 5, 9], T, bs
    )
    assert spec_kv_write_conflicts(pt, [], T, bs) == []
    # more positions than rows: handled by the old code
    assert _call(spec_kv_write_conflicts, pt[:2], [0, 0, 0], T, bs)[0] == "err"
    assert _call(spec_kv_write_conflicts, pt[:2], [0, 0, 0], T, bs) == _call(
        _spec_kv_write_conflicts_old, pt[:2], [0, 0, 0], T, bs
    )


def test_conflicts_flag_off_is_old(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_HOSTFIX", "0")
    pt = torch.full((4, 3), 9, dtype=torch.int32)
    assert spec_kv_write_conflicts(pt, [0, 1, 2, 3], 4, 64) == _spec_kv_write_conflicts_old(pt, [0, 1, 2, 3], 4, 64)


# ---- R7: row loops ----------------------------------------------------------------------------------------
from models.demos.blackhole.qwen36.tt.spec_engine import (  # noqa: E402
    MTPSpecEngine,
    _sanitize_verify_block_old,
    note_withheld_drafts,
    reseed_row_plan,
    reseed_rows_host,
    sanitize_verify_block,
)


def _run(fn, *args, **kw):
    try:
        return ("ok", fn(*args, **kw))
    except ValueError as e:
        return ("err", str(e))


def _same(a, b):
    if a[0] != b[0]:
        return False
    if a[0] == "err":
        return a[1] == b[1]
    return all(torch.equal(x, y) and x.dtype == y.dtype for x, y in zip(a[1], b[1]))


def _verify_block(g, B, K, corrupt):
    T = K + 1
    tok = torch.randint(0, 1000, (B, T), generator=g)
    start = torch.randint(0, 5000, (B,), generator=g)
    pos = start[:, None] + torch.arange(T)
    nv = torch.randint(0, K + 1, (B,), generator=g)
    dead = torch.rand(B, generator=g) < 0.3  # padding rows
    pos[dead] = -1
    nv[dead] = 0
    tok[dead] = torch.randint(-1, 3, (int(dead.sum()), T), generator=g)  # junk tokens in padding rows are fine
    if corrupt:
        u = int(torch.randint(0, B, (1,), generator=g))
        kind = int(torch.randint(0, 5, (1,), generator=g))
        if kind == 0:
            nv[u] = K + 1 if corrupt % 2 else -1
        elif kind == 1:
            tok[u, : int(nv[u]) + 1] = -5
        elif kind == 2:
            pos[u, int(nv[u])] += 3
        elif kind == 3:
            pos[u, 0] = -1  # inactive row that still claims drafts
            nv[u] = 1
        else:
            nv[u] = K + 2
        if int(torch.randint(0, 2, (1,), generator=g)):  # a second bad row after the first
            w = int(torch.randint(0, B, (1,), generator=g))
            pos[w, -1] += 1
            tok[w, 0] = -1
    return tok, pos, nv


@pytest.mark.parametrize("B", [1, 2, 8])
@pytest.mark.parametrize("K", [1, 3])
def test_sanitize_matches_old(B, K):
    g = torch.Generator().manual_seed(100 * B + K)
    n_err = 0
    for i in range(150):
        tok, pos, nv = _verify_block(g, B, K, corrupt=i % 3)
        a = _run(sanitize_verify_block, tok, pos, nv, K)
        b = _run(_sanitize_verify_block_old, tok, pos, nv, K)
        assert _same(a, b), (a, b)
        n_err += a[0] == "err"
    assert 0 < n_err < 150


def test_sanitize_edge_cases(monkeypatch):
    K, T = 3, 4
    # all rows inactive; all rows live with n = K; lists as input; empty batch
    tok = torch.arange(8).reshape(2, T)
    for pos, nv in (
        (torch.full((2, T), -1), torch.zeros(2, dtype=torch.int64)),
        (torch.arange(T).repeat(2, 1) + torch.tensor([[7], [9]]), torch.full((2,), K)),
    ):
        assert _same(_run(sanitize_verify_block, tok, pos, nv, K), _run(_sanitize_verify_block_old, tok, pos, nv, K))
    a = _run(sanitize_verify_block, tok.tolist(), pos.tolist(), nv.tolist(), K)
    assert _same(a, _run(_sanitize_verify_block_old, tok.tolist(), pos.tolist(), nv.tolist(), K))
    e = torch.zeros(0, T, dtype=torch.int64)
    assert _same(
        _run(sanitize_verify_block, e, e, torch.zeros(0, dtype=torch.int64), K),
        _run(_sanitize_verify_block_old, e, e, torch.zeros(0, dtype=torch.int64), K),
    )
    # shape errors keep their text
    assert _run(sanitize_verify_block, tok[:, :3], pos, nv, K) == _run(
        _sanitize_verify_block_old, tok[:, :3], pos, nv, K
    )
    monkeypatch.setenv("QWEN36_SPEC_HOSTFIX", "0")
    assert _same(_run(sanitize_verify_block, tok, pos, nv, K), _run(_sanitize_verify_block_old, tok, pos, nv, K))


@pytest.mark.parametrize("B", [1, 8])
def test_note_withheld_matches_old(B, monkeypatch):
    g = torch.Generator().manual_seed(B)
    for _ in range(200):
        nv = torch.randint(0, 3, (B,), generator=g)
        if int(torch.randint(0, 2, (1,), generator=g)):
            nv = nv.to(torch.int32).reshape(B, 1)  # any shape that flattens to [B]
        offered = {u for u in range(B) if int(torch.randint(0, 2, (1,), generator=g))}
        marked = {u for u in range(B) if int(torch.randint(0, 5, (1,), generator=g)) == 0}
        spec = {u for u in range(B) if int(torch.randint(0, 3, (1,), generator=g)) == 0}
        streak = {u: int(torch.randint(0, 17, (1,), generator=g)) for u in range(B) if u % 2}
        limit = int(torch.randint(1, 18, (1,), generator=g))
        out = {}
        for flag in ("1", "0"):
            monkeypatch.setenv("QWEN36_SPEC_HOSTFIX", flag)
            st = (set(offered), set(spec), dict(streak))
            m = note_withheld_drafts(st[0], marked, st[1], st[2], nv, limit=limit)
            out[flag] = (m, st[0], st[1], list(st[2].items()))  # dict order too
        assert out["1"] == out["0"]
        assert out["1"][1] == set()  # offered is consumed


def _propose_engine(B, K, pad, monkeypatch=None):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size = B, K, K + 1, 64
    e._offered, e._no_draft, e.skipped_propose_calls = set(), set(), 0
    e._row_active, e._last_pt, e._last_cap_blocks = (None if pad is None else [not p for p in pad]), None, 8
    e.calls = []

    def propose(ctok, cpos, acc, page_table, decline=None, cap_blocks=None, keep_kv=None):
        e.calls.append((ctok.clone(), cpos.clone(), acc.clone(), list(decline), list(keep_kv)))
        return torch.zeros(B, K, dtype=torch.int32)

    e.propose = propose
    return e


@pytest.mark.parametrize("B", [1, 8])
def test_propose_draft_tokens_row_checks_match_old(B, monkeypatch):
    K = 3
    T = K + 1
    g = torch.Generator().manual_seed(B)
    n_err = 0
    pt = torch.ones(B, 8, dtype=torch.int32)
    for i in range(200):
        acc = torch.randint(1, T + 1, (B,), generator=g)
        if i % 17 == 0:
            acc[:] = 1
        cpos = torch.randint(1, 3000, (B, 1), generator=g) + torch.arange(T)
        ctok = torch.randint(0, 900, (B, T), generator=g)
        for u in range(B):
            ctok[u, int(acc[u]) :] = -1  # runner pads past accepted_counts
            cpos[u, int(acc[u]) :] = -1
        pad = [bool(torch.rand(1, generator=g) < 0.3) for _ in range(B)]
        if i % 4 == 1:  # corrupt one used column
            u = int(torch.randint(0, B, (1,), generator=g))
            c = int(acc[u]) - 1
            if i % 8 == 1:
                ctok[u, c] = -2
            elif c > 0:
                cpos[u, c] += 2
            else:
                cpos[u, 0] = 0
        res = {}
        for flag in ("1", "0"):
            monkeypatch.setenv("QWEN36_SPEC_HOSTFIX", flag)
            e = _propose_engine(B, K, pad)
            r = _run(MTPSpecEngine.propose_draft_tokens, e, K, ctok, cpos, acc, page_table=pt)
            if r[0] == "ok":
                r = ("ok", (r[1].draft_token_ids, r[1].num_valid, *e.calls[0][:3]))
                r = ("ok", r[1] + (e.calls[0][3:],))
                res[flag] = r
            else:
                res[flag] = r
        assert res["1"][0] == res["0"][0]
        if res["1"][0] == "err":
            assert res["1"][1] == res["0"][1]
            n_err += 1
        else:
            assert all(
                torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y for x, y in zip(res["1"][1], res["0"][1])
            )
    assert n_err > 0


# ---- R6: one read of the drafter ids (fake ttnn) -----------------------------------------------------------
import ttnn  # noqa: E402


class _FakeDev:
    def __init__(self, data):
        self.data = data


class _FakeHost:
    def __init__(self, dev):
        self.dev, self.synced = dev, False


def _fake_ttnn(monkeypatch, log):
    def from_device(t, blocking=True, **kw):
        log.append(("from_device", blocking))
        h = _FakeHost(t)
        hosts_seen.append(h)
        return h

    def to_torch(t):
        if isinstance(t, _FakeHost):
            assert t.synced, "host copy read before the synchronize"
            return t.dev.data
        log.append(("blocking_read",))
        return t.data

    def synchronize_device(mesh):
        log.append(("sync",))
        for h in hosts_seen:
            h.synced = True

    hosts_seen = []
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t, _FakeDev(torch.full((1,), -1))])
    monkeypatch.setattr(ttnn, "from_device", from_device)
    monkeypatch.setattr(ttnn, "to_torch", to_torch)
    monkeypatch.setattr(ttnn, "synchronize_device", synchronize_device)


@pytest.mark.parametrize("B", [1, 8])
def test_ids_to_host_matches_per_tensor_reads(B, monkeypatch):
    g = torch.Generator().manual_seed(B)
    K = 3
    outs = [_FakeDev(torch.randint(0, 248320, (1, 1, B + 2), generator=g).to(torch.int64)) for _ in range(K)]
    # the engine reads the first B ids of a flattened [1,1,B]-like tensor; extra tail ids must be ignored
    e = _propose_engine(B, K, None)
    e.mesh = None
    res = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("QWEN36_SPEC_HOSTFIX", flag)
        log = []
        _fake_ttnn(monkeypatch, log)
        res[flag] = (MTPSpecEngine._ids_to_host(e, outs) if flag == "1" else [e._id_to_host(t) for t in outs], log)
    # old path in this test = per-tensor _id_to_host; _ids_to_host with flag 0 must delegate to it
    assert res["1"][0] == res["0"][0]
    assert [x for x in res["1"][1] if x[0] == "sync"] == [("sync",)] and ("blocking_read",) not in res["1"][1]
    assert all(x == ("from_device", False) for x in res["1"][1] if x[0] == "from_device")
    monkeypatch.setenv("QWEN36_SPEC_HOSTFIX", "0")
    log = []
    _fake_ttnn(monkeypatch, log)
    assert MTPSpecEngine._ids_to_host(e, outs) == res["0"][0]
    assert log.count(("blocking_read",)) == K and ("sync",) not in log
