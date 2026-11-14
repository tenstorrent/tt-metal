# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_SPEC_HOSTSET (work/notes/39_hostset_impl.md): the packed / derived inputs equal today's inputs
element for element, the vectorized plans equal the loops, the copies per step drop, the prestage is harmless.
No device. The in-trace ops themselves (ttnn.slice / typecast / embedding on the device) are checked by the device
tests of the note; here a torch emulation of their documented semantics is compared with the legacy host inputs.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_hostset_host.py -q   (also with QWEN36_SPEC_HOSTSET=0)
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_hostfix_host import _propose_engine
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf, _fake_ttnn, _vfy_model
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_page, spec_ctrl_words
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.rope import compute_rope_freqs
from models.demos.blackhole.qwen36.tt.spec_engine import (
    MTPSpecEngine,
    _anchor_selector_old,
    accepted_plan,
    anchor_selector,
    draft_step_plan,
    reseed_row_plan,
    reseed_rows_host,
    reseed_rows_vec,
    verify_write_table,
)
from models.demos.blackhole.qwen36.tt.tp_common import (
    pack_u32_rows,
    rope_row_index,
    spec_hostset_enabled,
    spec_hostset_trace_enabled,
)

RD, THETA = 64, 10000000.0
SHAPES = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (8, 4), (4, 8), (2, 12), (1, 1), (8, 2)]  # (B, T)


@pytest.fixture(params=["1", "0"])
def flag(request, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_HOSTSET", request.param)
    return request.param


def _verify_old_table(page_table, start_pos, nv, bs):
    pt = torch.as_tensor(page_table).clone()
    last = torch.as_tensor(start_pos).reshape(-1).to(torch.int64) + torch.as_tensor(nv).reshape(-1).to(torch.int64)
    cols = torch.arange(pt.shape[1]).reshape(1, -1)
    pt[cols > (last // bs).reshape(-1, 1)] = 0
    return pt


def _rope_table_bf16(max_seq_len):
    cos, sin = compute_rope_freqs(RD, max_seq_len, THETA)
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


def _fake_rope_model(max_seq_len=4096, rope_delta=0):
    m = types.SimpleNamespace(
        args=types.SimpleNamespace(rope_head_dim=RD, rope_theta=THETA),
        rope=types.SimpleNamespace(rope_delta=rope_delta, max_seq_len=max_seq_len),
    )
    for name in ("_rope_cos_sin_rows", "_rope_tp_cos_sin_decode_torch"):
        setattr(m, name, types.MethodType(getattr(Qwen36Model, name), m))
    return m


# ---- flags ----------------------------------------------------------------------------------------------------------
def test_flag_defaults_and_off(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_HOSTSET", raising=False)
    monkeypatch.delenv("QWEN36_SPEC_HOSTSET_TRACE", raising=False)
    assert spec_hostset_enabled() and spec_hostset_trace_enabled()
    monkeypatch.setenv("QWEN36_SPEC_HOSTSET_TRACE", "0")
    assert spec_hostset_enabled() and not spec_hostset_trace_enabled()
    monkeypatch.setenv("QWEN36_SPEC_HOSTSET", "0")
    monkeypatch.delenv("QWEN36_SPEC_HOSTSET_TRACE")
    assert not spec_hostset_enabled() and not spec_hostset_trace_enabled()


# ---- item 3: vectorized plans = the loops ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,T", SHAPES)
def test_anchor_selector_equals_loop(B, T, flag):
    g = torch.Generator().manual_seed(B * 100 + T)
    for _ in range(40):
        m = torch.randint(0, T, (B,), generator=g)
        for mm in (m, m.tolist(), [int(x) for x in m]):
            a, b = anchor_selector(mm, B, T), _anchor_selector_old(mm, B, T)
            assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape and torch.equal(a, b)


@pytest.mark.parametrize("B,T", SHAPES)
def test_accepted_plan_equals_loop(B, T, flag, expect_error):
    g = torch.Generator().manual_seed(B * 31 + T)
    for _ in range(40):
        ctok = torch.randint(0, 200000, (B, T), generator=g)
        cpos = torch.randint(1, 300000, (B, T), generator=g)
        acc = torch.randint(1, T + 1, (B,), generator=g)
        got = accepted_plan(ctok, cpos, acc, B, T)
        m = [int(a) - 1 for a in acc]
        want = (
            m,
            [u * T + m[u] for u in range(B)],
            [int(ctok[u][m[u]]) for u in range(B)],
            [int(cpos[u][m[u]]) - 1 for u in range(B)],
        )
        assert got == want and all(isinstance(x, int) for lst in got for x in lst)
        assert accepted_plan(ctok.tolist(), cpos.tolist(), acc.tolist(), B, T) == want  # list inputs (old path)
    with expect_error(AssertionError, ".*"):
        accepted_plan(ctok, cpos, torch.full((B,), T + 1), B, T)


@pytest.mark.parametrize("B,T", SHAPES)
def test_reseed_rows_vec_equals_plan_and_rows_host(B, T):
    g = torch.Generator().manual_seed(B * 7 + T)
    nb, scratch = 9, 4242
    for i in range(60):
        ctok = torch.randint(0, 248320, (B, T), generator=g)
        cpos = torch.randint(1, 9000, (B, T), generator=g)
        m = torch.randint(0, T + 1, (B,), generator=g).tolist()  # m_kv may reach T for a kept declined row
        if i % 5 == 0:
            m = [0] * B  # padding-only reseed
        pt = torch.randint(0, 3000, (B, nb), generator=g, dtype=torch.int32)
        plan = reseed_row_plan(ctok, cpos, m, B, T)
        tok, pos, rows = reseed_rows_host(plan, pt, T, nb, scratch)
        vt, vp, vr, real = reseed_rows_vec(ctok, cpos, m, pt, T, nb, scratch)
        assert vt.dtype == tok.dtype and vp.dtype == pos.dtype and vr.dtype == rows.dtype
        assert torch.equal(vt, tok) and torch.equal(vp, pos) and torch.equal(vr, rows)
        assert real.tolist() == [r for _, _, r in plan]
        t2, p2, r2, _ = reseed_rows_vec(ctok, cpos, m, None, T, 0, 0, need_pt=False)
        assert r2 is None and torch.equal(t2, tok) and torch.equal(p2, pos)


@pytest.mark.parametrize("B", [1, 2, 8])
def test_verify_write_table_equals_mask(B, flag):
    g = torch.Generator().manual_seed(B)
    bs = 64
    for _ in range(60):
        nb = int(torch.randint(1, 80, (1,), generator=g))
        pt = torch.randint(0, 900, (B, nb), generator=g, dtype=torch.int32)
        start = torch.randint(-3, nb * bs + 70, (B,), generator=g)  # incl. negative and past the table
        nv = torch.randint(0, 4, (B,), generator=g)
        for dt in (torch.int32, torch.int64):
            a = verify_write_table(pt.to(dt), start, nv, bs)
            b = _verify_old_table(pt.to(dt), start, nv, bs)
            assert a.dtype == b.dtype and torch.equal(a, b)
    # block boundary: last column exactly on / just before a block edge
    pt = torch.arange(1, 7, dtype=torch.int32).reshape(1, 6)
    for s in (0, 62, 63, 64, 190, 191, 192):
        assert torch.equal(verify_write_table(pt, [s], [3], bs), _verify_old_table(pt, [s], [3], bs))


# ---- packing ------------------------------------------------------------------------------------------------------
def test_pack_u32_rows_layout_and_errors(expect_error):
    p = pack_u32_rows([torch.arange(5), [7, 8], torch.arange(40)])
    assert p.dtype == torch.int32 and p.shape == (3, 64)
    assert p[0, :5].tolist() == list(range(5)) and not p[0, 5:].any()
    assert p[1, :2].tolist() == [7, 8] and not p[1, 2:].any()
    assert p[2, :40].tolist() == list(range(40)) and not p[2, 40:].any()
    assert pack_u32_rows([[1, 2, 3]]).shape == (1, 32)
    with expect_error(ValueError, "negative"):
        pack_u32_rows([[1, -1]])
    big = 0xFFFFFFF0  # a ring block / uint32 value above int32 max never occurs (ctrl words are small); int32 holds < 2^31
    assert pack_u32_rows([[2**31 - 1]])[0, 0] == 2**31 - 1


@pytest.mark.parametrize("delta", [0, 7])
def test_rope_row_index_gather_equals_host_cos_sin(delta):
    """The in-trace lookup is a row gather of the bf16 table = _rope_tp_cos_sin_decode_torch, bit for bit."""
    S = 4096
    tab_c, tab_s = _rope_table_bf16(S)
    m = _fake_rope_model(S, delta)
    g = torch.Generator().manual_seed(delta)
    pos_sets = [
        torch.arange(0, 33),
        torch.tensor([0, 1, 31, 32, 63, 64, 65, 4000, S - 1 - delta]),
        torch.randint(0, S - delta, (32,), generator=g),
        torch.randint(0, S - delta, (8,), generator=g),
    ]
    for pos in pos_sets:
        pos = pos.to(torch.int32)
        c0, s0 = m._rope_tp_cos_sin_decode_torch(pos)
        idx = rope_row_index(pos, delta, S).to(torch.int64)
        c1 = tab_c[idx].reshape(1, -1, 1, RD)
        s1 = tab_s[idx].reshape(1, -1, 1, RD)
        assert c1.dtype == c0.dtype == torch.bfloat16 and torch.equal(c0, c1) and torch.equal(s0, s1)


def test_rope_row_index_clamps_only_outside_the_table():
    assert rope_row_index([0, 5, 99], 0, 100).tolist() == [0, 5, 99]
    assert rope_row_index([100, 5000], 0, 100).tolist() == [99, 99]  # past max_model_len: junk window rows only
    assert rope_row_index([-1], 0, 100).tolist() == [0]
    assert rope_row_index([3], 4, 100).tolist() == [7]


# ---- drafter: packed inputs = unpacked inputs --------------------------------------------------------------------
def _drafter_engine(B, K, delta=0, S=4096, packed=True, nb=6):
    e = _propose_engine(B, K, None)
    e.T = K + 1
    e.nb, e._scratch_block = nb, 5000
    e.model = _fake_rope_model(S, delta)
    e.model._vfy_hostset = True
    e._dr_hostset = packed
    e._dr_trace = 11
    e._dr_outs = ["o"] * K
    e.mesh = "mesh"
    bufs = {n: _Buf(n) for n in ("pack", "anchor", "tok0", "rs_tok", "rs_pos", "rs_pt", "rs_cos", "rs_sin", "mtp")}
    e._dr_pack, e._anchor_sel, e._dr_tok0 = bufs["pack"], bufs["anchor"], bufs["tok0"]
    e._dr_rs_tok, e._dr_rs_pos, e._dr_rs_pt = bufs["rs_tok"], bufs["rs_pos"], bufs["rs_pt"]
    e._dr_rs_cos, e._dr_rs_sin = bufs["rs_cos"], bufs["rs_sin"]
    e._dr_pos = [_Buf(f"pos{k}") for k in range(K)]
    e._dr_cos = [_Buf(f"cos{k}") for k in range(K)]
    e._dr_sin = [_Buf(f"sin{k}") for k in range(K)]
    e._dr_rs_cache = (
        torch.zeros(B * (K + 1), dtype=torch.bool),
        None,
        torch.full((B * (K + 1), nb), e._scratch_block, dtype=torch.int32),
    )
    e._mtp_pt = bufs["mtp"]
    e.mtp_extra_steps = 0
    e._ids_to_host = lambda outs: [list(range(B)) for _ in outs]
    e._bufs = bufs
    e.staged = []
    return e


def _all_bufs(e):
    return [*e._bufs.values(), *e._dr_pos, *e._dr_cos, *e._dr_sin]


def _step_inputs(g, B, K, nb, declined=(), act=None):
    T = K + 1
    acc = torch.randint(1, T + 1, (B,), generator=g)
    m = (acc - 1).tolist()
    cpos = torch.randint(1, 4000, (B, 1), generator=g) + torch.arange(T)
    ctok = torch.randint(0, 248320, (B, T), generator=g)
    pt = torch.randint(1, 4000, (B, nb), generator=g, dtype=torch.int32)
    p = [int(cpos[u, m[u]]) - 1 for u in range(B)]
    pending = [int(ctok[u, m[u]]) for u in range(B)]
    for u in declined:  # a declined row drafts at position 0 with token 0 over an all-scratch table
        p[u], pending[u] = 0, 0
    keep = [False] * B
    m_kv = [m[u] + int(u in declined and keep[u]) for u in range(B)]
    return ctok, cpos, m, pending, p, pt, m_kv


def _emulate_device_derivation(e, pack):
    """Torch replica of _dr_inputs' documented device ops (slice, typecast, embedding row gathers, reshape)."""
    B, K, R = e.B, e.K, e.B * e.T
    tab_c, tab_s = _rope_table_bf16(e.model.rope.max_seq_len)
    eye = torch.eye(R, dtype=torch.bfloat16)
    row = lambda r, n: pack[r, :n].to(torch.int64)  # noqa: E731  (slice of the uint32 buffer)
    out = dict(
        tok0=row(0, B).reshape(B, 1),
        pos=[row(1 + k, B).to(torch.int32) for k in range(K)],
        rs_tok=row(K + 1, R).reshape(R, 1),
        rs_pos=row(K + 2, R).to(torch.int32),
        anchor_sel=eye[row(K + 3, B)].reshape(1, 1, B, R),
        rs_cos=tab_c[row(K + 4, R)].reshape(1, R, 1, RD),
        rs_sin=tab_s[row(K + 4, R)].reshape(1, R, 1, RD),
        cos=[tab_c[row(K + 5 + k, B)].reshape(1, B, 1, RD) for k in range(K)],
        sin=[tab_s[row(K + 5 + k, B)].reshape(1, B, 1, RD) for k in range(K)],
    )
    return out


@pytest.mark.parametrize("delta", [0, 5])
@pytest.mark.parametrize("B,K", [(1, 3), (2, 3), (3, 3), (5, 3), (8, 3), (4, 7), (2, 11), (1, 1)])
def test_drafter_pack_equals_unpacked_inputs(B, K, delta):
    g = torch.Generator().manual_seed(B * 100 + K + delta)
    nb = 6
    for i in range(50):
        declined = [u for u in range(B) if i % 3 == 1 and u % 2]
        ctok, cpos, m, pending, p, pt, m_kv = _step_inputs(g, B, K, nb, declined)
        e = _drafter_engine(B, K, delta, nb=nb)
        pack, rtok, rpos, real = MTPSpecEngine._dr_pack_host(e, ctok, cpos, m, pending, p, m_kv)
        d = _emulate_device_derivation(e, pack)
        T = K + 1
        # legacy host inputs
        plan = reseed_row_plan(ctok, cpos, m_kv, B, T)
        tok, pos, _ = reseed_rows_host(plan, pt, T, nb, e._scratch_block)
        rcos, rsin = e.model._rope_tp_cos_sin_decode_torch(pos)
        tok0, poss = draft_step_plan(pending, p, K)
        assert torch.equal(d["anchor_sel"], anchor_selector(m, B, T))
        assert d["anchor_sel"].dtype == torch.bfloat16
        assert torch.equal(d["rs_tok"].to(torch.int32), tok) and torch.equal(d["rs_pos"], pos)
        assert torch.equal(d["rs_cos"], rcos) and torch.equal(d["rs_sin"], rsin)
        assert torch.equal(d["tok0"].to(torch.int32), tok0)
        for k in range(K):
            c, s = e.model._rope_tp_cos_sin_decode_torch(poss[k])
            assert torch.equal(d["pos"][k], poss[k])
            assert torch.equal(d["cos"][k], c) and torch.equal(d["sin"][k], s)
        assert pack.shape == (2 * K + 5, 32 * (-(-max(B * T, B) // 32)))


@pytest.mark.parametrize("B", [1, 8])
def test_propose_recorded_packed_vs_unpacked_copies_and_content(B, monkeypatch):
    K, nb = 3, 6
    g = torch.Generator().manual_seed(B)
    counts = {True: [], False: []}
    contents = {True: [], False: []}
    for packed in (False, True):
        e = _drafter_engine(B, K, packed=packed, nb=nb)
        log = []
        _fake_ttnn(monkeypatch, log, e.model)
        e.model._vfy_gdn = []
        pre = []
        e.model.prestage_verify_aux = lambda pos, mi: pre.append((list(pos), list(mi)))
        monkeypatch.setattr(ttnn, "execute_trace", lambda *a, **k: log.append(("trace",)))
        g2 = torch.Generator().manual_seed(B)
        before = 0
        for i in range(40):
            ctok, cpos, m, pending, p, pt, m_kv = _step_inputs(g2, B, K, nb)
            if i % 4:  # steady batch: same page table for most steps
                pt = e._last_pt_keep if hasattr(e, "_last_pt_keep") else pt
            e._last_pt_keep = pt
            e._row_active = [u % 3 != 2 for u in range(B)] if i % 5 == 0 else None
            del log[:]
            e._propose_recorded(ctok, cpos, m, pending, p, pt, m_kv)
            n_copies = sum(1 for x in log if x[0] == "h2d")
            counts[packed].append(n_copies)
            if packed:
                # prestage guess: next start = cpos[u, m[u]] for live rows, mi = m[u]
                act = [True] * B if e._row_active is None else e._row_active
                assert pre[-1] == (
                    [int(cpos[u, m[u]]) if act[u] else 0 for u in range(B)],
                    [m[u] if act[u] else 0 for u in range(B)],
                )
                contents[True].append((e._dr_pack.data.clone(), e._dr_rs_pt.data.clone()))
            else:
                contents[False].append(
                    (
                        e._dr_rs_pt.data.clone(),
                        e._anchor_sel.data.clone(),
                        e._dr_tok0.data.clone(),
                        e._dr_rs_cos.data.clone(),
                    )
                )
    # unpacked: 1 anchor + 5 reseed + 1 tok0 + 3K step inputs (+ the table copy is among the 5 reseed ones) = 16
    assert set(counts[False]) <= {15, 16} and max(counts[False]) == 16  # 15 when the unchanged table copy is skipped
    assert max(counts[True]) <= 2 and min(counts[True]) >= 1  # the pack, plus the reseed table when it changed
    assert sum(counts[True]) < sum(counts[False]) / 6
    # the reseed table the device holds equals the unpacked path's after every step (skipped copies included)
    for (_, rs_new), (rs_old, *_rest) in zip(contents[True], contents[False]):
        assert torch.equal(rs_new, rs_old)


def test_reseed_table_cache_rules():
    """All-padding reseed ignores the page table; a changed table or a changed real-row mask is re-staged."""
    B, K, nb = 2, 3, 6
    e = _drafter_engine(B, K, nb=nb)
    T = K + 1
    copies = []
    e._stage = lambda host, dev, dt, lay: copies.append(host.clone())
    e._row_active = None
    mon = pytest.MonkeyPatch()
    log = []
    _fake_ttnn(mon, log, e.model)
    mon.setattr(ttnn, "execute_trace", lambda *a, **k: None)
    e.model.prestage_verify_aux = lambda *a: None
    try:
        ctok = torch.ones(B, T, dtype=torch.int64)
        cpos = torch.arange(1, T + 1).repeat(B, 1)
        pt = torch.arange(1, B * nb + 1, dtype=torch.int32).reshape(B, nb)
        run = lambda m, table: e._propose_recorded(ctok, cpos, m, [1] * B, [4] * B, table, m)  # noqa: E731
        run([0, 0], pt)
        assert copies == []  # all padding at init: nothing to stage, whatever the table
        run([0, 0], pt + 1)
        assert copies == []
        run([2, 0], pt)
        assert len(copies) == 1
        run([2, 0], pt)
        assert len(copies) == 1  # same mask, same table
        run([2, 0], pt + 1)
        assert len(copies) == 2  # table changed
        run([2, 1], pt + 1)
        assert len(copies) == 3  # mask changed
        run([0, 0], pt + 5)
        assert len(copies) == 4  # back to all padding: the device still holds real rows, so stage the scratch table
        run([0, 0], pt + 9)
        assert len(copies) == 4
    finally:
        mon.undo()


# ---- verify: packed aux = today's kvpos / cos / sin / ctrl -----------------------------------------------------------
def _hostset_vfy_model(B, T, fused=True, Nv=12, S=1024, delta=0, nb=8):
    m, bufs = _vfy_model(B, T, fused=fused, Nv=Nv, nb=nb)
    m.rope = types.SimpleNamespace(rope_delta=delta, max_seq_len=S)
    m._vfy_hostset = True
    m._vfy_aux_buf, m._vfy_aux_staged = _Buf("aux"), None
    m._vfy_ctrl_words = spec_ctrl_words(B, Nv)
    m._vfy_ctrl = None
    bufs["aux"] = m._vfy_aux_buf
    return m, bufs


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("delta", [0, 3])
@pytest.mark.parametrize("B,T", [(1, 4), (2, 4), (4, 4), (8, 4), (4, 8), (2, 12)])
def test_verify_aux_words_equal_legacy_inputs(B, T, fused, delta, monkeypatch):
    Nv, S = 12, 1024
    g = torch.Generator().manual_seed(B * 10 + T + delta)
    tab_c, tab_s = _rope_table_bf16(S)
    legacy = _fake_rope_model(S, delta)
    m, bufs = _hostset_vfy_model(B, T, fused=fused, Nv=Nv, S=S, delta=delta)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    tokens = [[u * 10 + j for j in range(T)] for u in range(B)]
    for i in range(30):
        positions = torch.randint(0, S - T - delta, (B,), generator=g).tolist()
        if i % 4 == 0:
            positions = [0] * B
        mi = torch.randint(0, T, (B,), generator=g).tolist()
        par = m._vfy_gdn[0]._spec_win_par
        del log[:]
        w_before = {n: b.writes for n, b in bufs.items()}
        m.verify_traced(tokens, positions, mi)
        copies = {n: b.writes - w_before[n] for n, b in bufs.items() if b.writes - w_before[n]}
        assert copies.get("aux") == 1 and copies.get("token") == 1
        for gone in ("kvpos", "cos", "sin", "ctrl"):
            assert gone not in copies, f"{gone} is derived in the trace under the packed layout"
        assert set(copies) == {"aux", "token"} | ({"state_idx", "conv_sel"} if not fused else set())
        aux = bufs["aux"].data
        assert bufs["aux"].dtype == ttnn.uint32 and bufs["aux"].layout == ttnn.ROW_MAJOR_LAYOUT
        rows = B * T
        pos_rows = torch.tensor([p + j for p in positions for j in range(T)], dtype=torch.int32)
        # row 0 = kvpos (typecast back to int32 on device), row 1 = RoPE row index -> table gather = legacy cos/sin
        assert torch.equal(aux[0, :rows], pos_rows)
        idx = aux[1, :rows].to(torch.int64)
        c0, s0 = legacy._rope_tp_cos_sin_decode_torch(pos_rows)
        assert torch.equal(tab_c[idx].reshape(1, rows, 1, RD), c0) and torch.equal(
            tab_s[idx].reshape(1, rows, 1, RD), s0
        )
        if fused:
            assert torch.equal(aux[2, : m._vfy_ctrl_words], spec_ctrl_page(mi, B, Nv, parity=par).reshape(-1))
        else:
            assert aux.shape[0] == 2
        assert not aux[0, rows:].any() and not aux[1, rows:].any()


def test_verify_prestage_hit_miss_and_flag_off(monkeypatch):
    B, T = 4, 4
    m, bufs = _hostset_vfy_model(B, T)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    tokens = [[1] * T] * B
    # a step to leave the GDN parity at the value the next verify will carry
    m.verify_traced(tokens, [10, 20, 30, 40], [0, 1, 2, 3])
    assert all(dn._spec_win_par == 1 for dn in m._vfy_gdn)
    w = bufs["aux"].writes
    assert m.prestage_verify_aux([14, 23, 31, 40], [3, 2, 1, 0]) is True and bufs["aux"].writes == w + 1
    # hit: verify_traced finds exactly these words on the device -> no second aux copy
    m.verify_traced(tokens, [14, 23, 31, 40], [3, 2, 1, 0])
    assert bufs["aux"].writes == w + 1
    # miss: prestaged guess differs from the real call -> the real words are staged and are what the device holds
    assert m.prestage_verify_aux([20, 30, 40, 50], [0, 0, 0, 0])
    w = bufs["aux"].writes
    want = m._vfy_aux_host(m._vfy_row_positions([20, 31, 40, 50]), [0, 0, 0, 0])  # at the parity going in
    m.verify_traced(tokens, [20, 31, 40, 50], [0, 0, 0, 0])
    assert bufs["aux"].writes == w + 1
    assert torch.equal(bufs["aux"].data.to(torch.int32), want) and torch.equal(m._vfy_aux_staged, want)
    # the host mirror always equals the device content, and an identical repeat of a non-ctrl word set still copies
    # because the parity word flips every replay
    # invalid predictions are refused without touching the device
    w = bufs["aux"].writes
    assert m.prestage_verify_aux([1, 2, 3], [0, 0, 0]) is False
    assert m.prestage_verify_aux([-1, 2, 3, 4], [0] * 4) is False
    assert m.prestage_verify_aux([1, 2, 3, 4], [0, 0, 0, T]) is False
    assert bufs["aux"].writes == w
    m._vfy_hostset = False
    assert m.prestage_verify_aux([1, 2, 3, 4], [0] * 4) is False


def test_verify_without_packed_layout_is_todays_path(monkeypatch):
    """A model prepared without the packed layout (flag 0 / TRACE 0) stages kvpos, cos, sin, ctrl as before."""
    B, T = 2, 4
    m, bufs = _vfy_model(B, T, fused=True)  # no _vfy_hostset attribute: the legacy model
    log = []
    _fake_ttnn(monkeypatch, log, m)
    m.verify_traced([[1] * T] * B, [5, 9], [0, 1])
    assert [bufs[n].writes for n in ("token", "kvpos", "cos", "sin", "ctrl")] == [1] * 5
    assert not m.prestage_verify_aux([5, 9], [0, 1])


def test_row_positions_vectorized_equals_loop(flag):
    m, _ = _vfy_model(8, 4, fused=True)
    g = torch.Generator().manual_seed(3)
    for _ in range(30):
        pos = torch.randint(0, 262000, (8,), generator=g)
        want = torch.tensor([int(pos[u]) + j for u in range(8) for j in range(4)], dtype=torch.int32)
        for p in (pos, pos.tolist()):
            got = m._vfy_row_positions(p)
            assert got.dtype == torch.int32 and torch.equal(got, want)


# ---- narrow step: column 0 only ----------------------------------------------------------------------------------
@pytest.mark.parametrize("B,T", [(1, 4), (4, 4), (8, 4), (2, 12)])
def test_narrow_read_equals_column_zero_of_the_full_read(B, T, monkeypatch):
    V = 64
    g = torch.Generator().manual_seed(B + T)
    m, bufs = _vfy_model(B, T, fused=True)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    full = torch.randn(1, 1, 32, V, generator=g).to(torch.bfloat16)
    bufs_logits = m._vfy_logits_rm_out
    bufs_logits.data = full
    m._vfy_narrow_logits = lambda x: _Buf("sl")
    sl = _Buf("sl")
    sl.data = x = full.reshape(-1, V)[: B * T].reshape(B, T, V)[:, :1].contiguous()
    m._vfy_narrow_logits = lambda x_: sl
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    tokens = [[1] * T] * B
    pos = list(range(5, 5 + B))
    _, _, lt_full = m.verify_traced(tokens, pos, [0] * B, read_logits=True)
    assert m._vfy_last_col0 is False and tuple(lt_full.shape) == (B * T, V)
    m._vfy_narrow_ready = False
    _, _, lt = m.verify_traced(tokens, pos, [0] * B, read_logits=True, read_col0=True)
    assert m._vfy_last_col0 is False and torch.equal(lt, lt_full)  # not ready: the full read
    m._vfy_narrow_ready = True
    _, _, lt = m.verify_traced(tokens, pos, [0] * B, read_logits=True, read_col0=True)
    assert m._vfy_last_col0 is True and tuple(lt.shape) == (B, V)
    assert torch.equal(lt, lt_full.reshape(B, T, V)[:, 0])
    _, _, lt = m.verify_traced(tokens, pos, [0] * B, read_logits=True)  # a plain logits read resets the marker
    assert m._vfy_last_col0 is False and tuple(lt.shape) == (B * T, V)


def test_narrow_slice_geometry_matches_torch():
    """The device slice is reshape [1,1,B*T,V] -> [B,T,V], slice [0:B, 0:1, :]: rows u*T."""
    B, T, V = 8, 4, 16
    x = torch.arange(B * T * V, dtype=torch.float32).reshape(1, 1, B * T, V)
    got = x.reshape(B, T, V)[0:B, 0:1, :]
    assert torch.equal(got.reshape(B, V), x.reshape(B * T, V)[::T])


def test_engine_verify_returns_b1v_for_narrow_and_btv_otherwise(monkeypatch):
    """Engine plumbing: narrow=True asks the model for column 0 and reshapes to [B,1,V]; any other call is unchanged."""
    monkeypatch.setenv("QWEN36_SPEC_HOSTSET", "1")
    B, K, V = 2, 3, 8
    T = K + 1
    e = _propose_engine(B, K, None)
    e.captured, e.fast_join, e.host_slotmap = True, True, True
    e._timing = False
    e._active, e._fresh, e._parked = {0, 1}, {}, {}
    e._gdn = []
    e.vocab = V
    calls = []

    def verify_traced(tokens, positions, mi_prev, **kw):
        calls.append(kw)
        col0 = kw.get("read_col0", False)
        e.model._vfy_last_col0 = col0
        lt = torch.arange((B if col0 else B * T) * V, dtype=torch.float32).reshape(-1, V)
        return list(range(B * T)), "feed", lt

    e.model = types.SimpleNamespace(verify_traced=verify_traced, num_devices=8)
    e._ints = types.MethodType(MTPSpecEngine._ints, e)
    e._fit = lambda pt: torch.as_tensor(pt).to(torch.int32)
    e.nb = 4
    tokens = torch.zeros(B, T, dtype=torch.int64)
    pos = torch.tensor([3, 9])
    pt = torch.ones(B, 4, dtype=torch.int32)
    ids, lg = MTPSpecEngine.verify(e, tokens, pos, [1, 1], pt, read_logits=True, narrow=True)
    assert calls[-1].get("read_col0") is True and tuple(lg.shape) == (B, 1, V) and tuple(ids.shape) == (B, T)
    ids, lg = MTPSpecEngine.verify(e, tokens, pos, [1, 1], pt, read_logits=True)
    assert "read_col0" not in calls[-1] and tuple(lg.shape) == (B, T, V)
    monkeypatch.setenv("QWEN36_SPEC_HOSTSET", "0")
    ids, lg = MTPSpecEngine.verify(e, tokens, pos, [1, 1], pt, read_logits=True, narrow=True)
    assert "read_col0" not in calls[-1] and tuple(lg.shape) == (B, T, V)
