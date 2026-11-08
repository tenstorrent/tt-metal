# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for the per-request M-RoPE delta on the MTP spec path (step 1b): every verify / draft / join host
index builder takes the row's OWN delta instead of the scalar ``rope.rope_delta`` (the last prefill's); the prefill routes
record the delta per engine row; fused / chain steps carry one delta word PER ROW (fused aux row 6, chain overlay last row) and
run on mixed deltas (QWEN36_SPEC_ROW_DELTA=1, default); with the flag 0 they take the common delta of the live rows and
decline on mixed deltas.
Text-only batches are compared byte for byte with a HEAD export of the same functions (skipped when the export is absent:
``git archive HEAD models/demos/blackhole/qwen36 | tar -x -C $QWEN36_HEAD_EXPORT``).

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_spec_rope_delta_host.py -q
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine_host import _ChainRig, _Flow
from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import (
    VOCAB,
    _dispatch_engine,
    _engine,
    _row,
    fake_ttnn,
)
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    _block,
    _expected_commit,
    _ids_for,
    _pt,
    _Rig,
)
from models.demos.blackhole.qwen36.tests.test_spec_hostset_host import (
    _drafter_engine,
    _fake_rope_model,
    _hostset_vfy_model,
    _step_inputs,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_chain import (
    OVR_DELTA,
    OVR_ROWS,
    chain_reference,
    pack_ovr_host,
    res_c_reference,
)
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine, _row_vision_of, join_mtp_warm_host
from models.demos.blackhole.qwen36.tt.spec_fused import (
    AUX_CONSTS,
    AUX_DELTA,
    CONST_ROPE_DELTA,
    FUSED_AUX_ROWS,
    aux_row_deltas,
    fused_accept_aux_host,
    fused_greedy_reference,
    signed_rope_delta,
)
from models.demos.blackhole.qwen36.tt.tp_common import rope_row_index

__all__ = ["fake_ttnn"]  # fixture re-export

HEAD_ROOT = os.environ.get("QWEN36_HEAD_EXPORT", "/home/ttuser/atupe/qwen-or/work/scratch/mmmtp/head")
VIS = "VISION_ROWS"


def _head(rel, name):
    path = os.path.join(HEAD_ROOT, "models/demos/blackhole/qwen36/tt", rel)
    if not os.path.exists(path):
        pytest.skip(f"no HEAD export at {HEAD_ROOT}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _with_row_deltas(m, deltas):
    """Give a fake model (SimpleNamespace or Qwen36Model.__new__ object) the per-engine-row delta vector and its readers."""
    m._slot_rope_delta = torch.tensor(deltas, dtype=torch.int64)
    for name in ("_spec_row_deltas", "_spec_pos_deltas"):
        setattr(m, name, types.MethodType(getattr(Qwen36Model, name), m))
    if not hasattr(m, "_vfy_T"):
        m._vfy_T = 1
    return m


# --------------------------------------------------------------------------- #
# rope_row_index
# --------------------------------------------------------------------------- #
def test_rope_row_index_scalar_and_per_row(expect_error):
    pos = torch.tensor([10, 20, 30, 40])
    scalar = rope_row_index(pos, 5, 1000)
    assert scalar.tolist() == [15, 25, 35, 45] and scalar.dtype == torch.int32
    per = rope_row_index(pos, [0, 5, -3, 7], 1000)
    assert per.tolist() == [10, 25, 27, 47] and per.dtype == torch.int32
    assert torch.equal(rope_row_index(pos, torch.tensor([5, 5, 5, 5]), 1000), scalar)
    assert rope_row_index([1, 2], [-5, 2000], 100).tolist() == [0, 99]  # clamped as before
    with expect_error(ValueError, "3 deltas for 4 positions"):
        rope_row_index(pos, [1, 2, 3], 1000)


def test_rope_row_index_scalar_equals_head():
    head = _head("tp_common.py", "head_tp_common")
    pos = torch.randint(0, 5000, (64,), generator=torch.Generator().manual_seed(1))
    for d in (0, 7, -9):
        assert torch.equal(rope_row_index(pos, d, 4096), head.rope_row_index(pos, d, 4096))


# --------------------------------------------------------------------------- #
# model builders
# --------------------------------------------------------------------------- #
def test_spec_row_deltas_fallback_and_padding():
    m = types.SimpleNamespace(rope=types.SimpleNamespace(rope_delta=4))
    assert Qwen36Model._spec_row_deltas(m, 3) == [4, 4, 4]  # no per-row vector (a fake): the scalar
    _with_row_deltas(m, [0, 9])
    assert Qwen36Model._spec_row_deltas(m, 3) == [0, 9, 0]  # rows past the vector: 0
    m._vfy_T = 3
    assert Qwen36Model._spec_pos_deltas(m, 6) == [0, 0, 0, 9, 9, 9]


@pytest.mark.parametrize("fused", [True, False])
def test_vfy_aux_host_rope_delta_none_is_per_row(fused):
    B, T = 3, 4
    m, _ = _hostset_vfy_model(B, T, fused=fused, S=2048)
    _with_row_deltas(m, [0, 7, -2])
    rows = m._vfy_row_positions([100, 200, 300])
    base = m._vfy_aux_host(rows, [0] * B)
    assert torch.equal(m._vfy_aux_host(rows, [0] * B, rope_delta=None), base)
    want = rope_row_index(rows, [0] * T + [7] * T + [-2] * T, 2048)
    assert torch.equal(base[1, : B * T].to(torch.int64), want.to(torch.int64))


@pytest.mark.parametrize("fused", [True, False])
def test_vfy_aux_host_scalar_delta_applies_to_every_row(fused):
    B, T, d = 3, 4, -300
    m, _ = _hostset_vfy_model(B, T, fused=fused, S=2048)
    _with_row_deltas(m, [0, 0, d])
    rows = m._vfy_row_positions([0, 0, 341])
    aux = m._vfy_aux_host(rows, [0] * B, rope_delta=d)
    got = aux[1, : B * T].tolist()
    assert got == [0] * 8 + [41, 42, 43, 44]
    assert aux[0, : B * T].tolist() == [0, 1, 2, 3] * 2 + [341, 342, 343, 344]
    assert m._vfy_aux_host(rows, [0] * B)[1, : B * T].tolist() == [0, 1, 2, 3] * 2 + [41, 42, 43, 44]  # per-row rule


@pytest.mark.parametrize("fused", [True, False])
def test_vfy_aux_host_scalar_zero_equals_text_only(fused):
    B, T = 3, 4
    m, _ = _hostset_vfy_model(B, T, fused=fused, S=2048)
    _with_row_deltas(m, [0, 0, 0])
    rows = m._vfy_row_positions([10, 20, 30])
    assert torch.equal(m._vfy_aux_host(rows, [0] * B, rope_delta=0), m._vfy_aux_host(rows, [0] * B))


def test_signed_rope_delta():
    assert signed_rope_delta(0) == 0
    assert signed_rope_delta(5) == 5
    assert signed_rope_delta((-300) & 0xFFFFFFFF) == -300
    assert signed_rope_delta(2**31 - 1) == 2**31 - 1


def test_chain_compare_accepts_scalar_delta_rule_for_inactive_rows(expect_error):
    """Only an image row (delta -300) is live: the device writes rope row clamp(pos + scalar delta) for the inactive rows
    too; the audit's expectation uses the same scalar, so no false failure."""
    B, T, d = 3, 4, -300
    m, _ = _hostset_vfy_model(B, T, fused=False, S=2048)
    _with_row_deltas(m, [0, 0, d])
    e = _engine(B, model=m)
    e.T = T
    rows = m._vfy_row_positions([0, 0, 341])
    dev = torch.zeros_like(m._vfy_aux_host(rows, [0] * B))  # padded to the packed width
    dev[0, : B * T], dev[1, : B * T] = rows, torch.tensor([0] * 8 + [41, 42, 43, 44], dtype=torch.int32)  # device rule
    tok = torch.arange(B * T, dtype=torch.int64)
    fa = torch.zeros(FUSED_AUX_ROWS, 8, dtype=torch.int64)
    expect = (tok, m._vfy_aux_host(rows, [0] * B, rope_delta=d), fa)
    e._chain_compare(expect, tok, dev, fa)
    stale = (tok, m._vfy_aux_host(rows, [0] * B), fa)  # the old per-row expectation fails
    with expect_error(RuntimeError, "rope row"):
        e._chain_compare(stale, tok, dev, fa)


@pytest.mark.parametrize("fused", [True, False])
def test_vfy_aux_host_uses_each_rows_delta(fused):
    B, T = 3, 4
    m, _ = _hostset_vfy_model(B, T, fused=fused, S=2048)
    _with_row_deltas(m, [0, 7, -2])
    rows = m._vfy_row_positions([100, 200, 300])
    aux = m._vfy_aux_host(rows, [0] * B)
    want = rows + torch.tensor([0] * T + [7] * T + [-2] * T, dtype=torch.int32)
    assert torch.equal(aux[0, : B * T], rows) and torch.equal(aux[1, : B * T], want)
    # the scalar the old code used would give one delta for every row
    m.rope.rope_delta = 7
    assert not torch.equal(aux[1, : B * T], rows + 7)


@pytest.mark.parametrize("fused", [True, False])
def test_vfy_aux_host_text_equals_head(fused):
    head = _head("model.py", "head_model")
    B, T = 4, 4
    m, _ = _hostset_vfy_model(B, T, fused=fused, S=2048)
    rows = m._vfy_row_positions([10, 20, 30, 40])
    want = head.Qwen36Model._vfy_aux_host(m, rows, [0, 1, 2, 3])  # HEAD code, scalar rope.rope_delta = 0
    _with_row_deltas(m, [0] * B)
    assert torch.equal(m._vfy_aux_host(rows, [0, 1, 2, 3]), want)
    got_fallback = Qwen36Model._vfy_aux_host(_hostset_vfy_model(B, T, fused=fused, S=2048)[0], rows, [0, 1, 2, 3])
    assert torch.equal(got_fallback, want)  # a model without the vector: the scalar, as HEAD


def test_cos_sin_decode_per_row_and_cached():
    m = _with_row_deltas(_fake_rope_model(512, rope_delta=99), [0, 6, 11])  # stale scalar must not matter
    m._rope_tp_cos_sin_decode_cached = types.MethodType(Qwen36Model._rope_tp_cos_sin_decode_cached, m)
    pos = torch.tensor([40, 50, 60], dtype=torch.int32)
    d = [0, 6, 11]
    c, s = m._rope_tp_cos_sin_decode_torch(pos, d)
    ref = _fake_rope_model(512, 0)
    for i in range(3):
        c1, s1 = ref._rope_tp_cos_sin_decode_torch(pos[i : i + 1] + d[i])
        assert torch.equal(c[0, i], c1[0, 0]) and torch.equal(s[0, i], s1[0, 0])
    cc, sc = m._rope_tp_cos_sin_decode_cached(pos, d)  # the cached table is keyed by the final RoPE position
    assert torch.equal(cc, c) and torch.equal(sc, s)
    c0, _ = m._rope_tp_cos_sin_decode_torch(pos)  # no deltas: the scalar, as before
    c99, _ = ref._rope_tp_cos_sin_decode_torch(pos + 99)
    assert torch.equal(c0, c99)


# --------------------------------------------------------------------------- #
# engine builders
# --------------------------------------------------------------------------- #
def test_join_mtp_warm_host_per_row_delta_and_scalar_equals_head():
    head = _head("spec_engine.py", "head_spec_engine")
    B, NB, SCR = 8, 4, 999
    tokens = torch.arange(B * 4).reshape(B, 4)
    pt = torch.randint(1, 500, (B, NB), dtype=torch.int32)
    fresh, phys = {1: 40, 6: 90}, {1: 4, 6: 0}
    for d in (0, 7):
        a, ta = join_mtp_warm_host(fresh, tokens, pt, phys, B, SCR, d, 1000)
        b, tb = head.join_mtp_warm_host(fresh, tokens, pt, phys, B, SCR, d, 1000)
        assert torch.equal(a, b) and torch.equal(ta, tb)
    deltas = [3, 11, 0, 0, 0, 0, 22, 0]  # per engine row
    pack, _ = join_mtp_warm_host(fresh, tokens, pt, phys, B, SCR, deltas, 1000)
    # fresh engine row 1 sits in physical row 4 (position 39), engine row 6 in physical row 0 (position 89)
    assert pack[1, 4] == 39 and pack[2, 4] == 39 + 11 and pack[1, 0] == 89 and pack[2, 0] == 89 + 22
    assert [int(pack[2, p]) for p in (1, 2, 3, 5, 6, 7)] == [0] * 6  # padding rows: position 0, delta 0


def test_dr_pack_host_per_row_delta_and_text_equals_head():
    head = _head("spec_engine.py", "head_spec_engine")
    B, K, nb = 3, 3, 6
    g = torch.Generator().manual_seed(3)
    ctok, cpos, m, pending, p, pt, m_kv = _step_inputs(g, B, K, nb, [])
    e = _drafter_engine(B, K, 0, nb=nb)
    want = head.MTPSpecEngine._dr_pack_host(e, ctok, cpos, m, pending, p, m_kv)[0]
    _with_row_deltas(e.model, [0] * B)
    assert torch.equal(MTPSpecEngine._dr_pack_host(e, ctok, cpos, m, pending, p, m_kv)[0], want)
    d = [0, 5, -4]
    _with_row_deltas(e.model, d)
    e.model.rope.rope_delta = 77  # stale scalar
    pack, _, rpos, _ = MTPSpecEngine._dr_pack_host(e, ctok, cpos, m, pending, p, m_kv)
    T = K + 1
    for r in range(B * T):  # reseed rows: user-major
        assert int(pack[K + 4, r]) == max(0, int(rpos[r]) + d[r // T])  # clamped into the table
    for k in range(K):  # step k of row u: p[u] + k + delta[u]
        for u in range(B):
            assert int(pack[K + 5 + k, u]) == max(0, int(p[u]) + k + d[u])
    # rows that equal in delta keep the equal index of the scalar form
    _with_row_deltas(e.model, [5] * B)
    e.model.rope.rope_delta = 5
    pack5 = MTPSpecEngine._dr_pack_host(e, ctok, cpos, m, pending, p, m_kv)[0]
    e.model = _fake_rope_model(4096, 5)  # a fake without the vector: the scalar path
    assert torch.equal(MTPSpecEngine._dr_pack_host(e, ctok, cpos, m, pending, p, m_kv)[0], pack5)


def test_warm_mtp_last_and_reseed_pass_the_rows_delta(monkeypatch):
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    calls = []
    model = types.SimpleNamespace(rope=types.SimpleNamespace(rope_delta=99, max_seq_len=64))
    _with_row_deltas(model, [0, 13, 0])
    model.ttnn_mtp_decode_forward = lambda *a, **k: calls.append(k) or ("L", "H")
    e = _engine(3, model=model)
    MTPSpecEngine._warm_mtp_last(e, "h", 5, 10, "pt", row=1)
    MTPSpecEngine._warm_mtp_last(e, "h", 5, 10, "pt")
    assert calls[0]["rope_deltas"] == [13] and calls[1]["rope_deltas"] == [0]
    # a model without the vector (fake): the call is exactly what it was, no new keyword
    model2 = types.SimpleNamespace(rope=types.SimpleNamespace(rope_delta=4))
    model2.ttnn_mtp_decode_forward = lambda *a, **k: calls.append(("plain", k)) or ("L", "H")
    MTPSpecEngine._warm_mtp_last(_engine(3, model=model2), "h", 5, 10, "pt", row=1)
    assert calls[2] == ("plain", {"need_logits": False})


# --------------------------------------------------------------------------- #
# prefill routes record the delta; re-prefill re-supplies the vision rows
# --------------------------------------------------------------------------- #
def _mm_engine(B=4, calls=None):
    calls = [] if calls is None else calls
    e = _dispatch_engine(B, fast=True, recorded=True, calls=calls)
    model = e.model
    model.rope = types.SimpleNamespace(rope_delta=0, max_seq_len=1024)
    _with_row_deltas(model, [0] * B)
    model.set_slot_rope_delta = lambda slot, d: model._slot_rope_delta.__setitem__(int(slot), int(d))
    seen = []

    def _fast(slot, pl, prompt, pt_row, Tp, on_chunk, captured, **kw):
        seen.append((slot, kw.get("vision_tokens")))
        # what the model's _build_request_rope leaves: the grid's delta for a multimodal prompt, 0 for text
        model.rope.rope_delta = 7 if kw.get("vision_tokens") is not None else 0
        captured[0] = True
        return torch.arange(VOCAB, dtype=torch.float32)

    e._prefill_fast = _fast
    e._row_gen = None
    return e, model, seen


def test_prefill_records_row_delta_and_text_prefill_keeps_it(fake_ttnn):
    e, model, seen = _mm_engine()
    mm = list(range(1, 40))
    MTPSpecEngine._prefill_row(e, 1, mm, _row(), None, vision_tokens=VIS)  # multimodal row
    assert model._slot_rope_delta.tolist() == [0, 7, 0, 0]
    MTPSpecEngine._prefill_row(e, 2, list(range(1, 30)), _row())  # a text prefill resets the scalar to 0 ...
    assert model.rope.rope_delta == 0 and model._slot_rope_delta.tolist() == [0, 7, 0, 0]  # ... not the mm row's
    assert model._spec_row_deltas(4) == [0, 7, 0, 0]
    # a text prefill onto the mm row itself clears it; release too
    MTPSpecEngine._prefill_row(e, 1, list(range(1, 30)), _row())
    assert model._slot_rope_delta.tolist() == [0, 0, 0, 0] and 1 not in _row_vision_of(e)
    MTPSpecEngine._prefill_row(e, 3, mm, _row(), None, vision_tokens=VIS)
    assert model._slot_rope_delta[3] == 7
    e._phys = list(range(4))
    e._row_gen = [0] * 4
    MTPSpecEngine.release(e, 3)
    assert model._slot_rope_delta.tolist() == [0, 0, 0, 0] and 3 not in _row_vision_of(e)


def test_reprefill_and_reconcile_pass_the_stored_vision_tokens(fake_ttnn):
    e, model, seen = _mm_engine()
    mm = list(range(1, 60))
    MTPSpecEngine._prefill_row(e, 2, mm, _row(), None, vision_tokens=VIS)
    assert _row_vision_of(e) == {2: VIS} and seen == [(2, VIS)]
    model.rope.rope_delta = 0  # another request's prefill moved the scalar meanwhile
    MTPSpecEngine._reprefill(e, 2)
    assert seen[-1] == (2, VIS) and model._slot_rope_delta[2] == 7  # recorded again, vision rows passed, start_pos None
    # first verify at a shorter position: the reconcile re-prefills with the stored rows too
    e._fresh = {2: len(mm)}
    out = MTPSpecEngine._reconcile_fresh_rows(e, {2: len(mm)}, [0, 0, len(mm) - 3, 0])
    assert seen[-1] == (2, VIS) and out[2] == len(mm) - 3 and model._slot_rope_delta[2] == 7
    # a text row re-prefills as text
    MTPSpecEngine._prefill_row(e, 0, list(range(1, 20)), _row())
    MTPSpecEngine._reprefill(e, 0)
    assert seen[-1] == (0, None)
    # the vision rows leave with the request / the join
    e._phys = list(range(4))
    MTPSpecEngine.release(e, 2)
    assert _row_vision_of(e) == {}


@pytest.mark.parametrize("host_slotmap", [True, False])
def test_remap_moves_deltas_only_without_the_host_slot_map(host_slotmap):
    B = 4
    model = types.SimpleNamespace(rope=types.SimpleNamespace(rope_delta=0))
    _with_row_deltas(model, [0, 7, 0, 9])
    model.remap_slot_rope_delta = types.MethodType(Qwen36Model.remap_slot_rope_delta, model)
    e = _engine(B, model=model)
    e.host_slotmap, e._gdn, e._row_vision_map = host_slotmap, [], {1: "V1", 3: "V3"}
    e._offered = set()
    e._set_slot_map = lambda m: setattr(e, "_phys", list(m))
    e._last_hidden = [f"h{u}" for u in range(B)]
    remap = [1, 0, 3, 2]
    MTPSpecEngine.remap_slots(e, remap)
    if host_slotmap:  # the rows never move: only the runner -> engine map changes
        assert model._slot_rope_delta.tolist() == [0, 7, 0, 9] and _row_vision_of(e) == {1: "V1", 3: "V3"}
    else:  # device remap: row i takes the state (and the delta, and the vision rows) of row remap[i]
        assert model._slot_rope_delta.tolist() == [7, 0, 9, 0]
        assert _row_vision_of(e) == {0: "V1", 2: "V3"}


# --------------------------------------------------------------------------- #
# fused / chain
# --------------------------------------------------------------------------- #
def _consts(aux):
    return int(aux[AUX_CONSTS, CONST_ROPE_DELTA]) & 0xFFFFFFFF


def _set_deltas(r, deltas, scalar=None):
    _with_row_deltas(r.m, deltas)
    r.m.rope.rope_delta = deltas[0] if scalar is None else scalar


def _row6(aux, B):
    return [int(w) & 0xFFFFFFFF for w in aux[AUX_DELTA, :B]]


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("delta", [0, 4, -3])
def test_fused_common_delta_word_and_text_equals_head(monkeypatch, delta, flag):
    monkeypatch.setenv("QWEN36_SPEC_ROW_DELTA", flag)
    B = 3
    r = _Rig(monkeypatch, B)
    _set_deltas(r, [delta] * B)
    tok, pos, nvd = _block(B, [3, 2, 1])
    ids = _ids_for(tok, nvd, [2] * B)
    r.set_ids(ids)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    assert r.e.fused_steps == 1 and not r.e.__dict__.get("fused_fallbacks")
    assert _consts(r.fa.aux_in.data) == delta & 0xFFFFFFFF  # the common delta of the live rows
    assert _row6(r.fa.aux_in.data, B) == [delta & 0xFFFFFFFF] * B  # every row's own word (all equal here)
    assert tuple(r.fa.aux_in.data.shape)[0] == FUSED_AUX_ROWS == 7
    committed, counts = _expected_commit(ids, tok, nvd)
    assert torch.equal(res.accepted_token_ids, committed)
    if delta == 0:  # text-only: the very words HEAD builds
        head = _head("spec_engine.py", "head_spec_engine")
        e = r.e
        nv = nvd.to(torch.int64)
        active = pos[:, 0] >= 0
        reason, aux = e._fused_ineligible(pos[:, 0].clamp(min=0), nv, active)
        h_reason, h_aux = head.MTPSpecEngine._fused_ineligible(e, pos[:, 0].clamp(min=0), nv, active)
        assert (
            reason is None and h_reason is None and torch.equal(aux, h_aux)
        )  # HEAD engine code, same spec_fused module
        assert not aux[AUX_DELTA].any()


@pytest.mark.parametrize("flag", ["0", "1"])
def test_fused_mixed_deltas_flag_declines_or_runs_per_row(monkeypatch, flag):
    monkeypatch.setenv("QWEN36_SPEC_ROW_DELTA", flag)
    B = 3
    r = _Rig(monkeypatch, B)
    _set_deltas(r, [4, 4, 9])
    tok, pos, nvd = _block(B, [3, 2, 1])
    ids = _ids_for(tok, nvd, [2] * B)
    r.set_ids(ids)
    res = r.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    committed, counts = _expected_commit(ids, tok, nvd)  # the host walk and the device accept give the same commit
    assert torch.equal(res.accepted_token_ids, committed) and torch.equal(res.accepted_counts, counts)
    if flag == "0":  # today: declined, counted, the plain verify
        assert r.e.__dict__["fused_fallbacks"] == {"mixed_rope_delta": 1}
        assert ("trace", 12) not in r.log and ("trace", 11) not in r.log  # no accept / drafter trace
        assert r.e.__dict__.get("row_delta_mixed_steps", 0) == 0
    else:  # per-row words: the step runs fused, counted as a mixed-delta step, the counter key stays 0
        assert not r.e.__dict__.get("fused_fallbacks") and r.e.fused_steps == 1
        assert r.e.row_delta_mixed_steps == 1
        aux = r.fa.aux_in.data
        assert _row6(aux, B) == [4, 4, 9] and _consts(aux) == 0  # row 5 col 1: no common delta
        assert aux_row_deltas(aux, B) == [4, 4, 9]
    # row 2 leaves the step (padding row): the live rows agree again and the step runs fused with their common delta
    r2 = _Rig(monkeypatch, B)
    _set_deltas(r2, [4, 4, 9])
    tok, pos, nvd = _block(B, [3, 2, -1])
    r2.set_ids(_ids_for(tok, nvd, [2] * B))
    r2.step(tok, pos, nvd, torch.ones(B, dtype=torch.int32), _pt(B))
    assert not r2.e.__dict__.get("fused_fallbacks") and _consts(r2.fa.aux_in.data) == 4
    assert r2.e.__dict__.get("row_delta_mixed_steps", 0) == 0
    # flag 1 keeps every row's own word (the parked row too); flag 0 spreads the common delta over all rows
    assert _row6(r2.fa.aux_in.data, B) == ([4, 4, 9] if flag == "1" else [4, 4, 4])


def test_fused_row6_words_and_reference_ridx_b8():
    """B=8, deltas [0, -1234, 0, 0, 77, 0, 0, 0]: row 6 holds each row's two's complement word, row 5 col 1 the common (0 for
    different rows), the references give clamp(pos + delta_row) for every drafter-pack ridx position, and a scalar still
    means 'all rows the same'."""
    B, K, rows, T = 8, 3, 500, 4
    deltas = [0, -1234, 0, 0, 77, 0, 0, 0]
    nv = [3, 2, 1, 0, 3, 3, 2, 1]
    start0 = [10, 20, 30, 40, 50, 60, 70, 80]
    act = [1] * B
    aux = fused_accept_aux_host(nv, start0, act, [False] * B, 10**6, K, 5000, deltas, rows)
    assert tuple(aux.shape)[0] == FUSED_AUX_ROWS
    assert [int(w) & 0xFFFFFFFF for w in aux[AUX_DELTA, :B]] == [d & 0xFFFFFFFF for d in deltas]
    assert int(aux[AUX_DELTA, 1]) == -1234 and not aux[AUX_DELTA, B:].any()  # columns B.. are 0
    assert int(aux[AUX_CONSTS, CONST_ROPE_DELTA]) == 0  # rows differ: no common delta
    assert aux_row_deltas(aux, B) == deltas
    # an explicit common delta (the live rows' common one) goes to row 5 col 1 only
    aux_c = fused_accept_aux_host(nv, start0, act, [False] * B, 10**6, K, 5000, deltas, rows, common_delta=-7)
    assert signed_rope_delta(aux_c[AUX_CONSTS, CONST_ROPE_DELTA]) == -7 and torch.equal(
        aux_c[AUX_DELTA], aux[AUX_DELTA]
    )
    # reference: ids make every draft accepted for row 0, rejected at once for the others
    toks = torch.arange(1, B * T + 1).reshape(B, T)
    ids = toks.clone()
    ids[:, :K] = toks[:, 1:]  # every valid draft matches: m = nv
    pt = torch.arange(B * 4, dtype=torch.int32).reshape(B, 4)
    out = fused_greedy_reference(ids.reshape(-1), toks.reshape(-1), aux, pt, B=B, K=K)
    pack = out.pack.to(torch.int64)
    R = B * T
    # pack rows: [pending, steps*K, rtok, rpos, anchor, ridx(R), ridx steps*K]
    rpos, ridx_r = pack[K + 2, :R], pack[K + 4, :R]
    steps_pos = pack[1 : 1 + K, :B]  # [K, B]
    ridx_s = pack[K + 5 :, :B]  # [K, B]
    dl = torch.tensor(deltas)
    want_r = (rpos.reshape(B, T) + dl[:, None]).clamp(0, rows - 1).reshape(-1)
    assert torch.equal(ridx_r, want_r)
    assert torch.equal(ridx_s, (steps_pos + dl[None, :]).clamp(0, rows - 1))
    assert int(ridx_r.reshape(B, T)[1].max()) == 0 and int(ridx_s[:, 1].max()) == 0  # row 1: clamped at 0
    # a scalar delta still means "all rows the same"
    aux_s = fused_accept_aux_host(nv, start0, act, [False] * B, 10**6, K, 5000, -9, rows)
    assert aux_row_deltas(aux_s, B) == [-9] * B and signed_rope_delta(aux_s[AUX_CONSTS, CONST_ROPE_DELTA]) == -9


def test_chain_ovr_last_row_and_reference_per_row_ridx():
    B, K, Nv, rows, T = 4, 3, 2, 400, 4
    deltas = [0, -300, 5, 0]
    committed = torch.tensor([[11, 12, -1, -1], [21, -1, -1, -1], [31, 32, 33, 34], [41, 42, 43, -1]])
    counts = [2, 1, 4, 3]
    p0_prev = [10, 20, 30, 40]
    res = res_c_reference(committed, counts, p0_prev, [0] * B)
    ovr = pack_ovr_host(
        [3] * B,
        [1] * B,
        [0] * B,
        [0] * B,
        p0_prev,
        [0] * B,
        None,
        [0] * B,
        0,
        K=K,
        scratch_block=5000,
        rope_delta=deltas,
        rope_rows=rows,
    )
    assert tuple(ovr.shape)[0] == OVR_ROWS == 11 and OVR_DELTA == OVR_ROWS - 1
    assert [int(w) & 0xFFFFFFFF for w in ovr[OVR_DELTA, :B]] == [d & 0xFFFFFFFF for d in deltas]
    assert int(ovr[AUX_CONSTS, CONST_ROPE_DELTA]) == 0
    drafts = torch.arange(B * K).reshape(B, K)
    tok, vaux, nxt = chain_reference(res, drafts, None, ovr, B=B, K=K, Nv=Nv, rope_rows=rows, ctrl_words=64)
    pos = vaux[0, : B * T].to(torch.int64).reshape(B, T)
    ridx = vaux[1, : B * T].to(torch.int64).reshape(B, T)
    assert torch.equal(ridx, (pos + torch.tensor(deltas)[:, None]).clamp(0, rows - 1))
    assert tuple(nxt.shape)[0] == FUSED_AUX_ROWS and aux_row_deltas(nxt, B) == deltas  # copied into the next aux row 6


@pytest.mark.parametrize("flag", ["0", "1"])
def test_chain_overlay_word_is_the_common_delta_and_mixed_chains_only_with_the_flag(monkeypatch, flag):
    monkeypatch.setenv("QWEN36_SPEC_ROW_DELTA", flag)
    B, d = 3, 5
    r = _ChainRig(monkeypatch, B)
    _set_deltas(r, [d] * B)  # the fake chain trace reads the overlay: the rows' common delta
    f = _Flow(r, B, [3, 2, 1])
    f.step([3, 1, 0])
    f.propose()
    f.next_block()
    f.step([1, 1, 1])  # chained step: overlay + audit (device-built words == _vfy_aux_host with the per-row deltas)
    assert r.e.chain_steps == 1 and r.e.chain_audits == 1 and r.e.chain_breaks == {"first": 1}
    ovr = r.e._ch_g.ovr_in.data
    assert [int(w) for w in ovr[OVR_DELTA, :B]] == [d] * B and int(ovr[AUX_CONSTS, CONST_ROPE_DELTA]) == d
    f.propose()
    f.next_block()
    _with_row_deltas(r.m, [d, d, d + 2])  # a mixed batch (a prefill gave row 2 another delta)
    f.step([1, 1, 1])
    if flag == "0":
        assert (
            r.e.fused_fallbacks == {"mixed_rope_delta": 1} and r.e.chain_steps == 1
        )  # declined before the chain is asked
    else:
        assert not r.e.__dict__.get("fused_fallbacks") and r.e.chain_steps == 2 and r.e.chain_audits == 2
        ovr = r.e._ch_g.ovr_in.data
        assert [int(w) for w in ovr[OVR_DELTA, :B]] == [d, d, d + 2] and int(ovr[AUX_CONSTS, CONST_ROPE_DELTA]) == 0
        assert r.e.row_delta_mixed_steps >= 1
