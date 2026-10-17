# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_SPEC_FAST_JOIN=1 (MTPSpecEngine._seed_fresh_rows + TPGatedDeltaNet.seed_spec_rows_select):
a fresh row's first verify runs inside the normal verify replay after its ring slot 0 / E_prev are seeded in place.
No device: the selector math is checked in torch, the engine bookkeeping with fakes.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_fastjoin_host.py -q
"""

import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt.gdn.tp import fresh_row_select, spec_conv_sel
from models.demos.blackhole.qwen36.tt.spec_engine import (
    PLACEHOLDER_TOKEN_ID,
    MTPSpecEngine,
    mask_unused_columns,
    sanitize_verify_block,
)
from models.demos.blackhole.qwen36.tt.tp_common import spec_fast_join_enabled

NB = 16  # page-table width in blocks of 64
C = 24  # fake conv channel width


# --------------------------------------------------------------------------- #
# The pure selector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B,T,kc", [(1, 4, 4), (2, 4, 4), (8, 4, 4), (4, 8, 4), (3, 2, 3), (8, 12, 4)])
def test_fresh_row_select_layout(B, T, kc):
    L = kc - 1 + T
    for fresh in ([], [0], list(range(B)), [u for u in range(B) if u % 2]):
        sel = fresh_row_select(fresh, B, T, kc)
        assert sel.shape == (B, L, L + kc - 1) and sel.dtype == torch.bfloat16
        assert set(sel.flatten().tolist()) <= {0.0, 1.0}
        assert bool((sel.sum(-1) <= 1).all()), "every output row takes at most one source row"
        for u in range(B):
            if u in fresh:
                # rows 1 .. kc-1 take the taps block (rows L .. L+kc-2); every other row is zero
                assert torch.equal(sel[u, 1:kc, L:], torch.eye(kc - 1, dtype=torch.bfloat16))
                assert not bool(sel[u, 0].any()) and not bool(sel[u, kc:].any()) and not bool(sel[u, :, :L].any())
            else:
                assert torch.equal(sel[u, :, :L], torch.eye(L, dtype=torch.bfloat16)) and not bool(sel[u, :, L:].any())


@pytest.mark.parametrize("B,T,kc,seed", [(2, 4, 4, 0), (8, 4, 4, 1), (4, 8, 4, 2), (3, 2, 3, 3)])
def test_fresh_row_select_matmul_seeds_fresh_rows_only(B, T, kc, seed):
    """E_new = sel @ [E_prev ; taps]: running rows keep E_prev bit for bit, fresh rows get taps 1 .. kc-1 at rows
    1 .. kc-1 and zeros elsewhere (one-hot gather in float: exact)."""
    g = torch.Generator().manual_seed(seed)
    L = kc - 1 + T
    e_prev = torch.randn(B, L, C, generator=g).to(torch.bfloat16)
    taps = torch.randn(B, kc - 1, C, generator=g).to(torch.bfloat16)
    fresh = [u for u in range(B) if (u * 7 + seed) % 3 == 0]
    sel = fresh_row_select(fresh, B, T, kc)
    new = (sel.float() @ torch.cat([e_prev, taps], dim=1).float()).to(torch.bfloat16)
    for u in range(B):
        if u in fresh:
            assert torch.equal(new[u, 1:kc], taps[u])
            assert not bool(new[u, 0].any()) and not bool(new[u, kc:].any())
        else:
            assert torch.equal(new[u], e_prev[u])


@pytest.mark.parametrize("B,T,kc", [(2, 4, 4), (8, 4, 4), (4, 8, 4)])
def test_fast_seed_window_reads_like_seed_spec_rows(B, T, kc):
    """What a mi = 0 verify READS from the window (spec_conv_sel: rows 1 .. kc-1 + the new rows) is identical whether a
    fresh user's E_prev row came from seed_spec_rows ([taps 0..kc-1 ; zeros]) or from the fast select ([0 ; taps
    1..kc-1 ; zeros]); running users read their kept rows either way."""
    g = torch.Generator().manual_seed(B * T)
    L = kc - 1 + T
    e_prev = torch.randn(B, L, C, generator=g).to(torch.bfloat16)  # the running rows' window
    taps_all = torch.randn(B, kc, C, generator=g).to(torch.bfloat16)  # conv_states[0..kc-1] of every user
    new_rows = torch.randn(B, T, C, generator=g).to(torch.bfloat16)  # this verify's projection rows
    fresh = [1, B - 1] if B > 2 else [1]
    # seed_spec_rows: fresh users' rows <- [taps(kc) ; zeros(T-1)]
    e_slow = e_prev.clone()
    for u in fresh:
        e_slow[u] = torch.cat([taps_all[u], torch.zeros(T - 1, C, dtype=torch.bfloat16)])
    # seed_spec_rows_select: E <- sel @ [E_prev ; taps(1..kc-1)]
    sel = fresh_row_select(fresh, B, T, kc)
    e_fast = (sel.float() @ torch.cat([e_prev, taps_all[:, 1:]], dim=1).float()).to(torch.bfloat16)
    conv_sel = spec_conv_sel([0] * B, B, T, kc).float()
    win_slow = conv_sel @ torch.cat([e_slow, new_rows], dim=1).float()
    win_fast = conv_sel @ torch.cat([e_fast, new_rows], dim=1).float()
    assert torch.equal(win_slow, win_fast)
    for u in range(B):
        if u not in fresh:
            assert torch.equal(e_fast[u], e_prev[u])


def test_fresh_row_select_rejects_out_of_range(expect_error):
    with expect_error(AssertionError, "outside range"):
        fresh_row_select([4], 4, 4, 4)


def test_flag_read(monkeypatch):
    """QWEN36_SPEC_FAST_JOIN is an explicit opt-in: only "1" turns the fast path on."""
    for val, want in (("1", True), ("0", False), (None, False), ("2", False)):
        if val is None:
            monkeypatch.delenv("QWEN36_SPEC_FAST_JOIN", raising=False)
        else:
            monkeypatch.setenv("QWEN36_SPEC_FAST_JOIN", val)
        assert spec_fast_join_enabled() is want


# --------------------------------------------------------------------------- #
# Engine bookkeeping with fakes
# --------------------------------------------------------------------------- #
class _GdnStub:
    """GDN layer stand-in: records the fast seeds (engine rows, selector) and parking; the eager-seed helpers raise,
    since the fast path must never call them."""

    K = 4

    def __init__(self, T):
        self.T = T
        self.selects, self.saved, self.loaded = [], [], []

    def seed_spec_rows_select(self, rows, sel):
        self.selects.append((sorted(int(u) for u in rows), sel))

    def save_spec_user(self, u):
        self.saved.append(u)
        return torch.full((self.T * 2, 1, 1, 1), float(u)), torch.full((1, 3 + self.T, 2), float(u))

    def load_spec_users(self, states):
        self.loaded.append(sorted(states))

    def _old_path(self, *a, **k):
        raise AssertionError("eager seed helper called on the fast-join path")

    seed_spec_state = seed_spec_rows = sync_conv_win = _old_path


class _FakeModel:
    """model.verify_traced stand-in: records every replay, returns ids = 10 * token + 1."""

    def __init__(self, B, T, V=12):
        self.B, self.T, self.V, self.calls = B, T, V, []

    def _fit_spec_page_table(self, pt, nb):
        return torch.as_tensor(pt, dtype=torch.int32)

    def verify_traced(self, tokens, positions, mi_prev, read_logits=False, page_tables=None, inactive=None):
        self.calls.append(
            dict(
                tokens=[[int(t) for t in row] for row in tokens],
                positions=[int(p) for p in positions],
                mi_prev=[int(m) for m in mi_prev],
                inactive=None if inactive is None else [bool(x) for x in inactive],
                pt=torch.as_tensor(page_tables).clone(),
            )
        )
        ids = [int(t) * 10 + 1 for row in tokens for t in row]
        lt = torch.zeros(self.B * self.T, self.V) if read_logits else None
        return ids, "feed", lt

    def seed_spec_step(self, *a, **k):
        raise AssertionError("model.seed_spec_step called on the fast-join path")


def _engine(B, K, model, gdn, fast_join=True, **kw):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size, e.nb = B, K, K + 1, 64, NB
    e.host_slotmap, e._phys, e._phys_perm = True, list(range(B)), None
    e._active, e._fresh, e._offered = set(), {}, set()
    e._parked, e._prompts, e._fresh_dirty = {}, {}, set()
    e._no_draft, e._speculated, e._withheld_streak = set(), set(), {}
    e._row_active, e._last_pt, e._last_cap_blocks, e.skipped_propose_calls = None, None, NB, 0
    e.no_draft_streak, e.captured, e._timing = 16, True, False
    e._gdn = [] if gdn is None else [gdn]
    e.model = model
    e._last_hidden, e._mtp_pt_rows, e._vfeed = [f"h{u}" for u in range(B)], [f"pt{u}" for u in range(B)], None
    e.fast_join, e._kc, e._fj_sel = fast_join, 4, "SEL"
    e.log = {"stage": [], "warm": [], "tables": [], "reprefill": []}
    e._stage = lambda host, dev, dtype, layout: e.log["stage"].append((host.clone(), dev, dtype, layout))
    e._stage_mtp_tables = lambda pt: e.log["tables"].append(pt.clone())
    e._warm_mtp_last = lambda h, tok, slot, pt: e.log["warm"].append((h, int(tok), int(slot), pt))
    e._reprefill = lambda slot: e.log["reprefill"].append(slot)
    e.__dict__.update(kw)
    return e


def _admit(e, slot, Tp):
    """What _prefill_row leaves behind for engine row ``slot`` (prompt length Tp)."""
    e._fresh[slot] = Tp
    e._active.add(slot)
    e._prompts[slot] = ([1] * Tp, torch.zeros(NB, dtype=torch.int32))


def _inputs(e, start):
    """tokens [B, T] (row u: 100 + u, 200 + u, ...), start_pos [B] from ``start``, a page table with distinct blocks."""
    B, T = e.B, e.T
    tok = torch.arange(B * T, dtype=torch.int64).reshape(B, T) + 100
    pos = torch.tensor(start, dtype=torch.int64)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    return tok, pos, pt


def test_fast_join_runs_fresh_rows_in_the_same_replay():
    """Two fresh rows join two running rows: ONE verify_traced call with every row active, the fresh rows at mi = 0,
    their MTP slot Tp-1 warmed, the selector staged once for the fresh engine rows, every GDN layer seeded for exactly
    those rows, and the bookkeeping cleared. No eager seed, no re-prefill."""
    B, K = 4, 3
    T = K + 1
    model, gdn = _FakeModel(B, T), _GdnStub(T)
    e = _engine(B, K, model, gdn)
    e._active = {0, 2}
    _admit(e, 1, 50)
    _admit(e, 3, 70)
    tok, pos, pt = _inputs(e, [20, 50, 30, 70])
    acc = torch.tensor([2, 1, 3, 1])
    ids, lt = e.verify(tok, pos, acc, pt, num_valid_drafts=[2, 0, 1, 0])
    assert len(model.calls) == 1, "the fresh rows must ride the running rows' replay"
    call = model.calls[0]
    assert call["mi_prev"] == [1, 0, 2, 0] and call["inactive"] == [False] * B
    assert call["tokens"] == tok.tolist() and call["positions"] == pos.tolist()
    assert e.log["warm"] == [("h1", int(tok[1, 0]), 49, "pt1"), ("h3", int(tok[3, 0]), 69, "pt3")]
    assert len(e.log["tables"]) == 1 and torch.equal(e.log["tables"][0], pt)
    assert gdn.selects == [([1, 3], "SEL")]
    (host, dev, dtype, layout), *rest = e.log["stage"]
    assert not rest and dev == "SEL" and torch.equal(host, fresh_row_select([1, 3], B, T, 4))
    assert e._fresh == {} and 1 not in e._prompts and 3 not in e._prompts and e._fresh_dirty == set()
    assert e.log["reprefill"] == []
    assert torch.equal(ids, (tok * 10 + 1).to(torch.int32)) and lt is None
    assert e._vfeed == "feed"


def test_fast_join_every_active_row_fresh_writes_the_whole_slot():
    """When every active row is fresh (a seed: the first step on an idle rank, or after the batch ran dry), the GDN
    layers get every row, i.e. the whole-slot ring write; an empty row is nobody's concern."""
    B, K = 3, 3
    T = K + 1
    model, gdn = _FakeModel(B, T), _GdnStub(T)
    e = _engine(B, K, model, gdn)
    _admit(e, 0, 10)
    _admit(e, 1, 12)
    tok, pos, pt = _inputs(e, [10, 12, 0])
    active = [True, True, False]
    ids, _ = e.verify(tok, pos, torch.ones(B, dtype=torch.int64), pt, active=active)
    assert gdn.selects == [([0, 1, 2], "SEL")]
    assert torch.equal(e.log["stage"][0][0], fresh_row_select([0, 1], B, T, 4))  # the selector names only the fresh
    call = model.calls[0]
    assert call["mi_prev"] == [0, 0, 0] and call["inactive"] == [False, False, True]
    assert bool((call["pt"][2] == 0).all()), "the empty row runs over the null block"
    assert e._fresh == {} and e._active == {0, 1}


def test_fast_join_left_out_fresh_row_keeps_its_prefill_state():
    """A fresh row left out of the step (padding) is not seeded and NOT marked dirty: nothing advanced rec_state, so
    it needs no re-prefill when it comes in (contrast test_spec_engine_host.test_verify_defers_fresh_row_left_out)."""
    B, K = 2, 2
    T = K + 1
    model, gdn = _FakeModel(B, T), _GdnStub(T)
    e = _engine(B, K, model, gdn)
    _admit(e, 0, 5)
    _admit(e, 1, 7)
    tok, pos, pt = _inputs(e, [5, 0])
    e.verify(tok, pos, torch.ones(B, dtype=torch.int64), pt, active=[True, False])
    assert gdn.selects == [([0, 1], "SEL")]  # row 1 is not active: every active row is fresh -> whole slot
    assert torch.equal(e.log["stage"][0][0], fresh_row_select([0], B, T, 4))
    assert e._fresh == {1: 7} and e._fresh_dirty == set() and 1 in e._prompts and 0 not in e._prompts
    assert e.log["reprefill"] == [] and e.log["warm"] == [("h0", int(tok[0, 0]), 4, "pt0")]
    tok, pos, pt = _inputs(e, [6, 7])
    e.verify(tok, pos, torch.tensor([1, 1]), pt)
    assert gdn.selects[-1] == ([1], "SEL") and e._fresh == {} and e.log["reprefill"] == []
    assert model.calls[-1]["mi_prev"] == [0, 0] and model.calls[-1]["inactive"] == [False, False]
    assert e.log["warm"][-1] == ("h1", int(tok[1, 0]), 6, "pt1")


def test_fast_join_with_a_returning_parked_row():
    """Running row 0, fresh row 1 and parked row 2 returning in one step: the parked row is restored first (ring
    slot 0, mi 0), the fresh row seeded per row (not the whole slot), and one replay runs all three."""
    B, K = 3, 3
    T = K + 1
    model, gdn = _FakeModel(B, T), _GdnStub(T)
    e = _engine(B, K, model, gdn)
    e._active = {0, 2}
    tok, pos, pt = _inputs(e, [20, 0, 40])
    e.verify(tok, pos, torch.tensor([2, 1, 3]), pt, active=[True, False, True])  # warm-up: both running
    e.verify(tok, pos, torch.tensor([2, 1, 3]), pt, active=[True, False, False])  # row 2 left out -> parked
    assert gdn.saved == [2] and set(e._parked) == {2} and gdn.selects == []
    _admit(e, 1, 33)
    tok, pos, pt = _inputs(e, [21, 33, 41])
    e.verify(tok, pos, torch.tensor([3, 1, 2]), pt)
    assert gdn.loaded == [[2]] and e._parked == {}
    assert gdn.selects == [([1], "SEL")]
    assert model.calls[-1]["mi_prev"] == [2, 0, 0] and model.calls[-1]["inactive"] == [False] * B


def test_fast_join_decode_forward_scatters_through_the_slot_map_and_masks_filler():
    """Runner slot 2 is fresh and sits at engine row 0 after a remap: the GDN seed names the ENGINE row, the fresh
    row's argmax_ids keep column 0 only (num_valid_drafts 0), in runner order."""
    B, K = 3, 3
    T = K + 1
    model, gdn = _FakeModel(B, T), _GdnStub(T)
    e = _engine(B, K, model, gdn)
    e._active = {1, 2}
    _admit(e, 0, 40)  # engine row 0 holds the fresh request
    remap = [1, 2, 0]  # runner row 2 now holds the request that sat at runner row 0 (engine row 0)
    tok = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    pos = torch.full((B, T), -1, dtype=torch.int32)
    nvd = torch.tensor([2, 1, 0], dtype=torch.int32)
    starts = [10, 20, 40]
    for s in range(B):
        tok[s, : nvd[s] + 1] = torch.arange(nvd[s] + 1) + 100 * (s + 1)
        pos[s, : nvd[s] + 1] = starts[s] + torch.arange(nvd[s] + 1)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    res = e.decode_forward(
        tok, pos, num_valid_drafts=nvd, accepted_counts=torch.tensor([2, 3, 1]), page_table=pt, slot_remap=remap
    )
    assert e._phys == remap
    assert gdn.selects == [([0], "SEL")] and e.log["warm"] == [("h0", 300, 39, "pt0")]
    call = model.calls[0]
    clean, start0, active = sanitize_verify_block(tok, pos, nvd, K)
    for s in range(B):
        r = remap[s]
        assert call["tokens"][r] == clean[s].tolist() and call["positions"][r] == int(start0[s])
    assert call["mi_prev"][0] == 0  # the fresh engine row verifies from ring slot 0 whatever the runner repeats
    want = mask_unused_columns((clean * 10 + 1).to(torch.int32), nvd)
    assert torch.equal(res.argmax_ids, want)
    assert res.argmax_ids[2, 1:].tolist() == [PLACEHOLDER_TOKEN_ID] * K


def test_fast_join_logits_mode_returns_the_replay_logits():
    B, K = 2, 2
    T = K + 1
    model, gdn = _FakeModel(B, T, V=7), _GdnStub(T)
    e = _engine(B, K, model, gdn)
    e._active = {0}
    _admit(e, 1, 9)
    tok, pos, pt = _inputs(e, [3, 9])
    ids, lt = e.verify(tok, pos, torch.tensor([1, 1]), pt, read_logits=True)
    assert lt is not None and tuple(lt.shape) == (B, T, 7) and ids.shape == (B, T)


def test_fast_join_without_gdn_layers_stages_nothing():
    """A model with no GDN layers (host doubles): the MTP warm still runs, no selector is staged."""
    B, K = 2, 2
    T = K + 1
    model = _FakeModel(B, T)
    e = _engine(B, K, model, None)
    _admit(e, 0, 4)
    tok, pos, pt = _inputs(e, [4, 0])
    e.verify(tok, pos, torch.tensor([1, 1]), pt, active=[True, False])
    assert e.log["stage"] == [] and e.log["warm"] == [("h0", int(tok[0, 0]), 3, "pt0")] and e._fresh == {}


def test_flag_off_keeps_the_eager_seed_path():
    """fast_join False: the fresh rows go through _seed / _join (stubbed) and seed_spec_rows_select is never called."""
    B, K = 2, 2
    T = K + 1
    model, gdn = _FakeModel(B, T), _GdnStub(T)
    e = _engine(B, K, model, gdn, fast_join=False)
    hits = []
    e._seed = lambda *a: (hits.append("seed"), (torch.zeros(B, T, dtype=torch.int32), None))[1]
    e._join = lambda *a: (hits.append("join"), (torch.zeros(B, T, dtype=torch.int32), None))[1]
    _admit(e, 0, 4)
    _admit(e, 1, 6)
    tok, pos, pt = _inputs(e, [4, 6])
    e.verify(tok, pos, torch.tensor([1, 1]), pt)
    e._active = {0}
    _admit(e, 1, 8)
    tok, pos, pt = _inputs(e, [5, 8])
    e.verify(tok, pos, torch.tensor([1, 1]), pt)
    assert hits == ["seed", "join"] and gdn.selects == [] and model.calls == []
    assert e.log["stage"] == []


def test_engine_init_reads_the_flag(monkeypatch):
    """MTPSpecEngine.__init__ reads QWEN36_SPEC_FAST_JOIN and the conv kernel size of the GDN layers."""
    import ttnn

    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda *a, **k: None)
    args = types.SimpleNamespace(vocab_size=8, dim=16, max_batch_size=2)
    gdn_layer = types.SimpleNamespace(is_full_attention=False, attention=types.SimpleNamespace(K=4))
    model = types.SimpleNamespace(
        mtp=object(),
        num_devices=2,
        args=args,
        mesh_device=types.SimpleNamespace(arch=lambda: None),
        gdn_fused_decode=True,
        layers=[gdn_layer],
    )
    for val, want in (("1", True), ("0", False), (None, False)):
        if val is None:
            monkeypatch.delenv("QWEN36_SPEC_FAST_JOIN", raising=False)
        else:
            monkeypatch.setenv("QWEN36_SPEC_FAST_JOIN", val)
        e = MTPSpecEngine(model, 2, 3)
        assert e.fast_join is want and e._kc == 4 and e._fj_sel is None
