# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the QWEN36_VERIFY_SPEC_STEP wiring that the host can check without the gdn_spec_step op (the
device tests are test_spec_verify_specstep.py and test_gdn_spec_step.py):

* the ctrl page helpers (spec_ctrl_page / spec_ctrl_words / HOLD) and their agreement with spec_state_blk_idx;
* the page built from ENGINE rows under the host slot map (QWEN36_SPEC_SLOTMAP, both settings): decode_forward
  scatters the runner's accepted_counts before the verify, parks / restores rows at their engine row, and the page
  (like state_blk_idx / conv_sel) carries runner slot s at engine row phys[s];
* Qwen36Model.verify_traced over fake persistent buffers: the page is staged on EVERY replay at the parity the GDN
  layers hold going in, next to the QWEN36_VERIFY_STAGING_CACHE page-table skip, with the parity flip after the
  synchronize (also with the QWEN36_VERIFY_SHARDED_ARGMAX logits trace); the composite selectors are untouched;
* the engine / model flag plumbing (gdn_verify_spec_step, set_gdn_fused_decode, spec_memory_costs).
* QWEN36_SPEC_STEP_VALUE_BLOCKS: parsed to 1 / 2 / 4, and the op's value_blocks kwarg is passed only when > 1 and
  only to a T > 1 verify (the T = 1 seed keeps one core per (user, head)).
* QWEN36_SPEC_STEP_OPT: the kernel_opts bitmask (decimal / 0x), unknown bits refused, passed only when non-zero and
  only with fp32_state.

No device, no op. Run: pytest models/demos/blackhole/qwen36/tests/test_spec_specstep_host.py -q
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import (
    HOLD_SENTINEL,
    SPEC_STEP_OPT_KNOWN,
    TPGatedDeltaNet,
    gdn_spec_step_kernel_opts,
    gdn_spec_step_value_blocks,
    spec_conv_sel,
    spec_ctrl_page,
    spec_ctrl_words,
    spec_state_blk_idx,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import (
    MTPSpecEngine,
    mask_unused_columns,
    rows_to_engine,
    sanitize_verify_block,
    slot_map_inverse,
    spec_memory_costs,
)

NB = 16  # page-table width in blocks of 64


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("Nv", [12, 24, 32])
def test_ctrl_words_cover_the_page_in_whole_lines(B, Nv):
    n = spec_ctrl_words(B, Nv)
    assert n >= 1 + B + B * Nv and n % 16 == 0 and n - (1 + B + B * Nv) < 16


@pytest.mark.parametrize("B,T,Nv", [(1, 4, 12), (8, 4, 12), (4, 8, 24), (2, 12, 12)])
def test_ctrl_page_layout_is_parity_mi_then_state_blk_idx(B, T, Nv):
    g = torch.Generator().manual_seed(B * 100 + T)
    for parity in (0, 1):
        mi = torch.randint(0, T, (B,), generator=g).tolist()
        page = spec_ctrl_page(mi, B, Nv, parity=parity)
        assert page.dtype == torch.int32 and tuple(page.shape) == (1, spec_ctrl_words(B, Nv))
        w = page[0]
        assert int(w[0]) == parity
        assert w[1 : 1 + B].tolist() == mi
        assert torch.equal(w[1 + B : 1 + B + B * Nv], spec_state_blk_idx(mi, B, Nv))
        assert not bool(w[1 + B + B * Nv :].any()), "padding words are zero"
        # ring block of (u, h) = (mi[u]*B + u)*Nv + h: the ring's (token slot, user, head)-major layout
        for u in range(B):
            assert w[1 + B + u * Nv : 1 + B + (u + 1) * Nv].tolist() == [(mi[u] * B + u) * Nv + h for h in range(Nv)]


def test_ctrl_page_hold_marks_every_head_of_the_user():
    B, Nv = 4, 12
    mi = [1, 3, 0, 2]
    page = spec_ctrl_page(mi, B, Nv, parity=1, hold={1, 2})[0]
    assert page[1 : 1 + B].tolist() == [1, 0, 0, 2], "a held user's mi word is 0 (unread)"
    blocks = page[1 + B : 1 + B + B * Nv].reshape(B, Nv)
    assert bool((blocks[[1, 2]] == -1).all()) and bool((blocks[[0, 3]] >= 0).all())
    plain = spec_state_blk_idx(mi, B, Nv).reshape(B, Nv)
    assert torch.equal(blocks[[0, 3]], plain[[0, 3]])
    assert torch.equal(spec_state_blk_idx(mi, B, Nv, hold={1, 2}).reshape(B, Nv), blocks)
    # staged as uint32, the int32 -1 is the op's sentinel
    assert HOLD_SENTINEL == 0xFFFFFFFF and (-1 & 0xFFFFFFFF) == HOLD_SENTINEL


def test_ctrl_page_rejects_bad_parity_and_row_count(expect_error):
    with expect_error(AssertionError, "parity"):
        spec_ctrl_page([0, 0], 2, 12, parity=2)
    with expect_error(AssertionError, "one mi per user"):
        spec_ctrl_page([0], 2, 12, parity=0)


@pytest.mark.parametrize("B,T,Nv", [(2, 4, 12), (5, 4, 12), (8, 4, 12), (4, 8, 24)])
def test_ctrl_page_from_engine_rows_follows_the_slot_map(B, T, Nv):
    """The page built from mi in ENGINE rows (rows_to_engine of the runner's accepted_counts - 1) carries runner
    slot s's mi and ring blocks at engine row phys[s] -- the rows state_blk_idx / conv_sel address too."""
    g = torch.Generator().manual_seed(B + T)
    Kc = 4
    for _ in range(10):
        phys = torch.randperm(B, generator=g).tolist()
        inv = slot_map_inverse(phys)
        acc = torch.randint(1, T + 1, (B,), generator=g)  # runner slots
        mi_rows = rows_to_engine(acc - 1, inv).tolist()
        page = spec_ctrl_page(mi_rows, B, Nv, parity=0)[0]
        blocks = page[1 + B : 1 + B + B * Nv].reshape(B, Nv)
        idx = spec_state_blk_idx(mi_rows, B, Nv).reshape(B, Nv)
        sel = spec_conv_sel(mi_rows, B, T, Kc)
        for s in range(B):
            r, m = phys[s], int(acc[s]) - 1
            assert int(page[1 + r]) == m
            assert blocks[r].tolist() == [(m * B + r) * Nv + h for h in range(Nv)]
            assert torch.equal(blocks[r], idx[r])
            assert torch.equal(sel[r], spec_conv_sel([m], 1, T, Kc)[0])


# --------------------------------------------------------------------------- #
# Engine: decode_forward -> verify -> model.verify_traced(mi_prev in engine rows)
# --------------------------------------------------------------------------- #
class _GdnStub:
    """GDN layer stand-in for MTPSpecEngine.verify's bookkeeping (no device): records which ENGINE rows are parked,
    restored and remapped; save_spec_user returns a small host state parked_rows_host can shift."""

    Nv, Dk, Dv, C, K = 2, 2, 2, 4, 4

    def __init__(self, T):
        self.T = T
        self.saved, self.loaded, self.remaps = [], [], []

    def save_spec_user(self, u):
        self.saved.append(u)
        ring = torch.full((self.T * self.Nv, self.Dk, self.Dv), float(u))  # [D*T, Nv, Dk, Dv] at D = 1
        e = torch.full((1, self.K - 1 + self.T, self.C), float(u))  # [D, K-1+T, C]
        return ring, e

    def load_spec_users(self, states):
        self.loaded.append(sorted(states))

    def remap_spec_slots(self, idx):
        self.remaps.append([int(i) for i in idx])


class _FakeVerifyModel:
    """model.verify_traced stand-in: builds the ctrl page from the mi_prev it is handed (engine rows), as
    Qwen36Model.verify_traced does under QWEN36_VERIFY_SPEC_STEP=1, flips its parity per replay, records the call."""

    def __init__(self, B, T, Nv):
        self.B, self.T, self.Nv, self.parity, self.calls = B, T, Nv, 0, []

    def _fit_spec_page_table(self, pt, nb):
        return torch.as_tensor(pt, dtype=torch.int32)

    def verify_traced(self, tokens, positions, mi_prev, read_logits=False, page_tables=None, inactive=None):
        assert all(0 <= int(m) < self.T for m in mi_prev)
        page = spec_ctrl_page(mi_prev, self.B, self.Nv, parity=self.parity)
        self.parity ^= 1
        self.calls.append(
            dict(
                tokens=[[int(t) for t in row] for row in tokens],
                positions=[int(p) for p in positions],
                mi_prev=[int(m) for m in mi_prev],
                inactive=None if inactive is None else [bool(x) for x in inactive],
                pt=torch.as_tensor(page_tables).clone(),
                page=page,
            )
        )
        return [int(t) * 10 + 1 for row in tokens for t in row], "feed", None


def _engine(B, K, slotmap, model):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size, e.nb = B, K, K + 1, 64, NB
    e.host_slotmap, e._phys, e._phys_perm = slotmap, list(range(B)), None
    e._active, e._fresh, e._offered = set(range(B)), {}, set()  # every row holds a prefilled request
    e._parked, e._prompts, e._fresh_dirty = {}, {}, set()
    e._no_draft, e._speculated, e._withheld_streak = set(), set(), {}
    e._row_active, e._last_pt, e._last_cap_blocks, e.skipped_propose_calls = None, None, NB, 0
    e.no_draft_streak, e.captured, e._timing = 16, True, False
    e._gdn = [_GdnStub(K + 1)]
    e.model = model
    e._last_hidden, e._mtp_pt_rows, e._vfeed = [None] * B, [None] * B, None
    e._stage_mtp_tables = e._warm_mtp_last = lambda *a, **k: None
    e.fast_join, e._kc, e._fj_sel = False, 4, None  # QWEN36_SPEC_FAST_JOIN=0: the eager seed path
    return e


def _block(B, T, nv, acc, base=50):
    """A runner verify block: row u has nv[u] drafts after its pending token at positions base + 10u ..;
    nv[u] < 0 marks a padding row (no request this step)."""
    tok = torch.arange(B * T, dtype=torch.int32).reshape(B, T) + 100
    pos = torch.full((B, T), -1, dtype=torch.int32)
    nvd = torch.zeros(B, dtype=torch.int32)
    for u in range(B):
        if nv[u] >= 0:
            pos[u, : nv[u] + 1] = base + 10 * u + torch.arange(nv[u] + 1, dtype=torch.int32)
            nvd[u] = nv[u]
    return tok, pos, nvd, torch.tensor(acc, dtype=torch.int32)


@pytest.mark.parametrize("slotmap", [True, False])
def test_decode_forward_builds_the_ctrl_page_from_engine_rows(slotmap):
    """The runner's accepted_counts are scattered to engine rows before the verify, so the page (like state_blk_idx
    / conv_sel) carries runner slot s's mi at engine row phys[s]; a padding row runs at mi = 0 and is parked at its
    engine row. Flag off, phys is the identity and the device remap moved the rows instead."""
    B, K, Nv = 4, 3, 12
    T = K + 1
    model = _FakeVerifyModel(B, T, Nv)
    e = _engine(B, K, slotmap, model)
    remap = [2, 0, 3, 1]
    e.remap_slots(remap)
    phys = list(e._phys)
    if slotmap:
        assert phys == remap and e._gdn[0].remaps == []
    else:
        assert phys == list(range(B)) and e._gdn[0].remaps == [remap]
    tok, pos, nvd, acc = _block(B, T, nv=[3, -1, 1, 2], acc=[1, 4, 2, 3])  # runner row 1 is padding
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    res = e.decode_forward(tok, pos, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt)
    call = model.calls[-1]
    page = call["page"][0]
    clean, start0, active = sanitize_verify_block(tok, pos, nvd, K)
    for s in range(B):
        r = phys[s]
        mi = int(acc[s]) - 1 if active[s] else 0
        assert call["mi_prev"][r] == mi and int(page[1 + r]) == mi
        assert page[1 + B + r * Nv : 1 + B + (r + 1) * Nv].tolist() == [(mi * B + r) * Nv + h for h in range(Nv)]
        assert call["inactive"][r] is (not bool(active[s]))
        assert call["tokens"][r] == clean[s].tolist() and call["positions"][r] == int(start0[s])
        # the verify write table (verify_write_table) of runner slot s sits at engine row r; a padding row is null
        nz = call["pt"][r] != 0
        assert bool(nz.any()) is bool(active[s]) and torch.equal(call["pt"][r][nz], pt[s][nz])
    assert torch.equal(page[1 + B : 1 + B + B * Nv], spec_state_blk_idx(call["mi_prev"], B, Nv))
    assert e._gdn[0].saved == [phys[1]] and set(e._parked) == {phys[1]}
    assert torch.equal(res.argmax_ids, mask_unused_columns(clean.to(torch.int32) * 10 + 1, nvd))


@pytest.mark.parametrize("slotmap", [True, False])
def test_returning_row_verifies_from_mi_zero_in_the_page(slotmap):
    """A parked row that returns is restored into ring slot 0 (load_spec_users at its ENGINE row), and the page
    (like state_blk_idx) sends it to mi = 0 whatever accepted_count the runner repeats for it."""
    B, K, Nv = 3, 3, 12
    T = K + 1
    model = _FakeVerifyModel(B, T, Nv)
    e = _engine(B, K, slotmap, model)
    e.remap_slots([1, 2, 0])
    phys = list(e._phys)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    # step 1: every row runs
    tok, pos, nvd, acc = _block(B, T, nv=[2, 1, 3], acc=[1, 1, 1])
    e.decode_forward(tok, pos, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt)
    assert model.calls[-1]["mi_prev"] == [0] * B
    # step 2: runner row 1 is left out -> parked at engine row phys[1], its page entry is mi = 0
    tok, pos, nvd, acc = _block(B, T, nv=[1, -1, 2], acc=[3, 2, 4], base=60)
    e.decode_forward(tok, pos, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt)
    assert e._gdn[0].saved == [phys[1]] and list(e._parked) == [phys[1]]
    page = model.calls[-1]["page"][0]
    assert (int(page[1 + phys[0]]), int(page[1 + phys[1]]), int(page[1 + phys[2]])) == (2, 0, 3)
    # step 3: it returns with a stale accepted count -> restored at its engine row, page says mi = 0
    tok, pos, nvd, acc = _block(B, T, nv=[0, 3, 1], acc=[2, 2, 3], base=70)
    e.decode_forward(tok, pos, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt)
    assert e._gdn[0].loaded == [[phys[1]]] and not e._parked
    call = model.calls[-1]
    page = call["page"][0]
    assert call["mi_prev"][phys[1]] == 0
    assert (int(page[1 + phys[0]]), int(page[1 + phys[1]]), int(page[1 + phys[2]])) == (1, 0, 2)
    r = phys[1]
    assert page[1 + B + r * Nv : 1 + B + (r + 1) * Nv].tolist() == [r * Nv + h for h in range(Nv)]
    # one page per replay, parity alternating like the model's window pair
    assert [int(c["page"][0, 0]) for c in model.calls] == [0, 1, 0]


# --------------------------------------------------------------------------- #
# Qwen36Model.verify_traced over fake persistent buffers (fake ttnn)
# --------------------------------------------------------------------------- #
class _Buf:
    """Fake persistent device buffer: holds what the last staged copy wrote and counts the copies."""

    def __init__(self, name):
        self.name, self.data, self.dtype, self.layout, self.writes = name, None, None, None, 0


class _Host:
    def __init__(self, t, dtype, layout):
        self.t, self.dtype, self.layout = torch.as_tensor(t).clone(), dtype, layout


def _fake_ttnn(monkeypatch, log, model):
    def from_torch(t, dtype=None, layout=None, device=None, mesh_mapper=None, **kw):
        assert device is None, "verify_traced stages into persistent buffers; it must not allocate on the device"
        return _Host(t, dtype, layout)

    def copy(h, buf):
        assert isinstance(h, _Host) and isinstance(buf, _Buf), (h, buf)
        buf.data, buf.dtype, buf.layout = h.t, h.dtype, h.layout
        buf.writes += 1
        log.append(("h2d", buf.name))

    def execute_trace(dev, tid, cq_id=0, blocking=False):
        log.append(("trace", tid))

    def synchronize_device(dev):
        log.append(("sync", tuple(dn._spec_win_par for dn in model._vfy_gdn)))

    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda dev: ("rep", dev))
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", copy)
    monkeypatch.setattr(ttnn, "execute_trace", execute_trace)
    monkeypatch.setattr(ttnn, "synchronize_device", synchronize_device)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t])
    monkeypatch.setattr(ttnn, "to_torch", lambda t: t.data)


def _vfy_model(B, T, fused, sharded=False, staging_cache=True, n_gdn=3, Nv=12, Kc=4, nb=8):
    """A Qwen36Model with only what verify_traced reads, as prepare_verify_trace / record_verify_trace leave it."""
    m = Qwen36Model.__new__(Qwen36Model)
    m.device = m.mesh_device = "mesh"
    m.args = types.SimpleNamespace(rope_head_dim=64, rope_theta=1e7)
    m.rope = types.SimpleNamespace(rope_delta=0, max_seq_len=1024)
    m.vocab_size = 64
    m._vfy_trace_id, m._vfy_logits_trace_id = 7, (8 if sharded else None)
    m._vfy_T, m._vfy_B, m._vfy_nb, m._vfy_spec_groups = T, B, nb, B
    m._vfy_kv_grouped = False
    m._vfy_staging_cache, m._vfy_sharded, m._vfy_fused = staging_cache, sharded, fused
    m._vfy_Nv, m._vfy_Kc = Nv, Kc
    m._vfy_gdn = [types.SimpleNamespace(_spec_win_par=0) for _ in range(n_gdn)]
    names = ("token", "kvpos", "cos", "sin", "kvpt", "kvpt_users", "kvpt1", "ctrl", "state_idx", "conv_sel")
    bufs = {n: _Buf(n) for n in names}
    m._vfy_token_buf, m._vfy_kvpos_buf, m._vfy_cos_buf, m._vfy_sin_buf = (bufs[n] for n in names[:4])
    m._vfy_kvpt_buf, m._vfy_kvpt_users, m._vfy_kvpt1_buf = (bufs[n] for n in names[4:7])
    m._vfy_ctrl = bufs["ctrl"] if fused else None
    m._vfy_state_idx, m._vfy_conv_sel = (None, None) if fused else (bufs["state_idx"], bufs["conv_sel"])
    m._vfy_pt_host = torch.zeros(B, nb, dtype=torch.int32)  # what prepare_verify_trace staged
    m._vfy_ids_out = _Buf("ids")
    m._vfy_ids_out.data = torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32)
    m._vfy_rows_out = "feed"
    m._vfy_logits_rm_out = _Buf("logits")
    m._vfy_logits_rm_out.data = torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)
    return m, bufs


@pytest.mark.parametrize("staging_cache", [True, False])
def test_verify_traced_stages_the_ctrl_page_on_every_replay(monkeypatch, staging_cache):
    """The page carries this replay's mi at the parity the layers hold going in, is copied on EVERY replay (the
    staging cache skips only an unchanged page table), and every layer's parity flips after the synchronize."""
    B, T, Nv, nb = 4, 4, 12, 8
    m, bufs = _vfy_model(B, T, fused=True, staging_cache=staging_cache, nb=nb)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    pt = torch.arange(1, B * nb + 1, dtype=torch.int32).reshape(B, nb)
    tokens = [[u * 10 + j for j in range(T)] for u in range(B)]
    positions = [100 + 7 * u for u in range(B)]
    steps = [([0, 1, 2, 3], pt), ([3, 0, 0, 1], pt), ([2, 2, 1, 0], None), ([1, 1, 1, 1], pt + 1)]
    want_par = 0
    for i, (mi, tbl) in enumerate(steps):
        del log[:]
        ctrl_writes, pt_writes = bufs["ctrl"].writes, bufs["kvpt"].writes
        ids, feed, lt = m.verify_traced(tokens, positions, mi, page_tables=tbl)
        assert bufs["ctrl"].writes == ctrl_writes + 1, "the ctrl page is staged on every replay"
        assert torch.equal(bufs["ctrl"].data, spec_ctrl_page(mi, B, Nv, parity=want_par))
        assert bufs["ctrl"].dtype == ttnn.uint32 and bufs["ctrl"].layout == ttnn.ROW_MAJOR_LAYOUT
        assert log[-2:] == [("trace", 7), ("sync", (want_par,) * len(m._vfy_gdn))], "flip after the synchronize"
        want_par ^= 1
        assert all(dn._spec_win_par == want_par for dn in m._vfy_gdn)
        # page tables: skipped only by the staging cache, and only when unchanged (step 1 repeats step 0's)
        changed = tbl is not None and not (staging_cache and i == 1)
        assert bufs["kvpt"].writes == pt_writes + int(changed)
        if changed:
            assert torch.equal(bufs["kvpt_users"].data, tbl) and torch.equal(
                bufs["kvpt"].data, tbl.repeat_interleave(T, 0)
            )
        assert bufs["state_idx"].writes == 0 and bufs["conv_sel"].writes == 0, "composite selectors unused"
        assert bufs["token"].data.reshape(-1).tolist() == [t for row in tokens for t in row]
        assert bufs["kvpos"].data.tolist() == [p + j for p in positions for j in range(T)]
        assert ids == list(range(B * T)) and feed == "feed" and lt is None
    assert torch.equal(m._vfy_pt_host, pt + 1) and m._vfy_pt_host is not steps[-1][1]


def test_verify_traced_composite_selectors_unchanged(monkeypatch):
    """Flag off: state_blk_idx + conv_sel are staged from mi_prev as before, no page, no parity flips; the ring
    blocks they carry are the ones the fused page would (one source: mi_prev in engine rows)."""
    B, T, Nv, Kc = 2, 8, 12, 4
    m, bufs = _vfy_model(B, T, fused=False)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    tokens = [[j for j in range(T)] for _ in range(B)]
    for mi in ([0, 0], [5, 2], [7, 0]):
        m.verify_traced(tokens, [10, 20], mi)
        assert torch.equal(bufs["state_idx"].data, spec_state_blk_idx(mi, B, Nv))
        assert torch.equal(bufs["conv_sel"].data, spec_conv_sel(mi, B, T, Kc))
        assert bufs["ctrl"].writes == 0 and all(dn._spec_win_par == 0 for dn in m._vfy_gdn)
    page = spec_ctrl_page([7, 0], B, Nv, parity=0)[0]
    assert torch.equal(page[1 + B : 1 + B + B * Nv], bufs["state_idx"].data)


@pytest.mark.parametrize("read_logits", [False, True])
def test_verify_traced_sharded_logits_trace_runs_before_the_parity_flip(monkeypatch, read_logits):
    """QWEN36_VERIFY_SHARDED_ARGMAX and QWEN36_VERIFY_SPEC_STEP together: the logits trace (sampled requests)
    replays right after the main trace, and the window parity flips only after both are synchronized."""
    B, T = 2, 4
    m, bufs = _vfy_model(B, T, fused=True, sharded=True)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    ids, _, lt = m.verify_traced([[1] * T] * B, [5, 9], [0, 3], read_logits=read_logits)
    assert [x for x in log if x[0] == "trace"] == ([("trace", 7), ("trace", 8)] if read_logits else [("trace", 7)])
    assert log[-1] == ("sync", (0,) * len(m._vfy_gdn)) and all(dn._spec_win_par == 1 for dn in m._vfy_gdn)
    assert bufs["ctrl"].writes == 1 and torch.equal(bufs["ctrl"].data, spec_ctrl_page([0, 3], B, 12, parity=0))
    assert ids == list(range(B * T))
    assert (lt is not None and tuple(lt.shape) == (B * T, 64)) if read_logits else lt is None


def test_verify_traced_fused_refuses_bad_mi_and_split_parities(monkeypatch, expect_error):
    B, T = 2, 4
    m, bufs = _vfy_model(B, T, fused=True)
    log = []
    _fake_ttnn(monkeypatch, log, m)
    with expect_error(AssertionError, "out of range"):
        m.verify_traced([[1] * T] * B, [5, 9], [0, T])
    assert bufs["ctrl"].writes == 0 and ("trace", 7) not in log
    assert all(dn._spec_win_par == 0 for dn in m._vfy_gdn)
    m._vfy_gdn[1]._spec_win_par = 1  # one page serves every layer: parities out of step are refused
    with expect_error(AssertionError, "out of step"):
        m.verify_traced([[1] * T] * B, [5, 9], [0, 1])
    assert bufs["ctrl"].writes == 0 and ("trace", 7) not in log


# --------------------------------------------------------------------------- #
# Flag plumbing
# --------------------------------------------------------------------------- #
def test_engine_accepts_spec_step_in_place_of_fused_decode(monkeypatch, expect_error):
    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda *a, **k: None)
    args = types.SimpleNamespace(vocab_size=8, dim=16, max_batch_size=2)

    def model(fused, step):
        return types.SimpleNamespace(
            mtp=object(),
            num_devices=2,
            args=args,
            mesh_device=types.SimpleNamespace(arch=lambda: None),
            gdn_fused_decode=fused,
            gdn_verify_spec_step=step,
            layers=[],
        )

    MTPSpecEngine(model(True, False), 2, 3)
    MTPSpecEngine(model(False, True), 2, 3)
    with expect_error(AssertionError, "set_gdn_fused_decode"):
        MTPSpecEngine(model(False, False), 2, 3)


def test_set_gdn_fused_decode_is_ignored_under_spec_step():
    def model(spec_fused):
        m = Qwen36Model.__new__(Qwen36Model)
        m.layers = [
            types.SimpleNamespace(
                is_full_attention=(i % 3 == 2),
                attention=types.SimpleNamespace(_spec_fused=spec_fused, use_fused_recurrent_decode=False),
            )
            for i in range(6)
        ]
        return m

    m = model(True)
    assert m.gdn_verify_spec_step and not m.gdn_fused_decode
    m.set_gdn_fused_decode(True)
    assert not m.gdn_fused_decode, "plain decode stays on gdn_decode_step"
    m.set_gdn_fused_decode(False)
    assert not m.gdn_fused_decode
    m = model(False)
    assert not m.gdn_verify_spec_step
    m.set_gdn_fused_decode(True)
    assert m.gdn_fused_decode


def test_spec_memory_costs_counts_the_window_pair_under_the_flag(monkeypatch):
    tc = dict(
        hidden_size=512,
        vocab_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
    )
    monkeypatch.setenv("QWEN36_VERIFY_SPEC_STEP", "0")
    off = spec_memory_costs(tc, 2, 2, 3, nb=16)
    monkeypatch.setenv("QWEN36_VERIFY_SPEC_STEP", "1")
    on = spec_memory_costs(tc, 2, 2, 3, nb=16)
    assert on["per_seq"]["gdn_conv_windows"] > off["per_seq"]["gdn_conv_windows"]
    assert on["per_seq"]["gdn_ring"] == off["per_seq"]["gdn_ring"] and on["fixed"] == off["fixed"]
    # the extra is one more [K-1+T, C] bf16 window per GDN layer and user (3 GDN layers of 4, B = 2)
    T, C = 4, (4 * 32 * 2 + 8 * 32) // 2
    extra = 3 * 2 * 32 * (-(-(3 + T) // 32)) * (-(-C // 32)) * 32 * 2
    assert on["per_seq"]["gdn_conv_windows"] - off["per_seq"]["gdn_conv_windows"] == extra


# --------------------------------------------------------------------------- #
# Bucketed verify (b rows on the B-wide state): the relaxed n_users checks and max_users
# --------------------------------------------------------------------------- #
def _bucket_layer(B, T, calls):
    g = types.SimpleNamespace(
        B=B, _spec_shape=(B, T), _spec_fused=True, _verify_win_pair=("wa", "wb"), _spec_ring="ring"
    )
    g._gdn_spec_step = lambda *a, **kw: calls.append((a[5:], kw)) or "gated"
    g._out_proj_tail = lambda gated, R, bucket, decode_ar=None: ("out", R)
    g._forward_verify_recurrent_fused = lambda x, valid_len, T, n_users, *a, **kw: ("fused", valid_len, n_users)
    return g


@pytest.mark.parametrize("b", [1, 2, 4, 8])
def test_verify_accepts_bucket_rows_and_passes_max_users(monkeypatch, b):
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    calls = []
    g = _bucket_layer(8, 4, calls)
    assert TPGatedDeltaNet.forward_verify_recurrent(g, "x", b * 4, n_users=b, spec_ctrl="ctrl") == ("fused", b * 4, b)
    assert TPGatedDeltaNet._verify_fullbatch_fused(g, "qkv", 4, b, 32, "ctrl") == ("out", b * 4)
    assert calls == [((4, b), {"max_users": 8})], "T, B=b rows, max_users = the shared state width"


def test_verify_rejects_rows_outside_one_to_B(expect_error):
    g = _bucket_layer(8, 4, [])
    for n in (0, 9):
        with expect_error(AssertionError, "rows must be in 1..B"):
            TPGatedDeltaNet.forward_verify_recurrent(g, "x", 4, n_users=n, spec_ctrl="ctrl")
    g._spec_shape = (4, 4)
    with expect_error(AssertionError, "buffers are sized for"):
        TPGatedDeltaNet.forward_verify_recurrent(g, "x", 8, n_users=2, spec_ctrl="ctrl")


# --------------------------------------------------------------------------- #
# QWEN36_SPEC_STEP_VALUE_BLOCKS (gdn_spec_step value_blocks: state column split over cores)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env,want", [(None, 1), ("", 1), ("1", 1), ("2", 2), (" 4 ", 4)])
def test_value_blocks_env(monkeypatch, env, want):
    if env is None:
        monkeypatch.delenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", env)
    assert gdn_spec_step_value_blocks() == want


@pytest.mark.parametrize("env", ["3", "8", "0", "-2"])
def test_value_blocks_env_rejects_other_values(monkeypatch, env):
    monkeypatch.setenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", env)
    with pytest.raises(ValueError, match="QWEN36_SPEC_STEP_VALUE_BLOCKS"):
        gdn_spec_step_value_blocks()


def _spec_step_layer():
    g = types.SimpleNamespace(
        tw={"conv_taps_rows": "taps", "dt_bias_fp32": "dtb", "neg_exp_A_fp32": "nea", "norm_w_1d": "w"},
        Nv=6,
        Nk=2,
        Dk=128,
        Dv=128,
        qkvz_dim_tp=2048,
        K=4,
        scale=128**-0.5,
        _decode_step_fp32_state=True,
        _spec_max_users=0,
    )
    g._spec_hnew_depth = TPGatedDeltaNet._spec_hnew_depth
    return g


@pytest.mark.parametrize("env,want", [(None, None), ("1", None), ("2", 2), ("4", 4)])
def test_gdn_spec_step_passes_value_blocks_only_when_split(monkeypatch, env, want):
    if env is None:
        monkeypatch.delenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", env)
    calls = []
    kda = types.SimpleNamespace(gdn_spec_step=lambda *a, **kw: calls.append((a, kw)) or "gated")
    monkeypatch.setattr(ttnn, "experimental", types.SimpleNamespace(kda=kda), raising=False)
    g = _spec_step_layer()
    assert TPGatedDeltaNet._gdn_spec_step(g, "qkv", "wa", "wb", "ring", "ctrl", 4, 8, max_users=8) == "gated"
    ((args, kw),) = calls
    assert args[:5] == ("qkv", "wa", "wb", "ring", "ctrl") and args[13:15] == (4, 8)
    assert kw.get("value_blocks") == want
    assert kw["max_users"] == 8 and kw["fast_mode"] is True and kw["fp32_state"] is True


@pytest.mark.parametrize("env,T,want", [("2", 1, None), ("4", 1, None), ("4", 2, 4), ("2", 12, 2)])
def test_gdn_spec_step_value_blocks_only_for_verify(monkeypatch, env, T, want):
    monkeypatch.setenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", env)
    monkeypatch.delenv("QWEN36_SPEC_STEP_OPT", raising=False)
    calls = []
    kda = types.SimpleNamespace(gdn_spec_step=lambda *a, **kw: calls.append((a, kw)) or "gated")
    monkeypatch.setattr(ttnn, "experimental", types.SimpleNamespace(kda=kda), raising=False)
    g = _spec_step_layer()
    TPGatedDeltaNet._gdn_spec_step(g, "qkv", "wa", "wb", "ring", "ctrl", T, 8, max_users=8)
    ((_, kw),) = calls
    assert kw.get("value_blocks") == want


# --------------------------------------------------------------------------- #
# QWEN36_SPEC_STEP_OPT (gdn_spec_step kernel_opts: bit-identical kernel restructurings)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "env,want", [(None, 0), ("", 0), ("0", 0), ("1", 1), ("7", 7), ("0xff", 255), (" 255 ", 255), ("130", 130)]
)
def test_kernel_opts_env(monkeypatch, env, want):
    if env is None:
        monkeypatch.delenv("QWEN36_SPEC_STEP_OPT", raising=False)
    else:
        monkeypatch.setenv("QWEN36_SPEC_STEP_OPT", env)
    assert gdn_spec_step_kernel_opts() == want


@pytest.mark.parametrize("env", ["256", "0x100", "-1", "abc", "1.5"])
def test_kernel_opts_env_rejects(monkeypatch, env):
    monkeypatch.setenv("QWEN36_SPEC_STEP_OPT", env)
    with pytest.raises(ValueError, match="QWEN36_SPEC_STEP_OPT"):
        gdn_spec_step_kernel_opts()


def test_kernel_opts_known_mask():
    assert SPEC_STEP_OPT_KNOWN == 255


@pytest.mark.parametrize(
    "env,fp32_state,want", [(None, True, None), ("0", True, None), ("255", True, 255), ("7", False, None)]
)
def test_gdn_spec_step_passes_kernel_opts_only_when_set(monkeypatch, env, fp32_state, want):
    monkeypatch.delenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", raising=False)
    if env is None:
        monkeypatch.delenv("QWEN36_SPEC_STEP_OPT", raising=False)
    else:
        monkeypatch.setenv("QWEN36_SPEC_STEP_OPT", env)
    calls = []
    kda = types.SimpleNamespace(gdn_spec_step=lambda *a, **kw: calls.append((a, kw)) or "gated")
    monkeypatch.setattr(ttnn, "experimental", types.SimpleNamespace(kda=kda), raising=False)
    g = _spec_step_layer()
    g._decode_step_fp32_state = fp32_state
    TPGatedDeltaNet._gdn_spec_step(g, "qkv", "wa", "wb", "ring", "ctrl", 4, 8, max_users=8)
    ((_, kw),) = calls
    assert kw.get("kernel_opts") == want
