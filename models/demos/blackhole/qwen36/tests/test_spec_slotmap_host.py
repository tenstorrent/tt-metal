# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the host slot map of MTPSpecEngine (QWEN36_SPEC_SLOTMAP): a runner slot_remap permutes the
runner-slot -> engine-row map instead of the GDN state on the device. No device.

The fakes replace the device with per-row host state that follows the device's row semantics: a verify advances
every active engine row from its ring slot mi_prev = accepted_counts - 1 and writes one ring slot per column, a
seed starts a fresh row from its prefill state, the drafter chains from the ring slot of the anchor, and (flag
off) remap_spec_slots permutes the rows. Both flag settings must give every request the same tokens and drafts
and leave the same state in the row that holds it.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_slotmap_host.py -q
"""

import random
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt.spec_engine import (
    PLACEHOLDER_TOKEN_ID,
    MTPSpecEngine,
    mask_unused_columns,
    reset_draft_skip,
    rows_to_engine,
    rows_to_runner,
    sanitize_verify_block,
    slot_map_after_remap,
    slot_map_inverse,
)

NB = 16  # page-table width in blocks of 64
V = 40  # fake vocabulary
MOD = 2**31 - 1


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B", [1, 2, 3, 5, 8])
def test_slot_map_round_trip(B):
    g = torch.Generator().manual_seed(B)
    for _ in range(20):
        phys = torch.randperm(B, generator=g).tolist()
        inv = slot_map_inverse(phys)
        assert slot_map_inverse(inv) == phys
        assert all(inv[phys[s]] == s for s in range(B))
        x = torch.randn(B, 3, generator=g)
        rows = rows_to_engine(x, inv)
        assert all(torch.equal(rows[phys[s]], x[s]) for s in range(B))
        assert torch.equal(rows_to_runner(rows, phys), x)


@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_slot_map_after_remap_tracks_requests(B):
    """Runner rows hold request ids and are permuted like the plugin's condense; a request's engine row never moves."""
    rng = random.Random(B)
    phys, rows = list(range(B)), list(range(B))  # rows[i] = request at runner row i
    home = {rid: phys[i] for i, rid in enumerate(rows)}
    for _ in range(50):
        remap = list(range(B))
        rng.shuffle(remap)
        rows = [rows[j] for j in remap]
        phys = slot_map_after_remap(phys, remap)
        assert sorted(phys) == list(range(B))
        assert all(phys[i] == home[rid] for i, rid in enumerate(rows))


# --------------------------------------------------------------------------- #
# Engine bookkeeping with fakes
# --------------------------------------------------------------------------- #
class _NoDevice:
    """GDN layer stand-in that fails on any call: a host slot map remap must not touch the device."""

    def __getattr__(self, name):
        raise AssertionError(f"device call {name} during a host slot map operation")


def _engine(B, K=3, slotmap=True, gdn=None, **kw):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size, e.nb = B, K, K + 1, 64, NB
    e.host_slotmap, e._phys, e._phys_perm = slotmap, list(range(B)), None
    e._active, e._fresh, e._offered = set(), {}, set()
    e._parked, e._prompts, e._fresh_dirty = {}, {}, set()
    e._no_draft, e._speculated, e._withheld_streak = set(), set(), {}
    e._row_active, e._last_pt, e._last_cap_blocks, e.skipped_propose_calls = None, None, NB, 0
    e.no_draft_streak, e.captured, e._timing = 16, True, False
    e._gdn = [_NoDevice() if gdn is None else gdn]
    e.model = types.SimpleNamespace(_fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt, dtype=torch.int32))
    e._last_hidden, e._mtp_pt_rows, e._vfeed = [None] * B, [None] * B, None
    e._stage_mtp_tables = e._warm_mtp_last = lambda *a, **k: None
    e.__dict__.update(kw)
    return e


def test_flag_read(monkeypatch):
    """QWEN36_SPEC_SLOTMAP is read like the other QWEN36_* flags: unset or anything but "0" turns the map on."""
    import ttnn

    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda *a, **k: None)
    args = types.SimpleNamespace(vocab_size=8, dim=16, max_batch_size=2)
    model = types.SimpleNamespace(
        mtp=object(),
        num_devices=2,
        args=args,
        mesh_device=types.SimpleNamespace(arch=lambda: None),
        gdn_fused_decode=True,
        layers=[],
    )
    for val, want in (("1", True), ("0", False), (None, True), ("2", True)):
        if val is None:
            monkeypatch.delenv("QWEN36_SPEC_SLOTMAP", raising=False)
        else:
            monkeypatch.setenv("QWEN36_SPEC_SLOTMAP", val)
        e = MTPSpecEngine(model, 2, 3)
        assert e.host_slotmap is want and e._phys == [0, 1] and e._phys_perm is None


@pytest.mark.parametrize("B", [2, 4, 8])
def test_remap_slots_host_map_touches_no_device_or_set(B, expect_error):
    state = dict(
        _active={0, 1},
        _fresh={1: 9},
        _offered={0},
        _no_draft={1},
        _speculated={0},
        _withheld_streak={0: 2},
        _parked={1: ["s"]},
        _prompts={1: ("p", None)},
        _fresh_dirty={1},
        _last_hidden=[f"h{u}" for u in range(B)],
    )
    e = _engine(B, **{k: (v.copy() if hasattr(v, "copy") else v) for k, v in state.items()})
    rng = random.Random(B)
    phys = list(range(B))
    for _ in range(10):
        remap = list(range(B))
        rng.shuffle(remap)
        e.remap_slots(remap)
        phys = slot_map_after_remap(phys, remap)
        assert e._phys == phys
        assert (e._phys_perm is None) == (phys == list(range(B)))
        if e._phys_perm is not None:
            assert e._phys_perm[0].tolist() == phys and e._phys_perm[1].tolist() == slot_map_inverse(phys)
    for k, v in state.items():
        assert getattr(e, k) == v, k
    with expect_error(ValueError, "not a permutation"):
        e.remap_slots([0] * B)
    assert e._phys == phys


def test_remap_slots_device_path_moves_state():
    """QWEN36_SPEC_SLOTMAP=0: the old path, every GDN layer remaps and the per-slot host keys are renumbered."""
    calls = []
    gdn = types.SimpleNamespace(remap_spec_slots=lambda idx: calls.append(list(idx)))
    e = _engine(
        3,
        slotmap=False,
        gdn=gdn,
        _active={0, 2},
        _fresh={2: 5},
        _offered={0},
        _no_draft={2},
        _speculated={0},
        _withheld_streak={0: 1},
        _parked={2: ["s"]},
        _prompts={2: ("p", None)},
        _fresh_dirty={2},
        _last_hidden=["h0", "h1", "h2"],
    )
    e.remap_slots([2, 0, 1])  # row 0 <- slot 2, row 1 <- slot 0, row 2 <- slot 1
    assert calls == [[2, 0, 1]] and e._phys == [0, 1, 2] and e._phys_perm is None
    assert e._active == {0, 1} and e._fresh == {0: 5} and e._offered == {1} and e._no_draft == {0}
    assert e._speculated == {1} and e._withheld_streak == {1: 1} and set(e._parked) == {0} and set(e._prompts) == {0}
    assert e._fresh_dirty == {0} and e._last_hidden == ["h2", "h0", "h1"]
    e.remap_slots([0, 1, 2])
    assert calls == [[2, 0, 1]], "identity remap must not reach the device"


@pytest.mark.parametrize("spec_mode", ["argmax_ids", "logits"])
def test_decode_forward_scatters_inputs_and_gathers_outputs(spec_mode):
    B, K = 4, 3
    T = K + 1
    seen = {}

    def verify(tokens, start_pos, accepted_counts, page_table, read_logits=False, active=None, num_valid_drafts=None):
        seen.update(
            tokens=tokens.clone(),
            start_pos=torch.as_tensor(start_pos).clone(),
            acc=torch.as_tensor(accepted_counts).clone(),
            pt=torch.as_tensor(page_table).clone(),
            active=torch.as_tensor(active).clone(),
            nv=torch.as_tensor(num_valid_drafts).clone(),
        )
        ids = tokens.to(torch.int32) * 10 + 1  # the row's content follows the row
        logits = None
        if read_logits:
            logits = torch.zeros(B, T, V)
            logits[..., 0] = tokens.float()
        return ids, logits

    e = _engine(B, K, verify=verify)
    e.remap_slots([2, 0, 3, 1])
    phys = list(e._phys)
    tok = torch.arange(B * T, dtype=torch.int32).reshape(B, T) + 100
    nv = torch.tensor([3, 0, 1, 2], dtype=torch.int32)
    pos = torch.full((B, T), -1, dtype=torch.int32)
    for u in (0, 2, 3):
        pos[u, : int(nv[u]) + 1] = 50 + 10 * u + torch.arange(int(nv[u]) + 1, dtype=torch.int32)
    # runner row 1 is a padding row
    acc = torch.tensor([1, 4, 2, 3], dtype=torch.int32)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB)
    res = e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt, spec_mode=spec_mode)
    clean, start0, active = sanitize_verify_block(tok, pos, nv, K)
    for s in range(B):
        r = phys[s]
        assert torch.equal(seen["tokens"][r], clean[s]) and int(seen["start_pos"][r]) == int(start0[s])
        assert bool(seen["active"][r]) is bool(active[s]) and int(seen["nv"][r]) == int(nv[s])
        assert int(seen["acc"][r]) == int(acc[s]) and torch.equal(seen["pt"][r], pt[s])
    assert torch.equal(res.argmax_ids, mask_unused_columns(clean.to(torch.int32) * 10 + 1, nv))
    if spec_mode == "logits":
        assert torch.equal(res.logits[..., 0], clean.float())
    else:
        assert res.logits is None


def test_decode_forward_identity_map_passes_objects_through():
    B, K = 2, 2
    T = K + 1
    seen = {}

    def verify(tokens, start_pos, accepted_counts, page_table, **kw):
        seen.update(pt=page_table, acc=accepted_counts, nv=kw["num_valid_drafts"])
        return torch.zeros(B, T, dtype=torch.int32), None

    e = _engine(B, K, verify=verify)
    e.remap_slots([1, 0])
    e.remap_slots([1, 0])  # back to the identity: no index tensors, no copies
    assert e._phys == [0, 1] and e._phys_perm is None
    tok = torch.ones(B, T, dtype=torch.int32)
    pos = (torch.arange(T) + 5).repeat(B, 1).to(torch.int32)
    nv, acc, pt = torch.zeros(B, dtype=torch.int32), torch.ones(B, dtype=torch.int32), torch.zeros(B, NB)
    e.decode_forward(tok, pos, num_valid_drafts=nv, accepted_counts=acc, page_table=pt)
    assert seen["pt"] is pt and seen["acc"] is acc and seen["nv"] is nv


def test_propose_draft_tokens_round_trip(expect_error):
    B, K = 4, 3
    T = K + 1
    seen = {}

    def propose(ctok, cpos, acc, page_table, decline=None, cap_blocks=None, keep_kv=None):
        seen.update(
            ctok=ctok.clone(),
            cpos=cpos.clone(),
            acc=acc.clone(),
            pt=page_table,
            decline=list(decline),
            keep=list(keep_kv),
        )
        d = torch.zeros(B, K, dtype=torch.int32)
        for r in range(B):
            d[r] = ctok[r, 0] * 10 + torch.arange(K)  # the row's content follows the row
        return d

    e = _engine(B, K, propose=propose)
    e.remap_slots([3, 1, 0, 2])
    phys = list(e._phys)
    # the last verify ran runner rows 0..2 and padded row 3; runner row 2's drafts are withheld
    row_active = torch.ones(B, dtype=torch.bool)
    row_active[phys[3]] = False
    e._row_active, e._no_draft = row_active, {phys[2]}
    ctok = torch.full((B, T), -1, dtype=torch.int32)
    cpos = torch.full((B, T), -1, dtype=torch.int32)
    acc = torch.tensor([2, 1, 3, 1], dtype=torch.int32)
    for u in range(3):
        c = int(acc[u])
        ctok[u, :c] = 10 * (u + 1) + torch.arange(c, dtype=torch.int32)
        cpos[u, :c] = 100 * (u + 1) + torch.arange(c, dtype=torch.int32)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB)
    res = e.propose_draft_tokens(K, ctok, cpos, acc, pt)
    for s in range(B):
        r = phys[s]
        assert torch.equal(seen["ctok"][r], ctok[s]) and torch.equal(seen["cpos"][r], cpos[s])
        assert torch.equal(seen["pt"][r], pt[s])
        assert int(seen["acc"][r]) == (1 if s == 3 else int(acc[s]))
    assert seen["decline"] == [r in (phys[2], phys[3]) for r in range(B)]
    assert seen["keep"] == [r == phys[2] for r in range(B)]
    assert res.num_valid.tolist() == [K, K, 0, 0] and e._offered == {phys[0], phys[1]}
    for s in (0, 1):
        assert res.draft_token_ids[s].tolist() == (ctok[s, 0] * 10 + torch.arange(K)).tolist()
    # page_table None: the last verify's table is already in engine rows and is passed as is
    e._last_pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1000
    e.propose_draft_tokens(K, ctok, cpos, acc, None)
    assert seen["pt"] is e._last_pt
    with expect_error(ValueError, "row 1: negative committed token"):
        bad = ctok.clone()
        bad[1, 0] = -1
        e.propose_draft_tokens(K, bad, cpos, acc, pt)


def test_prefill_and_release_translate_runner_slots(expect_error):
    B = 4
    rows = []
    e = _engine(B)

    def prefill_row(row, prompt_ids, page_table_row, _cap_blocks=None):
        rows.append(row)
        e._fresh[row] = len(prompt_ids)
        e._active.add(row)
        return torch.zeros(V)

    e._prefill_row = prefill_row
    e.remap_slots([1, 2, 3, 0])
    phys = list(e._phys)
    e.prefill(2, [1, 2, 3], torch.zeros(1, NB))
    assert rows == [phys[2]] and e._active == {phys[2]} and e._fresh == {phys[2]: 3}
    e._offered, e._withheld_streak, e._no_draft = {phys[2]}, {phys[2]: 4}, {phys[0]}
    e.release(2)
    assert e._active == set() and e._fresh == {} and e._offered == set() and e._no_draft == set()
    with expect_error(AssertionError, "out of range"):
        e.prefill(B, [1], None)


def test_withheld_draft_bookkeeping_in_engine_rows():
    B, K = 3, 2
    T = K + 1
    e = _engine(B, K, verify=lambda *a, **k: (torch.zeros(B, T, dtype=torch.int32), None), no_draft_streak=2)
    e.remap_slots([2, 0, 1])
    phys = list(e._phys)
    tok = torch.ones(B, T, dtype=torch.int32)
    pos = torch.full((B, T), -1, dtype=torch.int32)
    pos[:, 0] = 5
    pos[1] = torch.arange(5, 5 + T, dtype=torch.int32)
    nv = torch.tensor([0, 2, 0], dtype=torch.int32)  # runner row 0 withheld its drafts, row 1 verified with 2
    kw = dict(accepted_counts=torch.ones(B, dtype=torch.int32), page_table=torch.zeros(B, NB, dtype=torch.int32))
    e._offered = {phys[0], phys[1]}  # the last propose gave drafts to runner rows 0 and 1
    e.decode_forward(tok, pos, num_valid_drafts=nv, **kw)
    assert e._withheld_streak == {phys[0]: 1, phys[1]: 0} and e._speculated == {phys[1]} and e._no_draft == set()
    e._offered = {phys[0]}
    e.decode_forward(tok, pos, num_valid_drafts=nv, **kw)
    assert e._no_draft == {phys[0]}, "the streak limit marks the engine row of runner row 0"


# --------------------------------------------------------------------------- #
# Churn: flag on vs flag off on a fake device
# --------------------------------------------------------------------------- #
def _mix(s, t, p):
    return (s * 1000003 + int(t) * 7919 + int(p) * 104729 + 12345) % MOD


def _next(s, t):
    return (s * 31 + int(t) * 17 + 3) % V


def _draft_of(s, t):
    """The fake drafter: right unless the state says otherwise (about 3 of 4 depths accepted)."""
    n = _next(s, t)
    return n if (s // 7) % 4 else (n + 1) % V


class _FakeDevice:
    """Per-engine-row state as the device holds it: ``rec`` (durable prefill state) and ``ring`` (one state per
    verify column), both permuted by remap_spec_slots (flag off). ``attach`` binds it to an engine."""

    def __init__(self, B, T):
        self.B, self.T = B, T
        self.rec = [0] * B
        self.ring = [[0] * T for _ in range(B)]
        self.remaps = 0

    def remap_spec_slots(self, idx):
        idx = [int(i) for i in idx]
        if idx == list(range(self.B)):
            return
        self.rec = [self.rec[j] for j in idx]
        self.ring = [self.ring[j] for j in idx]
        self.remaps += 1

    def verify_traced(self, tokens, positions, mi_prev, read_logits=False, page_tables=None, inactive=None):
        B, T = self.B, self.T
        ids = [[0] * T for _ in range(B)]
        logits = torch.zeros(B * T, V) if read_logits else None
        for r in range(B):
            if inactive is not None and inactive[r]:
                continue
            s = self.ring[r][int(mi_prev[r])]
            for j in range(T):
                s = _mix(s, tokens[r][j], int(positions[r]) + j)
                self.ring[r][j] = s
                ids[r][j] = _next(s, tokens[r][j])
                if logits is not None:
                    logits[r * T + j, ids[r][j]] = 1.0 + s % 5
        return [x for row in ids for x in row], "feed", logits

    def attach(self, e):
        B, T, K = self.B, self.T, self.T - 1
        dev = self

        def prefill_row(row, prompt_ids, page_table_row, _cap_blocks=None):
            reset_draft_skip(e, row)
            e._parked.pop(row, None)
            e._fresh_dirty.discard(row)
            prompt = [int(t) for t in torch.as_tensor(prompt_ids).reshape(-1)]
            e._prompts[row] = (prompt, torch.as_tensor(page_table_row).clone())
            s = 0
            for p, t in enumerate(prompt):
                s = _mix(s, t, p)
            dev.rec[row] = s
            e._fresh[row] = len(prompt)
            e._active.add(row)
            logits = torch.zeros(V)
            logits[_next(s, prompt[-1])] = 1.0
            return logits

        def seed_rows(rows, tokens, start_pos, ids, logits):
            for r in rows:
                s = _mix(dev.rec[r], int(tokens[r, 0]), int(start_pos[r]))
                dev.ring[r][0] = s
                ids[r, 0] = _next(s, int(tokens[r, 0]))
                if logits is not None:
                    logits[r, 0, int(ids[r, 0])] = 1.0 + s % 5

        def seed(tokens, start_pos, pt, read_logits):
            ids = torch.zeros(B, T, dtype=torch.int32)
            logits = torch.zeros(B, T, V) if read_logits else None
            seed_rows(range(B), tokens, start_pos, ids, logits)  # the device seed runs every row
            return ids, logits

        def join(tokens, start_pos, mi_prev, pt, active, fresh, read_logits):
            run = [r for r in range(B) if bool(active[r]) and r not in fresh]
            flat, _, lt = dev.verify_traced(
                tokens.tolist(), start_pos.tolist(), mi_prev.tolist(), read_logits, pt, [r not in run for r in range(B)]
            )
            ids = torch.tensor(flat, dtype=torch.int32).reshape(B, T)
            logits = None if lt is None else lt.reshape(B, T, V)
            seed_rows(fresh, tokens, start_pos, ids, logits)
            return ids, logits

        def propose(ctok, cpos, acc, page_table, decline=None, cap_blocks=None, keep_kv=None):
            drafts = torch.zeros(B, K, dtype=torch.int32)
            for r in range(B):
                if decline[r]:
                    continue
                m = int(acc[r]) - 1
                s, tok, pos = dev.ring[r][m], int(ctok[r, m]), int(cpos[r, m])
                for k in range(K):
                    s = _mix(s, tok, pos + k)
                    tok = _draft_of(s, tok)
                    drafts[r, k] = tok
            return drafts

        e._prefill_row, e._seed, e._join, e.propose = prefill_row, seed, join, propose
        e._gdn = [dev]
        e.model = types.SimpleNamespace(
            _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt, dtype=torch.int32), verify_traced=dev.verify_traced
        )


def _churn(B, K, seed, slotmap, steps=40):
    """Host mirror of the plugin runner (runner rows, condense permutations, arrivals into the lowest free rows,
    finishes by output length) over a fake device. Returns a dict: finished outputs, live outputs, drafts per
    step, device state per live request, device remaps, multi-token commits, engine."""
    T = K + 1
    rng = random.Random(seed)
    dev = _FakeDevice(B, T)
    e = _engine(B, K, slotmap=slotmap)
    dev.attach(e)
    reqs = [None] * B  # runner rows
    outs, drafts_log, next_rid, multi = {}, [], 0, 0
    for _ in range(steps):
        for r in range(B):  # finished requests leave before the step
            q = reqs[r]
            if q is not None and len(q["out"]) >= q["max_new"]:
                e.release(r)
                outs[q["rid"]] = q["out"]
                reqs[r] = None
        for r in range(B):  # arrivals fill the lowest free rows
            if reqs[r] is None and rng.random() < 0.35:
                P = rng.randint(3, 20)
                prompt = [rng.randrange(V) for _ in range(P)]
                q = {"rid": next_rid, "P": P, "proposal": [], "acc": 1, "max_new": rng.randint(2, 12)}
                q["pt"] = torch.full((NB,), 1 + next_rid, dtype=torch.int32)
                next_rid += 1
                logits = e.prefill(r, prompt, q["pt"][None])
                q["out"] = [int(logits.argmax())]
                reqs[r] = q
        remap = None
        if rng.random() < 0.5:  # the runner condensed (or otherwise permuted) its rows
            remap = list(range(B))
            rng.shuffle(remap)
            reqs = [reqs[j] for j in remap]
        live = [r for r in range(B) if reqs[r] is not None]
        if not live:
            if remap is not None:
                e.remap_slots(remap)
            continue
        tokens = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        start = torch.full((B, T), -1, dtype=torch.int32)
        nvd, acc = torch.zeros(B, dtype=torch.int32), torch.ones(B, dtype=torch.int32)
        pt = torch.zeros(B, NB, dtype=torch.int32)
        pos0, drafts, ns = {}, {}, {}
        for r in live:
            q = reqs[r]
            pos0[r] = q["P"] + len(q["out"]) - 1
            drafts[r] = list(q["proposal"])
            n = ns[r] = len(drafts[r])
            tokens[r, : 1 + n] = torch.tensor([q["out"][-1]] + drafts[r], dtype=torch.int32)
            start[r, : 1 + n] = torch.arange(pos0[r], pos0[r] + 1 + n, dtype=torch.int32)
            nvd[r], acc[r], pt[r] = n, q["acc"], q["pt"]
        mode = "logits" if rng.random() < 0.3 else "argmax_ids"
        res = e.decode_forward(
            tokens, start, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt, spec_mode=mode, slot_remap=remap
        )
        ids_all = res.argmax_ids
        committed = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        cpos = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        acc_vec = torch.ones(B, dtype=torch.int32)
        for r in live:
            q, n = reqs[r], ns[r]
            ids = ids_all[r].tolist()
            assert all(i == PLACEHOLDER_TOKEN_ID for i in ids[n + 1 :]) and all(i >= 0 for i in ids[: n + 1])
            if mode == "logits":
                assert res.logits[r, : n + 1].argmax(-1).tolist() == ids[: n + 1], "logits rows do not follow ids"
            m = 0
            while m < n and drafts[r][m] == ids[m]:
                m += 1
            new = drafts[r][:m] + [ids[m]]
            multi += len(new) > 1
            q["out"].extend(new)
            q["acc"] = len(new)
            committed[r, : len(new)] = torch.tensor(new, dtype=torch.int32)
            cpos[r, : len(new)] = torch.arange(pos0[r] + 1, pos0[r] + 1 + len(new), dtype=torch.int32)
            acc_vec[r] = len(new)
        d = e.propose_draft_tokens(K, committed, cpos, acc_vec, None if rng.random() < 0.3 else pt)
        for r in range(B):
            nv = int(d.num_valid[r])
            if r in pos0:
                reqs[r]["proposal"] = d.draft_token_ids[r, :nv].tolist()
            else:
                assert nv == 0, f"padding row {r} proposed {nv} drafts"
        drafts_log.append([(reqs[r]["rid"], d.draft_token_ids[r, : int(d.num_valid[r])].tolist()) for r in live])
    live_out = {q["rid"]: list(q["out"]) for q in reqs if q is not None}
    # flag off: _phys is the identity, so the same expression reads engine row r
    state = {q["rid"]: (dev.rec[e._phys[r]], list(dev.ring[e._phys[r]])) for r, q in enumerate(reqs) if q is not None}
    assert e._active == {e._phys[r] for r in range(B) if reqs[r] is not None}
    return dict(outs=outs, live=live_out, drafts=drafts_log, state=state, remaps=dev.remaps, multi=multi, eng=e)


@pytest.mark.parametrize(
    "B,K,seed",
    [
        (1, 3, 0),
        (2, 3, 1),
        (3, 3, 2),
        (4, 3, 3),
        (5, 3, 4),
        (6, 3, 5),
        (7, 3, 6),
        (8, 3, 7),
        (1, 7, 8),
        (2, 7, 9),
        (4, 7, 10),
    ],
)
def test_churn_host_map_matches_device_remap(B, K, seed):
    """The same random churn schedule (arrivals, finishes, condense permutations, logits / argmax steps, drafter
    tables from the call or from the last verify) gives identical token streams, drafts and per-request device
    state with the host slot map and with the device remap; the host map never remaps the fake device."""
    on = _churn(B, K, seed, slotmap=True)
    off = _churn(B, K, seed, slotmap=False)
    for key in ("outs", "live", "drafts", "state"):
        assert on[key] == off[key], f"{key} differ between the host slot map and the device remap"
    assert on["remaps"] == 0, "the host slot map remapped the device"
    if B > 1:
        assert off["remaps"] > 0, "the schedule never moved a row"
    else:
        assert off["remaps"] == 0
    assert len(on["outs"]) >= 2, "the schedule finished too few requests"
    assert on["multi"] > 0, "no multi-token commit: drafts never accepted"


def test_churn_accepts_drafts():
    """Sanity of the fake: the drafter is right often enough that multi-token commits happen, not always."""
    run = _churn(4, 3, 42, slotmap=True, steps=30)
    offered = sum(len(d) > 0 for step in run["drafts"] for _, d in step)
    assert offered > 0 and run["multi"] > 0 and run["multi"] < offered
