# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): chained MTP verifies (tt/spec_chain.py, MTPBlockEngine, vllm-tt-plugin's
supports_chained_spec_verify).

* The chain program's arithmetic (spec_chain.chain_graph, run with torch ops) writes the verify step buffer the host
  stages for the same block (the predecessor's committed token at the device's count, the drafts its drafter made,
  the positions after that commit, the RoPE rows and the acceptance rows), and each ctrl page the GDN verify tables of
  that count's start blocks (GdnBlockState.ctrl_candidates and verify_tables), over random predecessors: every count,
  padding rows, rows the chained verify drops, declined rows, M-RoPE deltas, held rows.
* MTPBlockEngine: a chained verify replays the chain, verify, step and drafter traces with the host rows in one
  overlay; the predecessor's propose returns the device drafts and fills the chained verify's block, the chained
  verify's own propose checks it against the device; a commit unequal to the device's raises; a predecessor that ran
  no device drafter, or a chained propose that depends on the count, is built from the engine's own walk instead.
"""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm_tt_plugin.spec_decode import accept_greedy_drafts

import ttnn
from models.demos.blackhole.qwen36.tests.test_mtp_gdn_blocks_host import ATTENTION, NUM_BLOCKS, TP8, _layout
from models.demos.blackhole.qwen36.tests.test_spec_accept_host import ROWS, VOCAB, _case, _lanes, _step_buffer
from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.kv_cache_spec import SPEC_STEP_HOLD
from models.demos.blackhole.qwen36.tt.spec_accept import TorchOps, graph, step_accept_reference
from models.demos.blackhole.qwen36.tt.spec_chain import chain_overlay, chain_reference, step_rows_reference
from models.demos.blackhole.qwen36.tt.spec_step import (
    STEP_DECLINED,
    STEP_DRAWN,
    STEP_KEEP,
    STEP_NUM_VALID,
    STEP_P0,
    STEP_ROPE_BASE,
    pack_width,
    rope_rows,
)


def _ctrl_words(B, nv):
    return -(-(1 + B + 2 * B * nv) // 16) * 16


def _chain_case(rng, B, K, groups, nv_heads):
    """A random predecessor (its ids, the drafter pack its step trace wrote, its drafts) and a chained verify over it
    (the overlay and random ctrl candidates), with the block and ctrl pages the host stages for that verify."""
    T = K + 1
    ids, tokens, accept = _case(rng, B, K)
    for u in range(B):
        if accept[STEP_P0][u] == 0:
            accept[STEP_DECLINED][u] = 1
    pt = torch.randint(1, 900, (B, 5), dtype=torch.int32)
    _, m, pack, _ = step_accept_reference(ids.reshape(-1), tokens.reshape(-1), accept, pt, 999, K, ROWS)
    m = m.reshape(-1).tolist()
    drafts = [torch.tensor([rng.randrange(VOCAB) for _ in range(B)]) for _ in range(K)]
    pred_active = [p > 0 for p in accept[STEP_P0]]
    active = [a and rng.random() < 0.85 for a in pred_active]
    hint = [rng.randrange(K + 1) for _ in range(B)]
    nv = [min(h, K) if a and not accept[STEP_DECLINED][u] else 0 for u, (a, h) in enumerate(zip(active, hint))]
    deltas = [rng.choice([0, -rng.randrange(1, 30), rng.randrange(1, 5)]) for _ in range(B)]
    declined = [not a or rng.random() < 0.15 for a in active]
    keep = [d and a and rng.random() < 0.5 for d, a in zip(declined, active)]
    p0_prev = [accept[STEP_P0][u] if active[u] else 0 for u in range(B)]
    overlay = chain_overlay(p0_prev, active, deltas, nv, declined, keep)
    width = 2 * nv_heads * groups
    cand = torch.randint(0, 1 << 20, (T * B, width), dtype=torch.int64)
    for u in range(B):
        if not active[u]:
            cand[u::B] = SPEC_STEP_HOLD
    # The block the host stages for the chained verify.
    tok = torch.zeros(B, T, dtype=torch.int64)
    p0 = torch.zeros(B, dtype=torch.int64)
    for u in range(B):
        if not active[u]:
            continue
        tok[u] = int(ids[u, m[u]])
        tok[u, 1 : nv[u] + 1] = torch.tensor([int(drafts[k][u]) for k in range(nv[u])], dtype=torch.int64)
        p0[u] = p0_prev[u] + m[u] + 1
    pos = (p0.reshape(B, 1) + torch.arange(T)).reshape(-1)
    rows = {
        STEP_NUM_VALID: nv,
        STEP_DRAWN: [0] * B,
        STEP_P0: p0.tolist(),
        STEP_ROPE_BASE: rope_rows(p0, deltas, ROWS).tolist(),
        STEP_DECLINED: [int(d) for d in declined],
        STEP_KEEP: [int(k) for k in keep],
    }
    rope = rope_rows(pos, [d for d in deltas for _ in range(T)], ROWS)
    want_step = step_rows_reference(tok.reshape(-1), pos, rope, rows, B, T)
    words = _ctrl_words(B, nv_heads)
    want_ctrls = []
    for g in range(groups):
        ctrl = torch.zeros(1, words, dtype=torch.int64)
        for u in range(B):
            pick = cand[m[u] * B + u, 2 * nv_heads * g : 2 * nv_heads * (g + 1)]
            ctrl[0, 1 + B + u * nv_heads : 1 + B + (u + 1) * nv_heads] = pick[:nv_heads]
            base = 1 + B + B * nv_heads
            ctrl[0, base + u * nv_heads : base + (u + 1) * nv_heads] = pick[nv_heads:]
        want_ctrls.append(ctrl.to(torch.int32))
    return SimpleNamespace(
        ids=_lanes(ids), pack=pack, overlay=overlay, cand=cand, drafts=drafts, step=want_step, ctrls=want_ctrls, m=m
    )


@pytest.mark.parametrize("B,K", [(1, 3), (2, 3), (4, 3), (8, 3), (4, 7), (2, 11)])
def test_the_chain_program_writes_the_block_the_host_stages(B, K):
    """chain_graph (the device arithmetic, in torch) == the step buffer verify_enqueue stages for the chained block and
    the ctrl pages of each row's count, over random predecessors and chained verifies."""
    rng = random.Random(B * 13 + K)
    groups, nv_heads = 3, 6
    for _ in range(40):
        case = _chain_case(rng, B, K, groups, nv_heads)
        step, ctrls = chain_reference(
            case.ids,
            case.pack,
            case.overlay,
            case.cand,
            case.drafts,
            B,
            K,
            ROWS,
            _ctrl_words(B, nv_heads),
            nv_heads,
            groups,
        )
        assert torch.equal(step.to(torch.int64), case.step.to(torch.int64))
        for got, want in zip(ctrls, case.ctrls):
            assert torch.equal(got, want)


def _block_state(T=4, bmax=4):
    from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout

    plugin = PluginLayout(ATTENTION, 1, 256, TP8, True, num_blocks=NUM_BLOCKS, block_size=1024)
    st = object.__new__(gbs.GdnBlockState)
    st.mesh, st.bmax, st.num_blocks, st.dims, st.layout = None, bmax, NUM_BLOCKS, TP8, _layout()
    st.layers = {i: SimpleNamespace(state_blocks=SimpleNamespace()) for i, a in enumerate(ATTENTION) if not a}
    st._group_names = st._group_layers = None
    st._decode_blocks, st._index_refs = {}, {}
    st.verify_tokens = T
    st._bind_verify(plugin.per_layer_specs)
    return st, plugin


@pytest.fixture
def copies(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: SimpleNamespace(host=t.clone(), dtype=kw.get("dtype")))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: calls.append((host, dev)))
    return calls


def test_the_ctrl_candidates_select_the_verify_tables_of_each_count(copies):
    """GdnBlockState.ctrl_candidates holds, per count, the start words verify_tables writes for that count's start
    blocks; picking each row's count gives the ctrl page of the chosen blocks, in every table group. A chained verify
    stages the window tables only, the ctrl pages are int32, and a page the chain program wrote is copied again by the
    next host-staged verify."""
    st, plugin = _block_state()
    B, T = st.bmax, st.verify_tokens
    rng = random.Random(5)
    groups = []
    for names in plugin.group_names:
        windows = torch.randint(1, NUM_BLOCKS, (B, T), dtype=torch.int32)
        windows[2] = 0  # a padding row: held at every count
        by_count = torch.randint(1, NUM_BLOCKS, (B, T), dtype=torch.int32)
        by_count[2] = 0
        groups.append(
            SimpleNamespace(
                layer_names=names, windows=windows, start_blocks=by_count[:, -1], start_blocks_by_count=by_count
            )
        )
    state = SimpleNamespace(groups=tuple(groups))
    copies.clear()
    st.set_verify_blocks(state, stage_ctrl=False)
    assert {id(dev) for _, dev in copies} == {id(t) for t in st._verify_idx}
    cand = st.ctrl_candidates(state)
    nv = TP8.nv
    for _ in range(10):
        m = [rng.randrange(T) for _ in range(B)]
        for g, grp in enumerate(state.groups):
            k = st._verify_order[1][g]
            starts = grp.start_blocks_by_count[torch.arange(B), torch.tensor(m)]
            want, _ = st.layout.verify_tables(starts, grp.windows, NUM_BLOCKS, B)
            sel = torch.stack([cand[m[u] * B + u, 2 * nv * k : 2 * nv * (k + 1)] for u in range(B)])
            got = torch.zeros_like(want)
            got[0, 1 + B : 1 + B + B * nv] = sel[:, :nv].reshape(-1).to(torch.int32)
            got[0, 1 + B + B * nv : 1 + B + 2 * B * nv] = sel[:, nv:].reshape(-1).to(torch.int32)
            assert torch.equal(got, want)
    assert bool((cand[2::B] == SPEC_STEP_HOLD).all()), "a padding row is held at every count"
    copies.clear()
    st.set_verify_blocks(state)
    ctrl_copies = [(h, d) for h, d in copies if any(d is t for t in st._verify_ctrl)]
    assert len(ctrl_copies) == plugin.num_groups and {h.dtype for h, _ in ctrl_copies} == {ttnn.int32}
    copies.clear()
    st.set_verify_blocks(state)
    assert copies == []
    st.ctrl_written()
    st.set_verify_blocks(state)
    assert {id(d) for _, d in copies} == {id(t) for t in st._verify_ctrl}


# ----------------------------------------------------------------------------------------- MTPBlockEngine


@pytest.fixture
def chained(monkeypatch):
    """An MTPBlockEngine whose device is a stand-in: each verify's ids are the next of state["ids"]; the step trace runs
    as spec_accept.graph in torch over the step buffer the verify staged or the chain program wrote
    (spec_chain.chain_reference over the predecessor's ids, pack and drafts); the drafter answers state["drafts"]."""
    B, K = 4, 3
    T = K + 1
    log = {"traces": [], "traced": [], "enqueue": [], "staged": [], "overlay": [], "blocks": []}
    state = {"ids": [], "drafts": [torch.arange(100 * k, 100 * k + B) for k in range(K)]}
    gdn = SimpleNamespace(remap_spec_slots=lambda idx: None)
    blocks = SimpleNamespace(
        verify_tokens=T,
        set_verify_blocks=lambda s, stage_ctrl=True: log["blocks"].append(("set", stage_ctrl, s)),
        ctrl_candidates=lambda s: torch.zeros(T * B, 2 * 6 * 3),
        ctrl_written=lambda: log["blocks"].append(("written",)),
    )
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        mesh_device=MagicMock(),
        args=SimpleNamespace(vocab_size=VOCAB, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=gdn)],
        gdn_blocks=blocks,
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        spec_feed_rows=lambda h: h,
        rope=SimpleNamespace(rope_delta=0),
        rope_table_rows=lambda: ROWS,
        _vfy_rows_out="feed",
        _vfy_ids_out="ids",
        _vfy_sampler=SimpleNamespace(draws="draws"),
        _vfy_kv_grouped=False,
    )

    def verify_traced(tokens, positions, mi, **kw):
        log["traced"].append(kw)
        ids = state["ids"].pop(0)
        log["enqueue"].append(SimpleNamespace(tokens=tokens, positions=positions, accept=None, kw=kw, ids=ids))
        return [int(v) for v in ids.reshape(-1)], "feed", None

    def verify_enqueue(tokens, positions, mi, accept_rows=None, **kw):
        step = log["chain_step"].pop() if kw.get("step_staged") else None
        log["enqueue"].append(
            SimpleNamespace(
                tokens=tokens, positions=positions, accept=accept_rows, kw=kw, ids=state["ids"].pop(0), step=step
            )
        )

    model.verify_traced, model.verify_enqueue = verify_traced, verify_enqueue
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 8, 64, 999
    eng._active = set(range(B))
    eng._acc_trace, eng._dr_trace, eng._chain_trace = "step", "drafter", "chain"
    eng._accept = SimpleNamespace(stage_pt_rows=lambda rows: log.setdefault("pt_rows", []).append(rows))
    eng._chain = SimpleNamespace(stage=lambda overlay, cand: log["overlay"].append(overlay))
    eng._dr_pack, eng._mtp_pt, eng._dr_rs_pt, eng._dr_outs = "pack", "mtp_pt", "rs_pt", [f"d{k}" for k in range(K)]
    eng._stage = lambda host, buf, *a: log["staged"].append((buf, host))

    def execute_trace(mesh, trace, **kw):
        log["traces"].append(trace)
        if trace == "chain":
            pred = log["enqueue"][-1]
            step, _ = chain_reference(
                _lanes(pred.ids), pred.pack, log["overlay"][-1], torch.zeros(T * B, 36), state["drafts"],
                B, K, ROWS, _ctrl_words(B, 6), 6, 3,
            )  # fmt: skip
            log.setdefault("chain_step", []).append(step)

    monkeypatch.setattr(ttnn, "execute_trace", execute_trace)

    def read(tensors):
        e = log["enqueue"][-1]
        if getattr(e, "step", None) is not None:
            stepbuf = e.step
        else:
            stepbuf = _step_buffer(
                torch.tensor(e.tokens).reshape(-1), [0] * (B * T), [0] * (B * T), e.accept or {}, B, T
            )
        out = {"ids": _lanes(e.ids)}
        if e.accept is not None or getattr(e, "step", None) is not None:
            o = TorchOps(_lanes(e.ids), stepbuf, log["pt_rows"][-1], B, K, eng._scratch_block, ROWS)
            e.pack = graph(o, B, K, pack_width(B * T))[0].to(torch.int64)
            out["pack"] = e.pack
        out.update({f"d{k}": state["drafts"][k].reshape(1, 1, B, 1) for k in range(K)})
        return [out[t] for t in tensors]

    eng._read_device0 = read
    eng._read_device0_async = lambda tensors: SimpleNamespace(event="event", result=lambda r=read(tensors): r)
    return eng, log, state


def _host_verify(eng, tokens, starts, nv):
    T = eng.T
    start_pos = torch.tensor([[s + j if s >= 0 else -1 for j in range(T)] for s in starts])
    return eng.decode_forward(
        torch.as_tensor(tokens),
        start_pos,
        num_valid_drafts=torch.tensor(nv, dtype=torch.int32),
        accepted_counts=torch.ones(eng.B, dtype=torch.int32),
        page_table=torch.arange(1, eng.B * eng.nb + 1, dtype=torch.int32).reshape(eng.B, eng.nb),
        spec_mode="argmax_ids",
        mamba_state=_mamba(eng),
        defer=True,
    )


def _mamba(eng):
    by_count = torch.arange(eng.B * eng.T, dtype=torch.int32).reshape(eng.B, eng.T) + 1
    group = SimpleNamespace(layer_names=("l",), windows=torch.zeros(eng.B, eng.T), start_blocks=by_count[:, -1])
    group.start_blocks_by_count = by_count
    return SimpleNamespace(groups=(group,))


def _chained_verify(eng, res, upper_starts, hint, rows=None):
    """The runner's chained call: placeholders at the positions of a full acceptance, with ``res`` the predecessor."""
    B, T = eng.B, eng.T
    rows = [s >= 0 for s in upper_starts] if rows is None else rows
    tokens = torch.where(torch.tensor(rows).reshape(B, 1), -1, 0).expand(B, T)
    start_pos = torch.tensor([[s + j if r else -1 for j in range(T)] for s, r in zip(upper_starts, rows)])
    return eng.decode_forward(
        tokens,
        start_pos,
        num_valid_drafts=torch.tensor(hint, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=torch.arange(1, B * eng.nb + 1, dtype=torch.int32).reshape(B, eng.nb),
        spec_mode="argmax_ids",
        mamba_state=_mamba(eng),
        defer=True,
        predecessor=res.argmax_ids,
    )


def _walk(res, block, nv):
    ids = res.argmax_ids.result()
    return accept_greedy_drafts(ids, torch.as_tensor(block)[:, 1:].to(torch.int32), torch.tensor(nv, dtype=torch.int32))


def _propose(eng, starts, committed, counts):
    positions = torch.tensor([[s + 1 + j if s >= 0 else -1 for j in range(eng.T)] for s in starts])
    return eng.propose_draft_tokens(eng.K, committed, positions, counts)


def test_a_chained_verify_is_built_on_the_device_and_checked_at_both_proposes(chained):
    """The predecessor runs in one enqueue; the chained verify stages one overlay (the predecessor's column-0
    positions, its rows, their M-RoPE deltas, valid drafts and propose decisions), stages the window tables only, and
    replays the chain, verify, step and drafter traces. The predecessor's propose returns the device drafts and fills
    the chained block (its count's committed token, the drafts, the positions after the commit), whose own propose
    then checks the device's walk of it."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    ids0 = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [21, 22, 23, 24]])
    tokens0 = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [3, 21, 9, 9]])
    starts0, nv0 = [100, 200, -1, 300], [3, 3, 0, 2]
    drafts = torch.stack(state["drafts"], dim=1)  # [B, K]: what the drafter answers for every step
    ids1 = torch.zeros(B, T, dtype=torch.int64)
    ids1[0] = torch.tensor([int(drafts[0, 0]), int(drafts[0, 1]), 5, 6])  # accepts two drafts
    ids1[1] = torch.tensor([9, 9, 9, 9])  # accepts none
    ids1[3] = torch.tensor([int(drafts[3, 0]), 4, 4, 4])  # accepts one
    state["ids"] = [ids0, ids1]
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    # The runner's upper bound: column-0 position + 1 + K; the predecessor's padding row 2 stays padding.
    upper = [s + T if s >= 0 else -1 for s in starts0]
    res1 = _chained_verify(eng, res0, upper, [K] * B)
    assert log["traces"] == ["step", "drafter", "chain", "step", "drafter"]
    assert log["enqueue"][-1].kw["step_staged"] is True
    assert [b[1] for b in log["blocks"] if b[0] == "set"] == [True, False] and ("written",) in log["blocks"]
    overlay = log["overlay"][-1]
    assert overlay[0, :B].tolist() == [100, 200, 0, 300] and overlay[1, :B].tolist() == [1, 1, 0, 1]
    assert overlay[3, :B].tolist() == [3, 3, 0, 3]
    assert eng.chained_steps == 1 and eng.materialized_steps == 0
    committed0, counts0 = _walk(res0, tokens0, nv0)
    assert counts0.tolist() == [3, 4, 1, 2]
    out0 = _propose(eng, starts0, committed0, counts0)
    want = drafts.to(torch.int32).clone()
    want[2] = 0
    assert torch.equal(out0.draft_token_ids, want) and out0.num_valid.tolist() == [K, K, 0, K]
    succ = res0.argmax_ids.successor
    assert succ is res1.argmax_ids and succ.p0.tolist() == [103, 204, 0, 302]
    assert succ.tokens[0].tolist() == [9, *drafts[0].tolist()] and succ.tokens[1].tolist() == [14, *drafts[1].tolist()]
    # The runner fills the chained block from that propose and walks the chained ids.
    block1 = succ.tokens.clone()
    committed1, counts1 = _walk(res1, block1, [K, K, 0, K])
    assert counts1.tolist() == [3, 1, 1, 2]
    out1 = _propose(eng, succ.p0.where(succ.active, torch.tensor(-1)).tolist(), committed1, counts1)
    assert out1.num_valid.tolist() == [K, K, 0, K] and not eng._pending
    assert eng.one_enqueue_host_proposes == 0


def test_a_predecessor_count_unequal_to_the_runner_commit_raises(chained, expect_error):
    """The chained verify followed the device's count: a runner commit shorter than the device's (which an unchained
    step would draft again on the host) raises at the predecessor's propose."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    ids0 = torch.tensor([[7, 8, 9, 10]] * B)
    tokens0 = torch.tensor([[1, 7, 8, 9]] * B)
    starts0, nv0 = [100, 200, 300, 400], [3] * B
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    _chained_verify(eng, res0, [s + T for s in starts0], [K] * B)
    committed, counts = _walk(res0, tokens0, nv0)
    counts = counts.clone()
    counts[1] = 1
    committed = committed.clone()
    committed[1, 1:] = -1
    with expect_error(ValueError, "the chained verify and the one-enqueue drafts continue the device's commit"):
        _propose(eng, starts0, committed, counts)


def test_a_predecessor_without_a_device_drafter_is_read_and_walked_by_the_engine(chained, expect_error):
    """A predecessor that staged its drafter from the host (a row fits at some counts only) leaves no device drafts:
    the engine reads it, walks its ids, runs the propose itself and verifies the chained block on the host path. The
    runner's propose of the predecessor then returns those drafts, and raises for another commit."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    ids0 = torch.tensor([[7, 8, 9, 10]] * B)
    tokens0 = torch.tensor([[1, 7, 8, 9]] * B)
    starts0, nv0 = [100, 200, eng.nb * eng.block_size - 1 - K - 2, 400], [3] * B
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    assert res0.argmax_ids.plan is None
    proposes = []
    base = spec_engine.MTPSpecEngine.propose_draft_tokens

    def recorded(self, *args, **kw):
        proposes.append(args)
        return spec_engine.DraftResult(
            torch.full((B, K), 50, dtype=torch.int32), torch.tensor([K, K, 0, K], dtype=torch.int32)
        )

    spec_engine.MTPSpecEngine.propose_draft_tokens = recorded
    try:
        res1 = _chained_verify(eng, res0, [s + T for s in starts0], [K] * B, rows=[True, True, False, True])
    finally:
        spec_engine.MTPSpecEngine.propose_draft_tokens = base
    assert eng.materialized_steps == 1 and eng.chained_steps == 0 and len(proposes) == 1
    assert "chain" not in log["traces"]
    e = log["enqueue"][-1]
    assert e.tokens[0] == [10, 50, 50, 50] and e.positions[0] == 104  # all three drafts accepted: count 4
    committed, counts = _walk(res0, tokens0, nv0)
    out = _propose(eng, starts0, committed, counts)
    assert out.num_valid.tolist() == [K, K, 0, K]
    assert res1.argmax_ids is eng._pending[0]
    # Another commit of the predecessor raises.
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    spec_engine.MTPSpecEngine.propose_draft_tokens = recorded
    try:
        _chained_verify(eng, res0, [s + T for s in starts0], [K] * B, rows=[True, True, False, True])
    finally:
        spec_engine.MTPSpecEngine.propose_draft_tokens = base
    counts = counts.clone()
    counts[0] = 2
    with expect_error(ValueError, "the engine's walk"):
        _propose(eng, starts0, committed, counts)


def test_a_chained_propose_that_depends_on_the_count_is_built_from_the_read_predecessor(chained):
    """When the chained verify's own propose is not decided alike for every count the predecessor can accept (a row
    within reach of the paged KV end), the engine builds it from its read of the predecessor."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    ids0 = torch.tensor([[7, 8, 9, 10]] * B)
    tokens0 = torch.tensor([[1, 7, 8, 9]] * B)
    edge = eng.nb * eng.block_size - 9  # every count fits here, but not every count of the chained verify
    starts0, nv0 = [100, 200, edge, 400], [3] * B
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    assert res0.argmax_ids.plan is not None
    _chained_verify(eng, res0, [s + T for s in starts0], [K] * B)
    assert eng.materialized_steps == 1 and "chain" not in log["traces"]


def test_a_chained_row_must_be_a_predecessor_row(chained, expect_error):
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    state["ids"] = [torch.zeros(B, T, dtype=torch.int64)]
    res0 = _host_verify(eng, torch.zeros(B, T, dtype=torch.int64), [100, 200, -1, 300], [0, 0, 0, 0])
    with expect_error(ValueError, "are not rows of the predecessor"):
        _chained_verify(eng, res0, [104, 204, 300, 304], [K] * B)
