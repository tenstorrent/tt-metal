# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): chained MTP verifies (tt/spec_chain.py, MTPBlockEngine, vllm-tt-plugin's
supports_chained_spec_verify).

* The chain program's arithmetic (spec_chain.chain_graph, run with torch ops) writes the verify step buffer the host
  stages for the same block (the predecessor's committed token at the device's count, a draw for a drawn user, the
  drafts its drafter made, the positions after that commit, the RoPE rows and the acceptance rows with the chained
  verify's drawn users), each ctrl page the GDN verify tables of that count's start blocks
  (GdnBlockState.ctrl_candidates and verify_tables), and the sampler seeds of that count, over random predecessors:
  every count, padding rows, rows the chained verify drops, declined rows, M-RoPE deltas, held rows, drawn users.
* spec_sample.chained_sampling_plan: picked by any count, a seeded user's seeds are those an unchained verify (or a
  draftless narrow decode) at the same position draws with; an unseeded user's are fresh and the same at every
  count; a user that draws nothing seeds no core; a top-k the sampler cannot draw raises.
* MTPBlockEngine: a chained verify replays the chain, verify, step and drafter traces with the host rows in one
  overlay; the predecessor's propose returns the device drafts and fills the chained verify's block, the chained
  verify's own propose checks it against the device; a commit unequal to the device's raises; a predecessor that ran
  no device drafter, or a chained propose that depends on the count, is built from the engine's own walk instead. On
  a device-sampled step the same holds for sampled rows: the device counts over the draws, the chained verify
  stages no seeds (the chain program writes those of its positions), and a commit of the argmax instead of the draws
  raises.
"""

import inspect
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
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_accept import TorchOps, graph, step_accept_reference
from models.demos.blackhole.qwen36.tt.spec_chain import (
    CH_DRAWN,
    CH_DRAWN_PREV,
    chain_overlay,
    chain_reference,
    step_rows_reference,
)
from models.demos.blackhole.qwen36.tt.spec_engine import mask_unused_columns
from models.demos.blackhole.qwen36.tt.spec_sample import NO_CORE, chained_sampling_plan, sampling_plan
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


def _chain_case(rng, B, K, groups, nv_heads, sampler):
    """A random predecessor (its ids, with a device ``sampler`` its draws for the drawn users, the drafter pack its
    step trace wrote, its drafts) and a chained verify over it (the overlay, random ctrl candidates, with a sampler
    random seed candidates), with the block, ctrl pages and seeds the host stages for that verify."""
    T = K + 1
    ids, tokens, accept = _case(rng, B, K)
    for u in range(B):
        if accept[STEP_P0][u] == 0:
            accept[STEP_DECLINED][u] = 1
    draws = None
    if sampler:
        # A drawn user's draws repeat its ids for a random prefix, so the walk over the draws takes every count.
        draws = torch.tensor([rng.randrange(VOCAB) for _ in range(B * T)])
        for u in range(B):
            n = rng.randrange(T + 1)
            draws[u * T : u * T + n] = ids[u, :n]
    else:
        accept[STEP_DRAWN] = [0] * B
    pt = torch.randint(1, 900, (B, 5), dtype=torch.int32)
    merged, m, pack, _ = step_accept_reference(
        ids.reshape(-1), tokens.reshape(-1), accept, pt, 999, K, ROWS, draws=draws
    )
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
    drawn = [int(sampler and a and rng.random() < 0.5) for a in active]
    overlay = chain_overlay(p0_prev, active, deltas, nv, declined, keep, accept[STEP_DRAWN], drawn)
    seed_cand = torch.randint(1, 1_000_001, (T * B, T)).to(torch.float32) if sampler else None
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
        tok[u] = int(merged[u, m[u]])
        tok[u, 1 : nv[u] + 1] = torch.tensor([int(drafts[k][u]) for k in range(nv[u])], dtype=torch.int64)
        p0[u] = p0_prev[u] + m[u] + 1
    pos = (p0.reshape(B, 1) + torch.arange(T)).reshape(-1)
    rows = {
        STEP_NUM_VALID: nv,
        STEP_DRAWN: drawn,
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
    want_seeds = None
    if sampler:
        want_seeds = torch.stack([seed_cand[m[u] * B + u] for u in range(B)]).reshape(-1).to(torch.int64)
    return SimpleNamespace(
        ids=_lanes(ids),
        draws=None if draws is None else _lanes(draws),
        pack=pack,
        overlay=overlay,
        cand=cand,
        seed_cand=seed_cand,
        drafts=drafts,
        step=want_step,
        ctrls=want_ctrls,
        seeds=want_seeds,
        m=m,
        drawn_prev=accept[STEP_DRAWN],
        p0_prev=p0_prev,
        active=active,
    )


@pytest.mark.parametrize("sampler", [False, True])
@pytest.mark.parametrize("B,K", [(1, 3), (2, 3), (4, 3), (8, 3), (4, 7), (2, 11)])
def test_the_chain_program_writes_the_block_the_host_stages(B, K, sampler):
    """chain_graph (the device arithmetic, in torch) == the step buffer verify_enqueue stages for the chained block,
    the ctrl pages of each row's count and, with a device sampler, the seeds of that count, over random predecessors
    and chained verifies; a drawn predecessor user continues from its draw."""
    rng = random.Random(B * 13 + K + 1000 * sampler)
    groups, nv_heads = 3, 6
    counts, drawn_tokens = set(), 0
    for _ in range(40):
        case = _chain_case(rng, B, K, groups, nv_heads, sampler)
        step, ctrls, seeds = chain_reference(
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
            draws=case.draws,
            seed_cand=case.seed_cand,
        )
        assert torch.equal(step.to(torch.int64), case.step.to(torch.int64))
        for got, want in zip(ctrls, case.ctrls):
            assert torch.equal(got, want)
        assert (seeds is None) == (case.seeds is None)
        if seeds is not None:
            assert torch.equal(seeds, case.seeds)
        counts |= set(case.m)
        drawn_tokens += sum(case.drawn_prev)
    if K == 3 and B > 1:
        assert counts == set(range(K + 1)), "every count of the predecessor"
    assert bool(drawn_tokens) == sampler


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
    """An MTPBlockEngine whose device is a stand-in: each verify's ids are the next of state["ids"], and a device-sampled
    verify's draws the next of state["draws"]; the step trace runs as spec_accept.graph in torch over the step buffer
    the verify staged or the chain program wrote (spec_chain.chain_reference over the predecessor's ids, draws, pack
    and drafts, and the staged seed candidates); the drafter answers state["drafts"]."""
    B, K = 4, 3
    T = K + 1
    log = {
        "traces": [], "traced": [], "enqueue": [], "staged": [], "overlay": [], "seed_cand": [], "blocks": [],
        "chain_seeds": [],
    }  # fmt: skip
    state = {"ids": [], "draws": [], "drafts": [torch.arange(100 * k, 100 * k + B) for k in range(K)]}
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
        _vfy_sampler=SimpleNamespace(
            draws="draws", seeds="seeds", seeds_written=lambda: log["blocks"].append(("seeds_written",))
        ),
        _vfy_kv_grouped=False,
    )

    def verify_traced(tokens, positions, mi, **kw):
        log["traced"].append(kw)
        ids = state["ids"].pop(0)
        log["enqueue"].append(SimpleNamespace(tokens=tokens, positions=positions, accept=None, kw=kw, ids=ids))
        return [int(v) for v in ids.reshape(-1)], "feed", None

    def verify_enqueue(tokens, positions, mi, accept_rows=None, **kw):
        # Qwen36Model.verify_enqueue's own signature: a keyword it does not take raises here as on the device.
        inspect.signature(Qwen36Model.verify_enqueue).bind(model, tokens, positions, mi, accept_rows=accept_rows, **kw)
        step = log["chain_step"].pop() if kw.get("step_staged") else None
        draws = state["draws"].pop(0) if kw.get("sample_plan") is not None else None
        log["enqueue"].append(
            SimpleNamespace(
                tokens=tokens,
                positions=positions,
                accept=accept_rows,
                kw=kw,
                ids=state["ids"].pop(0),
                draws=draws,
                step=step,
            )
        )

    model.verify_traced, model.verify_enqueue = verify_traced, verify_enqueue
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 8, 64, 999
    eng._active = set(range(B))
    eng._acc_trace, eng._dr_trace, eng._chain_trace = "step", "drafter", "chain"
    eng._accept = SimpleNamespace(stage_pt_rows=lambda rows: log.setdefault("pt_rows", []).append(rows))
    eng.sample_on_device = True

    def stage(overlay, cand, seed_cand=None):
        log["overlay"].append(overlay)
        log["seed_cand"].append(seed_cand)

    eng._chain = SimpleNamespace(stage=stage)
    eng._dr_pack, eng._mtp_pt, eng._dr_rs_pt, eng._dr_outs = "pack", "mtp_pt", "rs_pt", [f"d{k}" for k in range(K)]
    eng._stage = lambda host, buf, *a: log["staged"].append((buf, host))

    def execute_trace(mesh, trace, **kw):
        log["traces"].append(trace)
        if trace == "chain":
            pred = log["enqueue"][-1]
            seed_cand = log["seed_cand"][-1]
            step, _, seeds = chain_reference(
                _lanes(pred.ids), pred.pack, log["overlay"][-1], torch.zeros(T * B, 36), state["drafts"],
                B, K, ROWS, _ctrl_words(B, 6), 6, 3,
                draws=_lanes(torch.zeros(B * T) if pred.draws is None else pred.draws),
                seed_cand=torch.zeros(T * B, T) if seed_cand is None else seed_cand,
            )  # fmt: skip
            log.setdefault("chain_step", []).append(step)
            log["chain_seeds"].append(seeds)

    monkeypatch.setattr(ttnn, "execute_trace", execute_trace)

    def read(tensors):
        e = log["enqueue"][-1]
        if getattr(e, "step", None) is not None:
            stepbuf = e.step
        else:
            stepbuf = _step_buffer(
                torch.tensor(e.tokens).reshape(-1), [0] * (B * T), [0] * (B * T), e.accept or {}, B, T
            )
        draws = getattr(e, "draws", None)
        out = {"ids": _lanes(e.ids), "draws": None if draws is None else _lanes(draws)}
        if e.accept is not None or getattr(e, "step", None) is not None:
            o = TorchOps(
                _lanes(e.ids),
                stepbuf,
                log["pt_rows"][-1],
                B,
                K,
                eng._scratch_block,
                ROWS,
                draws=_lanes(torch.zeros(B * T) if draws is None else draws),
            )
            e.pack = graph(o, B, K, pack_width(B * T))[0].to(torch.int64)
            out["pack"] = e.pack
        out.update({f"d{k}": state["drafts"][k].reshape(1, 1, B, 1) for k in range(K)})
        return [out[t] for t in tensors]

    eng._read_device0 = read
    eng._read_device0_async = lambda tensors: SimpleNamespace(event="event", result=lambda r=read(tensors): r)
    return eng, log, state


def _host_verify(eng, tokens, starts, nv, sampling=None):
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
        sampling=sampling,
        defer=True,
    )


def _mamba(eng):
    by_count = torch.arange(eng.B * eng.T, dtype=torch.int32).reshape(eng.B, eng.T) + 1
    group = SimpleNamespace(layer_names=("l",), windows=torch.zeros(eng.B, eng.T), start_blocks=by_count[:, -1])
    group.start_blocks_by_count = by_count
    return SimpleNamespace(groups=(group,))


def _chained_verify(eng, res, upper_starts, hint, rows=None, sampling=None):
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
        sampling=sampling,
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


def test_a_request_released_before_its_propose_keeps_the_rope_delta_it_ran_with(chained):
    """Under asynchronous scheduling a request can finish, and its slot be released (M-RoPE delta reset to 0), between
    its last verify and that verify's propose; with a chained verify that is the normal order (the plugin releases
    finished requests before it builds the chained step). The propose checks the device's drafter pack with the deltas
    the verify ran with, so a finished multimodal row is not a mismatch."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    eng._rope_deltas = [0, -7, 0, 5]
    ids0 = torch.tensor([[7, 8, 9, 10]] * B)
    tokens0 = torch.tensor([[1, 7, 8, 9]] * B)
    starts0, nv0 = [100, 200, -1, 300], [3, 3, 0, 3]
    ids0[2] = 0
    tokens0[2] = 0
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64), ids0, torch.zeros(B, T, dtype=torch.int64)]
    # O04: the next build drains the verify after the release.
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    eng.release(3)
    committed, counts = _walk(res0, tokens0, nv0)
    out = _propose(eng, starts0, committed, counts)
    assert out.num_valid.tolist() == [K, K, 0, K] and eng.one_enqueue_host_proposes == 0
    # O05: row 1 finishes; the chained verify runs it as a padding row.
    eng._rope_deltas = [0, -7, 0, 5]
    eng._active.add(3)
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    eng.release(1)
    res1 = _chained_verify(eng, res0, [s + T for s in starts0], [K] * B, rows=[True, False, False, True])
    assert eng.chained_steps == 1 and log["overlay"][-1][2, :B].tolist() == [0, 0, 0, 5]
    committed, counts = _walk(res0, tokens0, nv0)
    _propose(eng, starts0, committed, counts)
    succ = res1.argmax_ids
    committed1, counts1 = _walk(res1, succ.tokens, succ.num_valid.tolist())
    _propose(eng, succ.p0.where(succ.active, torch.tensor(-1)).tolist(), committed1, counts1)


# ----------------------------------------------------------------------------------------- sampled rows


def _sampling():
    """Row 0 seeded, row 1 unseeded, row 2 padding, row 3 greedy (vllm-tt-plugin's per-row sampling_params)."""
    return SimpleNamespace(
        temperature=[0.8, 1.0, 0.0, 0.0], top_k=[5, 20, 1, 1], top_p=[1.0, 0.9, 1.0, 1.0], seed=[77, None, None, None]
    )


def test_a_chained_plan_stages_the_seeds_of_every_position_a_count_gives(expect_error):
    """Whichever count c the predecessor commits, the candidates the chain program picks for a seeded user are the
    seeds sampling_plan stages for an unchained verify whose column 0 sits at p0_prev + c, and so for column 0 the seed
    of a draftless narrow decode at that position (Qwen36MTPContractForCausalLM._decode_narrow plans the same way);
    an unseeded user's are fresh and the same at every count; a user that draws nothing seeds no core."""
    T = 4
    sp = SimpleNamespace(
        temperature=[0.8, 1.0, 0.0, 0.7, 1.0],
        top_k=[5, 20, 1, 32, 3],
        top_p=[1.0, 0.9, 1.0, 1.0, 1.0],
        seed=[77, None, None, 2**40 + 5, 9],
    )
    p0_prev = [100, 200, 300, 4000, -1]
    B = len(p0_prev)
    fresh = iter(range(1, 1000))
    plan, cand = chained_sampling_plan(sp, p0_prev, T, fresh_seed=lambda: next(fresh))
    assert plan.seeds is None and cand.dtype == torch.float32
    assert [plan.sampled[u * T] for u in range(B)] == [True, True, False, True, False]
    assert plan.user_ids == [r if r // T in (0, 1, 3) else NO_CORE for r in range(B * T)]
    for c in range(1, T + 1):
        unchained = sampling_plan(sp, [p + c if p >= 0 else -1 for p in p0_prev], T, fresh_seed=lambda: 0)
        assert (unchained.k, unchained.p, unchained.temp, unchained.sampled) == (
            plan.k,
            plan.p,
            plan.temp,
            plan.sampled,
        )
        for u in (0, 3):
            assert cand[(c - 1) * B + u].to(torch.int64).tolist() == unchained.seeds[u * T : (u + 1) * T]
        assert torch.equal(cand[(c - 1) * B + 1], cand[1]), "an unseeded user's fresh seeds do not depend on c"
        assert not bool(cand[(c - 1) * B + 2].any()) and not bool(cand[(c - 1) * B + 4].any())
    assert len(set(cand[1].tolist())) == T, "each column of an unseeded user takes its own fresh seed"
    _, again = chained_sampling_plan(sp, p0_prev, T, fresh_seed=lambda: next(fresh))
    assert not torch.equal(again[1], cand[1]), "fresh seeds every step"
    with expect_error(ValueError, "top_k 64 outside"):
        chained_sampling_plan(SimpleNamespace(temperature=[1.0], top_k=[64], top_p=[1.0], seed=[None]), [10], T)


def _sampled_predecessor(eng, state):
    """A device-sampled one-enqueue predecessor whose drawn rows accept over their draws, not their argmax ids."""
    B, T = eng.B, eng.T
    ids0 = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [21, 22, 23, 24]])
    draws0 = torch.tensor([[40, 41, 42, 43], [50, 51, 52, 53], [0] * T, [0] * T]).reshape(-1)
    # Row 0's first two drafts equal its draws, row 3's first equals its argmax.
    tokens0 = torch.tensor([[1, 40, 41, 5], [2, 9, 9, 9], [0, 0, 0, 0], [3, 21, 9, 9]])
    starts0, nv0 = [100, 200, -1, 300], [3, 3, 0, 2]
    state["ids"] += [ids0]
    state["draws"] += [draws0]
    return _host_verify(eng, tokens0, starts0, nv0, _sampling()), tokens0, starts0, nv0


def test_a_chained_verify_draws_its_sampled_rows_at_the_positions_of_the_device_count(chained):
    """A device-sampled predecessor is chained on the device: its drawn rows continue from the device's count over
    their draws, the chained verify draws its sampled rows (no seeds staged: the chain program wrote them, and the
    seeded row's are those an unchained verify at the device's positions stages), and both proposes check the runner's
    walk of the draws."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    res0, tokens0, starts0, nv0 = _sampled_predecessor(eng, state)
    assert res0.argmax_ids.plan is not None and res0.argmax_ids.sample_plan is not None
    drafts = torch.stack(state["drafts"], dim=1)
    ids1 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 4, 4, 4]])
    draws1 = torch.zeros(B, T, dtype=torch.int64)
    draws1[0] = torch.tensor([int(drafts[0, 0]), int(drafts[0, 1]), 7, 7])  # accepts two drafts
    draws1[1] = torch.tensor([9, 9, 9, 9])  # accepts none
    state["ids"] += [ids1]
    state["draws"] += [draws1.reshape(-1)]
    upper = [s + T if s >= 0 else -1 for s in starts0]
    res1 = _chained_verify(eng, res0, upper, [K] * B, sampling=_sampling())
    assert eng.chained_steps == 1 and eng.materialized_steps == 0
    assert log["traces"] == ["step", "drafter", "chain", "step", "drafter"]
    overlay = log["overlay"][-1]
    assert overlay[CH_DRAWN_PREV, :B].tolist() == [1, 1, 0, 0] and overlay[CH_DRAWN, :B].tolist() == [1, 1, 0, 0]
    plan = log["enqueue"][-1].kw["sample_plan"]
    assert plan.seeds is None and plan.user_ids == [r if r < 2 * T else NO_CORE for r in range(B * T)]
    assert log["blocks"].index(("seeds_written",)) > log["blocks"].index(("written",))
    assert res1.argmax_ids.sample_plan is plan
    committed0, counts0 = _walk(res0, tokens0, nv0)
    assert counts0.tolist() == [3, 1, 1, 2], "the walk of the draws, not of the argmax ids"
    out0 = _propose(eng, starts0, committed0, counts0)
    assert out0.num_valid.tolist() == [K, K, 0, K]
    succ = res0.argmax_ids.successor
    assert succ.p0.tolist() == [103, 201, 0, 302]
    assert succ.tokens[0].tolist() == [42, *drafts[0].tolist()] and succ.tokens[1].tolist() == [50, *drafts[1].tolist()]
    seeds = log["chain_seeds"][-1]
    unchained = sampling_plan(_sampling(), [103, 201, -1, 302], T, fresh_seed=lambda: 0)
    assert seeds[:T].tolist() == unchained.seeds[:T], "the seeded row draws with the seeds of its positions"
    assert seeds[T : 2 * T].tolist() == log["seed_cand"][-1][B + 1].to(torch.int64).tolist(), "count 1's fresh seeds"
    committed1, counts1 = _walk(res1, succ.tokens.clone(), [K, K, 0, K])
    assert counts1.tolist() == [3, 1, 1, 2]
    out1 = _propose(eng, succ.p0.where(succ.active, torch.tensor(-1)).tolist(), committed1, counts1)
    assert out1.num_valid.tolist() == [K, K, 0, K] and not eng._pending and eng.one_enqueue_host_proposes == 0


def test_a_commit_of_the_argmax_instead_of_the_draws_raises_at_the_predecessor_propose(chained, expect_error):
    """The guard covers sampled rows: the chained verify continued the device's count over the draws, so a runner
    commit that walked the argmax ids of a drawn row raises."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    res0, tokens0, starts0, nv0 = _sampled_predecessor(eng, state)
    state["ids"] += [torch.zeros(B, T, dtype=torch.int64)]
    state["draws"] += [torch.zeros(B * T, dtype=torch.int64)]
    _chained_verify(eng, res0, [s + T if s >= 0 else -1 for s in starts0], [K] * B, sampling=_sampling())
    argmax = log["enqueue"][0].ids
    committed, counts = accept_greedy_drafts(
        mask_unused_columns(argmax, torch.tensor(nv0)).to(torch.int32),
        tokens0[:, 1:].to(torch.int32),
        torch.tensor(nv0, dtype=torch.int32),
    )
    assert counts.tolist() == [1, 1, 1, 2]
    with expect_error(ValueError, "the chained verify and the one-enqueue drafts continue the device's commit"):
        _propose(eng, starts0, committed, counts)


def test_a_chained_verify_refuses_a_top_k_the_sampler_cannot_draw(chained, expect_error):
    """Candidate overflow cannot reach a chained verify: a sampled row with more candidates than the sampler keeps
    (the plugin samples such a step on the host, max_device_top_k) raises instead of drawing from a cut list, and
    leaves the predecessor to a chained verify the sampler can draw."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    res0, tokens0, starts0, nv0 = _sampled_predecessor(eng, state)
    upper = [s + T if s >= 0 else -1 for s in starts0]
    wide = _sampling()
    wide.top_k = [64, 20, 1, 1]
    with expect_error(ValueError, "top_k 64 outside"):
        _chained_verify(eng, res0, upper, [K] * B, sampling=wide)
    assert eng.chained_steps == 0 and "chain" not in log["traces"]
    state["ids"] += [torch.zeros(B, T, dtype=torch.int64)]
    state["draws"] += [torch.zeros(B * T, dtype=torch.int64)]
    wide.top_k = [32, 20, 1, 1]
    _chained_verify(eng, res0, upper, [K] * B, sampling=wide)
    assert eng.chained_steps == 1 and log["enqueue"][-1].kw["sample_plan"].k[:T] == [32] * T


def test_a_sampled_chained_verify_built_from_the_read_predecessor_stages_the_seeds_of_its_positions(chained):
    """When the chain program cannot build the chained verify (its propose depends on the count), the engine reads
    the predecessor, walks its draws and verifies on the host path, staging the seeds of the positions it read."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    ids0 = torch.tensor([[7, 8, 9, 10]] * B)
    draws0 = torch.tensor([[40, 41, 42, 43]] * B).reshape(-1)
    tokens0 = torch.tensor([[1, 40, 41, 9]] * B)
    edge = eng.nb * eng.block_size - 9  # every count fits here, but not every count of the chained verify
    starts0, nv0 = [100, 200, edge, 400], [3] * B
    sampling = SimpleNamespace(temperature=[0.8] * B, top_k=[5] * B, top_p=[1.0] * B, seed=[77, 78, 79, 80])
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    state["draws"] = [draws0, torch.zeros(B * T, dtype=torch.int64)]
    res0 = _host_verify(eng, tokens0, starts0, nv0, sampling)
    assert res0.argmax_ids.plan is not None
    _chained_verify(eng, res0, [s + T for s in starts0], [K] * B, sampling=sampling)
    assert eng.materialized_steps == 1 and "chain" not in log["traces"]
    e = log["enqueue"][-1]
    assert e.positions[0] == 103 and e.tokens[0][0] == 42, "count 3 over the draws"
    want = sampling_plan(sampling, [103, 203, edge + 3, 403], T, fresh_seed=lambda: 0)
    assert e.kw["sample_plan"].seeds == want.seeds
