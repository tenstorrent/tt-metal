# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): one-enqueue MTP steps (tt/spec_accept.py, MTPBlockEngine).

* The step trace's arithmetic (spec_accept.graph, run with torch ops) gives the plugin's greedy walk
  (accept_greedy_drafts) and the drafter pack and reseed table the host builds from that walk, over random verifies:
  every count, padding rows, declined and kept rows, drawn rows, M-RoPE deltas.
* MTPBlockEngine: an argmax_ids verify enqueues the verify, the step trace and the drafter trace, stages the
  acceptance rows in the verify's one copy and reads once; the next propose returns the device's drafts when the
  runner commits what the device accepted, and raises otherwise. Steps whose propose depends on the count, logits
  and narrow steps keep the host-staged drafter.
"""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm_tt_plugin.spec_decode import accept_greedy_drafts

import ttnn
from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.spec_accept import TorchOps, graph, greedy_counts, step_accept_reference
from models.demos.blackhole.qwen36.tt.spec_step import (
    STEP_DECLINED,
    STEP_DRAWN,
    STEP_KEEP,
    STEP_NUM_VALID,
    STEP_P0,
    STEP_ROPE_BASE,
    VERIFY_POSITIONS,
    VERIFY_ROPE,
    VERIFY_STEP_ROWS,
    VERIFY_TOKENS,
    pack_u32_rows,
    pack_width,
    rope_rows,
)

ROWS = 4096
VOCAB = 248320


def _step_buffer(tokens, positions, rope, accept, B, T):
    rows = [[0] * B for _ in range(VERIFY_STEP_ROWS)]
    rows[VERIFY_TOKENS], rows[VERIFY_POSITIONS], rows[VERIFY_ROPE] = tokens, positions, rope
    for r, v in accept.items():
        rows[r] = v
    return pack_u32_rows(rows, width=pack_width(B * T))


def _lanes(ids):
    out = torch.zeros(1, 1, 1, 32, dtype=torch.int64)
    flat = torch.as_tensor(ids).reshape(-1)
    out[0, 0, 0, : flat.numel()] = flat
    return out


def _case(rng, B, K):
    """Random verify of B users: ids, tokens whose drafts match the ids for a random prefix, valid drafts, and the
    acceptance rows (padding rows, declined and kept rows, drawn rows)."""
    T = K + 1
    ids = torch.tensor([[rng.randrange(VOCAB) for _ in range(T)] for _ in range(B)])
    tokens = torch.tensor([[rng.randrange(VOCAB) for _ in range(T)] for _ in range(B)])
    nv, act = [], []
    for u in range(B):
        act.append(B == 1 or rng.random() < 0.8)
        n = rng.randrange(T) if act[u] else 0
        nv.append(n)
        agree = rng.randrange(n + 1)
        tokens[u, 1 : agree + 1] = ids[u, :agree]  # drafts 1 .. agree equal the id before them
        if rng.random() < 0.2:
            tokens[u, n + 1 :] = ids[u, n : T - 1]  # padding columns that would match must not count
    deltas = [rng.choice([0, -rng.randrange(1, 30), rng.randrange(1, 5)]) for _ in range(B)]
    p0 = [rng.randrange(40, ROWS - 3 * T) if a else 0 for a in act]
    skip = [a and rng.random() < 0.2 for a in act]
    declined = [not a or s or rng.random() < 0.1 for a, s in zip(act, skip)]
    accept = {
        STEP_NUM_VALID: nv,
        STEP_DRAWN: [int(a and rng.random() < 0.4) for a in act],
        STEP_P0: p0,
        STEP_ROPE_BASE: rope_rows(p0, deltas, ROWS).tolist(),
        STEP_DECLINED: [int(d) for d in declined],
        STEP_KEEP: [int(s) for s in skip],
    }
    return ids, tokens, accept


@pytest.mark.parametrize("B,K", [(1, 3), (2, 3), (4, 3), (8, 3), (1, 1), (4, 7), (2, 11), (16, 1)])
def test_the_step_graph_is_the_plugin_walk_and_the_host_pack(B, K):
    """graph (the device arithmetic, in torch) == step_accept_reference (the plugin's walk and the host's drafter pack
    and reseed table), and the walk's counts == accept_greedy_drafts on the merged ids."""
    T, R = K + 1, B * (K + 1)
    rng = random.Random(B * 31 + K)
    for _ in range(60):
        ids, tokens, accept = _case(rng, B, K)
        draws = torch.tensor([rng.randrange(VOCAB) for _ in range(R)])
        pt = torch.randint(1, 900, (B, 5), dtype=torch.int32)
        step = _step_buffer(tokens.reshape(-1), list(range(R)), list(range(R)), accept, B, T)
        o = TorchOps(_lanes(ids), step, pt.repeat_interleave(T, dim=0), B, K, 999, ROWS, draws=_lanes(draws))
        pack, rs = graph(o, B, K, pack_width(R))
        merged, m, want_pack, want_rs = step_accept_reference(
            ids.reshape(-1), tokens.reshape(-1), accept, pt, 999, K, ROWS, draws=draws
        )
        assert torch.equal(pack.to(torch.int64), want_pack.to(torch.int64))
        assert torch.equal(rs.to(torch.int32), want_rs)
        drawn = torch.tensor(accept[STEP_DRAWN], dtype=torch.bool)
        assert torch.equal(merged[drawn], draws.reshape(B, T)[drawn]) and torch.equal(merged[~drawn], ids[~drawn])
        _, counts = accept_greedy_drafts(
            merged.to(torch.int32), tokens[:, 1:].to(torch.int32), torch.tensor(accept[STEP_NUM_VALID])
        )
        assert torch.equal(m + 1, counts.to(torch.int64)), "the device walk is the plugin's greedy walk"


def test_greedy_counts_stop_at_the_first_mismatch_and_the_valid_drafts():
    ids = torch.tensor([[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]])
    tokens = torch.tensor([[1, 5, 6, 9], [1, 5, 0, 7], [1, 5, 6, 7]])
    assert greedy_counts(ids, tokens, [3, 3, 1]).tolist() == [2, 1, 1]


# ----------------------------------------------------------------------------------------- MTPBlockEngine


@pytest.fixture
def step(monkeypatch):
    """An MTPBlockEngine whose device is a stand-in: the step trace runs as spec_accept.graph in torch on what the
    verify staged, and the drafter answers state["drafts"]."""
    B, K = 4, 3
    T = K + 1
    log = {"traces": [], "traced": [], "enqueue": [], "staged": []}
    state = {"ids": None, "draws": None, "drafts": [torch.arange(100 * k, 100 * k + B) for k in range(K)]}
    gdn = SimpleNamespace(remap_spec_slots=lambda idx: None)
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        mesh_device=MagicMock(),
        args=SimpleNamespace(vocab_size=VOCAB, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=gdn)],
        gdn_blocks=SimpleNamespace(verify_tokens=T, set_verify_blocks=lambda s: None),
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        spec_feed_rows=lambda h: h,
        rope=SimpleNamespace(rope_delta=0),
        rope_table_rows=lambda: ROWS,
        _vfy_rows_out="feed",
        _vfy_ids_out="ids",
        _vfy_sampler=SimpleNamespace(draws="draws"),
    )

    def verify_traced(tokens, positions, mi, **kw):
        log["traced"].append(kw)
        return [int(v) for v in state["ids"].reshape(-1)], "feed", None

    def verify_enqueue(tokens, positions, mi, accept_rows=None, **kw):
        log["enqueue"].append(SimpleNamespace(tokens=tokens, positions=positions, accept=accept_rows, kw=kw))

    model.verify_traced, model.verify_enqueue = verify_traced, verify_enqueue
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 8, 64, 999
    eng._active = set(range(B))
    eng._acc_trace, eng._dr_trace = "step", "drafter"
    eng._accept = SimpleNamespace(stage_pt_rows=lambda rows: log.setdefault("pt_rows", []).append(rows))
    eng._dr_pack, eng._mtp_pt, eng._dr_rs_pt, eng._dr_outs = "pack", "mtp_pt", "rs_pt", [f"d{k}" for k in range(K)]
    eng._stage = lambda host, buf, *a: log["staged"].append((buf, host))
    monkeypatch.setattr(ttnn, "execute_trace", lambda mesh, trace, **kw: log["traces"].append(trace))

    def read(tensors):
        e = log["enqueue"][-1]
        tokens = torch.tensor(e.tokens).reshape(-1)
        stepbuf = _step_buffer(tokens, [0] * (B * T), [0] * (B * T), e.accept, B, T)
        draws = None if state["draws"] is None else _lanes(state["draws"])
        pt_rows = log["pt_rows"][-1]
        o = TorchOps(_lanes(state["ids"]), stepbuf, pt_rows, B, K, eng._scratch_block, ROWS, draws=draws)
        pack, _ = graph(o, B, K, pack_width(B * T))
        out = {"ids": _lanes(state["ids"]), "draws": draws, "pack": pack.to(torch.int64)}
        out.update({f"d{k}": state["drafts"][k].reshape(1, 1, B, 1) for k in range(K)})
        return [out[t] for t in tensors]

    eng._read_device0 = read
    return eng, log, state


def _verify(eng, tokens, starts, nv, spec_mode="argmax_ids", sampling=None, **kw):
    B, T = eng.B, eng.T
    start_pos = torch.tensor([[s + j if s >= 0 else -1 for j in range(T)] for s in starts])
    return eng.decode_forward(
        tokens,
        start_pos,
        num_valid_drafts=torch.tensor(nv, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=torch.arange(1, B * eng.nb + 1, dtype=torch.int32).reshape(B, eng.nb),
        spec_mode=spec_mode,
        mamba_state="S",
        sampling=sampling,
        **kw,
    )


def _walk_and_propose(eng, res, tokens, starts, nv):
    committed, counts = accept_greedy_drafts(
        res.argmax_ids, torch.as_tensor(tokens)[:, 1:].to(torch.int32), torch.tensor(nv, dtype=torch.int32)
    )
    positions = torch.tensor([[s + 1 + j if s >= 0 else -1 for j in range(eng.T)] for s in starts])
    return eng.propose_draft_tokens(eng.K, committed, positions, counts), counts


def test_a_greedy_step_runs_verify_acceptance_and_drafter_in_one_enqueue(step):
    """The verify, the step trace and the drafter trace replay in that order with one read; the step buffer carries
    the acceptance rows; the propose returns the device's drafts, with a padding row declined."""
    eng, log, state = step
    B, T = eng.B, eng.T
    state["ids"] = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [21, 22, 23, 24]])
    tokens = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [3, 9, 9, 9]])
    starts, nv = [100, 200, -1, 300], [3, 3, 0, 2]
    res = _verify(eng, tokens, starts, nv)
    assert log["traced"] == [] and log["traces"] == ["step", "drafter"]
    assert eng.one_enqueue_steps == eng.verify_steps == 1 and eng.one_enqueue_host_proposes == 0
    acc = log["enqueue"][-1].accept
    assert acc[STEP_NUM_VALID] == [3, 3, 0, 2] and acc[STEP_P0] == [100, 200, 0, 300]
    assert acc[STEP_DECLINED] == [0, 0, 1, 0] and acc[STEP_KEEP] == [0, 0, 0, 0] and acc[STEP_DRAWN] == [0] * B
    assert acc[STEP_ROPE_BASE] == [100, 200, 0, 300]
    assert res.argmax_ids[0].tolist() == [7, 8, 9, 10] and res.argmax_ids[3].tolist() == [21, 22, 23, -1]
    out, counts = _walk_and_propose(eng, res, tokens, starts, nv)
    assert counts.tolist() == [3, 4, 1, 1]
    want = torch.stack(state["drafts"], dim=1).to(torch.int32)
    want[2] = 0
    assert torch.equal(out.draft_token_ids, want) and out.num_valid.tolist() == [3, 3, 0, 3]
    assert eng._offered == {0, 1, 3} and eng._step_pending is None
    mtp = [h for buf, h in log["staged"] if buf == "mtp_pt"][-1]
    assert bool(mtp[2].eq(eng._scratch_block).all()) and torch.equal(mtp[0], eng._last_pt[0]), "padding row on scratch"


def test_a_commit_the_device_did_not_accept_raises(step, expect_error):
    """A commit that is not a prefix of the device's walk (another token, or more tokens than the device accepted)
    means a wrong device acceptance: the propose raises instead of returning drafts of another commit."""
    eng, log, state = step
    state["ids"] = torch.tensor([[7, 8, 9, 10]] * eng.B)
    tokens = torch.tensor([[1, 7, 8, 5]] * eng.B)
    starts, nv = [100, 200, 300, 400], [3] * eng.B
    res = _verify(eng, tokens, starts, nv)
    committed, counts = accept_greedy_drafts(res.argmax_ids, tokens[:, 1:].to(torch.int32), torch.tensor(nv))
    assert counts.tolist() == [3] * eng.B
    committed, counts = committed.clone(), counts.clone()
    counts[1], committed[1, 3] = 4, 10  # one token more than the device accepted
    positions = torch.tensor([[s + 1 + j for j in range(eng.T)] for s in starts])
    with expect_error(ValueError, "not a prefix of the device's"):
        eng.propose_draft_tokens(eng.K, committed, positions, counts)
    _verify(eng, tokens, starts, nv)
    committed, counts = accept_greedy_drafts(res.argmax_ids, tokens[:, 1:].to(torch.int32), torch.tensor(nv))
    committed = committed.clone()
    committed[2, 0] = 11  # another token
    with expect_error(ValueError, "not a prefix of the device's"):
        eng.propose_draft_tokens(eng.K, committed, positions, counts)


def test_a_commit_cut_short_by_the_runner_drafts_on_the_host(step):
    """A runner that cuts a row's commit short (a length or stop cut: a prefix of the device's walk) gets the base
    propose, which drafts from the runner's commit; the step counts as drafted again on the host."""
    eng, log, state = step
    state["ids"] = torch.tensor([[7, 8, 9, 10]] * eng.B)
    tokens = torch.tensor([[1, 7, 8, 9]] * eng.B)
    starts, nv = [100, 200, 300, 400], [3] * eng.B
    res = _verify(eng, tokens, starts, nv)
    committed, counts = accept_greedy_drafts(res.argmax_ids, tokens[:, 1:].to(torch.int32), torch.tensor(nv))
    assert counts.tolist() == [4] * eng.B
    committed, counts = committed.clone(), counts.clone()
    counts[1] = 1
    committed[1, 1:] = -1
    positions = torch.tensor([[s + 1 + j for j in range(eng.T)] for s in starts])
    out = eng.propose_draft_tokens(eng.K, committed, positions, counts)
    assert log["traces"] == ["step", "drafter", "drafter"], "the drafter runs again on the host's pack"
    assert eng.one_enqueue_host_proposes == 1 and out.num_valid.tolist() == [eng.K] * eng.B
    pack = [h for buf, h in log["staged"] if buf == "pack"][-1]
    assert int(pack[0, 1]) == 7 and int(pack[0, 0]) == 10, "row 1 drafts from its committed token 7, row 0 from 10"


def test_a_sampled_step_merges_the_draws_of_the_drawn_rows(step):
    eng, log, state = step
    B, T = eng.B, eng.T
    eng.sample_on_device = True
    state["ids"] = torch.arange(B * T).reshape(B, T) + 10
    state["draws"] = torch.arange(B * T).reshape(B, T) + 500
    tokens = torch.tensor([[1, 500, 501, 9], [1, 14, 15, 16], [2, 2, 2, 2], [3, 3, 3, 3]])
    params = SimpleNamespace(
        temperature=[1.0, 0.0, 0.0, 0.0], top_k=[20, 1, 1, 1], top_p=[1.0] * B, seed=[7] + [None] * 3
    )
    res = _verify(eng, tokens, [100, 200, 300, 400], [3, 3, 0, 0], sampling=params)
    assert log["enqueue"][-1].accept[STEP_DRAWN] == [1, 0, 0, 0] and "sample_plan" in log["enqueue"][-1].kw
    assert res.argmax_ids[0].tolist() == [500, 501, 502, 503] and res.argmax_ids[1].tolist() == [14, 15, 16, 17]
    out, counts = _walk_and_propose(eng, res, tokens, [100, 200, 300, 400], [3, 3, 0, 0])
    assert counts.tolist() == [3, 4, 1, 1] and out.num_valid.tolist() == [3] * B


@pytest.mark.parametrize("why", ["logits", "narrow", "mixed_fit", "all_declined"])
def test_steps_whose_propose_depends_on_the_count_stage_the_drafter_from_the_host(step, why):
    eng, log, state = step
    B = eng.B
    state["ids"] = torch.zeros(B, eng.T, dtype=torch.int64)
    tokens = torch.zeros(B, eng.T, dtype=torch.int64)
    starts, nv, kw = [100, 200, 300, 400], [3] * B, {}
    if why == "logits":
        kw["spec_mode"] = "logits"
        eng.model.verify_traced = lambda *a, **k: ([0] * (B * eng.T), "feed", torch.zeros(B * eng.T, 8))
    elif why == "narrow":
        kw["one_enqueue"] = False
    elif why == "mixed_fit":
        starts[2] = eng.nb * eng.block_size - 1 - eng.K - 2  # fits after 0 or 1 accepted drafts only
    elif why == "all_declined":
        eng._no_draft = set(range(B))
    _verify(eng, tokens, starts, nv, **kw)
    assert log["enqueue"] == [] and log["traces"] == [] and eng._step_pending is None


def test_a_b1_step_the_runner_withholds_or_that_could_reseed_eagerly_stays_on_the_host(monkeypatch):
    from models.demos.blackhole.qwen36.tt.spec_engine import MTPBlockEngine

    e = MTPBlockEngine.__new__(MTPBlockEngine)
    e.B, e.K, e.T, e.block_size, e._last_cap_blocks = 1, 3, 4, 64, 64
    e.eager_reseed_len, e._rope_deltas, e._scratch_block = 1000, [0], 999
    e._last_pt = torch.ones(1, 64, dtype=torch.int32)
    e.model = SimpleNamespace(rope_table_rows=lambda: ROWS)
    active = torch.ones(1, dtype=torch.bool)
    e._no_draft = {0}
    assert MTPBlockEngine._one_enqueue_plan(e, torch.tensor([100]), active, [3], None) is None
    e._no_draft = set()
    assert MTPBlockEngine._one_enqueue_plan(e, torch.tensor([998]), active, [3], None) is None
    plan = MTPBlockEngine._one_enqueue_plan(e, torch.tensor([997]), active, [3], None)
    assert plan is not None and plan.declined == [False]


def test_the_next_verify_drops_drafts_no_propose_took(step):
    """A one-enqueue verify the runner follows with another verify (no propose) leaves nothing behind: the next
    propose runs on the host path of its own verify."""
    eng, log, state = step
    state["ids"] = torch.zeros(eng.B, eng.T, dtype=torch.int64)
    tokens = torch.zeros(eng.B, eng.T, dtype=torch.int64)
    _verify(eng, tokens, [100, 200, 300, 400], [3] * eng.B)
    assert eng._step_pending is not None
    _verify(eng, tokens, [100, 200, 300, 400], [3] * eng.B, one_enqueue=False)
    assert eng._step_pending is None and (eng.one_enqueue_steps, eng.verify_steps) == (1, 2)


def test_a_propose_that_declines_other_rows_drafts_on_the_host(step):
    """A release between the verify and the propose re-offers a withheld row: the propose decides differently from the
    verify's plan, so the drafter runs again from the runner's commit and the row is offered drafts."""
    eng, log, state = step
    state["ids"] = torch.zeros(eng.B, eng.T, dtype=torch.int64)
    tokens = torch.zeros(eng.B, eng.T, dtype=torch.int64)
    starts, nv = [100, 200, 300, 400], [3] * eng.B
    eng._no_draft = {1}
    res = _verify(eng, tokens, starts, nv)
    assert log["enqueue"][-1].accept[STEP_DECLINED] == [0, 1, 0, 0] and log["enqueue"][-1].accept[STEP_KEEP] == [
        0,
        1,
        0,
        0,
    ]
    eng._no_draft.clear()
    out, _ = _walk_and_propose(eng, res, tokens, starts, nv)
    assert log["traces"] == ["step", "drafter", "drafter"] and out.num_valid.tolist() == [eng.K] * eng.B
    assert eng.one_enqueue_host_proposes == 1


def test_the_runners_commit_proposes_with_the_walks_counts(step):
    """The plugin runner's commit (TTModelRunner.commit_spec_acceptance, vllm-tt-plugin bfdb4d0) after a one-enqueue
    verify: a row that reaches max_model_len inside its accepted run has its written tokens trimmed, a padding row
    commits 1, and the propose still gets every row's walk count, so the device's drafts are returned without a
    mismatch. (EOS, stop strings and max_tokens are trimmed by vLLM's scheduler from the published tokens, after this.)
    """
    import inspect
    import types

    import numpy as np

    mr = pytest.importorskip("vllm_tt_plugin.model_runner")
    from vllm_tt_plugin.spec_decode import DraftOutput

    Runner = mr.TTModelRunner
    params = list(inspect.signature(Runner.commit_spec_acceptance).parameters)
    if params != ["self", "model_input", "committed", "counts", "spec_hidden", "skip_req_ids"]:
        pytest.skip("the plugin on the path is not the bfdb4d0 runner")
    eng, log, state = step
    B, T, K, max_len = eng.B, eng.T, eng.K, 400
    state["ids"] = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [21, 22, 23, 24], [0, 0, 0, 0]])
    tokens = torch.tensor([[1, 7, 8, 9], [2, 11, 5, 5], [3, 21, 22, 23], [0, 0, 0, 0]])
    starts, nv = [max_len - 3, 200, 300, -1], [3, 3, 3, 0]
    res = _verify(eng, tokens, starts, nv)
    committed, counts = accept_greedy_drafts(res.argmax_ids, tokens[:, 1:].to(torch.int32), torch.tensor(nv))
    assert counts.tolist() == [4, 2, 4, 1]
    seen = []

    def propose(num_drafts, c, positions, k, hidden=None):
        seen.append(k.clone())
        out = eng.propose_draft_tokens(num_drafts, c, positions, k, hidden=hidden)
        return DraftOutput(draft_token_ids=out.draft_token_ids, num_valid=out.num_valid)

    reqs = ["r0", "r1", "r2"]
    sampling = SimpleNamespace(
        logprobs=None,
        structured_outputs=None,
        allowed_token_ids=None,
        bad_words_token_ids=None,
        min_tokens=0,
        all_stop_token_ids=set(),
    )
    runner = SimpleNamespace(
        model=SimpleNamespace(propose_draft_tokens=propose, model_capabilities={}),
        model_config=SimpleNamespace(max_model_len=max_len),
        input_batch=SimpleNamespace(
            req_id_to_index={r: u for u, r in enumerate(reqs)},
            num_tokens=np.array([s + 1 for s in starts[:3]]),
            token_ids_cpu=np.zeros((3, max_len), dtype=np.int64),
            req_output_token_ids=[[] for _ in reqs],
            vocab_size=VOCAB,
        ),
        requests={r: SimpleNamespace(output_token_ids=[], sampling_params=sampling) for r in reqs},
        _req_state_slot={r: u for u, r in enumerate(reqs)},
        _req_accepted_counts={},
        _spec_drafts_from_model=True,
        _num_speculative_tokens=K,
        _proposed_draft_token_ids={},
        _spec_accept_modes=("argmax_ids", "logits"),
    )
    for name in (
        "_apply_committed_spec_tokens_to_state",
        "_propose_model_drafts",
        "_publish_draft",
        "_request_is_speculable",
        "_request_is_argmax_certifiable",
    ):
        setattr(runner, name, types.MethodType(getattr(Runner, name), runner))
    runner._committed_positions = Runner._committed_positions
    positions = torch.tensor([[s + j if s >= 0 else -1 for j in range(T)] for s in starts])
    model_input = SimpleNamespace(
        row_req_ids=reqs,
        live_rows=None,
        spec_mode="argmax_ids",
        input_positions=positions,
        perform_device_sampling=False,
    )
    Runner.commit_spec_acceptance(runner, model_input, committed, counts, None)
    assert [k.tolist() for k in seen] == [[4, 2, 4, 1]], "the propose gets the walk's counts, untrimmed"
    assert runner._req_accepted_counts == {"r0": 2, "r1": 2, "r2": 4}, "the runner writes r0 up to max_model_len only"
    assert int(runner.input_batch.num_tokens[0]) == max_len and eng.one_enqueue_host_proposes == 0
    assert "r0" not in runner._proposed_draft_token_ids, "no room left for r0's drafts"
    assert runner._proposed_draft_token_ids["r1"] == [int(state["drafts"][k][1]) for k in range(K)]


def test_the_step_trace_never_frees_an_input(monkeypatch):
    """A whole-tensor slice returns its input (the ids or the draws at B * T = 32): the step trace frees every
    intermediate but no tensor that aliases one of its inputs."""
    from models.demos.blackhole.qwen36.tt import spec_accept

    buf = lambda a: SimpleNamespace(buffer_address=lambda: a)
    acc = SimpleNamespace(
        ids=buf(1), draws=buf(2), step=buf(3), pt_rows=buf(4), iota_t=buf(5), iota_k=buf(6), user_rows=buf(7)
    )
    acc.c_top, acc.c_scratch = buf(8), buf(9)
    o = spec_accept._DeviceOps(acc)
    whole = buf(1)  # a slice over all of the ids: the same buffer
    part = buf(100)
    o._keep(whole)
    o._keep(part)
    o._keep(acc.draws)
    assert o.owned == [part]
