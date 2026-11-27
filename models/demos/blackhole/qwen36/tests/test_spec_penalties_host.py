# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the verify penalties of Qwen36MTPContractForCausalLM (tt/spec_penalties.py).

* The op graphs of VerifyPenalties, run on a torch stand-in of the ttnn ops they use (one device, whole vocabulary):
  apply_body scores column j of every row after the row's committed output and its first j drafts, as vLLM's
  apply_penalties does, rounds the penalized logits to bfloat16 once, leaves a neutral row bit for bit, and gives a
  greedy penalty row the float32 argmax; commit_body adds a predecessor's commit (its ids, or its draws, up to the
  drafter pack's count) to the tracked rows only; rebuild and add write a row from host lists.
* The records (PenaltyRow, reconcile): a join or a resume rewrites a row, a commit adds its tokens, a row move or a
  long gap rewrites, a chained commit resolves from the read verify.
* The engine: a host-staged verify brings every penalty row's counts to its output; a chained verify adds the
  predecessor's commit on the device and records it; a chained verify that moves a penalty row, or whose counts are
  not the runner's output, is built from the read predecessor with that commit added on the host. The device counts
  equal the bin counts of each request's committed output after a join, commits, a chained commit, a row move, a
  finish and a resume.
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.spec_penalties import (
    MAX_ADDED_TOKENS,
    PenaltyRow,
    VerifyPenalties,
    counts_of,
    reconcile,
    reference_penalties,
    step_penalties,
)
from models.demos.blackhole.qwen36.tt.spec_step import VERIFY_STEP_ROWS, draft_anchor, pack_width

V = 96


class TorchOps:
    """The ttnn ops VerifyPenalties uses, over torch tensors on one device: uint32 is int64, layouts and memory
    configs are tags, mesh mappers are None. Every op returns a new tensor, as the device ops do."""

    float32, bfloat16, uint32 = torch.float32, torch.bfloat16, torch.int64
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = "tile", "row_major", "dram"

    ShardTensorToMesh = staticmethod(lambda mesh, dim: None)
    ReplicateTensorToMesh = staticmethod(lambda mesh: None)
    ConcatMeshToTensor = staticmethod(lambda mesh, dim: None)

    @staticmethod
    def from_torch(t, dtype, layout=None, mesh_mapper=None, **kw):
        return torch.as_tensor(t).to(dtype).clone()

    @staticmethod
    def to_device(t, mesh, memory_config=None):
        return t.clone()

    @staticmethod
    def copy_host_to_device_tensor(host, dev):
        dev.copy_(host.to(dev.dtype))

    @staticmethod
    def slice(t, start, end):
        return t[tuple(slice(int(a), int(b)) for a, b in zip(start, end))].clone()

    @staticmethod
    def reshape(t, shape):
        return t.reshape(shape).clone()

    @staticmethod
    def to_layout(t, layout):
        return t.clone()

    @staticmethod
    def typecast(t, dtype):
        return t.to(dtype)

    multiply = staticmethod(lambda a, b: torch.mul(a, b))
    add = staticmethod(lambda a, b: torch.add(a, b))
    subtract = staticmethod(lambda a, b: torch.sub(a, b))
    transpose = staticmethod(lambda t, a, b: t.transpose(a, b).contiguous())

    @staticmethod
    def eq(a, b, dtype=torch.float32):
        return (a == b).to(dtype)

    @staticmethod
    def lt(a, b, dtype=torch.float32):
        return (a < b).to(dtype)

    @staticmethod
    def gt(a, b):
        return (a > b).to(a.dtype)

    @staticmethod
    def where(c, a, b):
        a = a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
        b = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.float32)
        return torch.where(c != 0, a, b)

    @staticmethod
    def copy(src, dst):
        dst.copy_(src.to(dst.dtype))

    @staticmethod
    def add_(a, b):
        a.add_(b)

    deallocate = staticmethod(lambda t: None)
    synchronize_device = staticmethod(lambda mesh: None)
    to_torch = staticmethod(lambda t, mesh_composer=None: t)
    get_device_tensors = staticmethod(lambda t: [t])


def _lanes_argmax(x):
    """The fused per-shard argmax in the verify ids' layout: lane r = row r's lowest id among its maxima."""
    lanes = torch.zeros(1, 1, 1, 32, dtype=torch.int64)
    rows = x.reshape(-1, x.shape[-1])
    lanes[0, 0, 0, : rows.shape[0]] = rows.argmax(dim=-1)
    return lanes


def _penalties(B, T, vocab=V):
    return VerifyPenalties(None, B, T, vocab, argmax=_lanes_argmax, ops=TorchOps)


def _step_buffer(tokens):
    """The verify step buffer with ``tokens`` [B, T] in its token row (the other rows are not read)."""
    B, T = tokens.shape
    buf = torch.zeros(VERIFY_STEP_ROWS, pack_width(B * T), dtype=torch.int64)
    buf[0, : B * T] = torch.as_tensor(tokens).reshape(-1)
    return buf


def _sampling(**rows):
    n = max(len(v) for v in rows.values())
    fill = dict(repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, temperature=0.0)
    return SimpleNamespace(**{k: list(rows.get(k, [fill[k]] * n)) for k in fill})


def _bf16_logits(rows, vocab=V, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, vocab, generator=g) * 4).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------------- apply


def test_each_column_is_scored_after_the_committed_output_and_its_drafts():
    """Column j of a penalty row is vLLM's apply_penalties over the committed output plus drafts 1 .. j, rounded once
    to bfloat16; a padded column repeats column 0, which only adds that token again; the neutral row is unchanged bit
    for bit and keeps its argmax ids; the greedy penalty row's ids are the float32 argmax of its penalized logits."""
    B, T = 2, 4
    pen = _penalties(B, T)
    prompt = {0: torch.tensor([3, 4, 4, 50])}
    output = {0: torch.tensor([7, 7, 9, 12])}
    tokens = torch.tensor([[12, 30, 7, 12], [5, 6, 7, 8]])  # row 0: tok0 = last committed token 12, drafts 30, 7, 12
    pen.rebuild({0: (prompt[0], output[0])})
    sp = step_penalties(
        _sampling(repetition_penalty=[1.3, 1.0], frequency_penalty=[0.4, 0.0], presence_penalty=[0.25, 0.0]),
        [True, True],
    )
    assert sp.users == [0]
    pen.stage(sp, [1] * T + [0] * T)
    ids_out = torch.arange(200, 232).reshape(1, 1, 1, 32)
    pen.bind(step=_step_buffer(tokens), ids_out=ids_out)
    x = _bf16_logits(B * T).reshape(1, 1, B * T, V)
    raw = x.clone()
    pen.apply_body(x)
    for j in range(T):
        history = torch.cat([output[0], tokens[0, 1 : j + 1]])
        want32 = reference_penalties(raw[0, 0, j], prompt[0], history, 1.3, 0.4, 0.25)
        assert torch.equal(x[0, 0, j], want32.to(torch.bfloat16)), f"column {j}"
        assert int(ids_out[0, 0, 0, j]) == int(want32.argmax())
    assert torch.equal(x[0, 0, T:], raw[0, 0, T:]), "a neutral row is unchanged bit for bit"
    assert ids_out[0, 0, 0, T:].tolist() == list(range(200 + T, 232)), "only the greedy penalty rows' lanes change"


def test_the_reference_is_vllms_own_penalty_op():
    pytest.importorskip("vllm")
    from vllm.v1.sample.ops.penalties import apply_all_penalties

    logits = _bf16_logits(3, seed=3).to(torch.float32)
    prompts = [[1, 2, 2], [5], [9, 10]]
    histories = [[1, 1, 3], [], [10, 10, 10, 4]]
    rep, freq, pres = [1.2, 0.8, 1.0], [0.3, 0.0, -0.5], [0.0, 0.7, 0.2]
    width = max(len(p) for p in prompts)
    prompt_ids = torch.tensor([p + [V] * (width - len(p)) for p in prompts])
    want = apply_all_penalties(
        logits.clone(), prompt_ids, torch.tensor(pres), torch.tensor(freq), torch.tensor(rep), histories
    )
    for r in range(3):
        got = reference_penalties(logits[r], prompts[r], histories[r], rep[r], freq[r], pres[r])
        assert torch.equal(got, want[r])


def test_a_bucket_scores_its_first_rows_only():
    """A verify row bucket of b users hands the penalty body its shard of the first b * T rows."""
    B, T, b = 4, 4, 2
    pen = _penalties(B, T)
    pen.rebuild({1: (torch.tensor([2]), torch.tensor([2, 2, 3]))})
    sp = step_penalties(_sampling(repetition_penalty=[1.0, 2.0, 1.0, 1.0]), [True, True, False, False])
    pen.stage(sp, [0] * (B * T))
    tokens = torch.tensor([[1, 2, 3, 4], [3, 2, 2, 5], [0] * 4, [0] * 4])
    pen.bind(step=_step_buffer(tokens), ids_out=torch.zeros(1, 1, 1, 32, dtype=torch.int64))
    x = _bf16_logits(b * T, seed=1).reshape(1, 1, b * T, V)
    raw = x.clone()
    pen.apply_body(x)
    assert torch.equal(x[0, 0, :T], raw[0, 0, :T])
    for j in range(T):
        history = torch.cat([torch.tensor([2, 2, 3]), tokens[1, 1 : j + 1]])
        want = reference_penalties(raw[0, 0, T + j], [2], history, 2.0, 0.0, 0.0).to(torch.bfloat16)
        assert torch.equal(x[0, 0, T + j], want)


# --------------------------------------------------------------------------------------------- state and commit


def _counts(pen, users):
    return pen.read(users)[0]


def test_rebuild_and_add_write_a_row_from_host_lists():
    B, T = 3, 2
    pen = _penalties(B, T)
    pen.rebuild({0: (torch.tensor([1, 1, 5]), torch.tensor([4, 4])), 2: (torch.tensor([]), torch.tensor([9]))})
    pen.add({0: torch.tensor([4, 8, 9]), 2: torch.tensor([9])})
    counts, prompt = pen.read([0, 1, 2])
    assert torch.equal(counts[0], counts_of([4, 4, 4, 8, 9], V)) and torch.equal(counts[2], counts_of([9, 9], V))
    assert not counts[1].any(), "a row neither rebuilt nor added stays zero"
    assert prompt[0].nonzero().reshape(-1).tolist() == [1, 5] and not prompt[2].any()
    # all T rows of a user hold the same counts
    assert torch.equal(pen.counts[0, 0, 0], pen.counts[0, 0, 1])
    pen.rebuild({0: (torch.tensor([7]), torch.tensor([3]))})
    counts, prompt = pen.read([0, 2])
    assert torch.equal(counts[0], counts_of([3], V)) and torch.equal(counts[1], counts_of([9, 9], V))


@pytest.mark.parametrize("drawn_user", [None, 1])
def test_the_commit_adds_each_tracked_users_commit_up_to_its_count(drawn_user):
    """The predecessor's commit of user u is its ids (its draws for a drawn user) in columns 0 .. m, with m the drafter
    pack's anchor u * T + m minus u * T; an untracked user's counts do not move."""
    B, T = 3, 4
    pen = _penalties(B, T)
    pen.rebuild({u: (torch.tensor([]), torch.tensor([1, 2])) for u in range(B)})
    ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    draws = ids + 40
    m = [2, 0, 3]
    pack = torch.zeros(draft_anchor(T - 1) + 1, pack_width(B * T), dtype=torch.int64)
    pack[draft_anchor(T - 1), :B] = torch.tensor([u * T + m[u] for u in range(B)])
    lanes = lambda t: torch.cat([t.reshape(-1), torch.zeros(32 - B * T, dtype=torch.int64)]).reshape(
        1, 1, 1, 32
    )  # noqa: E731
    pen.bind(ids_out=lanes(ids), draws=lanes(draws)[..., : B * T], pack=pack)
    sp = step_penalties(_sampling(presence_penalty=[0.5, 0.5, 0.5]), [True] * B)
    drawn = [int(u == drawn_user) for u in range(B)]
    pen.stage(sp, [0] * (B * T), track=[True, True, False], drawn=drawn)
    pen.commit_body()
    counts = _counts(pen, range(B))
    for u in range(2):
        commit = (draws if drawn[u] else ids)[u, : m[u] + 1]
        assert torch.equal(counts[u], counts_of(torch.cat([torch.tensor([1, 2]), commit]), V)), f"user {u}"
    assert torch.equal(counts[2], counts_of([1, 2], V))


# ------------------------------------------------------------------------------------------------------- records


def _ids(*v):
    return torch.tensor(v, dtype=torch.int64)


def test_reconcile_adds_a_commit_and_rewrites_a_join_a_move_a_new_prompt_and_a_long_gap():
    records = [None] * 4
    prompts = {u: _ids(1, 2, u) for u in range(4)}
    rebuild, add = reconcile(records, [0, 1], prompts, {0: _ids(5), 1: _ids(6, 6)})
    assert set(rebuild) == {0, 1} and add == {}, "a join rewrites"
    rebuild, add = reconcile(records, [0, 1], prompts, {0: _ids(5, 7, 8), 1: _ids(6, 6)})
    assert rebuild == {} and torch.equal(add[0], _ids(7, 8)) and 1 not in add, "a commit adds its tokens"
    records[2], records[0] = records[0], None  # a row move: the record names the row it was written to
    rebuild, add = reconcile(records, [2], prompts | {2: prompts[0]}, {2: _ids(5, 7, 8)})
    assert list(rebuild) == [2] and records[2].row == 2
    rebuild, _ = reconcile(records, [1], prompts | {1: _ids(9)}, {1: _ids(6, 6)})
    assert list(rebuild) == [1], "another prompt in the slot rewrites"
    long = torch.cat([_ids(6, 6), torch.arange(MAX_ADDED_TOKENS + 1)])
    rebuild, add = reconcile(records, [1], prompts | {1: _ids(9)}, {1: long})
    assert list(rebuild) == [1] and add == {}, "a gap past MAX_ADDED_TOKENS rewrites"
    rebuild, add = reconcile(records, [1], prompts | {1: _ids(9)}, {1: _ids(4)})
    assert list(rebuild) == [1], "an output that is not a continuation rewrites"


def test_a_record_resolves_the_commits_a_chained_verify_added():
    T = 4
    step = SimpleNamespace(K=T - 1, ids=None, pack=None)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    pack = torch.zeros(draft_anchor(T - 1) + 1, 32, dtype=torch.int64)
    pack[draft_anchor(T - 1), :2] = torch.tensor([0 * T + 1, 1 * T + 3])
    step.wait = lambda: ids
    step.pack = pack
    rec = PenaltyRow(_ids(9), _ids(4), 1, pending=[(step, 1)])
    assert not rec.matches(_ids(9), _ids(4, 5, 6, 7, 8), 1), "a pending commit is not resolved yet"
    rec.resolve(T)
    assert rec.matches(_ids(9), _ids(4, 5, 6, 7, 8), 1) and rec.pending == []


def test_step_penalties_finds_the_active_rows_with_a_penalty(expect_error):
    assert step_penalties(None, [True]) is None
    assert step_penalties(_sampling(repetition_penalty=[1.0, 1.0]), [True, True]) is None
    sp = step_penalties(
        _sampling(repetition_penalty=[1.0, 1.1, 1.2], frequency_penalty=[0.1, 0.0, 0.0]), [True, True, False]
    )
    assert sp.users == [0, 1] and sp.repetition.tolist()[2] == 1.0, "an inactive row is neutral"
    with expect_error(ValueError, "repetition above 0"):
        step_penalties(_sampling(repetition_penalty=[0.0]), [True])
    with expect_error(ValueError, "finite"):
        step_penalties(_sampling(presence_penalty=[float("inf")]), [True])


# -------------------------------------------------------------------------------------------------------- engine

ENGINE_VOCAB = 512  # every id the chained engine stand-in produces is below it


@pytest.fixture
def penalized(monkeypatch):
    """test_spec_chain_host's chained MTPBlockEngine with the verify penalties run in torch (TorchOps): the commit
    trace reads the predecessor's ids, draws and drafter pack where the stand-in device left them."""
    import ttnn
    from models.demos.blackhole.qwen36.tests import test_spec_chain_host as chain_host

    eng, log, state = chain_host._chained_engine(monkeypatch)
    pen = _penalties(eng.B, eng.T, ENGINE_VOCAB)
    eng.model._vfy_penalties = pen
    eng._pen_commit_trace = "pen_commit"
    replay = ttnn.execute_trace

    def execute_trace(mesh, trace, **kw):
        if trace == "pen_commit":
            pred = log["enqueue"][-1]
            draws = torch.zeros(eng.B * eng.T, dtype=torch.int64) if pred.draws is None else pred.draws
            pen.bind(ids_out=chain_host._lanes(pred.ids), draws=draws.reshape(1, 1, 1, -1), pack=pred.pack)
            pen.commit_body()
        replay(mesh, trace, **kw)

    monkeypatch.setattr(ttnn, "execute_trace", execute_trace)
    return eng, log, state, pen, chain_host


def _verify(eng, tokens, starts, nv, sampling, prompts, outputs, slot_remap=None):
    from models.demos.blackhole.qwen36.tests.test_spec_chain_host import _mamba

    T = eng.T
    return eng.decode_forward(
        torch.as_tensor(tokens),
        torch.tensor([[s + j if s >= 0 else -1 for j in range(T)] for s in starts]),
        num_valid_drafts=torch.tensor(nv, dtype=torch.int32),
        accepted_counts=torch.ones(eng.B, dtype=torch.int32),
        page_table=torch.arange(1, eng.B * eng.nb + 1, dtype=torch.int32).reshape(eng.B, eng.nb),
        spec_mode="argmax_ids",
        mamba_state=_mamba(eng),
        sampling=sampling,
        defer=True,
        slot_remap=slot_remap,
        prompt_tokens=_padded(prompts, eng.B),
        output_tokens=_padded(outputs, eng.B),
    )


def _chained(eng, res, upper, hint, sampling, prompts, outputs, rows=None, slot_remap=None):
    from models.demos.blackhole.qwen36.tests.test_spec_chain_host import _mamba

    B, T = eng.B, eng.T
    rows = [s >= 0 for s in upper] if rows is None else rows
    return eng.decode_forward(
        torch.where(torch.tensor(rows).reshape(B, 1), -1, 0).expand(B, T),
        torch.tensor([[s + j if r else -1 for j in range(T)] for s, r in zip(upper, rows)]),
        num_valid_drafts=torch.tensor(hint, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=torch.arange(1, B * eng.nb + 1, dtype=torch.int32).reshape(B, eng.nb),
        spec_mode="argmax_ids",
        mamba_state=_mamba(eng),
        sampling=sampling,
        defer=True,
        predecessor=res.argmax_ids,
        slot_remap=slot_remap,
        prompt_tokens=_padded(prompts, B),
        output_tokens=_padded(outputs, B),
    )


def _padded(rows, B):
    """The plugin's -1 padded token tensor of ``rows`` {row: ids}; rows without ids are all padding."""
    width = max([len(v) for v in rows.values()] + [1])
    out = torch.full((B, width), -1, dtype=torch.int32)
    for u, ids in rows.items():
        out[u, : len(ids)] = torch.as_tensor(ids, dtype=torch.int32)
    return out


def _assert_counts(pen, outputs):
    users = sorted(outputs)
    got = pen.read(users)[0]
    for i, u in enumerate(users):
        want = counts_of(torch.as_tensor(outputs[u]), ENGINE_VOCAB)
        assert torch.equal(got[i], want), f"row {u}: device counts {got[i].nonzero().reshape(-1).tolist()}"


def test_the_counts_follow_a_join_a_chained_commit_a_drained_commit_a_finish_and_a_resume(penalized):
    """Rows 0 and 3 carry penalties, row 1 none. Step 0 (host-staged) writes rows 0 and 3 from their outputs; step 1
    chains on it, and its commit trace adds step 0's commit on the device; step 2 is drained, and adds step 1's commit
    from the host lists after the record resolves step 0's commit from the read verify. Row 3's request then finishes
    and a resumed request takes slot 3: its first verify rewrites the row from its own output."""
    eng, log, state, pen, ch = penalized
    B, T, K = eng.B, eng.T, eng.K
    sampling = _sampling(
        repetition_penalty=[1.2, 1.0, 1.0, 1.0], presence_penalty=[0.0, 0.0, 0.0, 0.5], temperature=[0.0] * B
    )
    prompts = {0: [3, 3, 4], 1: [5], 3: [6, 7]}
    out = {0: [1], 1: [2], 3: [3]}
    ids0 = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [21, 22, 23, 24]])
    tokens0 = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [3, 21, 9, 9]])
    starts0, nv0 = [100, 200, -1, 300], [3, 3, 0, 2]
    drafts = torch.stack(state["drafts"], dim=1)
    ids1 = torch.zeros(B, T, dtype=torch.int64)
    ids1[0] = torch.tensor([int(drafts[0, 0]), int(drafts[0, 1]), 5, 6])
    ids1[1] = torch.tensor([9, 9, 9, 9])
    ids1[3] = torch.tensor([int(drafts[3, 0]), 4, 4, 4])
    ids2 = torch.tensor([[40, 41, 42, 43], [44, 45, 46, 47], [0, 0, 0, 0], [48, 49, 50, 51]])
    state["ids"] = [ids0, ids1, ids2]

    res0 = _verify(eng, tokens0, starts0, nv0, sampling, prompts, out)
    _assert_counts(pen, {0: out[0], 3: out[3]})
    assert pen._staged["greedy"][0, 0, 0].tolist() == [1.0] * T + [0.0] * (2 * T) + [1.0] * T + [0.0] * 16

    # Step 1 chains on the unread step 0: the runner's outputs still lack step 0's commit.
    res1 = _chained(eng, res0, [s + T if s >= 0 else -1 for s in starts0], [K] * B, sampling, prompts, out)
    assert eng.chained_steps == 1 and "pen_commit" in log["traces"]
    assert log["traces"].index("pen_commit") < log["traces"].index("chain")
    assert log["enqueue"][-1].kw["penalties"] is True
    committed0, counts0 = ch._walk(res0, tokens0, nv0)
    commit0 = {u: committed0[u, : int(counts0[u])].tolist() for u in (0, 1, 3)}
    _assert_counts(pen, {0: out[0] + commit0[0], 3: out[3] + commit0[3]})
    ch._propose(eng, starts0, committed0, counts0)
    out = {u: out[u] + commit0[u] for u in out}

    # Step 2 is drained: the runner read step 1 and sends every commit.
    succ = res0.argmax_ids.successor
    committed1, counts1 = ch._walk(res1, succ.tokens.clone(), [K, K, 0, K])
    ch._propose(eng, succ.p0.where(succ.active, torch.tensor(-1)).tolist(), committed1, counts1)
    out = {u: out[u] + committed1[u, : int(counts1[u])].tolist() for u in out}
    starts2 = [int(succ.p0[u]) + int(counts1[u]) if u != 2 else -1 for u in range(B)]
    tokens2 = torch.tensor([[out[0][-1], 0, 0, 0], [out[1][-1], 0, 0, 0], [0] * 4, [out[3][-1], 0, 0, 0]])
    _verify(eng, tokens2, starts2, [0] * B, sampling, prompts, out)
    _assert_counts(pen, {0: out[0], 3: out[3]})
    assert eng._pen_rows[0].pending == [] and eng._pen_rows[3].output.tolist() == out[3]

    # Row 3's request finishes; a resumed request (its own prompt and earlier output) is prefilled into slot 3.
    eng.release(3)
    assert eng._pen_rows[3] is None
    eng.finish_prefill(3, 500, True)
    eng._warm_fresh_rows = lambda fresh, tokens, pt: [eng._fresh.pop(u) for u in fresh]  # no drafter on the host
    prompts[3], out[3] = [8, 8, 9], [60, 61, 61]
    state["ids"] = [ids2]
    _verify(eng, torch.tensor([[out[0][-1], 0, 0, 0], [out[1][-1], 0, 0, 0], [0] * 4, [61, 0, 0, 0]]),
            [starts2[0] + 1, starts2[1] + 1, -1, 500], [0] * B, sampling, prompts, out)  # fmt: skip
    _assert_counts(pen, {3: out[3]})
    assert pen.read([3])[1][0].nonzero().reshape(-1).tolist() == [8, 9], "the resumed request's prompt mask"


def test_a_chained_verify_that_moves_a_penalty_row_adds_the_commit_on_the_host(penalized):
    """Row 0's request finished and the runner moved row 3's request into row 0 (slot_remap): the commit trace cannot
    follow the move, so the engine reads the predecessor, walks it, and rewrites row 0 from the runner's output plus
    that commit."""
    eng, log, state, pen, ch = penalized
    B, T, K = eng.B, eng.T, eng.K
    sampling = _sampling(presence_penalty=[0.0, 0.0, 0.0, 0.5], temperature=[0.0] * B)
    prompts, out = {1: [5], 3: [6, 7]}, {1: [2], 3: [3]}
    ids0 = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [21, 22, 23, 24]])
    tokens0 = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [3, 21, 9, 9]])
    starts0, nv0 = [100, 200, -1, 300], [3, 3, 0, 2]
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    res0 = _verify(eng, tokens0, starts0, nv0, sampling, prompts | {0: [1]}, out | {0: [1]})
    eng.release(0)
    moved = _sampling(presence_penalty=[0.5, 0.0, 0.0, 0.0], temperature=[0.0] * B)
    upper = [304, 204, -1, -1]
    _chained(eng, res0, upper, [K] * B, moved, {0: prompts[3], 1: prompts[1]}, {0: out[3], 1: out[1]},
             rows=[True, True, False, False], slot_remap=[3, 1, 2, 0])  # fmt: skip
    assert eng.materialized_steps == 1 and "pen_commit" not in log["traces"]
    committed0, counts0 = ch._walk(res0, tokens0, nv0)
    _assert_counts(pen, {0: out[3] + committed0[3, : int(counts0[3])].tolist()})
    assert eng._pen_rows[0].row == 0 and eng._pen_rows[3] is None


def test_a_chained_verify_whose_counts_are_not_the_runners_output_is_built_from_the_read_predecessor(penalized):
    eng, log, state, pen, ch = penalized
    B, T, K = eng.B, eng.T, eng.K
    sampling = _sampling(frequency_penalty=[0.3, 0.0, 0.0, 0.0], temperature=[0.0] * B)
    tokens0 = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [3, 21, 9, 9]])
    starts0, nv0 = [100, 200, -1, 300], [3, 3, 0, 2]
    state["ids"] = [torch.tensor([[7, 8, 9, 10]] * B), torch.zeros(B, T, dtype=torch.int64)]
    res0 = _verify(eng, tokens0, starts0, nv0, sampling, {0: [4]}, {0: [1]})
    _chained(eng, res0, [s + T if s >= 0 else -1 for s in starts0], [K] * B, sampling, {0: [4]}, {0: [1, 99]})
    assert eng.materialized_steps == 1 and eng.chained_steps == 0


def test_a_device_sampled_narrow_decode_with_logprobs_reads_raw_logits_and_penalized_ids(penalized):
    """The ordinary decode of a device-sampled step with logprobs reads the verify's logits: the step replays the
    penalty trace too, after the logits traces (raw logits, penalized ids)."""
    eng, log, state, pen, ch = penalized
    B, T = eng.B, eng.T
    state["ids"] = [torch.zeros(B, T, dtype=torch.int64)]
    eng.decode_forward(
        torch.tensor([[1, -1, -1, -1]] * B),
        torch.tensor([[100, -1, -1, -1]] * B),
        num_valid_drafts=torch.zeros(B, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=torch.arange(1, B * eng.nb + 1, dtype=torch.int32).reshape(B, eng.nb),
        spec_mode="logits",
        mamba_state=ch._mamba(eng),
        sampling=_sampling(presence_penalty=[0.5] * B, temperature=[0.0] * B),
        one_enqueue=False,
        prompt_tokens=_padded({u: [7] for u in range(B)}, B),
        output_tokens=_padded({u: [1] for u in range(B)}, B),
    )
    kw = log["traced"][-1]
    assert kw["read_logits"] is True and kw["penalties"] is True and "bucket" not in kw
    _assert_counts(pen, {u: [1] for u in range(B)})


def test_the_class_declares_verify_penalties_and_hands_over_the_histories():
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM

    caps = Qwen36MTPContractForCausalLM.model_capabilities
    assert caps["supports_spec_device_penalties"] is True and caps["spec_ids_are_sampled"] is True
    assert caps["supports_device_penalties"] is False, "an ordinary decode with a penalty stays host-sampled"


def test_a_token_masked_chained_verify_with_penalties_takes_its_greedy_penalty_rows_masked_argmax(penalized):
    """vllm-tt-plugin supports_spec_device_mask with supports_spec_device_penalties on a chained verify: the
    submission adds the predecessor's commit to the counts (the commit trace, before the chain program) and enqueues
    the verify alone; every live row draws but the greedy penalty rows 0 and 3, whose ids are the penalty trace's
    float32 argmax, replayed at sample_deferred after the mask. A mask may constrain them."""
    from models.demos.blackhole.qwen36.tt import spec_engine

    eng, log, state, pen, ch = penalized
    B, T, K = eng.B, eng.T, eng.K
    eng.model._vfy_mask = object()
    posts = []
    eng.model.verify_enqueue_post = lambda **kw: posts.append(kw)
    sampling = _sampling(
        repetition_penalty=[1.2, 1.0, 1.0, 1.0], presence_penalty=[0.0, 0.0, 0.0, 0.5], temperature=[0.0] * B
    )
    prompts = {0: [3, 3, 4], 1: [5], 3: [6, 7]}
    out = {0: [1], 1: [2], 3: [3]}
    ids0 = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [21, 22, 23, 24]])
    tokens0 = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [3, 21, 9, 9]])
    starts0, nv0 = [100, 200, -1, 300], [3, 3, 0, 2]
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    res0 = _verify(eng, tokens0, starts0, nv0, sampling, prompts, out)

    payload = _chained_deferred(eng, res0, [s + T if s >= 0 else -1 for s in starts0], sampling, prompts, out)
    assert isinstance(payload, spec_engine.DeferredVerify) and payload.chained and payload.penalties
    assert payload.penalty_argmax == (True, False, False, True)
    # The greedy row without a penalty draws with top-k 1; the greedy penalty rows take the penalized argmax.
    assert payload.step.sample_plan.sampled == [False] * T + [True] * T + [False] * (2 * T)
    assert log["traces"].index("pen_commit") < log["traces"].index("chain")
    assert log["enqueue"][-1].kw["post"] is False and log["enqueue"][-1].kw["penalties"] is False
    assert pen._staged["greedy"][0, 0, 0].tolist() == [1.0] * T + [0.0] * (2 * T) + [1.0] * T + [0.0] * 16

    committed0, counts0 = ch._walk(res0, tokens0, nv0)
    ch._propose(eng, starts0, committed0, counts0)
    mask = torch.full((B, T, ENGINE_VOCAB // 32), -1, dtype=torch.int32)
    mask[0, 1, 0] = 0b1100
    mask[1, 2, 0] = 0b0110
    eng.sample_deferred(payload, sampling=sampling, mask=mask, num_valid_drafts=torch.tensor([K, K, 0, K]))
    assert posts[-1]["penalties"] is True and posts[-1]["mask"] is not None
    assert posts[-1]["sample_plan"] is payload.step.sample_plan
    assert log["traces"][-2:] == ["step", "drafter"]


def _chained_deferred(eng, res, upper, sampling, prompts, outputs):
    from models.demos.blackhole.qwen36.tests.test_spec_chain_host import _mamba

    B, T, K = eng.B, eng.T, eng.K
    rows = [s >= 0 for s in upper]
    return eng.decode_forward(
        torch.where(torch.tensor(rows).reshape(B, 1), -1, 0).expand(B, T),
        torch.tensor([[s + j if r else -1 for j in range(T)] for s, r in zip(upper, rows)]),
        num_valid_drafts=torch.full((B,), K, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=torch.arange(1, B * eng.nb + 1, dtype=torch.int32).reshape(B, eng.nb),
        spec_mode="argmax_ids",
        mamba_state=_mamba(eng),
        sampling=sampling,
        defer=True,
        predecessor=res.argmax_ids,
        defer_sampling=True,
        prompt_tokens=_padded(prompts, B),
        output_tokens=_padded(outputs, B),
    )
