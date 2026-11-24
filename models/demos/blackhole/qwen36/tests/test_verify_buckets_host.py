# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the MTP verify row buckets of Qwen36MTPContractForCausalLM.

MTPBlockEngine runs every decode step as a verify of B users x T = K + 1 rows. A step whose live rows all sit below a
bucket b of spec_engine.verify_buckets(B) (1, 2, 4 at B = 8) replays the verify trace of users 0 .. b-1 instead
(Qwen36Model._verify_bucket_body), which hands its ids and drafter feed rows to the B-user outputs. Covered here:

* the bucket of a step is the smallest that holds its highest live row, chosen from that step's rows alone (it grows
  and drops at once), and a step that reads logits (logits verify) replays the B-user trace;
* a device-sampled step, chained or not, takes the bucket of its rows when the bucket has a sampler trace, and the
  B-user trace otherwise;
* rows join, finish, move (slot_remap) and are preempted across buckets; a chained verify takes the bucket of its own
  rows, also below its predecessor's, and both proposes still check out;
* Qwen36Model replays the bucket's trace, and with device draws the bucket's sampler trace after it, and refuses before
  it stages anything a bucket that would leave a live user out, device draws on a bucket without a sampler trace and a
  bucket whose logits are read;
* the bucket body runs on the first rows of the B-user page tables and hands the ids lanes and the feed rows over with
  zero rows past the bucket; it keeps its vocab shard only for a sampler; the GDN verify of b users reads its rows of
  the B-user ctrl page (gdn_spec_step max_users), which holds each user's start pages at 1 + B + u*Nv + h;
* a bucket keeps the full-width verify's SDPA core split (spec_sdpa_split_users), so its rows reduce their KV range
  as in the full-width verify;
* warmup runs every bucket body (and sampler body) into the warm outputs, capture records one trace per bucket (and
  one sampler trace) after the B-user trace, and release frees them.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_mtp_gdn_blocks_host import NUM_BLOCKS, _layout, _pages
from models.demos.blackhole.qwen36.tests.test_spec_chain_host import (  # noqa: F401 (chained is a fixture)
    _chained_verify,
    _host_verify,
    _propose,
    _sampled_predecessor,
    _sampling,
    _walk,
    chained,
)
from models.demos.blackhole.qwen36.tests.test_spec_step_host import _bare_verify_model, _rope_model
from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


# ----------------------------------------------------------------------------------------- bucket choice
def test_the_bucket_is_the_smallest_that_holds_the_highest_live_row():
    assert spec_engine.verify_buckets(8) == (1, 2, 4)
    assert spec_engine.verify_buckets(4) == (1, 2) and spec_engine.verify_buckets(2) == (1,)
    assert spec_engine.verify_buckets(1) == () and spec_engine.verify_buckets(6) == (1, 2, 4)
    rows = lambda *live: [u in live for u in range(8)]
    cases = [((0,), 1), ((1,), 2), ((0, 1), 2), ((2,), 4), ((0, 3), 4), ((4,), 8), ((0, 7), 8), ((), 1)]
    for live, want in cases:
        assert spec_engine.verify_bucket((1, 2, 4), rows(*live), 8) == want, live
    assert spec_engine.verify_bucket((), rows(0), 8) == 8, "no captured bucket: the B-user trace"
    assert spec_engine.verify_bucket((4,), rows(0), 8) == 4


@pytest.fixture
def bucket_engine(monkeypatch):
    """MTPBlockEngine of B = 8 users over a stand-in model with captured buckets 1, 2, 4: every verify records the
    keywords of its verify_traced call."""
    B, K = 8, 3
    calls = []
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        mesh_device=MagicMock(),
        args=SimpleNamespace(vocab_size=32, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=SimpleNamespace())],
        gdn_blocks=SimpleNamespace(verify_tokens=K + 1, set_verify_blocks=lambda s: None),
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        spec_feed_rows=lambda h: h,
        rope=SimpleNamespace(rope_delta=0),
        _vfy_bucket_traces={1: "v1", 2: "v2", 4: "v4"},
    )

    def verify_traced(tokens, positions, mi, after_enqueue=None, **kw):
        calls.append(kw)
        if after_enqueue is not None:
            after_enqueue()
        return [0] * (B * (K + 1)), "feed", (torch.zeros(B * (K + 1), 32) if kw.get("read_logits") else None)

    model.verify_traced = verify_traced
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 16, 64, 10**6
    eng._last_hidden = [f"h{u}" for u in range(B)]
    eng._mtp_pt_rows = [f"pt{u}" for u in range(B)]
    monkeypatch.setattr(eng, "_stage_mtp_tables", lambda pt: None)
    monkeypatch.setattr(eng, "_warm_mtp_last", lambda *a, **kw: None)
    return eng, calls


def _step(eng, starts, remap=None, spec_mode="argmax_ids"):
    """decode_forward of a [B, 1+K] block whose row u starts at starts[u] (None: padding), one valid draft per row."""
    B, T = eng.B, eng.T
    live = [s is not None for s in starts]
    start_pos = torch.tensor([[s + j if s is not None else -1 for j in range(T)] for s in starts])
    pt = torch.arange(1, B * eng.nb + 1, dtype=torch.int32).reshape(B, eng.nb)
    return eng.decode_forward(
        torch.ones(B, T, dtype=torch.int64),
        start_pos,
        num_valid_drafts=torch.tensor([1 if a else 0 for a in live], dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=pt,
        spec_mode=spec_mode,
        mamba_state="state",
        slot_remap=remap,
    )


def test_each_step_takes_the_bucket_of_its_own_rows_up_and_down_at_once(bucket_engine):
    """No hold: a bucket change costs no drain and no restaging (every program after the verify runs at B users), so
    each step takes the smallest bucket of its own rows, growing and dropping from one step to the next."""
    eng, calls = bucket_engine
    eng._active = set(range(eng.B))
    pattern = [
        (range(8), None),
        ((0,), 1),
        ((0, 1), 2),
        ((0,), 1),
        ((0, 2), 4),
        ((0, 1, 2, 3, 4), None),
        ((0, 1, 2, 3), 4),
        ((1,), 2),
    ]
    for live, want in pattern:
        _step(eng, [100 if u in live else None for u in range(eng.B)])
        assert calls[-1].get("bucket") == want, (list(live), calls[-1].get("bucket"))
        inactive = calls[-1]["inactive"]
        assert all(inactive[(want or eng.B) :]), "a live row past the bucket"
    assert eng.verify_bucket_steps == {1: 2, 2: 2, 4: 2, 8: 2}


def test_a_step_that_reads_logits_replays_the_full_width_trace(bucket_engine):
    """A bucket trace makes no logits to read: a logits verify (a host-sampled or narrow step) replays the B-user trace,
    whatever its rows, and so does a verify whose rows the device sampler draws when the buckets have no sampler
    trace."""
    eng, calls = bucket_engine
    eng._active = set(range(eng.B))
    _step(eng, [100] + [None] * (eng.B - 1), spec_mode="logits")
    assert "bucket" not in calls[-1] and calls[-1]["read_logits"] is True
    pt = torch.ones(eng.B, eng.nb, dtype=torch.int32)
    active = torch.tensor([True] + [False] * (eng.B - 1))
    eng.verify(torch.zeros(eng.B, eng.T), torch.tensor([100] + [0] * 7), None, pt, active=active, mamba_state="s",
               sample_plan=SimpleNamespace(sampled=[True] * 32))  # fmt: skip
    assert "bucket" not in calls[-1] and calls[-1]["sample_plan"] is not None
    eng.verify(torch.zeros(eng.B, eng.T), torch.tensor([100] + [0] * 7), None, pt, active=active, mamba_state="s")
    assert calls[-1]["bucket"] == 1
    assert eng.verify_bucket_steps == {1: 1, 8: 2}


def test_a_device_sampled_step_takes_the_bucket_that_has_a_sampler_trace(bucket_engine):
    """With the buckets' sampler traces, a verify whose rows the device sampler draws takes the bucket of its rows; a
    logits verify (a narrow step that asks for logprobs) stays on the B-user trace, and so do rows that fit only a
    bucket without a sampler trace."""
    eng, calls = bucket_engine
    eng._active = set(range(eng.B))
    eng.model._vfy_bucket_sample_traces = {1: "s1", 2: "s2", 4: "s4"}
    pt = torch.ones(eng.B, eng.nb, dtype=torch.int32)
    plan = SimpleNamespace(sampled=[True] * 32)

    def verify(live, **kw):
        active = torch.tensor([u in live for u in range(eng.B)])
        starts = torch.tensor([100 + u if u in live else 0 for u in range(eng.B)])
        eng.verify(torch.zeros(eng.B, eng.T), starts, None, pt, active=active, mamba_state="s", **kw)
        return calls[-1].get("bucket")

    assert verify({0}, sample_plan=plan) == 1 and calls[-1]["sample_plan"] is plan
    assert verify({0, 1}, sample_plan=plan) == 2
    assert verify({0, 3}, sample_plan=plan) == 4
    assert verify({0, 5}, sample_plan=plan) is None
    assert verify({0}, sample_plan=plan, read_logits=True) is None
    eng.model._vfy_bucket_sample_traces = {1: "s1"}
    assert verify({0, 1}, sample_plan=plan) is None, "bucket 2 has no sampler trace"
    assert verify({0, 1}) == 2, "a step without draws takes every bucket"
    assert eng.verify_bucket_steps == {1: 1, 2: 2, 4: 1, 8: 3}


def test_rows_join_finish_move_and_resume_across_buckets(bucket_engine):
    """Request A prefills into row 0, B into row 1, C into row 2 (each first verify writes its prompt's last drafter
    slot); A finishes and the runner moves C from row 2 into row 0 (slot_remap); B is preempted and resumes in row 1.
    Every step runs the bucket of its live rows; padding rows keep the null block in the verify's page table."""
    eng, calls = bucket_engine
    B = eng.B
    pos = {}

    def step(rows, remap=None):
        starts = [pos.get(u) for u in range(B)]
        _step(eng, starts, remap)
        for u in rows:
            pos[u] += eng.T
        kw = calls[-1]
        assert [not x for x in kw["inactive"]] == [u in rows for u in range(B)]
        assert all(int(kw["page_tables"][u].abs().sum()) == 0 for u in range(B) if u not in rows)
        return kw.get("bucket")

    eng.finish_prefill(0, 100, final=True)
    pos[0] = 100
    assert step({0}) == 1 and not eng._fresh
    eng.finish_prefill(1, 50, final=True)
    pos[1] = 50
    assert step({0, 1}) == 2
    eng.finish_prefill(2, 70, final=True)
    pos[2] = 70
    assert step({0, 1, 2}) == 4
    eng.release(0)  # A finishes; the runner condenses C into row 0
    pos[0] = pos.pop(2)
    assert step({0, 1}, remap=[2, 1, 0, 3, 4, 5, 6, 7]) == 2 and eng._active == {0, 1}
    eng.release(1)  # B is preempted
    pos.pop(1)
    assert step({0}) == 1
    eng.finish_prefill(1, 90, final=True)  # B resumes in row 1 after its prefill
    pos[1] = 90
    assert step({0, 1}) == 2
    assert eng.verify_bucket_steps == {1: 2, 2: 3, 4: 1}


def test_a_chained_verify_takes_the_bucket_of_its_own_rows(chained):
    """The predecessor runs rows 0 and 1 (bucket 2); row 1 then finishes, and the chained verify over row 0 alone
    replays bucket 1 after the chain program. The chain program, the step trace and the drafter run at B users, so
    both proposes check the device's commit as without buckets."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    eng.model._vfy_bucket_traces = {1: "v1", 2: "v2"}
    drafts = torch.stack(state["drafts"], dim=1)
    ids0 = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [0, 0, 0, 0]])
    tokens0 = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [0, 0, 0, 0]])
    starts0, nv0 = [100, 200, -1, -1], [3, 3, 0, 0]
    ids1 = torch.zeros(B, T, dtype=torch.int64)
    ids1[0] = torch.tensor([int(drafts[0, 0]), 4, 4, 4])
    state["ids"] = [ids0, ids1]
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    assert log["enqueue"][-1].kw["bucket"] == 2
    res1 = _chained_verify(eng, res0, [s + T if s >= 0 else -1 for s in starts0], [K] * B, rows=[True] + [False] * 3)
    assert log["enqueue"][-1].kw["bucket"] == 1 and log["enqueue"][-1].kw["step_staged"] is True
    assert log["traces"] == ["step", "drafter", "chain", "step", "drafter"] and eng.chained_steps == 1
    committed0, counts0 = _walk(res0, tokens0, nv0)
    out0 = _propose(eng, starts0, committed0, counts0)
    assert out0.num_valid.tolist() == [K, K, 0, 0]
    succ = res1.argmax_ids
    committed1, counts1 = _walk(res1, succ.tokens, succ.num_valid.tolist())
    assert counts1.tolist() == [2, 1, 1, 1]
    out1 = _propose(eng, succ.p0.where(succ.active, torch.tensor(-1)).tolist(), committed1, counts1)
    assert out1.num_valid.tolist() == [K, 0, 0, 0] and eng.one_enqueue_host_proposes == 0
    assert eng.verify_bucket_steps == {1: 1, 2: 1}


def test_a_sampled_chained_verify_replays_the_full_width_trace(chained):
    """A device-sampled chained verify on buckets without sampler traces replays the B-user trace even when its live
    rows (0 and 1) fit the captured bucket 2."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    eng.model._vfy_bucket_traces = {1: "v1", 2: "v2"}
    res0, tokens0, starts0, nv0 = _sampled_predecessor(eng, state)
    assert "bucket" not in log["enqueue"][-1].kw
    state["ids"] += [torch.zeros(B, T, dtype=torch.int64)]
    state["draws"] += [torch.zeros(B * T, dtype=torch.int64)]
    upper = [s + T if s >= 0 else -1 for s in starts0]
    _chained_verify(eng, res0, upper, [K] * B, rows=[True, True, False, False], sampling=_sampling())
    e = log["enqueue"][-1]
    assert e.kw["step_staged"] is True and e.kw["sample_plan"] is not None and "bucket" not in e.kw
    assert eng.chained_steps == 1 and eng.verify_bucket_steps == {B: 2}


def test_a_sampled_chained_verify_takes_the_bucket_of_its_rows_with_its_sampler_trace(chained):
    """With the buckets' sampler traces, a device-sampled one-enqueue predecessor over rows 0, 1 and 3 replays the
    B-user trace, and its chained verify over rows 0 and 1 takes bucket 2: it carries its chained plan (seeds from the
    chain program) and the bucket together, and the chain program, the step trace and the drafter still run at B
    users."""
    eng, log, state = chained
    B, T, K = eng.B, eng.T, eng.K
    eng.model._vfy_bucket_traces = {1: "v1", 2: "v2"}
    eng.model._vfy_bucket_sample_traces = {1: "s1", 2: "s2"}
    res0, tokens0, starts0, nv0 = _sampled_predecessor(eng, state)
    assert "bucket" not in log["enqueue"][-1].kw and log["enqueue"][-1].kw["sample_plan"] is not None
    state["ids"] += [torch.zeros(B, T, dtype=torch.int64)]
    state["draws"] += [torch.zeros(B * T, dtype=torch.int64)]
    upper = [s + T if s >= 0 else -1 for s in starts0]
    _chained_verify(eng, res0, upper, [K] * B, rows=[True, True, False, False], sampling=_sampling())
    e = log["enqueue"][-1]
    assert e.kw["step_staged"] is True and e.kw["bucket"] == 2
    assert e.kw["sample_plan"] is not None and e.kw["sample_plan"].seeds is None
    assert log["traces"][-3:] == ["chain", "step", "drafter"]
    assert eng.chained_steps == 1 and eng.verify_bucket_steps == {2: 1, B: 1}


def test_the_engine_prepares_the_buckets_below_its_width(bucket_engine):
    eng, _ = bucket_engine
    seen = {}
    eng.model.prepare_verify_trace = lambda pt, T, warm_positions, sample, buckets: seen.update(buckets=buckets, T=T)
    eng._prep_pt = torch.zeros(eng.B, eng.nb, dtype=torch.int32)
    eng._prepare_verify()
    assert seen == {"buckets": (1, 2, 4), "T": eng.T}


# ----------------------------------------------------------------------------------------- the model's replay
@pytest.fixture
def replays(monkeypatch):
    B, T = 4, 4
    m, copies = _bare_verify_model(B, T, monkeypatch)
    m._vfy_bucket_traces = {1: "v1", 2: "v2"}
    m._vfy_sampler = MagicMock(name="sampler")
    traces = []
    monkeypatch.setattr(ttnn, "execute_trace", lambda dev, trace, **kw: traces.append(trace))
    return m, copies, traces


def test_the_model_replays_the_buckets_trace(replays):
    m, copies, traces = replays
    tokens, pt = [[5, 6, 7, 8]] * 4, torch.ones(4, 4, dtype=torch.int32)
    m.verify_enqueue(tokens, [10, 0, 0, 0], [0] * 4, page_tables=pt, inactive=[False, True, True, True], bucket=1)
    m.verify_enqueue(tokens, [10, 20, 0, 0], [0] * 4, page_tables=pt, inactive=[False, False, True, True], bucket=2)
    m.verify_enqueue(tokens, [10, 20, 30, 0], [0] * 4, page_tables=pt, inactive=[False, False, False, True], bucket=4)
    m.verify_enqueue(tokens, [10, 20, 30, 0], [0] * 4, page_tables=pt, inactive=[False] * 4)
    assert traces == ["v1", "v2", "trace", "trace"]
    ids = m.verify_traced(tokens, [10, 0, 0, 0], [0] * 4, inactive=[False, True, True, True], bucket=1)[0]
    assert traces[-1] == "v1" and len(ids) == 16


def test_the_model_refuses_a_bucket_that_leaves_a_live_user_or_logits_out(replays, expect_error):
    """Every refusal comes before anything is staged, so the device keeps the last step's inputs."""
    m, copies, traces = replays
    tokens, pt = [[5, 6, 7, 8]] * 4, torch.ones(4, 4, dtype=torch.int32)
    plan = SimpleNamespace(any_sampled=True)
    for kw, msg in [
        (dict(inactive=[False, False, True, True], bucket=1), "leaves out a live user at or past row 1"),
        (dict(inactive=None, bucket=2), "leaves out a live user at or past row 2"),
        (dict(inactive=[False, True, True, True], bucket=8), r"no verify row bucket of 8 users \(captured: \[1, 2\]\)"),
        (dict(inactive=[False, True, True, True], bucket=1, sample_plan=plan), "has no sampler trace"),
        (dict(inactive=[False, True, True, True], bucket=1, read_logits=True), "makes no logits to read"),
    ]:
        with expect_error(ValueError, msg):
            m.verify_enqueue(tokens, [10, 20, 0, 0], [0] * 4, page_tables=pt, **kw)
    with expect_error(ValueError, "makes no logits to read"):
        m.verify_traced(tokens, [10, 0, 0, 0], [0] * 4, read_logits=True, inactive=[False, True, True, True], bucket=1)
    assert copies == [] and traces == []


def test_the_model_replays_the_buckets_sampler_trace_after_its_verify_trace(replays):
    """With device draws a bucket replays its verify trace, then its own sampler trace, with the plan staged into the
    B-user sampler; the B-user gather and sampler traces do not run. Without draws only the bucket's verify trace."""
    m, copies, traces = replays
    m._vfy_bucket_sample_traces = {1: "s1", 2: "s2"}
    m._vfy_gather_trace_id, m._vfy_sample_trace_id = "gather", "sample"
    tokens, pt = [[5, 6, 7, 8]] * 4, torch.ones(4, 4, dtype=torch.int32)
    plan = SimpleNamespace(any_sampled=True)
    m.verify_enqueue(tokens, [10, 20, 0, 0], [0] * 4, page_tables=pt, inactive=[False, False, True, True], bucket=2,
                     sample_plan=plan)  # fmt: skip
    assert traces == ["v2", "s2"] and m._vfy_sampler.stage.call_args.args == (plan,)
    m.verify_enqueue(tokens, [10, 0, 0, 0], [0] * 4, page_tables=pt, inactive=[False, True, True, True], bucket=1)
    m.verify_enqueue(tokens, [10, 20, 30, 0], [0] * 4, page_tables=pt, inactive=[False] * 4, sample_plan=plan)
    assert traces == ["v2", "s2", "v1", "trace", "sample"]


# ----------------------------------------------------------------------------------------- the bucket body
class FT:
    def __init__(self, shape, name, dtype="bf16"):
        self.shape, self.name, self.dtype = tuple(shape), name, dtype

    def memory_config(self):
        return "dram"

    def __repr__(self):
        return self.name


def _body_model(B=8, T=4, groups=None):
    m = object.__new__(Qwen36Model)
    m._vfy_B, m._vfy_T, m._vfy_nb = B, T, 16
    m._vfy_spec_groups = B if groups is None else groups
    m._vfy_att0 = SimpleNamespace(spec_sdpa_groups=lambda t, n: n * (m._vfy_spec_groups // B) or 1)
    m._vfy_kvpt_buf = FT((B * T, 16), "pt_rows")
    m._vfy_kvpt_users = FT((B, 16), "pt_users")
    m._vfy_kvpt1_buf = FT((m._vfy_spec_groups, 16), "pt_groups")
    m.device = "mesh"
    m._vfy_sampler = None
    return m


def test_the_bucket_body_runs_the_first_rows_and_hands_over_zero_padded_outputs(monkeypatch):
    """Bucket b runs the verify of b users on the first b*T rows of the per-row table, the first b of the per-user
    table and the first groups of the per-group table (a fresh slice each, freed after the layers), without logits;
    then the B-user ids output takes its lanes and the B-user feed output its rows followed by zeros, so the rows of
    users b .. B-1 that the drafter reads are zeros, not a stale step."""
    log = []
    monkeypatch.setattr(ttnn, "slice", lambda t, s, e: log.append(("slice", t.name, e)) or FT(e, f"{t.name}[:{e[0]}]"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("free", t.name)))
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: log.append(("copy", src.name, dst.name)))
    monkeypatch.setattr(
        ttnn,
        "concat",
        lambda ts, dim, memory_config: log.append(("concat", [t.name for t in ts], dim, memory_config))
        or FT((1, 1, 32, 640), "full"),
    )
    m = _body_model(groups=16)  # two SDPA groups per user (T = 8 would split so)
    m._vfy_bucket_zeros = {2: FT((1, 1, 24, 640), "zeros2")}

    def body(n_users, tables=None, logits=True):
        log.append(("body", n_users, [t.name for t in tables], logits))
        return None, FT((1, 1, 8, 640), "feed"), FT((1, 1, 1, 32), "ids"), None

    m._forward_verify_bucket_tp = body
    m._verify_bucket_body(2, FT((1, 1, 1, 32), "ids_out"), FT((1, 1, 32, 640), "rows_out"))
    assert log == [
        ("slice", "pt_rows", (8, 16)),
        ("slice", "pt_users", (2, 16)),
        ("slice", "pt_groups", (4, 16)),
        ("body", 2, ["pt_rows[:8]", "pt_users[:2]", "pt_groups[:4]"], False),
        ("free", "pt_rows[:8]"),
        ("free", "pt_users[:2]"),
        ("free", "pt_groups[:4]"),
        ("copy", "ids", "ids_out"),
        ("concat", ["feed", "zeros2"], 2, "dram"),
        ("copy", "full", "rows_out"),
        ("free", "ids"),
        ("free", "feed"),
        ("free", "full"),
    ]


def test_with_a_sampler_the_bucket_body_keeps_its_vocab_shard_and_its_sampler_body_draws_from_it(monkeypatch):
    """With a device sampler the bucket body runs the verify with logits and returns the bucket's vocab shard after the
    same hand-over; the bucket's sampler body runs the sampler over the shard's b*T rows, and the shard stays for the
    next replay."""
    log = []
    monkeypatch.setattr(ttnn, "slice", lambda t, s, e: FT(e, f"{t.name}[:{e[0]}]"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("free", t.name)))
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: log.append(("copy", src.name, dst.name)))
    monkeypatch.setattr(ttnn, "concat", lambda ts, dim, memory_config: FT((1, 1, 32, 640), "full"))
    m = _body_model()
    m._vfy_bucket_zeros = {2: FT((1, 1, 24, 640), "zeros2")}
    m._vfy_sampler = SimpleNamespace(body=lambda logits: log.append(("sampler", logits.name)))

    def body(n_users, tables=None, logits=True):
        log.append(("body", n_users, logits))
        return FT((1, 1, 8, 31040), "shard8"), FT((1, 1, 8, 640), "feed"), FT((1, 1, 1, 32), "ids"), None

    m._forward_verify_bucket_tp = body
    shard = m._verify_bucket_body(2, FT((1, 1, 1, 32), "ids_out"), FT((1, 1, 32, 640), "rows_out"))
    assert shard.name == "shard8" and ("body", 2, True) in log and ("free", "shard8") not in log
    assert [e for e in log if e[0] == "copy"] == [("copy", "ids", "ids_out"), ("copy", "full", "rows_out")]
    log.clear()
    m._bucket_sample_body(shard)
    assert log == [("sampler", "shard8")]


def test_the_bucket_body_never_slices_a_table_whole(monkeypatch):
    """A table whose first rows are all of it (one SDPA group in all, when the fused spec SDPA is off) is read as is:
    a full-span slice would alias the persistent table, and freeing it would free the table."""
    log = []
    monkeypatch.setattr(ttnn, "slice", lambda t, s, e: log.append(("slice", t.name)) or FT(e, "s"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("free", t.name)))
    monkeypatch.setattr(ttnn, "copy", lambda *a: None)
    monkeypatch.setattr(ttnn, "concat", lambda ts, dim, memory_config: FT((1, 1, 32, 640), "full"))
    m = _body_model(groups=1)
    m._vfy_att0 = SimpleNamespace(spec_sdpa_groups=lambda t, n: 1)
    m._vfy_bucket_zeros = {1: FT((1, 1, 28, 640), "zeros1")}
    seen = []
    m._forward_verify_bucket_tp = lambda n_users, tables, logits: seen.append(tables) or (
        None,
        FT((1,), "f"),
        FT((1,), "i"),
        None,
    )
    m._verify_bucket_body(1, FT((1,), "ids_out"), FT((1, 1, 32, 640), "rows_out"))
    assert [e for e in log if e[0] == "slice"] == [("slice", "pt_rows"), ("slice", "pt_users")]
    assert seen[0][2] is m._vfy_kvpt1_buf and ("free", "pt_groups") not in log


def test_the_verify_body_takes_the_given_tables_and_stops_at_the_shard_ids(monkeypatch, expect_error):
    """tables replace the persistent page tables in every attention layer; logits=False takes the ids from the fused
    per-shard argmax with no vocab all-gather and no untilize. The full-width verify with those ids ends at the vocab
    shard as well, and keeps it for the logits traces."""
    log = []
    for name in ("slice", "typecast", "reshape", "to_memory_config"):
        monkeypatch.setattr(ttnn, name, lambda t, *a, **kw: FT((4,), "t"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "untilize", lambda t, **kw: log.append("untilize") or "u")

    def layer(full):
        return SimpleNamespace(is_full_attention=full, forward=lambda x, **kw: log.append(("layer", full, kw)) or x)

    m = object.__new__(Qwen36Model)
    m._vfy_T, m._vfy_step_buf = 4, "step"
    m._vfy_kvpt_buf, m._vfy_kvpt_users, m._vfy_kvpt1_buf = "rows8", "users8", "groups8"
    m._vfy_state_idx = m._vfy_conv_sel = None
    m.args = SimpleNamespace(gdn_chunk_size=128)
    m.rope_lookup = lambda rows, n, row_view=False: (FT((1, 1, n, 64), "cos"), FT((1, 1, n, 64), "sin"))
    m.embd = lambda tok: FT((1, 1, 4, 64), "x")
    m._residual_ar = lambda rows: None
    m.layers = [layer(True), layer(False)]
    tails = []

    def tail(x, shard_ids, logits=True):
        tails.append(logits)
        return (None if not logits else "logits"), "feed", ("ids" if tails[-1] is not None else None)

    m._spec_tail_ids = tail
    out = m._forward_verify_bucket_tp(n_users=1, tables=("rows1", "users1", "groups1"), logits=False)
    assert out == (None, "feed", "ids", None) and tails == [False] and "untilize" not in log
    attn = next(kw for e, full, kw in (x for x in log if x[0] == "layer") if full)
    assert (attn["page_table"], attn["spec_user_page_table"], attn["spec_page_table"]) == ("rows1", "users1", "groups1")
    m._spec_tail_ids = lambda x, shard_ids, logits=True: (None, "feed", None)
    with expect_error(RuntimeError, "needs the fused per-shard argmax"):
        m._forward_verify_bucket_tp(n_users=1, tables=("r", "u", "g"), logits=False)
    log.clear()
    m._spec_tail_ids = lambda x, shard_ids: ("shard", "feed", "ids")
    assert m._forward_verify_bucket_tp(n_users=2) == ("shard", "feed", "ids", None) and "untilize" not in log
    attn = next(kw for e, full, kw in (x for x in log if x[0] == "layer") if full)
    assert (attn["page_table"], attn["spec_user_page_table"], attn["spec_page_table"]) == ("rows8", "users8", "groups8")


def test_the_lm_head_without_logits_frees_the_shard_once_it_has_the_ids(monkeypatch):
    freed, gathered = [], []
    monkeypatch.setattr(ttnn, "linear", lambda x, w: "shard")
    monkeypatch.setattr(ttnn, "deallocate", lambda t: freed.append(t))
    m = object.__new__(Qwen36Model)
    m._lmhead_vocab_sharded, m.lm_head_weight = True, "w"
    m._gather_vocab = lambda shard: gathered.append(shard) or "logits"
    assert m._lm_head_ids("normed", lambda s: "ids", logits=False) == (None, "ids")
    assert freed == ["shard"] and gathered == []
    assert m._lm_head_ids("normed", lambda s: None, logits=False) == ("logits", None), "no shard ids: gather"
    assert m._lm_head_ids("normed", lambda s: "ids") == ("shard", "ids"), "with logits the caller keeps the shard"
    assert gathered == ["shard"] and freed == ["shard"]


# ----------------------------------------------------------------------------------------- GDN rows of a bucket
def _gdn_layer(T, rows):
    from models.demos.blackhole.qwen36.tt.gdn import tp

    layer = SimpleNamespace(
        Nv=6, Nk=2, Dk=128, Dv=128, K=4, qkvz_dim_tp=64, scale=1.0, mesh=None, tt_ccl=None, cfg=None,
        tw=dict.fromkeys(["qkvz", "conv_taps_rows", "dt_bias_fp32", "neg_exp_A_fp32", "norm_w_1d", "out"]),
        args=SimpleNamespace(gdn_qkvz_decode_1d_progcfg=None, ccl_topology=lambda: None),
        state_blocks=SimpleNamespace(
            verify=SimpleNamespace(T=T, rows=rows, ctrl="ctrl", idx="idx"), rec_k="rk", rec_v="rv", hist="h"
        ),
        _row_proj=lambda gated, w: SimpleNamespace(shape=(1, 32, 64)),
        _spec_step_options=lambda t: {},
    )  # fmt: skip
    return tp, layer


@pytest.mark.parametrize("users, max_users", [(1, 8), (2, 8), (4, 8), (8, None)])
def test_the_gdn_verify_of_a_bucket_reads_its_rows_of_the_full_width_ctrl_page(monkeypatch, users, max_users):
    """The verify tables hold one row per decode row (bmax): a verify of fewer users passes gdn_spec_step the table's
    row count as max_users, so the op reads each user's start words where verify_tables put them; the full-width
    verify passes none (the program it always ran)."""
    calls = []
    monkeypatch.setattr(ttnn.experimental.gdn, "gdn_spec_step", lambda *a, **k: calls.append((a, k)) or "gated")
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: SimpleNamespace(shape=shape))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    T = 4
    tp, layer = _gdn_layer(T, rows=8)
    monkeypatch.setattr(tp.tpc, "matmul_1d_decode", lambda *a, **k: SimpleNamespace(shape=(1, 1, 32, 64)))
    monkeypatch.setattr(tp, "tt_all_reduce", lambda partial, *a, **k: partial)
    tp.TPGatedDeltaNet._forward_verify_blocks(layer, SimpleNamespace(shape=(1, 32, 64)), users * T, users)
    ((args, kw),) = calls
    assert args[13] == T and args[14] == users and kw.get("max_users") == max_users


def test_a_gdn_verify_of_more_users_than_its_tables_raises(monkeypatch, expect_error):
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: SimpleNamespace(shape=shape))
    tp, layer = _gdn_layer(4, rows=2)
    with expect_error(AssertionError, "4 verify users, the verify tables hold 2 rows"):
        tp.TPGatedDeltaNet._forward_verify_blocks(layer, SimpleNamespace(shape=(1, 16, 64)), 16, 4)


def test_the_ctrl_page_holds_each_users_start_words_at_its_full_width_offsets():
    """verify_tables lays the ctrl page out for all bmax rows: user u's start page of head h at word 1 + bmax + u*Nv + h
    and its start history page at 1 + bmax + bmax*Nv + u*Nv + h. gdn_spec_step with max_users = bmax reads exactly
    those for its first b users; the b-user offsets (max_users unset) would read another row's or a header word."""
    lay = _layout()
    nv, bmax, b, T = lay.dims.nv, 8, 2, 4
    starts, windows = [5, 9], torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    ctrl, _ = lay.verify_tables(starts, windows, NUM_BLOCKS, bmax)
    words = ctrl[0].tolist()
    for u, block in enumerate(starts):
        rec, hist = _pages(lay, block)
        for h in range(nv):
            assert words[1 + bmax + u * nv + h] == rec[h]
            assert words[1 + bmax + bmax * nv + u * nv + h] == hist[h]
    narrow = [words[1 + b + u * nv + h] for u in range(b) for h in range(nv)]
    assert narrow != [_pages(lay, starts[u])[0][h] for u in range(b) for h in range(nv)]


# ----------------------------------------------------------------------------------------- warmup, capture, release
def test_warmup_runs_every_bucket_body_into_the_warm_outputs(monkeypatch, expect_error):
    zeros, bodies = [], []
    monkeypatch.setattr(
        ttnn, "zeros", lambda shape, dtype, layout, device, memory_config: zeros.append((tuple(shape), dtype)) or "z"
    )
    m = _body_model(B=8, T=4)
    m._verify_bucket_body = lambda b, ids, rows: bodies.append((b, ids.name, rows.name))
    ids, rows = FT((1, 1, 1, 32), "ids", "u32"), FT((1, 1, 32, 640), "rows")
    assert m._warm_verify_buckets((4, 1, 2), ids, rows) == (1, 2, 4)
    assert zeros == [((1, 1, 28, 640), "bf16"), ((1, 1, 24, 640), "bf16"), ((1, 1, 16, 640), "bf16")]
    assert bodies == [(1, "ids", "rows"), (2, "ids", "rows"), (4, "ids", "rows")]
    assert set(m._vfy_bucket_zeros) == {1, 2, 4}
    for bad in ((8,), (0,)):
        with expect_error(ValueError, "are not below the decode width 8"):
            m._warm_verify_buckets(bad, ids, rows)


def test_with_a_sampler_warmup_runs_every_bucket_sampler_body(monkeypatch):
    """Before any capture each bucket's sampler body runs once over the warm bucket body's shard, which is then
    freed."""
    events = []
    monkeypatch.setattr(ttnn, "zeros", lambda shape, dtype, layout, device, memory_config: "z")
    monkeypatch.setattr(ttnn, "deallocate", lambda t: events.append(("free", t)))
    m = _body_model(B=8, T=4)
    m._vfy_sampler = SimpleNamespace()
    m._verify_bucket_body = lambda b, ids, rows: events.append(("bucket", b)) or f"shard{b}"
    m._bucket_sample_body = lambda shard: events.append(("sample", shard))
    m._warm_verify_buckets((1, 2, 4), FT((1, 1, 1, 32), "ids", "u32"), FT((1, 1, 32, 640), "rows"))
    assert events == [
        ("bucket", 1), ("sample", "shard1"), ("free", "shard1"),
        ("bucket", 2), ("sample", "shard2"), ("free", "shard2"),
        ("bucket", 4), ("sample", "shard4"), ("free", "shard4"),
    ]  # fmt: skip


@pytest.mark.parametrize("lanes", [True, False])
def test_the_warm_pass_prepares_buckets_only_with_the_shard_argmax_lanes(monkeypatch, lanes):
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    m = _body_model(B=8, T=4)
    m._vfy_sampler = None
    ids = FT((1, 1, 1, 32) if lanes else (1, 1, 32), "ids", ttnn.uint32)
    ids.layout = ttnn.ROW_MAJOR_LAYOUT
    m._forward_verify_bucket_tp = lambda n_users: ("logits", FT((1, 1, 32, 640), "rows"), ids, "u")
    warmed = []
    m._warm_verify_buckets = lambda buckets, i, r: warmed.append((buckets, i, r)) or tuple(sorted(buckets))
    m._warm_verify_body(8, (1, 2, 4))
    assert m._vfy_ids_lanes is lanes and m._vfy_buckets == ((1, 2, 4) if lanes else ())
    assert len(warmed) == int(lanes)


def test_capture_records_one_trace_per_bucket_after_the_full_width_trace_and_release_frees_them(monkeypatch):
    events, n = [], iter(range(100))
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda dev, cq_id: events.append("begin") or f"t{next(n)}")
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda dev, tid, cq_id: events.append(("end", tid)))
    monkeypatch.setattr(ttnn, "release_trace", lambda dev, tid: events.append(("release", tid)))
    m = _body_model(B=8, T=4)
    m.mesh_device, m._vfy_sampler, m._vfy_kv_grouped, m._vfy_prepared_key = "mesh", None, True, "key"
    m._vfy_trace_id = None
    m._vfy_buckets = (1, 2, 4)
    m._vfy_att0 = SimpleNamespace(
        spec_sdpa_enabled=lambda T: True, _spec_pack_heads=lambda tiles: 3, _forward_verify_ok=lambda *a: True
    )
    m._log_trace_region = lambda label: events.append(("region", label))
    m._forward_verify_bucket_tp = lambda n_users: events.append(("verify", n_users)) or ("lg", "rows", "ids", "rm")
    m._verify_bucket_body = lambda b, ids, rows: events.append(("bucket", b, ids, rows))
    m.record_verify_trace()
    assert events == [
        "begin", ("verify", 8), ("end", "t0"), ("region", "the verify trace"),
        "begin", ("bucket", 1, "ids", "rows"), ("end", "t1"),
        "begin", ("bucket", 2, "ids", "rows"), ("end", "t2"),
        "begin", ("bucket", 4, "ids", "rows"), ("end", "t3"),
        ("region", "the verify row bucket traces"),
    ]  # fmt: skip
    assert m._vfy_bucket_traces == {1: "t1", 2: "t2", 4: "t3"} and m._vfy_capture_count == 1
    events.clear()
    m.release_verify_trace()
    assert events == [("release", "t0"), ("release", "t1"), ("release", "t2"), ("release", "t3")]
    assert m._vfy_trace_id is None and m._vfy_bucket_traces == {}


def test_with_a_sampler_capture_records_each_buckets_sampler_trace_after_its_verify_trace(monkeypatch):
    """Each bucket's sampler trace is captured right after the bucket's verify trace, over the vocab shard that verify
    trace keeps allocated; release frees both kinds of traces and drops the shards."""
    events, n = [], iter(range(100))
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda dev, cq_id: events.append("begin") or f"t{next(n)}")
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda dev, tid, cq_id: events.append(("end", tid)))
    monkeypatch.setattr(ttnn, "release_trace", lambda dev, tid: events.append(("release", tid)))
    m = _body_model(B=8, T=4)
    m.mesh_device, m._vfy_kv_grouped, m._vfy_prepared_key, m._vfy_trace_id = "mesh", True, "key", None
    m._vfy_sampler = SimpleNamespace(body=lambda logits: events.append(("sampler", logits)))
    m._vfy_buckets = (1, 2)
    m._vfy_att0 = SimpleNamespace(
        spec_sdpa_enabled=lambda T: True, _spec_pack_heads=lambda tiles: 3, _forward_verify_ok=lambda *a: True
    )
    m._log_trace_region = lambda label: events.append(("region", label))
    m._forward_verify_bucket_tp = lambda n_users: ("lg", "rows", "ids", "rm")
    m._verify_bucket_body = lambda b, ids, rows: events.append(("bucket", b)) or f"shard{b}"
    m._bucket_sample_body = lambda shard: events.append(("sample", shard))
    m.record_verify_trace()
    assert events == [
        "begin", ("end", "t0"), "begin", ("sampler", "lg"), ("end", "t1"), ("region", "the verify trace"),
        "begin", ("bucket", 1), ("end", "t2"), "begin", ("sample", "shard1"), ("end", "t3"),
        "begin", ("bucket", 2), ("end", "t4"), "begin", ("sample", "shard2"), ("end", "t5"),
        ("region", "the verify row bucket traces and their sampler traces"),
    ]  # fmt: skip
    assert m._vfy_bucket_traces == {1: "t2", 2: "t4"} and m._vfy_bucket_sample_traces == {1: "t3", 2: "t5"}
    assert m._vfy_bucket_shards == {1: "shard1", 2: "shard2"}
    events.clear()
    m.release_verify_trace()
    assert events == [("release", t) for t in ("t0", "t1", "t2", "t4", "t3", "t5")]
    assert m._vfy_bucket_traces == m._vfy_bucket_sample_traces == m._vfy_bucket_shards == {}


# ----------------------------------------------------------------------------------------- SDPA core split, prepare
def test_a_bucket_keeps_the_full_width_sdpa_core_split(monkeypatch):
    """On the 110-core grid the fused spec SDPA of 8 users x T = 4 gives each user's group 13 cores; alone, 1, 2 and 4
    users would take 64, 55 and 27, which splits and reduces their KV range in another order. With the decode width
    set (spec_sdpa_split_users), every user count keeps 13 cores per group and only the group count follows the users;
    at T = 8 (two groups per user, 4 users) the same holds with the 4-user split."""
    from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod

    monkeypatch.setattr(ttnn, "SDPAProgramConfig", lambda **kw: SimpleNamespace(**kw))
    a = object.__new__(tp_mod.TPAttention)
    a.mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
    a._spec_sdpa_cfg_cache = {}
    cores = lambda T, n: a._spec_sdpa_plan(T, n)[0].max_cores_per_head_batch
    assert {n: cores(4, n) for n in (1, 2, 4, 8)} == {1: 64, 2: 55, 4: 27, 8: 13}
    a.spec_sdpa_split_users = 8
    for n in (1, 2, 4, 8):
        assert cores(4, n) == 13 and a._spec_sdpa_plan(4, n)[1:] == (n, 4)
    a.spec_sdpa_split_users = 4
    assert cores(8, 1) == cores(8, 4) == 13 and a._spec_sdpa_plan(8, 1)[1:] == (2, 4)


def test_prepare_sets_the_sdpa_split_and_warms_the_buckets(monkeypatch, expect_error):
    """prepare_verify_trace gives every attention layer of the verify the decode width as its SDPA split and warms the
    verify with its buckets, once per (B, T, nb, sample, buckets); the model-owned GDN state takes no bucket."""
    from models.demos.blackhole.qwen36.tt import model as model_mod

    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda dev: None)
    monkeypatch.setattr(model_mod, "get_block_size", lambda caches: 64)
    B, T = 8, 4
    m = _rope_model()
    m.num_devices, m._paged_kv_caches, m.args.max_batch_size = 8, "kv", B
    m.rope_rm_tables = lambda reach=0: None
    att = [SimpleNamespace(spec_sdpa_groups=lambda t, n: n) for _ in range(2)]
    m.layers = [
        SimpleNamespace(is_full_attention=True, attention=att[0]),
        SimpleNamespace(is_full_attention=False, attention="gdn"),
        SimpleNamespace(is_full_attention=True, attention=att[1]),
    ]
    m.gdn_blocks = SimpleNamespace(verify_tokens=T, clear_verify_blocks=lambda: None)
    warmed = []
    m._warm_verify_body = lambda width, buckets=(): warmed.append((width, buckets))
    pt = torch.arange(B * 4, dtype=torch.int32).reshape(B, 4) + 1
    m.prepare_verify_trace(pt, T, [0] * B, buckets=(4, 1, 2))
    assert [a.spec_sdpa_split_users for a in att] == [B, B]
    assert warmed == [(B, (4, 1, 2))] and m._vfy_prepared_key == (B, T, 4, False, (1, 2, 4))
    m.prepare_verify_trace(pt, T, [0] * B, buckets=(1, 2, 4))
    assert len(warmed) == 1, "the same key prepares nothing again"
    m.gdn_blocks = None
    with expect_error(ValueError, "verify row buckets need the GDN state in vLLM KV cache blocks"):
        m.prepare_verify_trace(pt, T, [0] * B, buckets=(1,))
