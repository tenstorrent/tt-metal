# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the MTP drafter row buckets of MTPBlockEngine.

The drafter trace (MTPSpecEngine._drafter_body) gathers each user's anchor hidden from the verify's feed rows, writes the
drafter KV of the committed rows (the batched reseed) and runs K chained draft steps, over all B users. A drafter replay
whose rows that draft or keep their drafter KV (spec_engine.drafter_rows) all sit below a bucket b of
spec_engine.drafter_buckets(B) (1, 2, 4 at B = 8) replays the drafter trace of users 0 .. b-1 instead, which hands its
drafts to the B-user outputs with zero ids past b. Covered here:

* the bucket of a replay is the smallest that holds its highest row that drafts or keeps its drafter KV, from that
  replay's own propose decisions; without bucket traces every replay takes the B-user trace;
* a one-enqueue step, a chained step and a host-staged propose replay the bucket's drafter trace, changing bucket from
  one step to the next;
* the bucket body reads the first b users of the pack, the feed rows and the tables, writes the drafter KV of b * T
  reseed rows, drafts b rows, and hands every step's ids over in place into the B-user outputs with zero rows past b;
  the B-user body is unchanged;
* warmup allocates each bucket's buffers and runs its body into the warm outputs before any capture, capture records
  one trace per bucket right after the B-user drafter's and before the step trace, and refuses a capture that moved
  a B-user output;
* the drafter's per-shard argmax keeps its constants per row count.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_accept_host import (  # noqa: F401 (step is a fixture)
    _verify,
    _walk_and_propose,
    step,
)
from models.demos.blackhole.qwen36.tests.test_spec_chain_host import (  # noqa: F401 (chained is a fixture)
    _chained_verify,
    _host_verify,
    _propose,
    _walk,
    chained,
)
from models.demos.blackhole.qwen36.tt import spec_engine
from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP
from models.demos.blackhole.qwen36.tt.spec_engine import MTPBlockEngine, MTPSpecEngine
from models.demos.blackhole.qwen36.tt.spec_step import (
    DRAFT_PENDING,
    draft_anchor,
    draft_reseed_positions,
    draft_reseed_rope,
    draft_reseed_tokens,
    draft_step_rope,
    draft_step_row,
)


# ----------------------------------------------------------------------------------------- bucket choice
def test_the_bucket_holds_every_row_that_drafts_or_keeps_its_drafter_kv():
    assert spec_engine.drafter_buckets(8) == (1, 2, 4)
    assert spec_engine.drafter_buckets(4) == (1, 2) and spec_engine.drafter_buckets(1) == ()
    assert spec_engine.drafter_rows([False, True, True, False], [False, True, False, False]) == [
        True,
        True,
        False,
        True,
    ]
    B = 8
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e._dr_bucket_traces = B, {1: "d1", 2: "d2", 4: "d4"}
    off = lambda *rows: [u not in rows for u in range(B)]  # declined: every row but ``rows``
    none = [False] * B
    cases = [
        (off(0), none, 1),
        (off(1), none, 2),
        (off(0, 3), none, 4),
        (off(0, 4), none, None),
        (off(0), [u == 2 for u in range(B)], 4),  # row 2 is declined but keeps its drafter KV
        (off(0), [u == 7 for u in range(B)], None),
        (off(), none, 1),  # every row declined: the smallest bucket (the propose replays nothing then)
    ]
    for declined, keep, want in cases:
        assert MTPSpecEngine._drafter_bucket(e, declined, keep) == want, (declined, keep)
    assert e.drafter_bucket_steps == {1: 2, 2: 1, 4: 2, 8: 2}
    e._dr_bucket_traces = {}
    assert MTPSpecEngine._drafter_bucket(e, off(0), none) is None, "no bucket trace: the B-user trace"
    e._dr_trace, e._dr_bucket_traces = "d", {2: "d2"}
    assert MTPSpecEngine._dr_trace_for(e, None) == "d" and MTPSpecEngine._dr_trace_for(e, 2) == "d2"


def test_only_the_served_engine_builds_drafter_buckets():
    """MTPBlockEngine (Qwen36MTPContractForCausalLM) builds the buckets below its width; the base engine none."""
    e = MTPBlockEngine.__new__(MTPBlockEngine)
    e.B = 8
    assert MTPBlockEngine.drafter_bucket_sizes(e) == (1, 2, 4)
    e.B = 1
    assert MTPBlockEngine.drafter_bucket_sizes(e) == ()
    assert MTPSpecEngine.drafter_bucket_sizes(MTPSpecEngine.__new__(MTPSpecEngine)) == ()


# ----------------------------------------------------------------------------------------- replays
def test_a_one_enqueue_step_replays_the_drafter_of_its_own_rows(step):
    """B = 4 (buckets 1 and 2). Each one-enqueue step replays the step trace and the drafter trace of the smallest
    bucket that holds its rows that draft or keep their drafter KV, up and down from one step to the next; the propose
    returns the device's drafts with the rows past the bucket declined."""
    eng, log, state = step
    eng._dr_bucket_traces = {1: "drafter1", 2: "drafter2"}
    B, T = eng.B, eng.T
    tokens = torch.zeros(B, T, dtype=torch.int64)
    pattern = [
        ([100, -1, -1, -1], set(), "drafter1"),
        ([100, 200, -1, -1], set(), "drafter2"),
        ([-1, 200, -1, -1], set(), "drafter2"),
        ([100, -1, 300, -1], set(), "drafter"),
        ([100, -1, -1, -1], set(), "drafter1"),
        ([100, 200, -1, -1], {1}, "drafter2"),  # row 1's drafts are withheld, but it keeps its drafter KV
    ]
    for starts, no_draft, want in pattern:
        log["traces"].clear()
        eng._no_draft = set(no_draft)
        state["ids"] = torch.zeros(B, T, dtype=torch.int64)
        nv = [3 if s >= 0 else 0 for s in starts]
        res = _verify(eng, tokens, starts, nv)
        assert log["traces"] == ["step", want], (starts, log["traces"])
        out, _ = _walk_and_propose(eng, res, tokens, starts, nv)
        live = [s >= 0 and u not in no_draft for u, s in enumerate(starts)]
        assert out.num_valid.tolist() == [eng.K if a else 0 for a in live]
        assert all(out.draft_token_ids[u].eq(0).all() for u in range(B) if not live[u])
    assert eng.drafter_bucket_steps == {1: 2, 2: 3, 4: 1}
    assert eng.one_enqueue_host_proposes == 0


def test_a_host_staged_propose_replays_the_drafter_of_its_own_rows(step):
    """A propose that decides differently from its verify's plan stages the drafter from the host: it replays the
    drafter trace of the bucket of its own decisions."""
    eng, log, state = step
    eng._dr_bucket_traces = {1: "drafter1", 2: "drafter2"}
    B, T = eng.B, eng.T
    tokens = torch.zeros(B, T, dtype=torch.int64)
    starts, nv = [100, 200, -1, -1], [3, 3, 0, 0]
    state["ids"] = torch.zeros(B, T, dtype=torch.int64)
    eng._no_draft = {1}
    res = _verify(eng, tokens, starts, nv)
    assert log["traces"] == ["step", "drafter2"]
    eng._no_draft.clear()  # a release re-offers row 1: the propose decides differently
    _walk_and_propose(eng, res, tokens, starts, nv)
    assert log["traces"] == ["step", "drafter2", "drafter2"] and eng.one_enqueue_host_proposes == 1


def test_a_chained_step_replays_the_drafter_of_its_own_rows(chained):
    """A chained verify of fewer rows than its predecessor replays the drafter of its own smaller bucket; the chain
    stays on the device and both proposes check out."""
    eng, log, state = chained
    eng._dr_bucket_traces = {1: "drafter1", 2: "drafter2"}
    B, T, K = eng.B, eng.T, eng.K
    ids0 = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14], [0, 0, 0, 0], [0, 0, 0, 0]])
    tokens0 = torch.tensor([[1, 7, 8, 5], [2, 11, 12, 13], [0, 0, 0, 0], [0, 0, 0, 0]])
    starts0, nv0 = [100, 200, -1, -1], [3, 3, 0, 0]
    state["ids"] = [ids0, torch.zeros(B, T, dtype=torch.int64)]
    res0 = _host_verify(eng, tokens0, starts0, nv0)
    upper = [starts0[0] + T, -1, -1, -1]  # row 1 leaves the batch after the predecessor
    _chained_verify(eng, res0, upper, [K, 0, 0, 0])
    assert log["traces"] == ["step", "drafter2", "chain", "step", "drafter1"]
    assert eng.chained_steps == 1 and eng.materialized_steps == 0
    committed0, counts0 = _walk(res0, tokens0, nv0)
    out0 = _propose(eng, starts0, committed0, counts0)
    assert out0.num_valid.tolist() == [K, K, 0, 0]


# ----------------------------------------------------------------------------------------- the trace bodies
def _body_engine(monkeypatch, B=8, K=3, dim=64):
    """A bare MTPSpecEngine whose ttnn calls and drafter forwards are recorded in ``calls``."""
    calls = []
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.nb = B, K, K + 1, 16
    e._dr_pack, e._dr_iota, e._dr_rs_pt, e._mtp_pt, e._hp_buf = "pack", "iota", "rs_pt", "mtp_pt", "hp"
    e._hp_bufs = {b: f"hp{b}" for b in (1, 2, 4)}
    e._dr_bucket_zeros = {b: f"zeros{b}" for b in (1, 2, 4)}
    e._anchor_cc = "cc"
    e.model = SimpleNamespace(rope_lookup=lambda rows, n: (f"cos{rows[1]}", f"sin{rows[1]}"))

    def forward_decode(h, tok, pos, cos, sin, pt, **kw):
        calls.append(("forward", h, tok, pos, cos, sin, pt, kw))
        return ("logits", f"h{len(calls)}") if kw.get("need_logits", True) else (None, None)

    e.mtp = SimpleNamespace(forward_decode=forward_decode)

    def op(name):
        def f(*a, **kw):
            calls.append((name, a, kw))
            return (name, len(calls))

        return f

    for name in ("typecast", "to_layout", "transpose", "add", "eq", "matmul", "copy", "concat", "deallocate"):
        monkeypatch.setattr(ttnn, name, op(name))
    monkeypatch.setattr(ttnn, "slice", lambda t, s, e_: calls.append(("slice", t, s, e_)) or ("slice", t, e_))
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: calls.append(("reshape", t, shape)) or ("reshape", shape))
    monkeypatch.setattr(spec_engine, "draft_argmax", lambda mtp, logits: ("ids", len(calls)))
    feed = SimpleNamespace(shape=(1, 1, B * (K + 1), dim), name="feed")
    return e, calls, feed


def test_the_bucket_body_runs_the_first_b_users_and_hands_its_ids_over_in_place(monkeypatch, expect_error):
    B, K, b = 8, 3, 2
    T = K + 1
    e, calls, feed = _body_engine(monkeypatch, B, K)
    dst = [f"out{k}" for k in range(K)]
    assert MTPSpecEngine._drafter_body(e, feed, b, dst) is dst
    slices = [(c[1], c[3]) for c in calls if c[0] == "slice"]
    pack = sorted(end[1] for t, end in slices if t == "pack")
    want = [b] + [b * T] * 3 + [b] + [b] * K + [b] * K  # anchor, reseed rope/tokens/positions, pending, steps
    assert pack == sorted(want), pack
    assert (feed, (1, 1, b * T, 64)) in slices, "the reseed reads the first b users' feed rows"
    assert ("rs_pt", (b * T, e.nb)) in slices and ("mtp_pt", (b, e.nb)) in slices
    onehot = next(c for c in calls if c[0] == "reshape" and c[2] == (1, 1, b, B * T))
    mm = next(c for c in calls if c[0] == "matmul")
    assert mm[1] == (("reshape", onehot[2]), feed), "the anchor one-hot [b, B*T] gathers from every feed row"
    assert next(c for c in calls if c[0] == "copy")[1][1] == f"hp{b}"
    fwd = [c for c in calls if c[0] == "forward"]
    assert fwd[0][1] == ("slice", feed, (1, 1, b * T, 64)) and fwd[0][6] == ("slice", "rs_pt", (b * T, e.nb))
    assert fwd[0][7] == {"need_logits": False, "alias_kv_write": True, "spec_n_users": b}
    assert all(f[6] == ("slice", "mtp_pt", (b, e.nb)) for f in fwd[1:]) and len(fwd) == 1 + K
    assert fwd[1][1] == f"hp{b}"
    toks = [c for c in calls if c[0] == "reshape" and c[2] == (b, 1)]
    assert len(toks) == K, "each draft step's ids feed the next step as [b, 1]"
    cats = [c for c in calls if c[0] == "concat"]
    copies = [c for c in calls if c[0] == "copy"][1:]
    assert [c[1][0][1] for c in cats] == [f"zeros{b}"] * K and all(c[2] == {"dim": 2} for c in cats)
    assert [c[1][1] for c in copies] == dst, "each step's ids are copied into its B-user output"
    with expect_error(ValueError, "hands its ids over"):
        MTPSpecEngine._drafter_body(e, feed, b)


def test_the_full_width_body_is_unchanged(monkeypatch):
    B, K = 8, 3
    T = K + 1
    e, calls, feed = _body_engine(monkeypatch, B, K)
    outs = MTPSpecEngine._drafter_body(e, feed)
    assert len(outs) == K and all(o[0] == "ids" for o in outs)
    assert not any(c[0] == "slice" and c[1] != "pack" for c in calls), "no feed or table slice"
    assert not any(c[0] in ("concat",) for c in calls)
    fwd = [c for c in calls if c[0] == "forward"]
    assert fwd[0][1] is feed and fwd[0][6] == "rs_pt" and fwd[0][7]["spec_n_users"] == B
    assert all(f[6] == "mtp_pt" for f in fwd[1:]) and fwd[1][1] == "hp"
    pack = sorted(c[3][1] for c in calls if c[0] == "slice")
    assert pack == sorted([B] + [B * T] * 3 + [B] + [B] * K + [B] * K)
    assert {(DRAFT_PENDING, B), (draft_anchor(K), B), (draft_reseed_tokens(K), B * T)} <= {
        (c[2][0], c[3][1]) for c in calls if c[0] == "slice"
    }
    assert {(draft_reseed_positions(K), B * T), (draft_reseed_rope(K), B * T), (draft_step_rope(K, 0), B)} <= {
        (c[2][0], c[3][1]) for c in calls if c[0] == "slice"
    }
    assert (draft_step_row(K - 1), B) in {(c[2][0], c[3][1]) for c in calls if c[0] == "slice"}


def test_warmup_allocates_and_runs_every_bucket_before_capture(monkeypatch, expect_error):
    B, K = 8, 3
    e = MTPBlockEngine.__new__(MTPBlockEngine)
    e.B, e.K, e.T, e._dim_frac, e._feed_dtype, e.mesh = B, K, K + 1, 640, "bf16", "mesh"
    e._hp_bufs, e._dr_bucket_zeros = {}, {}
    zeros, staged, bodies = [], [], []
    monkeypatch.setattr(ttnn, "zeros", lambda shape, **kw: zeros.append(shape) or f"hp{shape[2]}")
    e._alloc_host_replicated = lambda host, dtype, layout: staged.append(tuple(host.shape)) or f"z{host.shape[2]}"
    e._drafter_body = lambda feed, b=None, dst=None: bodies.append((feed, b, dst))
    assert MTPSpecEngine._warm_drafter_buckets(e, "feed", "outs") == (1, 2, 4)
    assert zeros == [[1, 1, b, 640] for b in (1, 2, 4)]
    assert staged == [(1, 1, B - b, 1) for b in (1, 2, 4)]
    assert e._hp_bufs == {1: "hp1", 2: "hp2", 4: "hp4"} and e._dr_bucket_zeros == {1: "z7", 2: "z6", 4: "z4"}
    assert bodies == [("feed", b, "outs") for b in (1, 2, 4)], "each bucket body hands its ids into the warm outputs"
    e.drafter_bucket_sizes = lambda: (8,)
    with expect_error(ValueError, "not below the decode width"):
        MTPSpecEngine._warm_drafter_buckets(e, "feed", "outs")


def test_capture_records_each_bucket_after_the_full_width_drafter(monkeypatch, expect_error):
    """capture_decode: the verify trace, the B-user drafter trace, one trace per bucket writing into the B-user
    outputs, then the step trace."""
    B, K = 8, 3
    log, n = [], [0]
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.prepared, e.captured, e.recorded_drafter = B, K, K + 1, True, False, True
    e.mesh = MagicMock()
    e._dr_buckets = (1, 2, 4)
    e.model = SimpleNamespace(record_verify_trace=lambda: log.append("verify"), _vfy_rows_out="feed")
    outs = [SimpleNamespace(buffer_address=lambda k=k: 1000 + k) for k in range(K)]

    def body(feed, b=None, dst=None):
        log.append(("drafter", b, dst is outs if dst is not None else None))
        return outs if b is None else dst

    def begin(mesh, cq_id):
        n[0] += 1
        return f"t{n[0]}"

    e._drafter_body = body
    e._capture_step = lambda: log.append("step")
    monkeypatch.setattr(ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, tid, cq_id: log.append(("end", tid)))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    MTPSpecEngine.capture_decode(e)
    assert log == [
        "verify",
        ("drafter", None, None),
        ("end", "t1"),
        ("drafter", 1, True),
        ("end", "t2"),
        ("drafter", 2, True),
        ("end", "t3"),
        ("drafter", 4, True),
        ("end", "t4"),
        "step",
    ]
    assert e._dr_trace == "t1" and e._dr_bucket_traces == {1: "t2", 2: "t3", 4: "t4"}

    # A bucket capture that moved a B-user output is refused.
    moved = [SimpleNamespace(buffer_address=lambda k=k: 1000 + k) for k in range(K)]
    e._dr_outs = moved

    def moving_body(feed, b, dst):
        moved[1].buffer_address = lambda: 7

    e._drafter_body = moving_body
    with expect_error(RuntimeError, r"moved the drafter outputs \[1\]"):
        MTPSpecEngine._capture_drafter_buckets(e, "feed")


# ----------------------------------------------------------------------------------------- per-shard argmax rows
def test_the_sharded_argmax_constants_follow_the_row_count(monkeypatch):
    """_argmax_sharded binds the shard offset and lane index of its shard's own row count, built once per count."""
    built = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: built.append(tuple(t.shape)) or t)
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices, m._vocab_shard, m.device, m._argmax_consts = 8, 100, "mesh", {}
    off, lane = m._argmax_constants(2)
    assert tuple(off.shape) == (8, 1, 2, m._LANES) and tuple(lane.shape) == (1, 1, 2, 8 * m._LANES)
    assert torch.equal(off[3, 0, 1], torch.full((m._LANES,), 300, dtype=torch.int32))
    assert torch.equal(lane[0, 0, 1], torch.arange(8 * m._LANES, dtype=torch.int32))
    assert m._argmax_constants(2)[0] is off and len(built) == 2, "built once per row count"
    m._argmax_constants(4)
    assert built[2:] == [(8, 1, 4, m._LANES), (1, 1, 4, 8 * m._LANES)]
