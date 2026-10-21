# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the model side of the verify row buckets (QWEN36_SPEC_BUCKETS), over fake buffers and a fake ttnn:

* verify_stage_and_enqueue / verify_after_external_sync take ``bucket=``: None and B replay today's traces, a bucket b < B
  replays its own main / logits / topk trace ids, the staging stays the B-row host state, and the readbacks are zero-padded
  to the B-row shapes the engine expects;
* prepare_verify_bucket / record_verify_bucket / release_verify_trace: warm before capture with stand-in hand-off targets,
  capture in the plan's order, hand off into the live B-row outputs, free the stand-ins, release every trace id.

No device. Run: pytest models/demos/blackhole/qwen36/tests/test_spec_bucket_model_host.py -q
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_specstep_host import _Buf, _fake_ttnn, _vfy_model
from models.demos.blackhole.qwen36.tt import model as model_mod
from models.demos.blackhole.qwen36.tt.model import _VfyBucket

B, T, V, M = 8, 4, 64, 2
MAIN, LOGITS, TOPK = 7, 8, 9  # the B-row trace ids of _vfy_model + the topk one


def _bucket_model(buckets=(1, 2, 4), sharded=True):
    m, bufs = _vfy_model(B, T, fused=True, sharded=sharded)
    m.num_devices = 2
    m._vfy_topk_trace_id, m._vfy_topk_m = TOPK, M
    m._vfy_topk_vals_out, m._vfy_topk_ids_out = _Buf("topk_vals"), _Buf("topk_ids")
    m._vfy_topk_vals_out.data = torch.arange(32 * 2 * M, dtype=torch.float32).reshape(1, 1, 32, 2 * M) + 1.0
    m._vfy_topk_ids_out.data = torch.arange(32 * 2 * M, dtype=torch.int32).reshape(1, 1, 32, 2 * M) + 1
    m._vfy_bk = {}
    for i, b in enumerate(buckets):
        bk = _VfyBucket(b, b * T)
        bk.trace_id, bk.logits_trace_id, bk.topk_trace_id = 20 + 10 * i, 21 + 10 * i, 22 + 10 * i
        bk.logits_rm_out = _Buf(f"logits{b}")
        bk.logits_rm_out.data = torch.full((1, 1, b * T, V), float(b), dtype=torch.bfloat16)
        bk.narrow_ready = True
        m._vfy_bk[b] = bk
    return m, bufs


def _args(positions=None):
    return [[u * 10 + j for j in range(T)] for u in range(B)], positions or [100 + u for u in range(B)], [0] * B


def _traces(log):
    return [x[1] for x in log if x[0] == "trace"]


def test_bucket_none_and_B_replay_the_B_row_traces(monkeypatch):
    for bucket in (None, B):
        m, bufs = _bucket_model()
        log = []
        _fake_ttnn(monkeypatch, log, m)
        h = m.verify_stage_and_enqueue(*_args(), read_logits=True, bucket=bucket)
        assert _traces(log) == [MAIN, LOGITS] and h.bucket is None and m._vfy_last_bucket is None
        ids, feed, lt = m.verify_sync_and_read(h)
        assert ids == list(range(B * T)) and tuple(lt.shape) == (B * T, V)


@pytest.mark.parametrize("b,tids", [(1, (20, 21, 22)), (2, (30, 31, 32)), (4, (40, 41, 42))])
def test_bucket_replays_its_own_traces_with_the_B_row_staging(monkeypatch, b, tids):
    m, bufs = _bucket_model()
    log = []
    _fake_ttnn(monkeypatch, log, m)
    tokens, positions, mi = _args()
    # reference staging of the B-row call
    m.verify_stage_and_enqueue(tokens, positions, mi)
    ref = {k: (v.data.clone() if v.data is not None else None, v.writes) for k, v in bufs.items()}
    m2, bufs2 = _bucket_model()
    log2 = []
    _fake_ttnn(monkeypatch, log2, m2)
    h = m2.verify_stage_and_enqueue(tokens, positions, mi, read_logits=True, execute_topk=True, bucket=b)
    assert _traces(log2) == list(tids[:2]) + [tids[2]] and h.bucket == b and m2._vfy_last_bucket is m2._vfy_bk[b]
    for k, v in bufs2.items():  # the same B-row host state is staged, whatever the bucket
        assert v.writes == ref[k][1], k
        if ref[k][0] is not None:
            assert torch.equal(v.data, ref[k][0]), k
    # main trace only when no logits / topk read is asked
    m4, _ = _bucket_model()
    log4 = []
    _fake_ttnn(monkeypatch, log4, m4)
    m4.verify_stage_and_enqueue(tokens, positions, mi, bucket=b)
    assert _traces(log4) == [tids[0]]


def test_bucket_flips_the_parity_like_the_B_row_step(monkeypatch):
    m, _ = _bucket_model()
    log = []
    _fake_ttnn(monkeypatch, log, m)
    m.verify_traced(*_args(), bucket=2)
    assert all(dn._spec_win_par == 1 for dn in m._vfy_gdn)
    m.verify_traced(*_args(), bucket=1)
    assert all(dn._spec_win_par == 0 for dn in m._vfy_gdn)


@pytest.mark.parametrize("b", [1, 2, 4])
def test_bucket_readbacks_are_padded_to_the_B_row_shapes(monkeypatch, b):
    m, _ = _bucket_model()
    _fake_ttnn(monkeypatch, [], m)
    # ids: the shared buffer, B*T lanes (a bucket leaves lanes >= b*T at zero on the device)
    ids, feed, lt = m.verify_traced(*_args(), bucket=b)
    assert ids == list(range(B * T)) and feed == "feed" and lt is None
    # full logits: [b*T, V] from the bucket copy, zero rows to B*T
    ids, feed, lt = m.verify_traced(*_args(), read_logits=True, bucket=b)
    assert tuple(lt.shape) == (B * T, V)
    assert bool((lt[: b * T] == float(b)).all()) and bool((lt[b * T :] == 0).all())
    # narrow read: [b, V] column-0 rows zero-padded to [B, V]
    m._vfy_narrow_logits = lambda t, bb=None: types.SimpleNamespace(
        data=t.data.reshape(bb, T, V)[:, 0:1].contiguous(), shape=(bb, 1, V)
    )
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    ids, feed, lt = m.verify_traced(*_args(), read_logits=True, read_col0=True, bucket=b)
    assert m._vfy_last_col0 and tuple(lt.shape) == (B, V)
    assert bool((lt[:b] == float(b)).all()) and bool((lt[b:] == 0).all())
    # topk: shared buffers, first b*T rows, zero rows to B*T
    ids, feed, (vals, idx) = m.verify_traced(*_args(), read_topk=True, bucket=b)
    assert tuple(vals.shape) == (B * T, 2 * M) == tuple(idx.shape)
    assert torch.equal(vals[: b * T], m._vfy_topk_vals_out.data.reshape(-1, 2 * M)[: b * T])
    assert torch.equal(idx[: b * T], m._vfy_topk_ids_out.data.reshape(-1, 2 * M)[: b * T])
    assert bool((vals[b * T :] == 0).all()) and bool((idx[b * T :] == 0).all())


def test_fetch_verify_logits_follows_the_last_replay(monkeypatch):
    m, _ = _bucket_model()
    log = []
    _fake_ttnn(monkeypatch, log, m)
    m.device = "dev"
    m._vfy_logits_rm_out.data = torch.full((1, 1, B * T, V), 9.0, dtype=torch.bfloat16)
    m.verify_traced(*_args(), bucket=2)
    del log[:]
    lt = m.fetch_verify_logits()
    assert _traces(log) == [31] and tuple(lt.shape) == (B * T, V)
    assert bool((lt[: 2 * T] == 2.0).all()) and bool((lt[2 * T :] == 0).all())
    m.verify_traced(*_args())  # a B-row replay resets the lazy read to the B-row logits trace
    del log[:]
    lt = m.fetch_verify_logits()
    assert _traces(log) == [LOGITS] and bool((lt == 9.0).all())


def test_unknown_or_uncaptured_bucket_is_refused(monkeypatch, expect_error):
    m, _ = _bucket_model(buckets=(2,))
    _fake_ttnn(monkeypatch, [], m)
    with expect_error(AssertionError, "bucket 4 is not captured"):
        m.verify_stage_and_enqueue(*_args(), bucket=4)
    m._vfy_bk[2].trace_id = None
    with expect_error(AssertionError, "bucket 2 is not captured"):
        m.verify_stage_and_enqueue(*_args(), bucket=2)
    m._vfy_topk_trace_id = TOPK
    m._vfy_bk[2].trace_id = 5
    m._vfy_bk[2].topk_trace_id = None
    with expect_error(ValueError, "read_topk"):
        m.verify_stage_and_enqueue(*_args(), execute_topk=True, bucket=2)


# --------------------------------------------------------------------------- #
# prepare / record / release over fakes
# --------------------------------------------------------------------------- #
class _T:
    """Fake device tensor with the attributes _tensor_spec reads."""

    live = set()

    def __init__(self, name, shape=(1, 1, 32, 8), dtype="bf16", layout="TILE", mc="DRAM"):
        self.name, self.shape, self.dtype, self.layout, self._mc = name, shape, dtype, layout, mc
        _T.live.add(self)

    def memory_config(self):
        return self._mc


def _flow_model(monkeypatch, topk=True):
    """A model with prepare_verify_trace done and the B-row trace captured, the bodies replaced by recording fakes."""
    m, _ = _vfy_model(B, T, fused=True, sharded=True)
    m.device = m.mesh_device = "mesh"
    m._vfy_prepared_key = (B, T, 8)
    m._vfy_topk, m._vfy_topk_m, m._vfy_topk_trace_id = topk, M, None
    m._vfy_bk = None
    m._vfy_ids_out = _T("ids8", (1, 1, 1, 32), "u32", "RM")
    m._vfy_rows_out = _T("rows8", (1, 1, 32, 8))
    m._vfy_topk_vals_out = _T("tkv8", (1, 1, 32, 4))
    m._vfy_topk_ids_out = _T("tki8", (1, 1, 32, 4), "u32")
    m._vfy_specs = {
        "ids": ((1, 1, 1, 32), "u32", "RM", "DRAM"),
        "rows": ((1, 1, 32, 8), "bf16", "TILE", "DRAM"),
    }
    if topk:
        m._vfy_specs.update(
            topk_vals=((1, 1, 32, 4), "bf16", "TILE", "DRAM"), topk_ids=((1, 1, 32, 4), "u32", "TILE", "DRAM")
        )
    log, state = [], {"next_id": 100, "open": None}

    def new_dl(rows):
        d = types.SimpleNamespace(rows=rows, released=False)
        d.release = lambda: setattr(d, "released", True)
        log.append(("dl", rows))
        return d

    def from_torch(t, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None, **kw):
        x = _T("standin", tuple(t.shape), dtype, layout, memory_config)
        log.append(("alloc", x))
        return x

    def body(n_users, bk=None):
        log.append(("body", n_users, dict(bk.handoff) if bk else None, state["open"]))
        return None, None, None, None, _T(f"normed{n_users}"), (_T(f"shard{n_users}") if topk else None)

    def logits_body(bk=None):
        log.append(("logits_body", bk.b, state["open"]))
        return _T("lg"), _T(f"lgrm{bk.b}")

    def topk_body(bk=None):
        log.append(("topk_body", bk.b, state["open"], dict(bk.handoff)))
        return None, None

    def begin(dev, cq_id=0):
        state["next_id"] += 1
        state["open"] = state["next_id"]
        log.append(("begin", state["open"]))
        return state["open"]

    def end(dev, tid, cq_id=0):
        assert state["open"] == tid
        state["open"] = None
        log.append(("end", tid))

    monkeypatch.setattr(m, "_vfy_new_dl", new_dl)
    monkeypatch.setattr(m, "_forward_verify_bucket_tp", body)
    monkeypatch.setattr(m, "_forward_verify_logits_body", logits_body)
    monkeypatch.setattr(m, "_forward_verify_topk_body", topk_body)
    monkeypatch.setattr(m, "_vfy_warm_narrow", lambda t, b=None: True)
    monkeypatch.setattr(m, "_log_trace_memory", lambda label: log.append(("mem", label)))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda dev: ("rep", dev))
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(ttnn, "end_trace_capture", end)
    monkeypatch.setattr(ttnn, "release_trace", lambda dev, tid: log.append(("release", tid)))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: (log.append(("free", t.name)), _T.live.discard(t)))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda dev: log.append(("sync",)))
    monkeypatch.setattr(ttnn, "bfloat16", "bf16", raising=False)
    monkeypatch.setattr(ttnn, "uint32", "u32", raising=False)
    return m, log


def _names(log, kind):
    return [x for x in log if x[0] == kind]


def test_prepare_warms_before_any_capture_with_standins(monkeypatch):
    _T.live.clear()
    m, log = _flow_model(monkeypatch)
    m.prepare_verify_buckets((1, 2, 4, 8))  # 8 is the B-row trace: skipped
    assert sorted(m._vfy_bk) == [1, 2, 4] and not _names(log, "begin"), "nothing is captured while warming"
    assert [x[1] for x in _names(log, "body")] == [4, 2, 1], "descending"
    assert [("dl", b * T) for b in (4, 2, 1)] == _names(log, "dl")
    for x in _names(log, "body"):
        assert set(x[2]) == {"ids", "rows", "topk_vals", "topk_ids"} and x[3] is None
        assert all(t.name == "standin" for t in x[2].values()), "the warm hand-off targets are stand-ins"
    for b, bk in m._vfy_bk.items():
        assert bk.trace_id is None and bk.dl.rows == b * T and bk.narrow_ready and len(bk.standins) == 4
    # stand-ins carry the shape / dtype of the live outputs
    st = m._vfy_bk[4].handoff
    assert st["ids"].shape == (1, 1, 1, 32) and st["rows"].shape == (1, 1, 32, 8) and st["topk_ids"].dtype == "u32"
    n = len(log)
    m.prepare_verify_buckets((1, 2, 4))  # idempotent
    assert len(log) == n
    # the warm pass frees its own normed / shard logits outputs
    assert {"normed4", "shard4", "lg", "lgrm4"} <= {x[1] for x in _names(log, "free")}


def test_record_orders_the_captures_and_hands_off_to_the_live_outputs(monkeypatch):
    _T.live.clear()
    m, log = _flow_model(monkeypatch)
    m.prepare_verify_buckets((1, 2, 4))
    del log[:]
    m.record_verify_buckets((1, 2, 4, 8))
    seq = [
        x[0] if x[0] != "body" else f"body{x[1]}"
        for x in log
        if x[0] in ("begin", "end", "body", "logits_body", "topk_body", "mem")
    ]
    want = []
    for b in (4, 2, 1):
        want += ["begin", f"body{b}", "end", "begin", "logits_body", "end", "begin", "topk_body", "end", "mem"]
    assert seq == want
    assert [x[1] for x in _names(log, "mem")] == ["V4", "V2", "V1"]
    for x in _names(log, "body"):  # captured with the live B-row outputs as hand-off targets, inside a trace
        h = x[2]
        assert h["ids"] is m._vfy_ids_out and h["rows"] is m._vfy_rows_out and x[3] is not None
        assert h["topk_vals"] is m._vfy_topk_vals_out and h["topk_ids"] is m._vfy_topk_ids_out
    for x in _names(log, "topk_body"):
        assert x[3]["topk_vals"] is m._vfy_topk_vals_out and x[2] is not None
    for b, bk in m._vfy_bk.items():
        assert None not in (bk.trace_id, bk.logits_trace_id, bk.topk_trace_id) and bk.standins == []
        assert bk.normed is not None and bk.logits_rm_out.name == f"lgrm{b}"
    assert len({bk.trace_id for bk in m._vfy_bk.values()} | {bk.logits_trace_id for bk in m._vfy_bk.values()}) == 6
    # the stand-ins were freed after their bucket's capture, the live outputs never
    freed = [x[1] for x in _names(log, "free")]
    assert freed.count("standin") == 12 and not {"ids8", "rows8", "tkv8", "tki8"} & set(freed)
    n = len(log)
    m.record_verify_buckets((1, 2, 4))  # idempotent
    assert len(log) == n


def test_record_needs_the_B_row_trace_and_a_prepare(monkeypatch, expect_error):
    m, log = _flow_model(monkeypatch)
    with expect_error(AssertionError, "prepare_verify_bucket"):
        m.record_verify_bucket(2)
    m.prepare_verify_bucket(2)
    m._vfy_trace_id = None
    with expect_error(AssertionError, "record_verify_trace"):
        m.record_verify_bucket(2)
    with expect_error(AssertionError, "must be in"):
        m.prepare_verify_bucket(8)


def test_release_frees_every_bucket_trace_and_buffer(monkeypatch):
    _T.live.clear()
    m, log = _flow_model(monkeypatch)
    m.prepare_verify_buckets((1, 2, 4))
    m.record_verify_buckets((1, 2, 4))
    ids = sorted(t for bk in m._vfy_bk.values() for t in (bk.trace_id, bk.logits_trace_id, bk.topk_trace_id))
    dls = [bk.dl for bk in m._vfy_bk.values()]
    m._vfy_last_bucket = m._vfy_bk[2]
    del log[:]
    m.release_verify_trace()
    assert sorted(x[1] for x in _names(log, "release")) == sorted(ids + [7, 8]), "bucket ids plus the B-row ones"
    assert m._vfy_bk == {} and m._vfy_last_bucket is None and all(d.released for d in dls)
    assert not [t for t in _T.live if t.name in ("normed4", "normed2", "normed1", "lgrm4", "lgrm2", "lgrm1")]
    del log[:]
    m.release_verify_trace()  # idempotent
    assert not _names(log, "release")


def test_record_logs_memory_for_the_B_row_trace_only_with_buckets(monkeypatch):
    """record_verify_trace's own log line appears only when buckets exist (the B-row path stays byte-equal otherwise)."""
    src = open(model_mod.__file__).read()
    assert 'if getattr(self, "_vfy_bk", None):\n            self._log_trace_memory(f"V{B}")' in src


def test_log_trace_memory_reports_and_never_raises(monkeypatch):
    m, _ = _vfy_model(B, T, fused=True)
    views = {
        "TRACE": types.SimpleNamespace(
            total_bytes_allocated_per_bank=2 << 20, total_bytes_per_bank=64 << 20, num_banks=8
        ),
        "DRAM": types.SimpleNamespace(
            total_bytes_allocated_per_bank=100 << 20, total_bytes_per_bank=1 << 30, num_banks=8
        ),
    }
    monkeypatch.setattr(ttnn, "get_memory_view", lambda mesh, bt: views[bt], raising=False)
    monkeypatch.setattr(ttnn, "BufferType", types.SimpleNamespace(TRACE="TRACE", DRAM="DRAM"), raising=False)
    assert m._log_trace_memory("V8") == (16 << 20, 800 << 20)
    views["TRACE"] = types.SimpleNamespace(
        total_bytes_allocated_per_bank=5 << 20, total_bytes_per_bank=64 << 20, num_banks=8
    )
    assert m._log_trace_memory("V4")[0] == 40 << 20 and list(m._vfy_mem_log) == ["V8", "V4"]
    monkeypatch.setattr(
        ttnn, "get_memory_view", lambda *a: (_ for _ in ()).throw(RuntimeError("no view")), raising=False
    )
    assert m._log_trace_memory("V2") is None
