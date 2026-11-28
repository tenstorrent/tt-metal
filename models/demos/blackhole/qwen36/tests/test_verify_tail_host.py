# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the end of the TP MTP verify trace (Qwen36Model, tt/model.py) and the logits traces.

* With the fused per-shard argmax (tt/spec_argmax_fused.py), _forward_verify_bucket_tp ends at this device's vocab
  shard of the logits: no vocab all-gather, untilize or argmax. Without it, the trace all-gathers, untilizes and
  argmaxes as before.
* record_verify_trace then captures two more traces: the gather trace all-gathers a copy of the very shard the fused
  argmax read (so the logits are the ones the ids came from), the ROW_MAJOR trace untilizes those logits. The sampler
  trace reads the gathered logits.
* verify_enqueue replays the gather trace only for a device-sampled or logits step and the ROW_MAJOR trace only for a
  logits step; a greedy step replays the verify trace only. verify_traced(read_logits=True) reads the ROW_MAJOR copy.
* _warm_verify_body compiles the gather, the untilize and the sampler over the gathered logits before any capture.
* release_verify_trace releases every verify trace.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_sample import sampling_plan

DIM = 64


class FT:
    def __init__(self, shape, name):
        self.shape, self.name = tuple(shape), name

    def __repr__(self):
        return f"FT({self.name})"


def _body_model(tail):
    """A Qwen36Model whose verify body runs two stand-in layers and ``tail`` as _spec_tail_ids."""
    model = object.__new__(Qwen36Model)
    model._vfy_T, model._vfy_step_buf = 4, "step"
    model.args = SimpleNamespace(gdn_chunk_size=128)
    model.rope_lookup = lambda rows, n, row_view=False: (FT((1, 1, n, 8), "cos"), FT((1, 1, n, 8), "sin"))
    model._vfy_kvpt_buf = model._vfy_kvpt_users = model._vfy_kvpt1_buf = None
    model._vfy_state_idx = model._vfy_conv_sel = None
    model.embd = lambda tok: FT((1, 1, 8, DIM), "x")
    model._residual_ar = lambda rows: None
    model._spec_tail_ids = tail
    layer = lambda full: SimpleNamespace(is_full_attention=full, forward=lambda x, **kw: FT(x.shape, "h"))
    model.layers = [layer(False), layer(True)]
    return model


@pytest.fixture
def body_ops(monkeypatch):
    log = []
    monkeypatch.setattr(ttnn, "slice", lambda t, s, e: FT((1, 8), "slice"))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "reshape", lambda t, s: t)
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, m: t)
    monkeypatch.setattr(ttnn, "typecast", lambda t, dtype: "kvpos")
    monkeypatch.setattr(ttnn, "untilize", lambda t, **kw: log.append(("untilize", t)) or "rm")
    monkeypatch.setattr(ttnn, "argmax", lambda t, **kw: log.append(("argmax", t)) or "argmax ids")
    return log


def test_a_verify_with_shard_ids_ends_at_the_vocab_shard(body_ops):
    model = _body_model(lambda x, shard_ids: ("shard", "feed", "lanes"))
    assert model._forward_verify_bucket_tp(n_users=2) == ("shard", "feed", "lanes", None)
    assert body_ops == [], "no untilize and no argmax of the logits"


def test_a_verify_without_shard_ids_gathers_untilizes_and_argmaxes(body_ops):
    model = _body_model(lambda x, shard_ids: ("logits", "feed", None))
    assert model._forward_verify_bucket_tp(n_users=2) == ("logits", "feed", "argmax ids", "rm")
    assert body_ops == [("untilize", "logits"), ("argmax", "rm")]


def test_the_verify_logits_gather_a_copy_of_the_shard(monkeypatch):
    log = []
    model = object.__new__(Qwen36Model)
    model._gather_vocab = lambda t: log.append(("gather", t)) or "logits"
    monkeypatch.setattr(ttnn, "clone", lambda t, memory_config=None: log.append(("clone", t, memory_config)) or "copy")
    assert model._verify_logits("shard") == "logits"
    assert log == [("clone", "shard", ttnn.DRAM_MEMORY_CONFIG), ("gather", "copy")]


# --------------------------------------------------------------------------------------------- capture and replay


@pytest.fixture
def capture(monkeypatch):
    """record_verify_trace on a model whose verify body is a stand-in; ``log`` holds the captures and the ops."""
    log = []
    ids = iter(range(100))
    current = []

    def begin(dev, cq_id):
        current.append(next(ids))
        return current[-1]

    monkeypatch.setattr(ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda dev, tid, cq_id: log.append(("capture", tid)))
    monkeypatch.setattr(ttnn, "untilize", lambda t, **kw: log.append(("untilize", t)) or "rm")

    def make(verify_out, sample=True):
        model = object.__new__(Qwen36Model)
        model._vfy_prepared_key, model.device = ("key",), "dev"
        model._vfy_B, model._vfy_T, model._vfy_spec_groups, model._vfy_kv_grouped = 2, 4, 2, False
        model._vfy_kvpt_buf = model._vfy_kvpt_users = model._vfy_kvpt1_buf = None
        model._vfy_att0 = SimpleNamespace(
            spec_sdpa_enabled=lambda T: False, _forward_verify_ok=lambda *a: True, _spec_pack_heads=lambda r: 0
        )
        model._forward_verify_bucket_tp = lambda n_users: log.append(("verify body", n_users)) or verify_out
        model._verify_logits = lambda shard: log.append(("verify logits", shard)) or "gathered"
        model._vfy_sampler = (
            SimpleNamespace(body=lambda logits: log.append(("sampler", logits)) or "draws") if sample else None
        )
        return model

    return make, log


def test_the_capture_adds_a_gather_and_a_row_major_trace_after_a_verify_that_ends_at_the_shard(capture):
    make, log = capture
    model = make(("shard", "rows", "lanes", None))
    model.record_verify_trace()
    assert log == [
        ("verify body", 2),
        ("capture", 0),
        ("verify logits", "shard"),
        ("capture", 1),
        ("untilize", "gathered"),
        ("capture", 2),
        ("sampler", "shard"),
        ("capture", 3),
    ]
    assert (model._vfy_trace_id, model._vfy_gather_trace_id, model._vfy_rm_trace_id, model._vfy_sample_trace_id) == (
        0,
        1,
        2,
        3,
    )
    assert model._vfy_shard_out == "shard" and model._vfy_ids_out == "lanes" and model._vfy_rows_out == "rows"
    assert model._vfy_logits_out == "gathered" and model._vfy_logits_rm_out == "rm"


def test_the_capture_keeps_one_trace_for_a_verify_that_gathers_itself(capture):
    make, log = capture
    model = make(("logits", "rows", "argmax ids", "logits rm"), sample=False)
    model.record_verify_trace()
    assert log == [("verify body", 2), ("capture", 0)]
    assert model._vfy_gather_trace_id is None and model._vfy_rm_trace_id is None and model._vfy_shard_out is None
    assert model._vfy_logits_out == "logits" and model._vfy_logits_rm_out == "logits rm"


def test_the_sampler_of_a_verify_that_gathers_itself_reads_the_gathered_logits(capture):
    make, log = capture
    model = make(("logits", "rows", "argmax ids", "logits rm"))
    model.record_verify_trace()
    assert log == [("verify body", 2), ("capture", 0), ("sampler", "logits"), ("capture", 1)]


def test_the_warm_pass_compiles_the_logits_traces_and_the_sampler_over_the_shard(monkeypatch):
    log = []
    monkeypatch.setattr(ttnn, "untilize", lambda t, **kw: log.append(("untilize", t)) or "rm")
    monkeypatch.setattr(ttnn, "deallocate", lambda t: log.append(("deallocate", t)))
    lanes = SimpleNamespace(shape=(1, 1, 1, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    model = object.__new__(Qwen36Model)
    model._forward_verify_bucket_tp = lambda n_users: ("shard", "rows", lanes, None)
    model._verify_logits = lambda shard: log.append(("verify logits", shard)) or "gathered"
    model._vfy_sampler = SimpleNamespace(body=lambda logits: log.append(("sampler", logits)))
    model._warm_verify_body(2)
    assert model._vfy_ids_lanes
    assert log == [
        ("verify logits", "shard"),
        ("untilize", "gathered"),
        ("sampler", "shard"),
        ("deallocate", "shard"),
        ("deallocate", "rows"),
        ("deallocate", lanes),
        ("deallocate", "gathered"),
        ("deallocate", "rm"),
    ]


def _replay_model(monkeypatch, gather=True):
    B, T = 2, 4
    m = Qwen36Model.__new__(Qwen36Model)
    m._vfy_trace_id, m._vfy_sample_trace_id, m._vfy_T, m._vfy_B, m._vfy_kv_grouped = "verify", "sample", T, B, False
    m._vfy_gather_trace_id, m._vfy_rm_trace_id = ("gather", "rm") if gather else (None, None)
    m._vfy_pt_host, m.gdn_blocks, m.device, m.vocab_size = torch.zeros(B, 4, dtype=torch.int32), object(), None, 16
    for name in ("_vfy_step_buf", "_vfy_rows_out"):
        setattr(m, name, MagicMock())
    m._vfy_ids_out, m._vfy_logits_rm_out = "ids", "logits rm"
    m.args = SimpleNamespace(rope_head_dim=4, max_seq_len=4096)
    m.rope = SimpleNamespace(rope_delta=0)
    m._vfy_sampler = SimpleNamespace(stage=lambda plan: None, read=lambda: torch.arange(500, 500 + B * T))
    replays = []
    monkeypatch.setattr(ttnn, "execute_trace", lambda dev, trace, **kw: replays.append(trace))
    return m, replays


def _plan():
    """Row 0 greedy, row 1 sampled."""
    return sampling_plan(
        SimpleNamespace(
            temperature=[0.0, 1.0],
            top_k=[1, 20],
            top_p=[1.0, 1.0],
            seed=[None, None],
            enable_log_probs=[False, False],
        ),
        [10, 20],
        4,
        fresh_seed=lambda: 7,
    )


@pytest.mark.parametrize(
    "sampled, read_logits, gather, expected",
    [
        (False, False, True, ["verify"]),
        (True, False, True, ["verify", "sample"]),
        (False, True, True, ["verify", "gather", "rm"]),
        (True, True, True, ["verify", "gather", "rm", "sample"]),
        (True, True, False, ["verify", "sample"]),
    ],
)
def test_only_a_step_that_reads_logits_replays_the_logits_traces(monkeypatch, sampled, read_logits, gather, expected):
    m, replays = _replay_model(monkeypatch, gather)
    plan = _plan() if sampled else None
    m.verify_enqueue([[1] * 4] * 2, [10, 20], [0, 0], sample_plan=plan, read_logits=read_logits)
    assert replays == expected


def test_a_logits_verify_reads_the_row_major_copy(monkeypatch):
    m, replays = _replay_model(monkeypatch)
    lanes = torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32)
    rm = torch.arange(32 * 16, dtype=torch.float32).reshape(1, 1, 32, 16)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t])
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: {"ids": lanes, "logits rm": rm}[t])
    ids, _, logits = m.verify_traced([[1] * 4] * 2, [10, 20], [0, 0], read_logits=True)
    assert replays == ["verify", "gather", "rm"]
    assert ids == list(range(8)) and torch.equal(logits, rm.reshape(-1, 16)[:8])
    replays.clear()
    ids, _, logits = m.verify_traced([[1] * 4] * 2, [10, 20], [0, 0])
    assert replays == ["verify"] and logits is None


def test_work_enqueued_after_the_replays_runs_before_the_wait(monkeypatch):
    """``after_enqueue`` runs once every replay of the step is enqueued (here the verify and the sampler; a sampled
    step without ``read_logits`` replays no logits gather), and the wait for the ids follows it."""
    m, replays = _replay_model(monkeypatch)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda dev: replays.append("wait"))
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t])
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: torch.arange(32, dtype=torch.int32))
    m.verify_traced([[1] * 4] * 2, [10, 20], [0, 0], sample_plan=_plan(), after_enqueue=lambda: replays.append("after"))
    assert replays == ["verify", "sample", "after", "wait"]


def test_the_release_frees_every_verify_trace(monkeypatch):
    released = []
    monkeypatch.setattr(ttnn, "release_trace", lambda dev, tid: released.append(tid))
    m = object.__new__(Qwen36Model)
    m.mesh_device = "mesh"
    m._vfy_trace_id, m._vfy_gather_trace_id, m._vfy_rm_trace_id, m._vfy_sample_trace_id = 1, 2, 3, 4
    m.release_verify_trace()
    assert released == [1, 2, 3, 4]
    assert (m._vfy_trace_id, m._vfy_gather_trace_id, m._vfy_rm_trace_id, m._vfy_sample_trace_id) == (None,) * 4
