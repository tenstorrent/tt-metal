# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the big chunk-trace set of Qwen36Model (tt/model.py prepare_prefill_trace_chunked_big,
record_prefill_trace_chunked_big, _prefill_traced_chunked_tp).

prefill_traced_chunked replays a captured 2048-row chunk trace for each full chunk of a prefill segment and runs the
partial last chunk as one masked-bucket forward. With a captured big chunk trace (Qwen36Model._PREFILL_BIG_CHUNK rows),
the whole big blocks of the full chunks replay the big trace first and the remaining full chunks the 2048-row one; the
tail, its checkpoint and its logits stay those of the 2048-row plan. These tests replace the ttnn calls with the
recorders of test_masked_prefill_trace_host and check the replay order, the inputs each set's replay stages, the
logits of a prefill that ends on a big chunk, the MTP chunk hook, and the prepare, capture and release lifecycle.
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_masked_prefill_trace_host import (  # noqa: F401 (device_calls fixture)
    Dev,
    _captured,
    _model,
    device_calls,
)

BIG = 8192
BIG_SET = {
    "_chunk_token_buf": Dev("big token"),
    "_chunk_start_idx_tensor": Dev("big csi"),
    "_chunk_full_page_table_buf": Dev("big full pt", (1, 32)),
    "_chunk_page_table_buf": Dev("big page_table"),
    "_chunk_cos_buf": Dev("big cos"),
    "_chunk_sin_buf": Dev("big sin"),
    "_chunked_trace_id": "big trace",
    "_chunked_trace_output": Dev("big hidden"),
    "_chunked_chunk_size": BIG,
    "_chunked_prepared_key": "big prepared",
    "_vis_buf": Dev("big vis"),
    "_vis_mask_buf": Dev("big vis mask"),
    "_vis_zero_mask_host": None,
}


def _with_big(monkeypatch, calls, captured=True):
    """test_masked_prefill_trace_host's captured model (2048-row chunk trace and masked-bucket traces) plus a big set."""
    m, gdn = _captured(monkeypatch, calls)
    m._chunk_sets = {"big": dict(BIG_SET, _chunked_trace_id="big trace" if captured else None)}
    return m


def _prefill(m, end, start=0, checkpoint=0, **kw):
    tokens = torch.arange(100, 100 + end, dtype=torch.int64).reshape(1, end)  # no image placeholder
    return m.prefill_traced_chunked(
        tokens,
        torch.arange(1, 33, dtype=torch.int32).reshape(1, 32),
        actual_len=end,
        start_pos=start,
        checkpoint_pos=checkpoint,
        **kw,
    )


def _replays(m, calls):
    """The replays in ``calls``: ("chunk" or "big", staged chunk start) or (bucket, checkpoint, staged chunk start)."""
    names = {tid: (b, ck) for b, tr in m._masked_traces.items() for ck, tid in tr.trace_ids.items()}
    csi_bufs = {"chunk csi": None, "big csi": None, **{tr.csi.name: None for tr in m._masked_traces.values()}}
    out, csi = [], None
    for c in calls:
        if c[0] == "stage" and getattr(c[1], "name", None) in csi_bufs:
            csi = int(c[2].t[0])
        if c[0] == "replay":
            if c[1] in ("chunk trace", "big trace"):
                out.append(("chunk" if c[1] == "chunk trace" else "big", csi))
            else:
                out.append((*names[c[1]], csi))
    return out


@pytest.mark.parametrize(
    "case, kw, want",
    [
        ("below one big block", dict(end=6000), [("chunk", 0), ("chunk", 2048), (2048, False, 4096)]),
        ("one big block and a tail", dict(end=BIG + 300), [("big", 0), (512, False, BIG)]),
        (
            "big, 2048 and tail",
            dict(end=BIG + 2048 + 500),
            [("big", 0), ("chunk", BIG), (512, False, BIG + 2048)],
        ),
        ("two big blocks exactly", dict(end=2 * BIG), [("big", 0), ("big", BIG)]),
        ("resumed off a big boundary", dict(start=1024, end=1024 + BIG + 300), [("big", 1024), (512, False, 9216)]),
        (
            "checkpoint in the tail",
            dict(end=10000, checkpoint=9216),
            [("big", 0), (2048, 1024, BIG)],
        ),
        (
            "whole last chunk with a checkpoint",
            dict(end=BIG + 2048, checkpoint=9216),
            [("big", 0), (2048, 1024, BIG)],
        ),
    ],
)
def test_whole_big_blocks_replay_the_big_trace_first(monkeypatch, device_calls, case, kw, want):
    """The full chunks of a segment replay the big trace for each whole big block from the segment's start, then the
    2048-row trace; the tail (with the checkpoint inside it) is the masked replay the 2048-row plan runs. The 2048-row
    set is active again afterwards."""
    m = _with_big(monkeypatch, device_calls)
    device_calls.clear()
    _prefill(m, **kw)
    assert _replays(m, device_calls) == want, case
    assert m._chunked_trace_id == "chunk trace" and m._chunked_chunk_size == 2048
    assert m._chunk_sets["big"]["_chunked_trace_id"] == "big trace"
    assert getattr(m, "prefill_big_chunk_replays", 0) == sum(r[0] == "big" for r in want)


def test_each_set_stages_its_own_inputs(monkeypatch, device_calls):
    """A big replay stages its tokens, chunk start, RoPE rows and the 8 blocks of its chunk page table into the big
    set's buffers, and each set's full page table once; the 2048-row replay after it stages the 2048-row set's."""
    m = _with_big(monkeypatch, device_calls)
    device_calls.clear()
    _prefill(m, BIG + 2048 + 500, start=0)
    staged = [(c[1].name, c[2].t) for c in device_calls if c[0] == "stage" and isinstance(c[1], Dev)]
    by_buf = {}
    for name, t in staged:
        by_buf.setdefault(name, []).append(t)
    assert len(by_buf["big full pt"]) == 1 and len(by_buf["chunk full pt"]) == 1
    assert by_buf["big full pt"][0].tolist() == [list(range(1, 33))]
    assert by_buf["big token"][0].reshape(-1).tolist() == list(range(100, 100 + BIG))
    assert by_buf["big csi"][0].tolist() == [0]
    assert by_buf["big page_table"][0].tolist() == [list(range(1, 9))]
    assert by_buf["big cos"][0].reshape(-1).tolist() == list(range(BIG))
    assert by_buf["chunk token"][0].reshape(-1).tolist() == list(range(100 + BIG, 100 + BIG + 2048))
    assert by_buf["chunk page_table"][0].tolist() == [[9, 10]]
    assert by_buf["chunk csi"][0].tolist() == [BIG]
    vision = [c for c in device_calls if c[0] == "vision"]
    assert vision[:2] == [("vision", BIG, False, 0), ("vision", 2048, False, 0)]


def test_a_prefill_that_ends_on_a_big_chunk_takes_its_logits_from_the_big_output(monkeypatch, device_calls):
    m = _with_big(monkeypatch, device_calls)
    device_calls.clear()
    _prefill(m, 2 * BIG)
    assert [c for c in device_calls if c[0] == "logits"] == [("logits", BIG, BIG, False)]
    device_calls.clear()
    _prefill(m, BIG + 2048)
    assert [c for c in device_calls if c[0] == "logits"] == [("logits", 2048, 2048, False)]


def test_the_chunk_hook_gets_each_replays_output_and_rows(monkeypatch, device_calls):
    """Qwen36MTPContractForCausalLM's prefill hook (on_chunk) gets the big trace's output with its row count, then the
    2048-row trace's, then the masked tail's."""
    m = _with_big(monkeypatch, device_calls)
    seen = []
    _prefill(m, BIG + 2048 + 500, on_chunk=lambda hidden, start, n: seen.append((hidden.name, start, n)))
    tail = m._masked_traces[512].outputs[False][0].name
    assert seen == [("big hidden", 0, BIG), ("chunk hidden", BIG, 2048), (tail, BIG + 2048, 500)]


def test_an_uncaptured_big_set_is_not_replayed(monkeypatch, device_calls):
    m = _with_big(monkeypatch, device_calls, captured=False)
    assert m.prefill_big_chunk_size == BIG and m.big_chunk_rows() == 0
    device_calls.clear()
    _prefill(m, BIG + 300)
    assert _replays(m, device_calls) == [
        ("chunk", 0),
        ("chunk", 2048),
        ("chunk", 4096),
        ("chunk", 6144),
        (512, False, BIG),
    ]


def test_the_sets_swap_every_chunk_trace_attribute(monkeypatch, device_calls):
    m = _with_big(monkeypatch, device_calls)
    small = {a: getattr(m, a, None) for a in m._CHUNK_SET_ATTRS}
    with m._big_chunk_set():
        assert {a: getattr(m, a) for a in m._CHUNK_SET_ATTRS} == BIG_SET
        assert m.prefill_big_chunk_size == 0 and m.big_chunk_rows() == 0
    assert {a: getattr(m, a, None) for a in m._CHUNK_SET_ATTRS} == small
    assert m.big_chunk_rows() == BIG


# ------------------------------------------------------------------------------- prepare, capture and release


def _preparable(monkeypatch, calls, max_seq_len=131072, kv_blocks=64):
    """A model whose 2048-row chunk set is prepared (not captured); prepare_prefill_trace_chunked records its calls."""
    m, _ = _model(monkeypatch, calls)
    m._chunked_trace_id = None
    m._chunked_trace_output = None
    m.args.max_seq_len = max_seq_len
    m._paged_kv_caches = [(Dev("k", (kv_blocks, 1, 1024, 256)), Dev("v"))]

    def prepare(device, page_table, chunk_size=2048, warmup_masked_buckets=True):
        calls.append(("prepare", chunk_size, tuple(page_table.shape), warmup_masked_buckets))
        m._chunked_chunk_size = chunk_size
        m._chunked_prepared_key = (chunk_size, tuple(page_table.shape), bool(warmup_masked_buckets))
        m._chunk_full_page_table_buf = Dev("big full pt", tuple(page_table.shape))

    def record(device):
        calls.append(("record", m._chunked_chunk_size))
        m._chunked_trace_id = f"trace {m._chunked_chunk_size}"

    m.prepare_prefill_trace_chunked = prepare
    m.record_prefill_trace_chunked = record
    m._forward_prefill_chunk_tp = lambda *bufs: calls.append(("forward", m._chunked_chunk_size)) or Dev("out")
    return m


def test_prepare_big_compiles_the_set_and_its_logits_before_any_capture(monkeypatch, device_calls):
    m = _preparable(monkeypatch, device_calls)
    pt = torch.zeros(1, 32, dtype=torch.int32)
    m.prepare_prefill_trace_chunked_big("mesh", pt)
    assert [c for c in device_calls if c[0] in ("prepare", "forward", "logits")] == [
        ("prepare", BIG, (1, 32), False),
        ("forward", BIG),
        ("logits", BIG, BIG, False),
    ]
    assert not any(c[0] in ("begin", "replay", "record") for c in device_calls)
    assert m._chunked_chunk_size == 2048 and m.prefill_big_chunk_size == BIG and m.big_chunk_rows() == 0
    device_calls.clear()
    m.prepare_prefill_trace_chunked_big("mesh", pt)
    assert device_calls == []

    m.record_prefill_trace_chunked_big("mesh")
    assert [c for c in device_calls if c[0] == "record"] == [("record", BIG)]
    assert m.big_chunk_rows() == BIG and m._chunked_trace_id is None
    device_calls.clear()
    m.record_prefill_trace_chunked_big("mesh")
    assert device_calls == []

    m.release_prefill_trace_chunked_big()
    assert ("release", f"trace {BIG}") in device_calls
    assert m.prefill_big_chunk_size == 0 and m.big_chunk_rows() == 0


@pytest.mark.parametrize(
    "why, kw, columns",
    [
        ("short max_seq_len", dict(max_seq_len=8191), 32),
        ("short page table", dict(), 7),
        ("small KV cache", dict(kv_blocks=7), 32),
    ],
)
def test_prepare_big_skips_a_model_that_never_prefills_a_big_block(monkeypatch, device_calls, why, kw, columns):
    m = _preparable(monkeypatch, device_calls, **kw)
    m._chunk_full_page_table_buf = Dev("chunk full pt", (1, columns))
    m.prepare_prefill_trace_chunked_big("mesh", torch.zeros(1, columns, dtype=torch.int32))
    assert not any(c[0] in ("prepare", "forward") for c in device_calls), why
    assert m.prefill_big_chunk_size == 0
    m.record_prefill_trace_chunked_big("mesh")
    assert not any(c[0] == "record" for c in device_calls), why


def test_prepare_big_refusals(monkeypatch, device_calls, expect_error):
    m = _preparable(monkeypatch, device_calls)
    with expect_error(ValueError, "page table"):
        m.prepare_prefill_trace_chunked_big("mesh", torch.zeros(1, 64, dtype=torch.int32))
    m._chunked_trace_id = "chunk trace"
    with expect_error(ValueError, "before the 2048-row chunk trace is captured"):
        m.prepare_prefill_trace_chunked_big("mesh", torch.zeros(1, 32, dtype=torch.int32))
    m._chunked_prepared_key = None
    with expect_error(ValueError, "before the big one"):
        m.prepare_prefill_trace_chunked_big("mesh", torch.zeros(1, 32, dtype=torch.int32))
    m.num_devices = 1
    m.prepare_prefill_trace_chunked_big("mesh", torch.zeros(1, 32, dtype=torch.int32))


def test_warm_big_chunk_output_runs_one_forward_of_the_big_set(monkeypatch, device_calls):
    m = _preparable(monkeypatch, device_calls)
    m.prepare_prefill_trace_chunked_big("mesh", torch.zeros(1, 32, dtype=torch.int32))
    device_calls.clear()
    seen = []
    m.warm_big_chunk_output(lambda hidden: seen.append((hidden.name, m._chunked_chunk_size)))
    assert seen == [("out", BIG)]
    assert [c[0] for c in device_calls if c[0] in ("reset", "forward", "free", "sync")] == [
        "reset",
        "forward",
        "free",
        "sync",
        "reset",
    ]
    assert m._chunked_chunk_size == 2048
    m.release_prefill_trace_chunked_big()
    device_calls.clear()
    m.warm_big_chunk_output(lambda hidden: seen.append(hidden))
    assert device_calls == [] and len(seen) == 1
