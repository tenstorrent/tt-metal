# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the masked 4096- and 8192-row traces that can run the rows of a prefill segment after its
whole big blocks (tt/prefill_split.py, Qwen36Model._masked_big_plan, prepare_prefill_traces_masked with a big chunk
set).

prefill_traced_chunked replays the big chunk trace for each whole 8192-row block of a segment's full chunks. The rows
after them ran as 2048-row chunk replays and one masked tail; prefill_split.plan_remainder models each replay's time as
A[rows] + B[rows] * depth and may pick one masked 4096- or 8192-row replay of all of them instead. These tests check the
plans against hand-computed costs, and with the ttnn recorders of test_masked_prefill_trace_host the routing of rows
with and without a checkpoint, the vision-splice buffers the big masked traces read, and their prepare and capture.
"""

import pytest

from models.demos.blackhole.qwen36.tests.test_masked_prefill_trace_host import (  # noqa: F401 (device_calls fixture)
    _model,
    device_calls,
)
from models.demos.blackhole.qwen36.tests.test_prefill_big_chunk_host import BIG, BIG_SET, _prefill, _replays
from models.demos.blackhole.qwen36.tt import prefill_split as ps

# Expected plans from the cost table (seconds, A + B * depth): 2048: (0.238, 3.6e-6), 4096: (0.42, 4.0e-6), 8192:
# (0.78, 4.6e-6); a tail bucket up to 2048 rows scales the 2048 entry's A by bucket / 2048. Example: 2100 rows at depth
# 131072: chunks = 0.238 + 3.6e-6 * 131072 + 0.238 / 16 + 3.6e-6 * 133120 = 1.204 s, one masked 4096 = 0.42 + 4.0e-6 *
# 131072 = 0.944 s.
DEPTHS = (0, 32768, 131072)
EXPECTED = {
    100: ("chunks", "chunks", "chunks"),
    2048: ("chunks", "chunks", "chunks"),
    2100: ("chunks", "chunks", "m4096"),
    4096: ("m4096", "m4096", "m4096"),
    5000: ("chunks", "m8192", "m8192"),
    8191: ("m8192", "m8192", "m8192"),
}


@pytest.mark.parametrize("r", sorted(EXPECTED))
def test_plans_match_the_hand_computed_costs(r):
    for depth, want in zip(DEPTHS, EXPECTED[r]):
        plan, costs = ps.plan_remainder(r, depth)
        assert plan == want, (r, depth, plan, costs)
        assert costs[plan] == min(costs.values())


def test_hand_computed_costs():
    _, c = ps.plan_remainder(2100, 131072)
    assert c["chunks"] == pytest.approx(0.238 + 3.6e-6 * 131072 + 0.238 / 16 + 3.6e-6 * (131072 + 2048))
    assert c["m4096"] == pytest.approx(0.42 + 4.0e-6 * 131072)
    assert c["m8192"] == pytest.approx(0.78 + 4.6e-6 * 131072)
    _, c = ps.plan_remainder(8191, 0)  # 3 chunks and a 2048-row tail bucket
    assert c["chunks"] == pytest.approx(0.238 * 4 + 3.6e-6 * (0 + 2048 + 4096 + 6144))
    assert "m4096" not in c


def test_only_buckets_with_a_trace_are_planned():
    plan, costs = ps.plan_remainder(4096, 131072, buckets=(8192,))
    assert "m4096" not in costs and plan in ("chunks", "m8192")
    plan, costs = ps.plan_remainder(5000, 131072, buckets=())
    assert plan == "chunks" and list(costs) == ["chunks"]
    assert ps.plan_remainder(0, 131072)[0] == "chunks"


# ------------------------------------------------------------------------------- routing in prefill_traced_chunked


def _with_masked_big(monkeypatch, calls):
    """test_masked_prefill_trace_host's model with a captured big chunk set, then its masked traces prepared and
    captured (the big set prepared first, as Qwen36ForCausalLM.warmup_model_prefill does). The traced forward records
    the rows of the chunk set active when it runs."""
    m, gdn = _model(monkeypatch, calls)
    m._chunk_sets = {"big": dict(BIG_SET)}
    forward = m._forward_prefill_chunk_masked_tp

    def recorded(*a, **kw):
        calls.append(("set", m._chunked_chunk_size))
        return forward(*a, **kw)

    m._forward_prefill_chunk_masked_tp = recorded
    vision = m._set_vision_merge
    m._set_vision_merge = lambda ids, v, offset=0: calls.append(("vision set", m._chunked_chunk_size)) or vision(
        ids, v, offset
    )
    m.prepare_prefill_traces_masked()
    m.record_prefill_traces_masked()
    calls.clear()
    return m


def _expected(start, end):
    """The replays prefill_traced_chunked should run for the rows [start, end) without a checkpoint."""
    rows = (end - start) // 2048 * 2048
    n_big = rows // BIG
    rest = start + n_big * BIG
    out = [("big", start + i * BIG) for i in range(n_big)]
    plan, _ = ps.plan_remainder(end - rest, rest)
    if end - rest > 2048 and plan != "chunks":
        return out + [(int(plan[1:]), False, rest)]
    out += [("chunk", rest + c * 2048) for c in range((rows - n_big * BIG) // 2048)]
    tail = end - start - rows
    if tail:
        out.append((ps.tail_bucket(tail), False, start + rows))
    return out


@pytest.mark.parametrize(
    "start, end",
    [(0, 4096), (0, 5000), (0, BIG + 6144), (0, BIG + 2048), (0, 2 * BIG + 4096), (1024, 1024 + BIG + 4000)],
)
def test_the_rows_after_the_big_blocks_take_the_cheapest_plan(monkeypatch, device_calls, start, end):
    m = _with_masked_big(monkeypatch, device_calls)
    _prefill(m, end, start=start)
    assert _replays(m, device_calls) == _expected(start, end)


def test_the_routing_cases_cover_every_plan():
    """The rows above: one masked 4096 replay of a whole 4096-token prompt, one masked 8192 replay of the 6144 rows
    after a big block, and 2048-row chunks and a tail where those cost more."""
    assert _expected(0, 4096) == [(4096, False, 0)]
    assert _expected(0, BIG + 6144) == [("big", 0), (8192, False, BIG)]
    assert _expected(0, 5000) == [("chunk", 0), ("chunk", 2048), (1024, False, 4096)]


def test_a_checkpoint_keeps_the_2048_row_chunks(monkeypatch, device_calls):
    """A checkpoint inside the rows after the big blocks keeps the 2048-row plan: the masked big traces have no
    checkpoint variant."""
    m = _with_masked_big(monkeypatch, device_calls)
    _prefill(m, 4096, checkpoint=3072)
    assert _replays(m, device_calls) == [("chunk", 0), (2048, 1024, 2048)]


def test_a_masked_big_replay_reads_the_big_sets_vision_buffers_and_feeds_the_chunk_hook(monkeypatch, device_calls):
    m = _with_masked_big(monkeypatch, device_calls)
    seen = []
    _prefill(m, BIG + 6144, on_chunk=lambda hidden, start, n: seen.append((hidden.name, start, n)))
    staged = [c[1] for c in device_calls if c[0] == "vision set"]
    assert staged == [BIG, BIG]  # the big chunk's, then the masked 8192 replay's
    out = m._masked_traces[8192].outputs[0][0].name
    assert seen == [("big hidden", 0, BIG), (out, BIG, 6144)]
    assert m._chunked_chunk_size == 2048
    assert not any(c[0] == "free" and getattr(c[1], "name", None) == out for c in device_calls)


def test_prepare_adds_the_big_buckets_on_the_big_set(monkeypatch, device_calls):
    """With a prepared big chunk set the masked traces also cover 4096 and 8192 rows, without checkpoint variants,
    warmed and captured while the big set (its vision-splice buffers) is active; without one they do not."""
    m, _ = _model(monkeypatch, device_calls)
    m._chunk_sets = {"big": dict(BIG_SET)}
    forward = m._forward_prefill_chunk_masked_tp
    m._forward_prefill_chunk_masked_tp = lambda *a, **kw: device_calls.append(
        ("set", a[4], m._chunked_chunk_size)
    ) or forward(*a, **kw)
    m.prepare_prefill_traces_masked()
    assert m.masked_big_buckets() == (4096, 8192)
    assert sorted(m._masked_traces[4096].checkpoints) == [] and sorted(m._masked_traces[8192].checkpoints) == []
    assert sorted(m._masked_traces[2048].checkpoints) == [1024]
    sets = {(c[1], c[2]) for c in device_calls if c[0] == "set"}
    assert sets == {(b, 2048) for b in (128, 256, 512, 1024, 2048)} | {(4096, BIG), (8192, BIG)}
    device_calls.clear()
    m.record_prefill_traces_masked()
    sets = {(c[1], c[2]) for c in device_calls if c[0] == "set"}
    assert {(4096, BIG), (8192, BIG)} <= sets and (2048, 2048) in sets
    assert m._chunked_chunk_size == 2048

    m, _ = _model(monkeypatch, device_calls)
    m.prepare_prefill_traces_masked()
    assert m.masked_big_buckets() == () and sorted(m._masked_traces) == [128, 256, 512, 1024, 2048]


def test_warm_masked_big_outputs_runs_each_big_bucket_once(monkeypatch, device_calls):
    m = _with_masked_big(monkeypatch, device_calls)
    seen = []
    m.warm_masked_big_outputs(lambda hidden: seen.append((hidden.shape[2], m._chunked_chunk_size)))
    assert seen == [(4096, BIG), (8192, BIG)]
    assert ("reset",) in device_calls and m._chunked_chunk_size == 2048
