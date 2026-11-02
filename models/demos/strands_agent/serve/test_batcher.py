# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only: batch collection and answer mapping of the server worker (no ttnn, no tokenizer)."""

import queue
from types import SimpleNamespace

import pytest

from models.demos.strands_agent.serve import batcher


def item(i, names=("q",), t=0.0, length=10):
    qs = [SimpleNamespace(name=n, input_ids=[1] * (length + len(n))) for n in names]
    return {"id": i, "qs": qs, "t_submit": t}


class Clock:
    """Fake time; get() advances it by the time it would block."""

    def __init__(self, arrivals):
        self.now = 0.0
        self.arrivals = list(arrivals)  # (arrival time, item)

    def clock(self):
        return self.now

    def get(self, timeout):
        if not self.arrivals:
            if timeout is None:
                raise AssertionError("would block forever")
            self.now += timeout
            raise queue.Empty
        at, it = self.arrivals[0]
        if timeout is not None and at > self.now + timeout:
            self.now += timeout
            raise queue.Empty
        self.now = max(self.now, at)
        self.arrivals.pop(0)
        return it


def collect(arrivals, max_wait_s=0.01, max_questions=4):
    c = Clock(arrivals)
    return batcher.collect_batch(c.get, max_wait_s, max_questions, clock=c.clock), c


def test_collect_stops_at_max_questions():
    arrivals = [(0.0, item(i, ("a", "b"))) for i in range(5)]
    (items, stop), c = collect(arrivals, max_questions=4)
    assert [x["id"] for x in items] == [0, 1] and not stop
    assert len(c.arrivals) == 3


def test_collect_stops_at_deadline():
    arrivals = [(0.0, item(0)), (0.004, item(1)), (0.02, item(2))]
    (items, stop), c = collect(arrivals, max_wait_s=0.01)
    assert [x["id"] for x in items] == [0, 1] and not stop
    assert c.now == pytest.approx(0.01)


def test_collect_waits_from_first_item_arrival():
    (items, _), _ = collect([(5.0, item(0)), (5.008, item(1)), (5.02, item(2))], max_wait_s=0.01)
    assert [x["id"] for x in items] == [0, 1]


def test_collect_stop_sentinel():
    (items, stop), _ = collect([(0.0, None)])
    assert items == [] and stop
    (items, stop), _ = collect([(0.0, item(0)), (0.001, None), (0.002, item(1))])
    assert [x["id"] for x in items] == [0] and stop


class FakeDecider:
    def __init__(self, fail=False):
        self.tokenizer = None
        self.config = SimpleNamespace(temperature_by_kind={}, temperature=1.0, ordinal_smoothing=0.0)
        self.fail = fail
        self.calls = []

    def question_logits_batch(self, flat):
        self.calls.append([q.name for q in flat])
        if self.fail:
            raise RuntimeError("device hung")
        return [(f"logits:{q.name}", None) for q in flat]


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(batcher, "build_answer", lambda q, logits, *rest: {"for": q.name, "logits": logits})


class Queued:
    """get() over a list: pops items, raises queue.Empty when empty; records the timeouts it was given."""

    def __init__(self, items):
        self.items = list(items)
        self.timeouts = []

    def get(self, timeout):
        self.timeouts.append(timeout)
        if not self.items:
            if timeout is None:
                raise AssertionError("would block forever")
            raise queue.Empty
        return self.items.pop(0)


def test_collect_zero_wait_drains_queued_items_without_waiting():
    q = Queued([item(i) for i in range(6)])
    items, stop = batcher.collect_batch(q.get, 0, 4)
    assert [x["id"] for x in items] == [0, 1, 2, 3] and not stop
    assert q.timeouts == [None, 0, 0, 0]
    q = Queued([item(i) for i in range(2)])
    items, _ = batcher.collect_batch(q.get, 0, 4)
    assert [x["id"] for x in items] == [0, 1]
    assert q.timeouts == [None, 0, 0]


def test_collect_after_deadline_still_drains_queued_items():
    late = Queued([item(i) for i in range(3)])
    clock = iter([0.0, 5.0, 5.0, 5.0, 5.0])  # every check after the first item is past the deadline
    items, _ = batcher.collect_batch(late.get, 0.5, 4, clock=lambda: next(clock))
    assert [x["id"] for x in items] == [0, 1, 2]
    assert late.timeouts == [None, 0, 0, 0]


def test_run_batch_one_device_call_in_item_order(fakes):
    decider = FakeDecider()
    results = batcher.run_batch(decider, [item(1, ("a",)), item(2, ("c", "d"))], 5)
    assert decider.calls == [["a", "c", "d"]]
    assert [r["id"] for r in results] == [1, 2] and all(r["ok"] for r in results)
    timing = results[0]["timing"]
    assert timing["batch_requests"] == 2 and timing["batch_questions"] == 3
    assert {"batch_id", "device_ms", "t_batch_start"} <= set(timing)


def test_run_batch_passes_batch_id_through(fakes):
    results = batcher.run_batch(FakeDecider(), [item(1), item(2)], 42)
    assert results[0]["timing"]["batch_id"] == results[1]["timing"]["batch_id"] == 42


def test_run_batch_device_error_fails_all_items(fakes):
    results = batcher.run_batch(FakeDecider(fail=True), [item(1), item(3)], 1)
    assert [(r["id"], r["status"]) for r in results] == [(1, 500), (3, 500)]
    assert "device hung" in results[0]["detail"]


def test_run_batch_maps_results_to_requests(fakes):
    results = batcher.run_batch(FakeDecider(), [item(7, ("x", "y")), item(9, ("z",))], 1)
    assert list(results[0]["result"]["answers"]) == ["x", "y"]
    assert results[0]["result"]["answers"]["y"] == {"for": "y", "logits": "logits:y"}
    assert results[1]["result"]["answers"] == {"z": {"for": "z", "logits": "logits:z"}}
    assert results[0]["result"]["usage"] == {"input_tokens": 2 * 10 + 2, "output_tokens": 2}
    assert results[1]["result"]["usage"] == {"input_tokens": 11, "output_tokens": 1}
