# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only: batch collection and result split of the rerank server worker (no ttnn, no tokenizer).
Fake clock as in atupe/strands-decider-fastapi serve/test_batcher.py. Run: pytest -q test_rerank_batcher.py"""

import os
import queue
import sys

import pytest

# the server runs as a script from this folder; tt-metal's pytest does not put the test's folder on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rerank_batcher as batcher  # noqa: E402


def item(i, pairs=1, length=5):
    return {"id": i, "ids": [[i + 1] * length for _ in range(pairs)], "t_submit": 0.0}


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


def collect(arrivals, max_wait_s=0.01, max_pairs=4, carry=None):
    c = Clock(arrivals)
    return batcher.collect_batch(c.get, max_wait_s, max_pairs, carry=carry, clock=c.clock), c


def test_hard_cap_carries_the_item_that_does_not_fit():
    arrivals = [(0.0, item(0, 2)), (0.0, item(1, 1)), (0.0, item(2, 2)), (0.0, item(3, 1))]
    (items, stop, carry), c = collect(arrivals, max_pairs=4)
    assert [x["id"] for x in items] == [0, 1] and not stop
    assert carry["id"] == 2 and len(c.arrivals) == 1


def test_carry_starts_the_next_batch():
    (items, stop, carry), _ = collect([(0.0, item(3, 1))], carry=item(2, 2), max_pairs=4)
    assert [x["id"] for x in items] == [2, 3] and carry is None and not stop


def test_oversized_item_runs_alone():
    (items, stop, carry), _ = collect([(0.0, item(0, 40)), (0.0, item(1, 1))], max_pairs=32)
    assert [x["id"] for x in items] == [0] and carry is None and not stop


def test_stops_at_deadline():
    arrivals = [(0.0, item(0)), (0.004, item(1)), (0.02, item(2))]
    (items, stop, carry), c = collect(arrivals, max_wait_s=0.01)
    assert [x["id"] for x in items] == [0, 1] and not stop and carry is None
    assert c.now == pytest.approx(0.01)


def test_zero_wait_takes_only_queued():
    (items, _, _), c = collect([(0.0, item(0)), (0.0, item(1)), (0.5, item(2))], max_wait_s=0.0)
    assert [x["id"] for x in items] == [0, 1] and len(c.arrivals) == 1


def test_stop_sentinel():
    (items, stop, _), _ = collect([(0.0, None)])
    assert items == [] and stop
    (items, stop, _), _ = collect([(0.0, item(0)), (0.0, None)])
    assert [x["id"] for x in items] == [0] and stop


def test_run_batch_splits_logits_in_order():
    items = [item(0, 2), item(1, 3)]
    res = batcher.run_batch(lambda flat: [float(ids[0]) * 10 + k for k, ids in enumerate(flat)], items, 7)
    assert [r["id"] for r in res] == [0, 1]
    assert res[0]["logits"] == [10.0, 11.0] and res[1]["logits"] == [22.0, 23.0, 24.0]
    assert res[0]["timing"]["batch_pairs"] == 5 and res[0]["timing"]["batch_tasks"] == 2


def test_run_batch_error_fails_every_item():
    def boom(flat):
        raise RuntimeError("device")

    res = batcher.run_batch(boom, [item(0), item(1)], 1)
    assert all(not r["ok"] and r["status"] == 500 for r in res)


def test_run_batch_wrong_logit_count_fails():
    res = batcher.run_batch(lambda flat: [0.0], [item(0, 2)], 1)
    assert not res[0]["ok"] and "1 logits for 2 pairs" in res[0]["detail"]
