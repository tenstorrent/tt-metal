# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host only (no device, no model): requests that find no slot wait instead of getting a 429, and the overhead table.

* ``test_queued_not_rejected`` -- more requests than slots: every one is served, first come first served.
* ``test_waiter_gives_up`` -- a queued request whose client has gone leaves the queue.
* ``test_blocked_head_lets_others_try`` -- a waiter whose own slot is busy does not hold up those behind it.
* ``test_overhead_table`` -- stages by kind, nesting and a loop's untracked time.
* ``test_tokenize_memo_matches_the_tokenizer`` / ``test_incremental_detokenize`` -- with the checkpoint's tokenizer.

    pytest models/experimental/deepseek_v41_flash/tests/test_request_queue.py
"""

import os
import random
import threading
import time
from types import SimpleNamespace

import pytest

from models.experimental.deepseek_v4_flash.demo import overhead
from models.experimental.deepseek_v4_flash.demo import server
from models.experimental.deepseek_v4_flash.demo.server import RequestError, _ClaimQueue, _ClientGone


class _Slots:
    def __init__(self, n: int):
        self.free = n
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            if not self.free:
                raise RequestError(429, "no slot", "rate_limit_error")
            self.free -= 1

    def give(self):
        with self.lock:
            self.free += 1


def test_queued_not_rejected():
    queue, slots = _ClaimQueue(), _Slots(2)
    served, errors = [], []
    order_lock = threading.Lock()

    def request(i: int):
        def attempt(claimed):
            slots.take()
            claimed()
            with order_lock:
                served.append(i)
            time.sleep(0.05)
            return i

        try:
            queue.run(attempt)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            slots.give() if i in served else None
            queue.changed()

    threads = []
    for i in range(10):
        threads.append(threading.Thread(target=request, args=(i,)))
        threads[-1].start()
        time.sleep(0.005)  # arrival order
    for t in threads:
        t.join(timeout=10)
    assert not errors and sorted(served) == list(range(10)) and len(queue) == 0
    # The two that found a slot came first; the queued ones were served in their arrival order.
    assert served[2:] == sorted(served[2:])


def test_waiter_gives_up(expect_error):
    queue, slots = _ClaimQueue(), _Slots(0)
    gone = threading.Event()
    threading.Timer(0.2, gone.set).start()

    def attempt(claimed):
        slots.take()
        claimed()

    with expect_error(_ClientGone, ""):
        queue.run(attempt, gone.is_set)
    assert len(queue) == 0


def test_blocked_head_lets_others_try():
    queue = _ClaimQueue()
    first_in = threading.Event()
    result = {}

    def blocked(claimed):  # a conversation whose own instance is busy for the whole test
        first_in.set()
        raise RequestError(429, "its instance is busy", "rate_limit_error")

    stop = threading.Event()
    head = threading.Thread(target=lambda: _quietly(queue.run, blocked, stop.is_set))
    head.start()
    first_in.wait(5)
    time.sleep(0.05)

    def other(claimed):
        claimed()
        return "served"

    # Joins behind the head; once the head has tried and failed, it may try.
    result["other"] = queue.run(other)
    stop.set()
    head.join(5)
    assert result["other"] == "served" and len(queue) == 0


def _quietly(fn, *args):
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        pass


def test_overhead_table():
    o = overhead.Overhead(period=3600)
    with o.loop_iteration("sched"):
        with o.time("http.sse_write"):
            with o.time("compute.inner"):  # nested: in the table, not twice against the loop
                time.sleep(0.01)
        o.add("wait.queue_for_slot", 5.0, own_thread=False)
        o.add("idle.scheduler", 0.02, own_thread=False)  # (not spent here: not against the loop)
        time.sleep(0.02)  # the loop's own
    wall, stats = o.take()
    assert {"http.sse_write", "compute.inner", "wait.queue_for_slot", "idle.scheduler", "sched.untracked"} <= set(stats)
    untracked = stats["sched.untracked"][1]
    assert 0.015 < untracked < 0.2, untracked
    o.add("compute.verify_read", 0.5, count=10)
    o.add("http.recv_body", 0.01)
    text = o.report()
    assert "-- overhead" in text and "-- compute" in text and "compute.verify_read" in text
    assert o.report() is None  # a new window: nothing recorded since
    assert [overhead.kind(n) for n in ("compute.x", "idle.x", "wait.x", "http.x")] == [
        "compute",
        "idle",
        "wait",
        "overhead",
    ]


def _tokenizer():
    import glob

    snapshots = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/*")
    )
    if not snapshots:
        pytest.skip("no V4.1 checkpoint (tokenizer) here")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshots[0])


def test_tokenize_memo_matches_the_tokenizer():
    """Texts sharing long starts (a conversation rendered again, requests sharing a document) reuse earlier ids."""
    tok = _tokenizer()
    source = open(server.__file__).read()
    base = (source * 8)[:1_200_000]
    rng = random.Random(1)
    for _ in range(6):
        cut = rng.randrange(200_000, len(base))
        tail = "".join(rng.choice(["Hello world ", "日本語 ", "x=1;\n", " foo bar", "ab cd"]) for _ in range(500))
        text = base[:cut] + tail
        assert server._tokenize(tok, text) == list(tok(text, add_special_tokens=False)["input_ids"])


def test_incremental_detokenize():
    """The streamer decodes only past its kept prefix, and close() with the whole reply flushes the rest."""
    tok = _tokenizer()
    fmt = SimpleNamespace(calls_open="<｜DSML｜function_calls>", calls_close="</｜DSML｜function_calls>")
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog.", "😀", "日本語", "é", "x=1;\n"]
    text = " ".join(random.Random(4).choice(words) for _ in range(3000))  # (no think tags: all of it is content)
    ids = tok.encode(text, add_special_tokens=False)
    out = []
    stream = server._Streamer(tok, lambda r, c: out.append(c), fmt)
    rng = random.Random(2)
    k = 0
    while k < len(ids) - 50:
        k = min(len(ids), k + rng.randrange(1, 6))
        stream.push(ids[:k])
    stream.close(ids)  # (the last pushes never came: close decodes them)
    stream.push(ids)  # after close: nothing
    assert "".join(out) == stream.content == tok.decode(ids, skip_special_tokens=False)
