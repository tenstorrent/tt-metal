# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host only (no device, no model): prefill ahead -- the prompts of requests waiting for a decode slot prefilled on the
disaggregated prefill server meanwhile, into its prefix cache.

* ``test_device_queue_order`` -- the prefill server serves real prefills before warm-ups, each kind in arrival order.
* ``test_warm_yields_to_a_prefill_and_resumes`` -- a warm-up stops after a chunk when a prefill comes in; that prefill
  takes what it computed from the cache, and the warm-up sent again continues from there.
* ``test_claim_queue_on_queued`` -- the queue's hook runs for a request that waits, and its leave() once it has a slot.
* ``test_prefill_ahead_order_and_depth`` -- the decode side's worker: oldest first, ``depth`` ahead, a yielded warm-up
  sent again, a request that left never sent, a prompt kept already not sent twice.
* ``test_ahead_prompt_matches_admission`` -- the predicted prompt is what admission renders, for a new conversation and
  for one continued from host RAM.

    pytest models/experimental/deepseek_v41_flash/tests/test_prefill_ahead.py
"""

import socket
import threading
import time
from types import SimpleNamespace

import torch

from models.experimental.deepseek_v4_flash.demo.server import RequestError, _ClaimQueue
from models.experimental.deepseek_v41_flash.demo import prefill_server as ps
from models.experimental.deepseek_v41_flash.demo import server as srv
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import recv_msg


def _until(cond, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


# -- prefill server ------------------------------------------------------------------------------ #
def test_device_queue_order():
    device = ps.DeviceQueue()
    order, started = [], threading.Event()
    release = threading.Event()

    def holder():
        with device.hold(warm=False):
            started.set()
            release.wait(5)
            order.append("first")

    def request(name: str, warm: bool):
        with device.hold(warm=warm):
            order.append(name)

    threads = [threading.Thread(target=holder)]
    threads[0].start()
    started.wait(5)
    assert not device.urgent()
    for name, warm in (("warm1", True), ("real1", False), ("warm2", True), ("real2", False)):
        threads.append(threading.Thread(target=request, args=(name, warm)))
        threads[-1].start()
        _until(lambda: len(device._waiting) == len(threads) - 1)
    assert device.urgent()
    release.set()
    for t in threads:
        t.join(5)
    assert order == ["first", "real1", "real2", "warm1", "warm2"]
    assert not device.urgent()


class _FakeDP:
    """DPPrefill's host-visible behaviour: chunks on a grid, slots that keep their prompt's end, ``resume_start`` only
    from (near) that end, ``stop`` asked after every chunk but the last."""

    def __init__(self, chunk: int = 128, max_seq: int = 4096, slots: int = 4, chunk_seconds: float = 0.03):
        self.chunk, self.max_seq, self.chunk_seconds = chunk, max_seq, chunk_seconds
        self.slots = [object() for _ in range(slots)]
        self.ends = [0] * slots
        self.layers = [0]
        self.cfg = SimpleNamespace(vocab_size=1 << 20, ratios=[0], window=128, is_kv_source=lambda i: False)
        self.mesh_device = SimpleNamespace(shape=(8, 4))
        self.runs: list[tuple[int, int, int]] = []  # (slot, start, end) of every prefill
        self.first_chunk = threading.Event()

    def resume_start(self, slot: int, held: int, end: int) -> int:
        r = min(held, end) // 32 * 32
        return r if r > 0 and self.ends[slot] - 64 <= r <= self.ends[slot] else 0

    def prefill(self, ids, on_chunk=None, slot=0, start=0, times=None, sync_chunks=False, stop=None) -> int:
        n, pos = len(ids), start
        while pos < n:
            end = min(n, (pos // self.chunk + 1) * self.chunk)
            time.sleep(self.chunk_seconds)
            self.ends[slot] = end
            self.first_chunk.set()
            if times is not None:
                times["chunks"] = times.get("chunks", 0) + 1
            if on_chunk is not None:
                on_chunk(pos, end, self.chunk_seconds)
            pos = end
            if pos < n and stop is not None and stop():
                break
        self.runs.append((slot, start, pos))
        return pos

    def host_states(self, n, parts=None, slot=0, base=0) -> dict:
        return {"seq_len": n, "base": base, "layers": {}}


def _final(sock: socket.socket) -> dict:
    while True:
        header, _ = recv_msg(sock)
        if "progress" not in header:
            return header


def test_warm_yields_to_a_prefill_and_resumes():
    d_p = _FakeDP()
    service = ps.PrefillService(d_p)
    assert service.info["warm"] is True
    ids = torch.arange(1, 1025, dtype=torch.int32)  # 8 chunks
    warm_conn, warm_peer = socket.socketpair()
    real_conn, real_peer = socket.socketpair()
    with warm_conn, warm_peer, real_conn, real_peer:
        warm = threading.Thread(target=service.prefill, args=(warm_conn, ids, 0, True, "warm"), kwargs={"warm": True})
        warm.start()
        d_p.first_chunk.wait(5)
        # The request that has its decode slot now: the same prompt, a real prefill.
        real = threading.Thread(target=service.prefill, args=(real_conn, ids, 0, True, "real"))
        real.start()
        warm_reply = _final(warm_peer)
        real_reply = _final(real_peer)
        warm.join(5)
        real.join(5)
        done = warm_reply["done"]
        assert 0 < done < ids.numel() and done % d_p.chunk == 0, warm_reply
        assert warm_reply["cached"] == 0
        # The prefill continued from where the warm-up stopped.
        assert real_reply["cached"] == done and real_reply["seq_len"] == ids.numel()
        assert d_p.runs[-1][1:] == (done, ids.numel())
        # The warm-up sent again finds the whole prompt kept: nothing computed.
        runs = len(d_p.runs)
        service.prefill(warm_conn, ids, 0, True, "warm", warm=True)
        again = _final(warm_peer)
        assert again["done"] == again["cached"] == ids.numel() and len(d_p.runs) == runs
    stats = service.warm_totals.snapshot()
    assert stats["requests"] == 2 and stats["tokens"] == done
    assert service.totals.snapshot()["tokens"] == ids.numel() - done


def test_warm_alone_runs_to_the_end(expect_error):
    d_p = _FakeDP(chunk_seconds=0.0)
    service = ps.PrefillService(d_p)
    ids = torch.arange(1, 641, dtype=torch.int32)
    conn, peer = socket.socketpair()
    with conn, peer:
        service.prefill(conn, ids, 0, True, "warm", warm=True)
        reply = _final(peer)
    assert reply["done"] == ids.numel() and reply["cached"] == 0
    with expect_error(ValueError, "a warm-up takes a text prompt only"):
        service.prefill(None, ids, 0, True, "warm", targets={"seq_len": 640, "start": 0}, warm=True)


# -- decode server ------------------------------------------------------------------------------- #
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


def test_claim_queue_on_queued():
    queue, slots = _ClaimQueue(), _Slots(1)
    events: list[str] = []
    hold = threading.Event()

    def hook(name: str):
        def on_queued():
            events.append(f"queued {name}")
            return lambda: events.append(f"left {name}")

        return on_queued

    def request(name: str, wait: bool):
        def attempt(claimed):
            slots.take()
            claimed()
            events.append(f"claimed {name}")
            if wait:
                hold.wait(5)
            return name

        try:
            queue.run(attempt, on_queued=hook(name))
        finally:
            slots.give()
            queue.changed()

    first = threading.Thread(target=request, args=("a", True))
    first.start()
    _until(lambda: "claimed a" in events)
    second = threading.Thread(target=request, args=("b", False))
    second.start()
    _until(lambda: "queued b" in events)
    hold.set()
    first.join(5)
    second.join(5)
    assert events == ["claimed a", "queued b", "left b", "claimed b"]


class _FakePrefiller:
    """What _PrefillAhead uses of a _RemotePrefiller. The first warm-up of a 512-token prompt yields halfway."""

    info = {"cache_slots": 4}
    address = "fake:9041"

    def __init__(self):
        self.warmed = srv._WarmSet()
        self.calls: list[int] = []
        self.gate = threading.Event()
        self.gate.set()

    def warm_span(self, ids):
        return list(ids[: len(ids) // 128 * 128])

    def is_warm(self, ids):
        return ids in self.warmed

    def warm(self, ids):
        self.gate.wait(5)
        n = len(ids)
        first = n not in self.calls
        self.calls.append(n)
        done = n // 2 if n == 512 and first else n
        if done >= n:
            self.warmed.add(ids)
        return {"done": done, "cached": n // 2 if n == 512 and not first else 0, "seconds": 0.01}


def test_prefill_ahead_order_and_depth():
    prefiller = _FakePrefiller()
    ahead = srv._PrefillAhead(prefiller, depth=8)
    try:
        assert ahead.depth == 2  # half the server's 4 cache slots
        prefiller.gate.clear()
        leave_a = ahead.add(lambda: list(range(512)), "a")
        leave_b = ahead.add(lambda: list(range(256 + 50)), "b")
        leave_c = ahead.add(lambda: list(range(384)), "c")
        prefiller.gate.set()
        _until(lambda: prefiller.calls == [512, 512, 256])  # a yielded once and went again, then b
        time.sleep(0.1)
        assert 384 not in prefiller.calls  # c is third: not while a and b still wait
        leave_a()
        _until(lambda: 384 in prefiller.calls)

        # A request that leaves before its turn is never sent.
        predicted = []
        prefiller.gate.clear()
        leave_d = ahead.add(lambda: predicted.append("d") or list(range(640)), "d")
        leave_e = ahead.add(lambda: predicted.append("e") or list(range(768)), "e")
        leave_b()
        leave_c()
        _until(lambda: predicted == ["d"])
        leave_e()
        prefiller.gate.set()
        _until(lambda: 640 in prefiller.calls)
        time.sleep(0.1)
        assert predicted == ["d"] and 768 not in prefiller.calls
        leave_d()

        # A prompt the server keeps already is not sent again.
        calls = len(prefiller.calls)
        leave_f = ahead.add(lambda: list(range(256 + 7)), "f")
        _until(lambda: ahead.snapshot()["skipped"] == 1)  # (e left before its turn: never picked)
        assert len(prefiller.calls) == calls
        leave_f()
        snap = ahead.snapshot()
        assert snap["waiting"] == 0 and snap["warmed"] == 4 and snap["failed"] == 0
    finally:
        ahead.close()


class _CharTokenizer:
    """One id per character (a stand-in for the checkpoint's tokenizer)."""

    bos_token_id = 0

    def get_added_vocab(self):
        return {}

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [[ord(c) for c in t] for t in text]}
        return {"input_ids": [ord(c) for c in text]}


def _fake_server(prefill_rate_tokens: int = 128):
    engine = SimpleNamespace(
        chat_format=srv.V41_CHAT_FORMAT,
        tokenizer=_CharTokenizer(),
        eos_id=1,
        image_token_id=-1,
        images=None,
        users=[],
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=1 << 20)),
    )
    prefiller = object.__new__(srv._RemotePrefiller)
    prefiller.align, prefiller.window, prefiller.decode_window, prefiller.max_len = 128, 128, True, 1 << 20
    prefiller.fill_seconds, prefiller.chunk_seconds, prefiller.chunk_size = 0.5, 0.15, 5120
    prefiller.move_seconds_per_token = 1.5e-6
    prefiller.can_warm, prefiller.warmed = True, srv._WarmSet()
    server = object.__new__(srv.V41GenerationServer)
    server.engine, server.prefiller = engine, prefiller
    server.pool = SimpleNamespace(_cv=threading.Condition(), _busy=[], device_parked=[], host=[])
    server.scheduler = SimpleNamespace(replay_rate=100.0)  # tokens a second through decode
    server.thinking_mode, server.reasoning_effort = "chat", None
    return server


def test_ahead_prompt_matches_admission():
    server = _fake_server()
    doc = "lorem ipsum dolor sit amet " * 200
    body = {
        "messages": server.normalized_messages(
            {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": doc}]}
        ),
        "thinking": {"type": "enabled"},
    }
    predicted = server.ahead_prompt(body)

    # Admission of a new conversation (_Scheduler._admit): the server's defaults, the request's options, every
    # message rendered from 0.
    user = SimpleNamespace(
        engine=server.engine,
        messages=list(body["messages"][:-1]) + [body["messages"][-1]],
        _next_render=0,
        pending_id=None,
        thinking_mode=server.thinking_mode,
        reasoning_effort=server.reasoning_effort,
    )
    server._apply_options(user, body)
    assert user.thinking_mode == "thinking"
    expected = server._render_ids(user, include_assistant=True)
    assert predicted == expected
    span = server.prefiller.warm_span(predicted)
    assert len(span) == server.prefiller.aligned(len(expected)) > 0

    # The same conversation, later, continued from host RAM: its tokens, then the new turn as a continuation.
    reply = "Sure."
    held = list(body["messages"]) + [{"role": "assistant", "content": reply}]
    tokens = expected + [ord(c) for c in reply]
    server.pool.host.append(
        SimpleNamespace(
            messages=list(held),
            tokens=list(tokens),
            pending_id=1,
            next_render=len(held),
            thinking_mode="thinking",
            reasoning_effort=None,
        )
    )
    follow_up = {
        "messages": server.normalized_messages(
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": doc},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "And now in detail: " + doc},
                ]
            }
        )
    }
    predicted = server.ahead_prompt(follow_up)
    user = SimpleNamespace(
        engine=server.engine,
        messages=list(held),
        _next_render=len(held),
        pending_id=1,
        thinking_mode="thinking",
        reasoning_effort=None,
    )
    server._apply_options(user, follow_up)
    assert srv._continues(follow_up["messages"], user.messages)  # (_sync_messages: a continuation)
    user.messages.extend(follow_up["messages"][len(user.messages) : -1])
    user.messages.append(follow_up["messages"][-1])
    assert predicted == tokens + server._render_ids(user, include_assistant=False)

    # A short follow-up is appended through decode: nothing to prefill ahead.
    short = {"messages": follow_up["messages"][:3] + [{"role": "user", "content": "ok"}]}
    assert server.ahead_prompt({"messages": server.normalized_messages(short)}) is None
