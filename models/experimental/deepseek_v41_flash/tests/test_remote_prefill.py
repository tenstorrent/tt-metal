# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only unit tests for disaggregated TCP prefill (``d_p_wire`` and ``_RemotePrefiller``).
Opens no device."""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from types import SimpleNamespace

import torch

from models.experimental.deepseek_v41_flash.demo.server import _PromptStopped, _RemotePrefiller
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import (
    VERSION,
    PrefillServerError,
    recv_msg,
    send_msg,
    send_states,
)


class _MockPrefillServer:
    """Minimal loopback TCP server speaking the ``d_p_wire`` protocol."""

    def __init__(
        self,
        *,
        layers: list[int] | None = None,
        kv_sources: dict[str, int] | None = None,
        window: int = 128,
        align: int = 128,
        chunk: int = 256,
        max_seq: int = 4096,
        vocab: int = 1000,
        hold: threading.Event | None = None,
        refuse_have: bool = False,
        push_error: str | None = None,
    ):
        self.info = {
            "version": VERSION,
            "max_seq": max_seq,
            "chunk": chunk,
            "align": align,
            "layers": layers if layers is not None else [0, 1, 2],
            "kv_sources": kv_sources if kv_sources is not None else {"2": 2},
            "window": window,
            "vocab": vocab,
            "mesh": "mock",
            "cache_slots": 4,
        }
        self.hold = hold  # a prefill replies only once this is set
        self.refuse_have = refuse_have  # answer any have > 0 with the server's have error
        self.push_error = push_error  # answer a prefill_push with this error, after its progress
        self.pushes: list[dict] = []  # the targets of each prefill_push
        self.prefill_calls: list[list[int]] = []
        self.prefill_haves: list[int] = []
        self.kept: list[list[int]] = []  # the prompts its prefix cache holds
        self.connections = 0
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self.host, self.port = self._sock.getsockname()
        self.address = f"{self.host}:{self.port}"
        self.url = f"tcp://{self.address}"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connections += 1
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            while not self._stop.is_set():
                try:
                    header, blobs = recv_msg(conn)
                    op = header.get("op")
                    if op == "ping":
                        send_msg(conn, {"pong": True})
                    elif op == "info":
                        send_msg(conn, self.info)
                    elif op == "prefill_push":
                        ids = torch.frombuffer(blobs[0], dtype=torch.int32).tolist()
                        self.prefill_calls.append(ids)
                        have = int(header.get("have", 0))
                        self.prefill_haves.append(have)
                        self.pushes.append(header["targets"])
                        if self.hold is not None:
                            self.hold.wait()
                        n = len(ids)
                        for end in range(self.info["chunk"], n + 1, self.info["chunk"]):
                            send_msg(conn, {"progress": end})
                        if self.push_error:
                            send_msg(conn, {"error": self.push_error})
                            continue
                        reply = {"seq_len": n, "base": have, "cached": 0, "seconds": 0.05, "queued": 0.01}
                        send_msg(conn, {**reply, "bytes": 4096, "pages": 4, "push_seconds": 0.01})
                    elif op == "prefill":
                        ids = torch.frombuffer(blobs[0], dtype=torch.int32).tolist()
                        self.prefill_calls.append(ids)
                        have = int(header.get("have", 0))
                        self.prefill_haves.append(have)
                        if self.hold is not None:
                            self.hold.wait()
                        n = len(ids)
                        if n <= 0 or n % self.info["align"] != 0 or n > self.info["max_seq"]:
                            send_msg(conn, {"error": f"bad length {n}"})
                            continue
                        if any(t < 0 or t >= self.info["vocab"] for t in ids):
                            send_msg(conn, {"error": "token out of vocab"})
                            continue
                        if have % self.info["align"] or have > n or (self.refuse_have and have):
                            send_msg(conn, {"error": f"have {have}: a multiple of 128 up to the prompt's {n} tokens"})
                            continue
                        cached = 0
                        if header.get("cache", True):
                            cached = max((len(k) for k in self.kept if ids[: len(k)] == k), default=0)
                        self.kept.append(ids)
                        chunk = self.info["chunk"]
                        for end in range(chunk, n + 1, chunk):
                            send_msg(conn, {"progress": end})
                        if n % chunk:
                            send_msg(conn, {"progress": n})
                        layers_out = {}
                        w = self.info["window"]
                        for li in self.info["layers"]:
                            entry = {"kv_tail": torch.full((w, 16), float(li + 1), dtype=torch.bfloat16)}
                            ratio = self.info["kv_sources"].get(str(li))
                            if ratio:
                                e = (n - have) // int(ratio)
                                entry["compressed_kv"] = torch.full((e, 16), float(li + 10), dtype=torch.bfloat16)
                                entry["idx_keys"] = torch.full((e, 8), float(li + 20), dtype=torch.bfloat16)
                            layers_out[li] = entry
                        send_states(
                            conn,
                            {"seq_len": n, "layers": layers_out},
                            seconds=0.0 if cached == n else 0.05,
                            queued=0.01,
                            base=have,
                            cached=cached,
                        )
                    else:
                        send_msg(conn, {"error": f"unknown op {op!r}"})
                except (ConnectionError, OSError):
                    return

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=2)

    def __enter__(self) -> "_MockPrefillServer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _fake_engine(num_layers: int = 4, dspark: bool = False, max_seq: int = 2048):
    committed = []

    class _FakeModel:
        sliding_window = 128
        config = SimpleNamespace(
            kv_source_layer_ids=[2],
            compress_ratios={2: 2},
            vocab_size=1000,
        )

        def __init__(self):
            self.num_layers = num_layers
            self.resets = []
            self.starts = []
            self.static_resets = 0
            self.in_flight = {}  # session -> (start, total, ring layers) of its remote commit
            self.finished = []
            self.aborted = []

        def reset_session(self, sid: int) -> None:
            self.resets.append(sid)

        def reset_static_caches(self) -> None:
            self.static_resets += 1

        def commit_host_prefill_state(self, host_states: dict, session_id: int, ids=None, start: int = 0) -> int:
            assert host_states["base"] == start
            committed.append((session_id, host_states, list(ids) if ids is not None else None))
            self.starts.append(start)
            return int(host_states["seq_len"])

        def remote_commit_targets(self, session_id: int, total: int, ring_layers, start: int = 0) -> dict:
            assert session_id not in self.in_flight
            self.in_flight[session_id] = (start, total, list(ring_layers))
            return {"session": session_id, "seq_len": total, "start": start}

        def finish_remote_commit(self, session_id: int, ids=None) -> int:
            start, total, _ = self.in_flight.pop(session_id)
            self.finished.append((session_id, start, list(ids) if ids is not None else None))
            return total

        def abort_remote_commit(self, session_id: int) -> None:
            del self.in_flight[session_id]
            self.aborted.append(session_id)
            self.reset_session(session_id)

    engine = SimpleNamespace(
        model=_FakeModel(),
        max_seq=max_seq,
        dspark=dspark,
        _replay_q=deque(),
    )
    return engine, committed


def test_remote_prefiller_aligned_and_run():
    # KV-only server (layers 0..2 of a 4-layer decode model) leaves sliding_window (128) tokens for decode.
    with _MockPrefillServer(layers=[0, 1, 2], max_seq=4096) as srv:
        engine, committed = _fake_engine(num_layers=4, dspark=False, max_seq=2048)
        prefiller = _RemotePrefiller(engine, srv.url)
        try:
            assert prefiller.kv_only is True
            assert prefiller.decode_window is True
            assert prefiller.aligned(128) == 0
            assert prefiller.aligned(256) == 128
            assert prefiller.aligned(500) == 256
            # A session continues from any position on the 128 grid; a window replay is not offered.
            assert prefiller.continue_alignment == 128
            assert prefiller.aligned(500, start=128) == 256
            assert prefiller.aligned(500, start=64) == 0
            assert prefiller.aligned(500, bounded=True) == 0

            user = SimpleNamespace(index=0, sid=7, pos=0, tokens=[], activated=False)
            user.activate = lambda: setattr(user, "activated", True)
            user.reset = lambda: setattr(user, "pos", 0)
            seen = []
            out = prefiller.run(user, list(range(256)), on_progress=seen.append)
            assert out is None
            assert user.activated
            assert user.pos == 256
            assert seen == [256]
            assert engine.model.resets == [7]
            assert engine.model.static_resets == 1
            assert len(committed) == 1
            sid, states, ids = committed[0]
            assert sid == 7
            assert states["seq_len"] == 256
            assert ids == list(range(256))
        finally:
            prefiller.close()

    # Full-layer server without DSpark leaves at least 1 token for decode to produce logits.
    with _MockPrefillServer(layers=[0, 1, 2, 3]) as srv:
        engine, _ = _fake_engine(num_layers=4, dspark=False, max_seq=2048)
        prefiller = _RemotePrefiller(engine, srv.address)
        try:
            assert prefiller.kv_only is False
            assert prefiller.decode_window is False
            assert prefiller.aligned(128) == 0
            assert prefiller.aligned(129) == 128
        finally:
            prefiller.close()

        # With DSpark, even a full-layer server leaves sliding_window (128) tokens for drafter rings.
        engine_spec, _ = _fake_engine(num_layers=4, dspark=True, max_seq=2048)
        prefiller_spec = _RemotePrefiller(engine_spec, srv.url)
        try:
            assert prefiller_spec.decode_window is True
            assert prefiller_spec.aligned(129) == 0
            assert prefiller_spec.aligned(256) == 128
        finally:
            prefiller_spec.close()


def test_remote_prefiller_refused_have_resends_from_zero():
    with _MockPrefillServer(refuse_have=True) as srv:
        engine, _ = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url)
        try:
            user = SimpleNamespace(index=0, sid=2, pos=256, tokens=list(range(256)))
            user.activate = lambda: None
            user.reset = lambda: setattr(user, "pos", 0)
            prefiller.run(user, list(range(256, 512)))
            assert srv.prefill_haves == [256, 0]
            # Every entry came back: the session is rewound and committed from 0.
            assert engine.model.resets == [2]
            assert engine.model.starts == [0]
            assert user.pos == 512
        finally:
            prefiller.close()


def _push_user(sid: int, pos: int = 0):
    user = SimpleNamespace(index=0, sid=sid, pos=pos, tokens=list(range(pos)), resets=0)
    user.activate = lambda: None

    def reset():
        user.pos = 0
        user.resets += 1

    user.reset = reset
    return user


def test_remote_prefiller_push_reserves_pushes_and_finishes():
    with _MockPrefillServer(layers=[0, 1, 2], chunk=128) as srv:
        engine, committed = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url, push=True)
        try:
            assert prefiller.ring_layers == [0, 1, 2]
            user = _push_user(sid=5)
            seen = []
            prefiller.run(user, list(range(256)), on_progress=seen.append)
            assert seen == [128, 256]
            assert srv.pushes == [{"session": 5, "seq_len": 256, "start": 0}]
            assert engine.model.resets == [5] and engine.model.static_resets == 1  # rewound before the reservation
            assert engine.model.finished == [(5, 0, list(range(256)))]
            assert not committed and not engine.model.in_flight
            assert user.pos == 256

            # A continued session keeps its entries: the push starts at its position, no rewind.
            user.tokens = list(range(256))
            prefiller.run(user, list(range(256, 512)))
            assert srv.pushes[-1] == {"session": 5, "seq_len": 512, "start": 256}
            assert srv.prefill_haves[-1] == 256 and engine.model.resets == [5]
            assert engine.model.finished[-1] == (5, 256, list(range(512)))
            assert user.pos == 512
        finally:
            prefiller.close()


def test_remote_prefiller_push_error_aborts_the_commit():
    with _MockPrefillServer(push_error="RuntimeError: no fabric route") as srv:
        engine, _ = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url, push=True)
        try:
            user = _push_user(sid=3)
            try:
                prefiller.run(user, list(range(256)))
            except PrefillServerError as exc:
                assert "no fabric route" in str(exc)
            else:
                raise AssertionError("the error reply did not raise")
            assert engine.model.aborted == [3] and not engine.model.in_flight and not engine.model.finished
            assert user.resets == 1 and user.pos == 0
        finally:
            prefiller.close()


def test_remote_prefiller_push_holds_a_cancellation_until_the_reply():
    with _MockPrefillServer(chunk=128) as srv:
        engine, _ = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url, push=True)
        try:
            user = _push_user(sid=4)
            seen = []

            def progress(end: int) -> None:
                seen.append(end)
                raise _PromptStopped()

            try:
                prefiller.run(user, list(range(384)), on_progress=progress)
            except _PromptStopped:
                pass
            else:
                raise AssertionError("the cancellation was dropped")
            # Asked once, then the reply was waited for (its writes may land until then) before giving the pages up.
            assert seen == [128]
            assert engine.model.aborted == [4] and not engine.model.finished
            # The connection is still in sync: the next push on it works.
            user = _push_user(sid=6)
            prefiller.run(user, list(range(256)))
            assert engine.model.finished == [(6, 0, list(range(256)))]
        finally:
            prefiller.close()


def test_remote_prefiller_push_ahead_reserves_at_start_and_run_finishes():
    hold = threading.Event()
    with _MockPrefillServer(hold=hold, chunk=128) as srv:
        engine, committed = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url, push=True)
        try:
            wake = threading.Condition()
            user = _push_user(sid=8)
            job = prefiller.start(user, list(range(256)), wake=wake)
            # Reserved on the calling (scheduler) thread, sent from the job's own.
            assert engine.model.in_flight == {8: (0, 256, [0, 1, 2])} and engine.model.resets == [8]
            assert _wait_until(lambda: len(srv.pushes) == 1)
            assert not job.done()
            hold.set()
            with wake:
                assert wake.wait_for(job.done, timeout=5)
            prefiller.run(user, list(range(256)))
            assert engine.model.finished == [(8, 0, list(range(256)))]
            assert len(srv.pushes) == 1 and not committed and not engine.model.in_flight
            assert user.pos == 256
        finally:
            prefiller.close()


def test_remote_prefiller_forgotten_push_keeps_its_pages_until_the_reply():
    hold = threading.Event()
    with _MockPrefillServer(hold=hold) as srv:
        engine, _ = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url, push=True)
        try:
            user = _push_user(sid=9)
            job = prefiller.start(user, list(range(256)))
            assert _wait_until(lambda: len(srv.pushes) == 1)
            prefiller.forget(user, job)  # the turn ended before loading it
            prefiller.reap()
            assert 9 in engine.model.in_flight and not engine.model.aborted  # its writes may still land
            hold.set()
            prefiller.reap(block=True)
            assert engine.model.aborted == [9] and not engine.model.in_flight and not engine.model.finished
            assert user.resets == 1 and not prefiller._orphans
            # The connection stayed in sync: the next push on it works.
            user = _push_user(sid=9)
            prefiller.run(user, list(range(256)))
            assert engine.model.finished == [(9, 0, list(range(256)))]
        finally:
            prefiller.close()


def test_remote_prefiller_push_ahead_holds_a_cancellation_until_the_reply():
    with _MockPrefillServer(chunk=128) as srv:
        engine, _ = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url, push=True)
        try:
            user = _push_user(sid=4)
            seen = []

            def progress(end: int) -> None:
                seen.append(end)
                raise _PromptStopped()

            job = prefiller.start(user, list(range(384)), progress)
            job.wait()
            assert seen == [128] and job.aborted
            try:
                prefiller.run(user, list(range(384)))
            except _PromptStopped:
                pass
            else:
                raise AssertionError("the cancellation was dropped")
            assert engine.model.aborted == [4] and not engine.model.finished and user.resets == 1
        finally:
            prefiller.close()


def test_remote_prefiller_start_runs_off_thread_and_run_loads_it():
    hold = threading.Event()
    with _MockPrefillServer(hold=hold) as srv:
        engine, committed = _fake_engine(num_layers=4)
        prefiller = _RemotePrefiller(engine, srv.url)
        try:
            wake = threading.Condition()
            users = []
            for i in range(2):
                user = SimpleNamespace(index=i, sid=10 + i, pos=0, tokens=[])
                user.activate = lambda: None
                user.reset = lambda u=user: setattr(u, "pos", 0)
                users.append(user)
            # Two prompts in flight at once, each on a connection of its own, neither loaded yet.
            jobs = [prefiller.start(user, list(range(256)), wake=wake) for user in users]
            assert _wait_until(lambda: len(srv.prefill_calls) == 2)
            assert srv.connections == 2
            assert not any(job.done() for job in jobs)
            assert committed == []

            hold.set()
            with wake:
                assert wake.wait_for(lambda: all(job.done() for job in jobs), timeout=5)
            for user in users:
                prefiller.run(user, list(range(256)))
                assert user.pos == 256
            assert [sid for sid, _, _ in committed] == [10, 11]
            assert len(srv.prefill_calls) == 2  # run took the replies, it sent nothing again
            assert len(prefiller._idle) == 2
        finally:
            prefiller.close()


def test_prefill_first_decodes_while_a_prompt_is_at_the_prefill_server(monkeypatch):
    from models.experimental.deepseek_v4_flash.demo.server import _Scheduler, _Turn
    from models.experimental.deepseek_v41_flash.demo.server import _PrefillFirst

    class _T:
        def __init__(self, phase, fetch=None):
            self.phase = phase
            self.error = None
            self.cancelled = threading.Event()
            self.remote_fetch = fetch

    reply = threading.Event()
    fetch = SimpleNamespace(done=reply.is_set, abort=lambda: None)
    decoding = _T(_Turn.DECODE)
    fetching = _T(_Turn.PREFILL, fetch)

    sched = _PrefillFirst.__new__(_PrefillFirst)
    sched.server = SimpleNamespace(pool=SimpleNamespace(max_running=8), prefiller=None)
    sched._new, sched._chores, sched._stop = deque(), deque(), False
    sched._wake = threading.Condition()
    sched._inflight, sched._posted = deque(), deque()
    sched._active = [fetching, decoding]
    rounds = []
    monkeypatch.setattr(_Scheduler, "_round_turns", lambda self: rounds.append(list(self._active)))

    # The prompt at the prefill server holds nothing up: the decoding turn takes its round without it.
    assert not sched._prefill_waiting()
    sched._round_turns()
    assert rounds == [[decoding]]
    assert sched._active == [fetching, decoding]

    # Alone, a prompt at the prefill server leaves the scheduler waiting, not spinning through rounds.
    sched._active = [fetching]
    sched._round_turns()
    assert len(rounds) == 1

    # Its KV is back: now it is a prompt to load, which decode waits for.
    reply.set()
    sched._active = [fetching, decoding]
    assert sched._prefill_waiting()


def test_block_verify_variants_cover_any_position():
    from models.experimental.deepseek_v41_flash.tt.model import DeepSeekV41Model

    cfg = SimpleNamespace(
        compress_ratios=[0, 1, 1, 2, 2, 1],
        index_source_layer_ids=[1, 3, 5],
        index_topk=512,
        candidate_source_layer_id=1,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    fake = SimpleNamespace(
        config=cfg,
        num_layers=6,
        verify_rows=4,
        _decode_max_seq=1_048_576,
        _candidate_group=1,
        _split_from={1: 4096, 3: 8192, 5: 4096},
        submeshes_io=[{"kv_block_layers": [0, 1, 2, 3]}],  # layer 3 the last kv source
    )
    fake._two_level = lambda p: DeepSeekV41Model._two_level(fake, p)
    fake._split_layers = lambda p: DeepSeekV41Model._split_layers(fake, p)
    fake._block_dense_ratios = lambda p: DeepSeekV41Model._block_dense_ratios(fake, p)
    fake._block_two_level = lambda p: DeepSeekV41Model._block_two_level(fake, p)
    fake._block_variant_key = lambda p, kv_only=False: DeepSeekV41Model._block_variant_key(fake, p, kv_only)

    reachable = set(DeepSeekV41Model._reachable_block_variants(fake))
    # Every transition region (dense, r=1 sparse / r=2 dense, both sparse, split, two-level, 1M) must be covered.
    sample_positions = [
        0,
        1,
        2,
        507,
        508,
        509,
        510,
        511,
        512,
        513,
        1019,
        1020,
        1021,
        1022,
        1023,
        1024,
        4094,
        4095,
        4096,
        4097,
        8190,
        8191,
        8192,
        8193,
        16379,
        16380,
        16381,
        16382,
        16383,
        16384,
        16385,
        100_000,
        100_001,
        500_000,
        1_048_570,
        1_048_571,
    ]
    for p in sample_positions:
        key = fake._block_variant_key(p)
        assert key in reachable, f"pos {p} variant {key} missing from reachable {reachable}"

    # Verify exact mode transitions.
    assert fake._block_variant_key(508) == (False, frozenset({1, 2}), False, frozenset())
    assert fake._block_variant_key(509) == (True, frozenset({2}), False, frozenset())
    assert fake._block_variant_key(1021) == (True, frozenset({2}), False, frozenset())
    assert fake._block_variant_key(1022) == (False, frozenset(), False, frozenset())
    assert fake._block_variant_key(1023) == (True, frozenset(), False, frozenset())
    assert fake._block_variant_key(16380)[2] is False
    assert fake._block_variant_key(16381)[2] is True
    assert fake._block_variant_key(1_048_570) == (False, frozenset(), True, frozenset({1, 3, 5}))

    # The KV-only block's variants split only its own layers, and cover any position too.
    kv_reachable = set(DeepSeekV41Model._reachable_block_variants(fake, kv_only=True))
    for p in sample_positions:
        key = fake._block_variant_key(p, kv_only=True)
        assert key in kv_reachable, f"pos {p} KV-only variant {key} missing from reachable {kv_reachable}"
    assert fake._block_variant_key(1_048_570, kv_only=True) == (False, frozenset(), True, frozenset({1, 3}))
    assert len(kv_reachable) <= len(reachable)


def test_spec_window_block_verify_any_position():
    import threading
    from models.experimental.deepseek_v41_flash.demo.server import _SpecScheduler, _Turn

    sched = _SpecScheduler.__new__(_SpecScheduler)
    sched._drafts = 3
    sched._flushing = False
    sched.engine = SimpleNamespace(
        max_seq=1_048_576,
        eos_id=2,
        block_verify=True,
        users=[SimpleNamespace(pos=0)],
    )
    turn = SimpleNamespace(
        phase=_Turn.DECODE,
        pending=0,
        cancelled=threading.Event(),
        next_id=42,
        ignore_eos=False,
        slot=0,
        max_tokens=100,
        generated=[],
    )
    for pos, expected in [
        (0, 100),
        (508, 100),
        (509, 100),
        (20_000, 100),
        (500_000, 100),
        (1_048_560, 8),
        (1_048_568, 0),
    ]:
        sched.engine.users[0].pos = pos
        assert sched._spec_budget(turn) == expected, f"pos={pos}"


def test_prefix_cache_with_remote_prefiller():
    from models.experimental.deepseek_v41_flash.demo.server import _PrefillFirst
    from models.experimental.deepseek_v4_flash.demo.server import _PrefixCache

    cache = _PrefixCache(1 << 30)
    prefix_tokens = list(range(1, 2001))
    fake_snap = SimpleNamespace(pos=len(prefix_tokens), nbytes=1024)
    assert cache.store(prefix_tokens, fake_snap)

    imported = []
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=129280),
        import_session=lambda sid, snap: imported.append((sid, snap)),
        reset_session=lambda sid: None,
    )
    prefiller = SimpleNamespace(
        replay_window=128,
        continue_alignment=0,
        aligned=lambda n, start=0, bounded=False: 0 if (start or bounded or n <= 128) else (n - 128) // 128 * 128,
        seconds=lambda n: 0.5 + max(0, -(-n // 4096) - 1) * 0.15,
    )
    sched = _PrefillFirst.__new__(_PrefillFirst)
    sched.engine = SimpleNamespace(model=model)
    sched.server = SimpleNamespace(prefiller=prefiller, prefix_cache=cache)
    sched.replay_rate = 300.0

    # Case 1: 2000-token cached prefix + 50-token tail -> hit (50/300 = 0.17s < remote prefill ~1.1s).
    user = SimpleNamespace(sid=1, pos=0, tokens=[])
    turn = SimpleNamespace(
        user_key="default",
        ids=prefix_tokens + list(range(5000, 5050)),
        from_prefix_cache=False,
        resumed_from=0,
    )
    assert sched._share_prefix(user, turn) is True
    assert user.pos == 2000
    assert len(turn.ids) == 50
    assert turn.from_prefix_cache is True
    assert imported == [(1, fake_snap)]

    # Case 2: 2000-token cached prefix + 20000-token tail -> skip (20000/300 = 66s >> remote prefill ~2.0s).
    imported.clear()
    user2 = SimpleNamespace(sid=2, pos=0, tokens=[])
    turn2 = SimpleNamespace(
        user_key="default",
        ids=prefix_tokens + list(range(10000, 30000)),
        from_prefix_cache=False,
        resumed_from=0,
    )
    assert sched._share_prefix(user2, turn2) is False
    assert user2.pos == 0
    assert imported == []


def test_block_replay_prompt_respects_cuts_and_final_token():
    import threading
    from models.experimental.deepseek_v4_flash.demo.server import _Turn
    from models.experimental.deepseek_v41_flash.demo.server import _SpecScheduler

    blocks = []
    dirty = {0}
    model = SimpleNamespace(
        verify_rows=4,
        _block_verify=True,
        _spec_dirty=dirty,
        config=SimpleNamespace(vocab_size=129280),
        decode_verify_block=lambda chunk, pos, kv_only=False: blocks.append((list(chunk), pos)) or [0] * len(chunk),
    )
    user = SimpleNamespace(index=0, sid=0, pos=100, tokens=list(range(100)), activate=lambda: None)
    turn = SimpleNamespace(
        slot=0,
        phase=_Turn.PREFILL,
        ids=list(range(100, 111)),  # 11 tokens: positions 100..110; cut at 105, final prompt token at 110
        fed=0,
        device_prefill=0,
        replay_ids=[],
        prefix_cuts={105, 110},
        cancelled=threading.Event(),
    )
    turn.prompt_left = len(turn.ids) - turn.fed

    sched = _SpecScheduler.__new__(_SpecScheduler)
    sched.engine = SimpleNamespace(model=model, users=[user])
    sched.server = SimpleNamespace(prefix_cache=object())
    sched.steps = 0
    sched._spec_accounted = 0
    sched._settle = lambda: True
    sched._account = lambda n: None

    # Cut at 105 leaves 5 tokens before the cut -> 1 block of 4 (100..103), leaving 104 for single-row decode.
    assert sched._block_replay_prompt(turn) is True
    assert blocks == [([100, 101, 102, 103], 100)]
    assert (user.pos, turn.fed, len(user.tokens)) == (104, 4, 104)
    assert 0 not in dirty

    # Next call before cut 105 has only 1 token left (< 4), so falls back to single-row decode.
    turn.prompt_left = len(turn.ids) - turn.fed
    assert sched._block_replay_prompt(turn) is False

    # After 104 is replayed and cut 105 is captured/discarded, 6 tokens remain (105..110):
    # final token 110 is preserved -> 5 eligible -> 1 block of 4 (105..108), leaving 109..110.
    user.pos = 105
    user.tokens.append(104)
    turn.fed = 5
    turn.prefix_cuts.discard(105)
    turn.prompt_left = len(turn.ids) - turn.fed
    assert sched._block_replay_prompt(turn) is True
    assert blocks[-1] == ([105, 106, 107, 108], 105)
    assert (user.pos, turn.fed) == (109, 9)


def test_keep_headroom_evicts_parked_before_a_round_and_block_replay_fails_only_its_turn():
    import threading
    from models.experimental.deepseek_v4_flash.demo.server import _Turn
    from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import PagedCacheFull
    from models.experimental.deepseek_v41_flash.demo.server import _ROUND_HEADROOM_TOKENS, _SpecScheduler

    sched = _SpecScheduler.__new__(_SpecScheduler)
    free = [10_000]
    room, settled = [], []
    sched._settle = lambda: settled.append(True) or True
    sched._active = []
    sched._posted = []
    sched._spec_live = None

    def make_room(n, keep, frees=0):
        room.append((n, keep))
        free[0] += frees

    sched.engine = SimpleNamespace(tokens_left=lambda: free[0], _replay_q=[])
    sched._make_room = make_room
    sched._keep_headroom()
    assert room == []  # nothing active, nothing to keep room for

    prompt = SimpleNamespace(prompt_left=3000)
    reply = SimpleNamespace(prompt_left=0)
    sched._active = [prompt, reply]
    need = 3000 + 2 * _ROUND_HEADROOM_TOKENS
    free[0] = need
    sched._keep_headroom()
    assert room == [] and settled == []
    free[0] = need - 1
    sched._keep_headroom()
    # Settled first (no step waits on a packet while the parked KV is read back), then evicted.
    assert room == [(need, -1)] and settled == [True]

    # Mid-turn: a block whose capacity check finds the pools full evicts the oldest conversations and goes on.
    full = [1]
    blocks = []

    def verify_block(chunk, pos, kv_only=False):
        if full[0]:
            full[0] -= 1
            raise PagedCacheFull("need 1 blocks, 0 free of 32769")
        blocks.append(pos)
        return [0] * len(chunk)

    model = SimpleNamespace(
        verify_rows=4, _block_verify=True, _spec_dirty=set(), config=SimpleNamespace(vocab_size=129280)
    )
    model.decode_verify_block = verify_block
    user = SimpleNamespace(index=0, sid=0, pos=0, tokens=[], activate=lambda: None)

    def prompt_turn():
        return SimpleNamespace(
            user_key="u",
            slot=0,
            phase=_Turn.PREFILL,
            ids=list(range(1000, 1012)),
            fed=0,
            device_prefill=0,
            replay_ids=[],
            prefix_cuts=set(),
            cancelled=threading.Event(),
            prompt_left=12,
        )

    sched.engine = SimpleNamespace(model=model, users=[user], tokens_left=lambda: free[0], _replay_q=[])
    sched.server = SimpleNamespace(prefix_cache=None)
    sched.steps = 0
    sched._spec_accounted = 0
    sched._account = lambda n: None
    sched._cut_short = set()
    failed, aborted = [], []
    sched._fail = lambda t, e: failed.append((t, e))
    sched._abort_all = lambda e: aborted.append(e)
    room.clear()
    free[0] = 0
    sched._make_room = lambda n, keep: make_room(n, keep, frees=4096)
    turn = prompt_turn()
    assert sched._block_replay_prompt(turn) is True
    assert room == [(4 + _ROUND_HEADROOM_TOKENS, 0)]  # the turn's own slot is kept
    assert blocks == [0, 4] and (user.pos, turn.fed) == (8, 8)
    assert failed == [] and aborted == []

    # Nothing to evict: only this turn fails, its slot cut short; the scheduler is not wedged.
    user.pos, user.tokens, blocks[:] = 0, [], []
    full[0] = 10
    free[0] = 0
    sched._make_room = lambda n, keep: make_room(n, keep)
    turn = prompt_turn()
    assert sched._block_replay_prompt(turn) is True
    assert [t for t, _ in failed] == [turn] and isinstance(failed[0][1], PagedCacheFull)
    assert aborted == [] and sched._cut_short == {0}
    assert (user.pos, turn.fed) == (0, 0)

    # With steps posted ahead (waiting for their packets) the eviction's device reads would queue behind them.
    sched._posted = [turn]
    assert sched._room_mid_step(1, turn) is False


def test_post_evicts_mid_turn_when_the_pool_is_full():
    from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import PagedCacheFull
    from models.experimental.deepseek_v41_flash.demo.server import _ROUND_HEADROOM_TOKENS, _V41Scheduler

    free = [0]
    posted = []

    def post_traced(user, positions, token_ids):
        if not free[0]:
            raise PagedCacheFull("need 1 blocks, 0 free of 32769")
        posted.append(list(positions))

    def make_room(n, keep):
        rooms.append((n, keep))
        free[0] = 4096

    rooms = []
    sched = _V41Scheduler.__new__(_V41Scheduler)
    sched.engine = SimpleNamespace(
        users=[SimpleNamespace()], post_traced=post_traced, tokens_left=lambda: free[0], _replay_q=[]
    )
    sched._posted = []
    sched._reader = SimpleNamespace(post=lambda turn, positions: None)
    sched._make_room = make_room
    turn = SimpleNamespace(user_key="u", slot=0)
    sched._post(turn, range(10, 12))
    assert rooms == [(2 + _ROUND_HEADROOM_TOKENS, 0)]
    assert posted == [[10, 11]] and sched._posted == [turn, turn]


def test_block_replay_prompt_runs_kv_only_blocks_before_the_tail():
    import threading
    from models.experimental.deepseek_v4_flash.demo.server import _Turn
    from models.experimental.deepseek_v41_flash.demo.server import _SpecScheduler

    blocks = []
    model = SimpleNamespace(
        verify_rows=4,
        _block_verify=True,
        _spec_dirty=set(),
        kv_only_blocks=True,
        sliding_window=8,
        dspark=SimpleNamespace(ring_rows=12),  # the tail is the larger of the two: 12 positions
        config=SimpleNamespace(vocab_size=129280),
        decode_verify_block=lambda chunk, pos, kv_only=False: blocks.append((pos, kv_only)) or [0] * len(chunk),
    )

    def run(prefix_cuts, capture_at=None):
        blocks.clear()
        user = SimpleNamespace(index=0, sid=0, pos=0, tokens=[], activate=lambda: None)
        turn = SimpleNamespace(
            slot=0,
            phase=_Turn.PREFILL,
            ids=list(range(1000, 1040)),  # positions 0..39
            fed=0,
            device_prefill=0,
            replay_ids=[],
            prefix_cuts=set(prefix_cuts),
            capture_at=capture_at,
            cancelled=threading.Event(),
            prompt_left=40,
        )
        sched = _SpecScheduler.__new__(_SpecScheduler)
        sched.engine = SimpleNamespace(model=model, users=[user])
        sched.server = SimpleNamespace(prefix_cache=object())
        sched.steps = 0
        sched._spec_accounted = 0
        sched._settle = lambda: True
        sched._account = lambda n: None
        assert sched._block_replay_prompt(turn) is True
        return list(blocks)

    # Prompt end 40, tail 12: blocks ending by 28 skip layers 21+, the last 12 positions' blocks run every layer.
    assert run([]) == [(p, p + 4 <= 28) for p in range(0, 36, 4)]
    # A prefix snapshot at 20 is read too: only blocks ending by 8 are KV-only before it.
    assert run([20]) == [(0, True), (4, True), (8, False), (12, False), (16, False)]
    # So is a capture_at point.
    assert run([], capture_at=24) == [(p, p + 4 <= 12) for p in range(0, 36, 4)]
    model.kv_only_blocks = False
    assert all(not kv_only for _, kv_only in run([]))
