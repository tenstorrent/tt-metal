# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0309: the async export's socket guard (no device, no ttnn).

    pytest --noconftest models/demos/glm53_flash/tests/test_export_guard.py

The policy tests drive ``ExportDrainGuard`` with a fake queue / probe / clock. ``test_serving_loop_*`` model the live
deadlock with real threads: an engine that sends the next prompt only after every layer ack of the previous one, a
runner main thread whose blocking socket wait holds a lock the export worker needs (on the device: the GIL, held by
``ttnn.from_device`` in the runner's ``_decode_metadata``, G53F-0309), and a worker that packs + acks. Unguarded, the
last chunk's acks never go out (the 22:02 hang of G53F-0256); guarded, every prompt completes and the export overlaps
every chunk but the last of each prompt.
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from models.demos.glm53_flash.tt.runners.export_guard import ExportDrainGuard


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _guard(jobs, probe=None, on_sleep=None):
    clock = _Clock()
    sleeps = []

    def sleep(dt):
        sleeps.append(dt)
        clock.t += dt
        if on_sleep is not None:
            on_sleep(len(sleeps))

    logs = []
    return ExportDrainGuard(jobs, probe, poll_s=0.001, clock=clock, sleep=sleep, log=logs.append), sleeps, logs


def _finish_one(jobs):
    jobs.get_nowait()
    jobs.task_done()


def test_idle_returns_at_once():
    jobs = queue.Queue()
    g, sleeps, logs = _guard(jobs, probe=lambda: pytest.fail("no probe needed with nothing pending"))
    assert g.before_socket_wait() == "idle" and not sleeps and not logs


def test_a_waiting_chunk_overlaps():
    jobs = queue.Queue()
    jobs.put("chunk 0")
    g, sleeps, _ = _guard(jobs, probe=lambda: (0, 12288, 24576))
    assert g.before_socket_wait() == "overlap"
    assert jobs.unfinished_tasks == 1 and not sleeps and g.overlapped == 1


def test_no_chunk_waiting_drains_first():
    jobs = queue.Queue()
    jobs.put("chunk 2")
    g, sleeps, logs = _guard(jobs, probe=lambda: None, on_sleep=lambda n: n == 3 and _finish_one(jobs))
    assert g.before_socket_wait() == "drained"
    assert jobs.unfinished_tasks == 0 and len(sleeps) == 3 and g.drained == 1
    assert "no chunk waiting, drained 1 export job(s)" in logs[0]


def test_a_full_last_chunk_is_not_mistaken_for_mid_prompt():
    """The header of the chunk just computed says nothing about what comes next: a prompt may end on a chunk
    boundary. Only the socket's own state decides."""
    jobs = queue.Queue()
    jobs.put("chunk [24576, 36864)")
    g, _, _ = _guard(jobs, probe=lambda: None, on_sleep=lambda n: _finish_one(jobs))
    assert g.before_socket_wait() == "drained"


def test_a_sentinel_waiting_drains_first():
    for sentinel in ((-1, -1, -1), (-2, -2, -2)):
        jobs = queue.Queue()
        jobs.put("chunk")
        g, sleeps, logs = _guard(jobs, probe=lambda s=sentinel: s, on_sleep=lambda n: n == 2 and _finish_one(jobs))
        assert g.before_socket_wait() == "drained" and len(sleeps) == 2
        assert f"a sentinel {sentinel} waiting" in logs[0]


def test_a_chunk_arriving_mid_drain_stops_the_wait():
    jobs = queue.Queue()
    jobs.put("chunk")
    arrived = []
    g, sleeps, _ = _guard(
        jobs, probe=lambda: (0, 0, 4092) if arrived else None, on_sleep=lambda n: n == 4 and arrived.append(1)
    )
    assert g.before_socket_wait() == "overlap" and len(sleeps) == 4 and jobs.unfinished_tasks == 1


def test_no_probe_drains_always():
    jobs = queue.Queue()
    jobs.put("chunk")
    g, _, logs = _guard(jobs, probe=None, on_sleep=lambda n: _finish_one(jobs))
    assert g.before_socket_wait() == "drained" and "NO probe: drain-always" in logs[0]


def test_a_failing_probe_degrades_to_drain_always():
    jobs = queue.Queue()
    jobs.put("chunk")

    def probe():
        raise RuntimeError("read_from_core: chip 7 unreachable")

    g, _, logs = _guard(jobs, probe=probe, on_sleep=lambda n: _finish_one(jobs))
    assert g.before_socket_wait() == "drained"
    assert g.probe is None and "chip 7 unreachable" in g.probe_error and "probe failed" in logs[0]


# ---------------------------------------------------------------------------------------------- threaded serving model
CHUNK, LAYERS = 4, 46


class _Socket:
    """The H2D socket: the engine pushes headers, the runner's wait blocks HOLDING ``held`` (as the live wait held what
    the export worker needed), and ``probe`` peeks without blocking."""

    def __init__(self, held: threading.Lock):
        self.q: list = []
        self.cv = threading.Condition()
        self.held = held

    def push(self, hdr):
        with self.cv:
            self.q.append(hdr)
            self.cv.notify_all()

    def probe(self):
        with self.cv:
            return self.q[0] if self.q else None

    def wait_next(self, timeout_s: float):
        with self.held:
            with self.cv:
                if not self.cv.wait_for(lambda: self.q, timeout=timeout_s):
                    return None
                return self.q.pop(0)


def _serve(prompts, *, guarded: bool, timeout_s: float = 2.0):
    held = threading.Lock()  # stands for the GIL / driver lock of the blocked socket wait
    sock = _Socket(held)
    jobs: queue.Queue = queue.Queue()
    acks = []
    acks_cv = threading.Condition()
    guard = ExportDrainGuard(jobs, sock.probe if guarded else None, poll_s=0.0005)
    stop = threading.Event()

    def worker():  # pack + write + acks of one chunk; needs `held` (the GIL) for its Python work
        while not stop.is_set():
            try:
                hdr = jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            time.sleep(0.02)  # pack + write (longer than a prefill here, so every guard call finds a job pending)
            with held:
                with acks_cv:
                    acks.extend((hdr, layer) for layer in range(LAYERS))
                    acks_cv.notify_all()
            jobs.task_done()

    def engine(result):  # pushes a prompt's chunks back to back, then waits for all their acks
        for n in prompts:
            chunks = [(0, s, min(n, s + CHUNK)) for s in range(0, n, CHUNK)]
            for hdr in chunks:
                sock.push(hdr)
            want = {(hdr, layer) for hdr in chunks for layer in range(LAYERS)}
            with acks_cv:
                if not acks_cv.wait_for(lambda: want <= set(acks), timeout=timeout_s):
                    result.append(("HUNG", n, len(want & set(acks)), len(want)))
                    sock.push((-1, -1, -1))
                    return
            result.append(("ok", n))
        sock.push((-1, -1, -1))

    def runner():  # the common runner's request loop with the GLM async runtime
        while True:
            if guarded:
                guard.before_socket_wait()
            hdr = sock.wait_next(timeout_s * 3)
            if hdr is None or hdr[0] < 0:
                return
            time.sleep(0.01)  # prefill
            jobs.put(hdr)

    result = []
    threads = [threading.Thread(target=f, daemon=True) for f in (worker, runner)]
    threads.append(threading.Thread(target=engine, args=(result,), daemon=True))
    for t in threads:
        t.start()
    threads[2].join(timeout=timeout_s * len(prompts) + 5)
    threads[1].join(timeout=timeout_s * 3 + 1)
    stop.set()
    return result, guard


def test_serving_loop_unguarded_hangs_at_the_last_chunk():
    result, _ = _serve([3 * CHUNK + 2], guarded=False, timeout_s=0.5)
    # the earlier chunks' acks can go out while the runner's next socket wait returns at once (mid-prompt); the last
    # chunk's never do: the runner sits in the wait holding what the worker needs, and the engine sends nothing more
    ((status, n, got, want),) = result
    assert (status, n, want) == ("HUNG", 3 * CHUNK + 2, 4 * LAYERS) and got <= 3 * LAYERS


def test_serving_loop_guarded_completes_every_prompt():
    prompts = [3 * CHUNK + 2, 3 * CHUNK, CHUNK - 1, 2 * CHUNK]  # ragged tail, FULL last chunk, one short chunk, full
    result, guard = _serve(prompts, guarded=True)
    assert result == [("ok", n) for n in prompts]
    chunks = sum(-(-n // CHUNK) for n in prompts)
    # every socket wait found an export pending (a pack outlasts a prefill here) and either overlapped the next chunk or
    # drained; every chunk but a prompt's last overlapped. A last chunk may count as an overlap too: its acks are out,
    # the engine has already sent the next prompt, and only the worker's task_done is still to come
    assert guard.drained + guard.overlapped == chunks
    assert guard.overlapped >= chunks - len(prompts) and 1 <= guard.drained <= len(prompts)
