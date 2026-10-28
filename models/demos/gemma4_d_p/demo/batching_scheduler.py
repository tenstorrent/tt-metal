# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-process continuous-batching prefill scheduler on the 8x4 Galaxy, driven by batching_showcase.py and
agentx_replay.py.

Modes:
- batched: each traced step has up to num_slots 2k lanes; every active request gets one and the spare lanes go to
  requests with chunks left (Scheduler._plan). Every chunk is 2k, so every request's KV has the 2k layout;
- serial 8k / 4k / 2k: one request at a time at a fixed chunk size, as an unbatched server runs. The last chunk is
  padded to the full chunk size, as in the serving producer.
A request resumes at its cached prefix rounded down to its chunk width (align "chunk") or to the model's resume
granularity at that width (align "resume": a per-rank slab, 256 tokens at 2k, 512 at 4k, 1024 at 8k).
Running requests keep their mode; after a switch, new ones wait until those finish. A request keeps one chunk width
for its whole life. Latencies and step times are host wall clock around each step (staging + trace replay). Projections
take each request's own chunk fidelity: HiFi2 for 2k lanes, LoFi for serial 4k and 8k, as an unbatched server runs them.
"""

import collections
import itertools
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from loguru import logger

from models.demos.gemma4_d_p.demo.batched_steps import StepRunner, build_batched_model
from models.demos.gemma4_d_p.demo.text_demo_prefill import _hf_model_id, _text_token_stream

CHUNK = 2048
MID = 4096  # serial 4k's chunk
WIDE = 8192  # serial 8k's chunk
BATCHED = "batched"
SERIAL_CHUNKS = {"serial 8k": WIDE, "serial 4k": MID, "serial 2k": CHUNK}  # mode -> its fixed chunk size
MODES = (*SERIAL_CHUNKS, BATCHED)


def step_layouts(num_slots):
    """Every step layout the scheduler can produce, widest first (the compile order). A layout is the lanes' widths
    in order; the trace is keyed by it alone, slots and prefixes are staged per step."""
    return [(WIDE,), (MID,)] + [(CHUNK,) * n for n in range(num_slots, 0, -1)]


@dataclass
class Request:
    rid: int
    label: str
    tokens: torch.Tensor
    arrived: float
    start: int = 0  # tokens already cached; _admit rounds it down (per Scheduler.align) into done
    slot: int = -1
    mode: str = BATCHED
    width: int = CHUNK
    done: int = 0  # tokens prefilled
    started: float = 0.0
    finished: float = 0.0
    error: str = ""
    event: threading.Event = field(default_factory=threading.Event)

    def result(self):
        return dict(
            label=self.label,
            tokens=len(self.tokens),
            queue_ms=round((self.started - self.arrived) * 1000, 1),
            prefill_ms=round((self.finished - self.started) * 1000, 1),
            latency_ms=round((self.finished - self.arrived) * 1000, 1),
        )


def plan_step(active, max_lanes, chunks_left):
    """[(slot, lanes)] for one batched step: one lane per active slot in the given order (at most max_lanes), then
    the spare lanes to those slots in the same order, each up to its chunks left (chunks_left(slot))."""
    plan = [[slot, 1] for slot in active[:max_lanes]]
    free = max_lanes - len(plan)
    for entry in plan:
        extra = min(free, chunks_left(entry[0]) - 1)
        entry[1] += extra
        free -= extra
    return [tuple(entry) for entry in plan]


def _lanes(request, n):
    """The request's next n chunks as lanes (slot, start, tokens); a last chunk shorter than the request's width is
    padded, as the serving producer pads it."""
    lanes = []
    for start in range(request.done, request.done + n * request.width, request.width):
        tokens = request.tokens[start : start + request.width]
        lanes.append((request.slot, start, torch.cat([tokens, tokens.new_zeros(request.width - len(tokens))])))
    return lanes


def _chunks_left(request):
    return -(-(len(request.tokens) - request.done) // request.width)


class Scheduler:
    """Owns the device. Admits requests into free KV slots and runs one traced step at a time."""

    def __init__(self, runner, num_slots, capacity, stream, resume_headroom=False):
        assert capacity % WIDE == 0, f"capacity {capacity} must be a multiple of {WIDE}: serial 8k pads prompts to it"
        self.runner = runner
        self.layouts = set(step_layouts(num_slots))
        self.num_slots = num_slots
        self.capacity = capacity
        self.stream = stream
        self.mode = BATCHED
        self.align = "chunk"  # or "resume" (needs resume_headroom); switchable between plays, like mode
        self.resume_headroom = resume_headroom
        cp = runner.mesh_config.cp_degree
        self.resume_steps = {w: w // cp for w in (CHUNK, MID, WIDE)}  # Gemma4PrefillRuntime.resume_granularity
        self.incoming = queue.Queue()  # the only state the submitting thread touches
        self.waiting, self.active = collections.deque(), []
        self.ids = itertools.count()
        self.busy_s, self.steps, self.replay_ms = 0.0, 0, 0.0
        self.stop = threading.Event()

    def submit(self, n_tokens, label, start=0):
        """Queue a prompt of n_tokens (rounded up to whole 2k chunks). start: tokens of it already cached; the request
        resumes there, rounded down per self.align (the KV below it is whatever the slot holds)."""
        n_tokens = -(-int(n_tokens) // CHUNK) * CHUNK
        if not CHUNK <= n_tokens <= self.capacity:
            raise ValueError(f"prompt of {n_tokens} tokens outside [{CHUNK}, {self.capacity}] (DEMO_CAPACITY)")
        rid = next(self.ids)
        # A different window of real text per request; the content does not change the step time.
        offset = (rid * 37_000) % (len(self.stream) - n_tokens)
        request = Request(
            rid=rid,
            label=label,
            tokens=self.stream[offset : offset + n_tokens].clone(),
            arrived=time.perf_counter(),
            start=min(int(start), n_tokens - 1),
        )
        self.incoming.put(request)
        if self.stop.is_set():
            # run() sets stop before its final drain, so a request that missed the drain fails here instead of
            # waiting for an event nobody will set.
            raise RuntimeError("scheduler stopped")
        return request

    def play(self, arrivals):
        """Submit each (seconds after now, tokens, label, cached tokens) at its time, then wait for all; returns
        (start time, requests)."""
        t0, requests = time.perf_counter(), []
        for at, n_tokens, label, start in arrivals:
            time.sleep(max(0.0, t0 + at - time.perf_counter()))
            requests.append(self.submit(n_tokens, label, start=start))
        for request in requests:
            request.event.wait()
        assert not any(r.error for r in requests), [r.error for r in requests if r.error][:3]
        return t0, requests

    def _admit(self):
        # Block briefly when idle instead of polling; then drain everything that has arrived.
        try:
            first = [self.incoming.get(timeout=0.1)] if not self.active and not self.waiting else []
        except queue.Empty:
            return
        self.waiting.extend(first)
        while not self.incoming.empty():
            self.waiting.append(self.incoming.get_nowait())
        if any(r.mode != self.mode for r in self.active):
            return  # after a mode switch, the requests of the old mode finish first
        if self.align == "resume" and not self.resume_headroom:
            raise ValueError("align 'resume' needs the KV headroom of start_scheduler(..., resume=True)")
        limit = self.num_slots if self.mode == BATCHED else 1
        free = [s for s in range(self.num_slots) if all(r.slot != s for r in self.active)]
        admitted = []
        while self.waiting and len(self.active) + len(admitted) < limit and free:
            request = self.waiting.popleft()
            request.slot, request.mode, request.started = free.pop(0), self.mode, time.perf_counter()
            # Serial modes: one at a time, at the mode's chunk size (see _lanes for the last chunk). Resume at the
            # cached prefix rounded down to that width, or to the resume granularity at that width.
            request.width = SERIAL_CHUNKS.get(self.mode, CHUNK)
            step = self.resume_steps[request.width] if self.align == "resume" else request.width
            request.done = request.start // step * step
            admitted.append(request)
        self.active.extend(admitted)

    def _plan(self):
        """[(request, lanes it gets)] for the next step.

        Serial: the one active request, one chunk. Batched: up to num_slots lanes. There are never more active
        requests than lanes, so each gets one lane, and the spare lanes go to requests with chunks left, in turn: a
        lone prompt takes every lane, and a long one never makes a short one wait a step. self.active rotates by one
        each step, so the spare lanes alternate between long prompts.
        """
        if self.active[0].mode in SERIAL_CHUNKS:  # _admit never mixes modes
            return [(self.active[0], 1)]
        plan = plan_step(self.active, self.num_slots, _chunks_left)
        self.active.append(self.active.pop(0))
        return plan

    def _step(self):
        plan = self._plan()
        lanes = [lane for request, n in plan for lane in _lanes(request, n)]
        assert StepRunner.layout(lanes) in self.layouts, f"layout {StepRunner.layout(lanes)} was not captured"
        t0 = time.perf_counter()
        replay_ms = self.runner.step(lanes)
        step_s = time.perf_counter() - t0
        logger.info(f"[demo] step lanes={len(lanes)} total={step_s * 1000:.1f}ms replay={replay_ms:.1f}ms")
        self.busy_s += step_s
        self.steps += 1
        self.replay_ms += replay_ms
        for request, n in plan:
            request.done += n * request.width
            if request.done >= len(request.tokens):
                request.finished = time.perf_counter()
                self.active.remove(request)
                request.event.set()

    def run(self):
        try:
            while not self.stop.is_set():
                self._admit()
                if self.active:
                    self._step()
        finally:
            self.stop.set()
            for request in [*self.active, *self.waiting, *list(self.incoming.queue)]:
                request.error = request.error or "scheduler stopped"
                request.event.set()


def _zero_lanes(layout):
    return [(slot, 0, torch.zeros(w, dtype=torch.int32)) for slot, w in enumerate(layout)]


def start_scheduler(mesh_device, num_slots, capacity, resume=False):
    """Build the batched model, compile and capture every step layout, and return (scheduler, runner). The caller runs
    scheduler.run() on this thread (it owns the device) and releases the runner's traces afterwards. resume: the KV
    slots get one wide chunk of headroom, since a request that resumes mid-chunk pads its last chunk past its prompt."""
    mesh_config, _, model = build_batched_model(mesh_device, CHUNK, num_slots, capacity + (WIDE if resume else 0))
    runner = StepRunner(mesh_device, mesh_config, model, CHUNK)
    # The layouts mix widths (serial 8k / 4k next to 2k steps), so every step takes the PrefillLanes path
    # (StepRunner.always_lanes), and the widest layout compiles first, so lazily created all-gather semaphores land
    # above every layout's SDPA buffers.
    layouts = step_layouts(num_slots)
    if os.environ.get("DEMO_BATCHED_ONLY") == "1":
        # Batched mode only (SP4 x 2 study replays): 2k layouts alone, so steps take the same path as
        # test_batched_step_perf's and no 8k layout has to fit a 4x4 half.
        layouts = [layout for layout in layouts if set(layout) == {CHUNK}]
    logger.info(f"[demo] compiling and capturing {len(layouts)} step traces")
    runner.capture_all([_zero_lanes(layout) for layout in layouts])
    # One untimed replay of every trace, so no request pays a trace's first replay.
    for layout in layouts:
        runner.step(_zero_lanes(layout))
    stream = _text_token_stream(_hf_model_id())[0]
    return Scheduler(runner, num_slots, capacity, stream, resume_headroom=resume), runner
