# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Interactive continuous-batching prefill demo: a Gemma4 prefill server on the 8x4 Galaxy.

Start the server (it owns the mesh; compiling and capturing the traces takes a few minutes):

    pytest models/demos/gemma4_d_p/demo/batching_server.py -s

then fire prompts from another terminal with demo/batching_client.py, e.g.

    python models/demos/gemma4_d_p/demo/batching_client.py compare

Two modes, switchable at any time (requests already running keep the mode they were admitted with):
- continuous batching: every active request (up to DEMO_SLOTS) gets its next 2k chunk in one traced step;
- serial: one request at a time in arrival order, the way an unbatched server runs. A prompt whose length is a
  multiple of 8k is prefilled in 8k chunks (one 8k-wide lane, the faster unbatched chunk size), others in 2k chunks.
  A request keeps one chunk width for its whole life.

The server draws the lanes of each step and every request's queue / prefill / total latency. Latencies and step times
are host wall clock around each step (staging + trace replay).

Env: DEMO_PORT (default 8765), DEMO_SLOTS (requests in flight, default 4), DEMO_CAPACITY (max prompt tokens, default
65536).
"""

import collections
import itertools
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
from loguru import logger
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import ttnn
from models.demos.gemma4_d_p.demo.batched_steps import StepRunner, build_batched_model
from models.demos.gemma4_d_p.demo.text_demo_prefill import _hf_model_id, _text_token_stream
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention import operations as attention_operations

CHUNK = 2048
WIDE = 8192  # serial mode's chunk for prompts that are a multiple of it
TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 600_000_000))
COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red", "bright_cyan", "bright_magenta"]
BATCHED, SERIAL = "batched", "serial"


@dataclass
class Request:
    rid: int
    label: str
    tokens: torch.Tensor
    arrived: float
    color: str
    slot: int = -1
    mode: str = BATCHED
    width: int = CHUNK
    done: int = 0  # tokens prefilled
    started: float = 0.0
    finished: float = 0.0
    error: str = ""
    event: threading.Event = field(default_factory=threading.Event)

    def result(self):
        if self.error:
            return dict(id=self.rid, label=self.label, error=self.error)
        return dict(
            id=self.rid,
            label=self.label,
            tokens=len(self.tokens),
            mode=self.mode,
            queue_ms=round((self.started - self.arrived) * 1000, 1),
            prefill_ms=round((self.finished - self.started) * 1000, 1),
            latency_ms=round((self.finished - self.arrived) * 1000, 1),
        )


def _bar(fraction, width, full="█", empty="░"):
    filled = max(0, min(width, int(round(width * fraction))))
    return full * filled + empty * (width - filled)


class Scheduler:
    """Owns the device. Admits requests into free KV slots and runs one traced step at a time."""

    def __init__(self, runner, num_slots, capacity, stream):
        self.runner = runner
        self.num_slots = num_slots
        self.capacity = capacity
        self.stream = stream
        self.mode = BATCHED
        self.incoming = queue.Queue()
        self.lock = threading.Lock()  # guards waiting / active / recent / steps against the view thread
        self.waiting, self.active = collections.deque(), []
        self.recent = collections.deque(maxlen=12)
        self.steps = collections.deque(maxlen=14)  # (lanes as [(color, label)], step ms)
        self.ids = itertools.count()
        self.tokens_done, self.busy_s = 0, 0.0
        self.stop = threading.Event()

    def submit(self, n_tokens, label):
        n_tokens = -(-int(n_tokens) // CHUNK) * CHUNK
        if not CHUNK <= n_tokens <= self.capacity:
            raise ValueError(f"prompt of {n_tokens} tokens outside [{CHUNK}, {self.capacity}] (DEMO_CAPACITY)")
        rid = next(self.ids)
        # A different window of real text per request; the content does not change the step time.
        offset = (rid * 37_000) % (len(self.stream) - n_tokens)
        request = Request(
            rid=rid,
            label=label or f"{n_tokens // 1024}k",
            tokens=self.stream[offset : offset + n_tokens].clone(),
            arrived=time.perf_counter(),
            color=COLORS[rid % len(COLORS)],
        )
        self.incoming.put(request)
        return request

    def _admit(self):
        # Block briefly when idle instead of polling; then drain everything that has arrived.
        try:
            first = [self.incoming.get(timeout=0.1)] if not self.active and not self.waiting else []
        except queue.Empty:
            return
        with self.lock:
            self.waiting.extend(first)
            while not self.incoming.empty():
                self.waiting.append(self.incoming.get_nowait())
            limit = self.num_slots if self.mode == BATCHED else 1
            free = [s for s in range(self.num_slots) if all(r.slot != s for r in self.active)]
            while self.waiting and len(self.active) < limit and free:
                request = self.waiting.popleft()
                request.slot, request.mode, request.started = free.pop(0), self.mode, time.perf_counter()
                request.width = WIDE if self.mode == SERIAL and len(request.tokens) % WIDE == 0 else CHUNK
                self.active.append(request)

    def _step(self):
        lanes = [(r.slot, r.done, r.tokens[r.done : r.done + r.width]) for r in self.active]
        t0 = time.perf_counter()
        replay_ms = self.runner.step(lanes)
        step_s = time.perf_counter() - t0
        logger.info(f"[demo] step lanes={len(lanes)} total={step_s * 1000:.1f}ms replay={replay_ms:.1f}ms")
        with self.lock:
            self.busy_s += step_s
            self.tokens_done += sum(r.width for r in self.active)
            self.steps.append(([(r.color, r.label) for r in self.active], step_s * 1000))
            for request in list(self.active):
                request.done += request.width
                if request.done == len(request.tokens):
                    request.finished = time.perf_counter()
                    self.active.remove(request)
                    self.recent.append(request)
                    request.event.set()

    def run(self):
        try:
            while not self.stop.is_set():
                self._admit()
                if self.active:
                    self._step()
        finally:
            self.stop.set()
            with self.lock:
                for request in [*self.active, *self.waiting, *list(self.incoming.queue)]:
                    request.error = request.error or "server stopped"
                    request.event.set()

    # ── Live view ───────────────────────────────────────────────────────────────

    def _lanes_table(self, steps):
        table = Table(title="lanes (one traced step per row, newest at the bottom)", expand=True)
        table.add_column("step ms", justify="right", width=8)
        for slot in range(self.num_slots):
            table.add_column(f"lane {slot}", ratio=1)
        for lanes, step_ms in steps:
            cells = [Text(label, style=f"reverse {color}") for color, label in lanes]
            table.add_row(f"{step_ms:.0f}", *cells, *[Text("·", style="dim")] * (self.num_slots - len(cells)))
        return table

    def _inflight_table(self, active, waiting, now):
        table = Table(title="in flight", expand=True)
        for col in ("request", "tokens", "progress", "waited"):
            table.add_column(col)
        for r in active:
            table.add_row(
                Text(r.label, style=r.color),
                str(len(r.tokens)),
                _bar(r.done / len(r.tokens), 20),
                f"{r.started - r.arrived:.2f}s",
            )
        for r in waiting:
            table.add_row(Text(r.label, style=r.color), str(len(r.tokens)), "queued", f"{now - r.arrived:.2f}s")
        return table

    def _finished_table(self, recent):
        table = Table(title="finished (latency = queue + prefill)", expand=True)
        for col in ("request", "tokens", "mode", "queue", "prefill", "latency", ""):
            table.add_column(col)
        longest = max([r.finished - r.arrived for r in recent] + [1e-6])
        for r in reversed(recent):
            latency = r.finished - r.arrived
            table.add_row(
                Text(r.label, style=r.color),
                str(len(r.tokens)),
                r.mode,
                f"{r.started - r.arrived:.2f}s",
                f"{r.finished - r.started:.2f}s",
                f"{latency:.2f}s",
                Text(_bar(latency / longest, 30, "▇", ""), style=r.color),
            )
        return table

    def view(self):
        with self.lock:
            steps, active, waiting, recent = list(self.steps), list(self.active), list(self.waiting), list(self.recent)
            tokens_done, busy_s = self.tokens_done, self.busy_s
        mode = (
            Text("continuous batching ON", style="bold green")
            if self.mode == BATCHED
            else Text("continuous batching OFF: one request at a time, 8k chunks where they fit", style="bold yellow")
        )
        rate = tokens_done / busy_s if busy_s else 0.0
        footer = Text(f"prefilled {tokens_done:,} tokens, {rate:,.0f} tok/s while busy", style="dim")
        body = Group(
            mode,
            self._lanes_table(steps),
            self._inflight_table(active, waiting, time.perf_counter()),
            self._finished_table(recent),
            footer,
        )
        return Panel(body, title="Gemma4 prefill, 8x4 Galaxy")


def _handler(scheduler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/prefill":
                try:
                    request = scheduler.submit(body.get("tokens", CHUNK), body.get("label"))
                except ValueError as error:
                    return self._reply({"error": str(error)}, 400)
                request.event.wait()
                self._reply(request.result(), 503 if request.error else 200)
            elif self.path == "/mode":
                scheduler.mode = BATCHED if body.get("batching", True) else SERIAL
                self._reply({"batching": scheduler.mode == BATCHED})
            elif self.path == "/shutdown":
                scheduler.stop.set()
                self._reply({"stopping": True})
            else:
                self._reply({"error": "unknown path"}, 404)

    return Handler


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_batching_server(mesh_device, reset_seeds, monkeypatch):
    num_slots = int(os.environ.get("DEMO_SLOTS", "4"))
    capacity = int(os.environ.get("DEMO_CAPACITY", "65536"))
    port = int(os.environ.get("DEMO_PORT", "8765"))
    # Batched steps keep HiFi2 projections, as an unbatched 2k chunk does (see BATCHING_POC.md).
    monkeypatch.setattr(attention_operations, "PROJECTION_FIDELITY_OVERRIDE", ttnn.MathFidelity.HiFi2)

    mesh_config, _, model = build_batched_model(mesh_device, CHUNK, num_slots, capacity)
    runner = StepRunner(mesh_device, mesh_config, model, CHUNK)
    # Mixed widths (the serial 8k lane next to 2k steps): every step takes the PrefillLanes path, and the widest
    # layout compiles first, so lazily created all-gather semaphores land above every layout's SDPA buffers.
    runner.always_lanes = True
    layouts = [[(0, 0, torch.zeros(WIDE, dtype=torch.int32))]]
    layouts += [
        [(slot, 0, torch.zeros(CHUNK, dtype=torch.int32)) for slot in range(n)] for n in range(num_slots, 0, -1)
    ]
    logger.info(f"[demo] compiling and capturing {len(layouts)} step traces")
    runner.capture_all(layouts)

    scheduler = Scheduler(runner, num_slots, capacity, _text_token_stream(_hf_model_id())[0])
    # The default listen backlog (5) makes a burst of clients wait out a TCP SYN retry (~0.5-1 s).
    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler(scheduler))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"[demo] serving on port {port}; fire prompts with demo/batching_client.py")
    try:
        with Live(get_renderable=scheduler.view, refresh_per_second=4):
            scheduler.run()
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.stop.set()
        server.shutdown()
        runner.release()
