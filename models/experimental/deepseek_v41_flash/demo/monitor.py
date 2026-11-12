# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Web monitor for ``demo/server.py`` on the Rust host runtime (the default; ``--no-rust-host`` or
``DEEPSEEK_V41_RUST_HOST=0`` turns it off, and with it the monitor).

It runs as a process of its own, so watching the server costs the server nothing beyond what it publishes:

* the Rust host runtime keeps its counters (steps posted and read back, per-thread timings) in a small shared
  file, ``<prefix>.counters``, which its threads update anyway and this process maps read-only;
* the server writes a raw snapshot, ``<prefix>.json``, once a second (:class:`MonitorFeed`): the slots and who
  holds them, the active turns' token counts, the KV page pool, totals and the turns that ended, with ``--dspark``
  the drafts per verify round (the server's and each speculating turn's), and a log tail.
  Counts and timestamps only, and replaced by a rename, so a reader never sees half of one.

Every rate, average and time series is worked out here: decode tok/s in total and per user, prefill tok/s,
device steps/s, the runtime's per-step host timings, the KV pool's use over time. Then it serves them as a web
page (``monitor.html``) on ``--port``::

    python models/experimental/deepseek_v41_flash/demo/monitor.py --port 8090

``<prefix>`` is ``DEEPSEEK_V41_MONITOR`` (or ``--prefix``), by default ``/dev/shm/dsv41-monitor-<uid>``, so the
server and the monitor started by the same user find each other. Standard library only: it imports neither torch
nor ttnn, and never opens the device.
"""

from __future__ import annotations

import argparse
import itertools
import json
import mmap
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

COUNTERS_MAGIC = int.from_bytes(b"DSV41MON", "little")
COUNTERS_VERSION = 1
# host_rs/src/counters.rs's `Counters`, one u64 each, in order.
TIMERS = ("dispatch", "packet", "hash", "engram", "output_read")
COUNTER_FIELDS = (
    "magic",
    "version",
    "pid",
    "started_ns",
    "posted",
    "queued",
    "packets_written",
    "outputs_read",
    "last_output_ns",
    "closed",
) + tuple(f"{timer}_{part}" for timer in TIMERS for part in ("count", "ns"))

# Log records each snapshot carries (the newest); the monitor keeps a longer tail, merged by sequence number.
_FEED_LOG_LINES = 200
# A record's text is cut here before the monitor keeps it. The server logs a request's messages whole, which for a
# long conversation is a megabyte a record: a page's first fetch would then be some tens of megabytes, and the
# browser would still be reading it when the next one is due.
_LOG_MESSAGE_CHARS = 2000
# Ended turns each snapshot carries (the newest, which also holds the running totals).
FEED_ENDED_TURNS = 32
# Seconds over which a turn's live decode and prefill rates are taken.
_RATE_WINDOW = 5.0
_PAGE = Path(__file__).with_name("monitor.html")


def default_prefix() -> str:
    return os.environ.get("DEEPSEEK_V41_MONITOR") or f"/dev/shm/dsv41-monitor-{os.getuid()}"


def snapshot_path(prefix: str) -> str:
    return f"{prefix}.json"


def counters_path(prefix: str) -> str:
    return f"{prefix}.counters"


class MonitorFeed:
    """The server's side: every ``period`` seconds, ``snapshot()`` (a JSON-able dict of raw state) plus the newest
    log records goes to ``path``, written beside it and renamed over it. :meth:`log_sink` is a loguru sink."""

    def __init__(self, snapshot, path: str, period: float = 1.0):
        self.snapshot = snapshot
        self.path = path
        self.period = period
        self._log: deque = deque(maxlen=_FEED_LOG_LINES)
        self._log_seq = itertools.count(1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="monitor-feed", daemon=True)
        self._failed = False

    def log_sink(self, message) -> None:
        record = message.record
        text = record["message"]
        if len(text) > _LOG_MESSAGE_CHARS:
            text = f"{text[:_LOG_MESSAGE_CHARS]}... [+{len(text) - _LOG_MESSAGE_CHARS} chars]"
        self._log.append((next(self._log_seq), record["time"].timestamp(), record["level"].name, text))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        staged = f"{self.path}.{os.getpid()}.tmp"
        while not self._stop.wait(self.period):
            try:
                data = self.snapshot()
                data["log"] = list(self._log)
                with open(staged, "w") as f:
                    json.dump(data, f, default=str)
                os.replace(staged, self.path)
            except Exception as e:  # noqa: BLE001 - the monitor is never worth a server thread
                if not self._failed:
                    self._failed = True
                    print(f"[monitor feed: {e}]", file=sys.stderr, flush=True)


class _Counters:
    """``<prefix>.counters``, mapped read-only and remapped whenever the server replaces the file."""

    def __init__(self, path: str):
        self.path = path
        self._inode = None
        self._map = None
        self._words = None

    def read(self) -> dict | None:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        if st.st_ino != self._inode:
            self._words = None
            if self._map is not None:
                self._map.close()
                self._map = None
            if st.st_size < 8 * len(COUNTER_FIELDS):
                return None
            with open(self.path, "rb") as f:
                self._map = mmap.mmap(f.fileno(), 8 * len(COUNTER_FIELDS), prot=mmap.PROT_READ)
            self._words = memoryview(self._map).cast("Q")
            self._inode = st.st_ino
        values = dict(zip(COUNTER_FIELDS, self._words.tolist()))
        if values["magic"] != COUNTERS_MAGIC or values["version"] != COUNTERS_VERSION:
            return None
        return values


def _rate(points: deque, now: float) -> float:
    """Units per second over ``points`` (``(t, cumulative)``), dropping those older than the rate window."""
    while len(points) > 2 and now - points[1][0] >= _RATE_WINDOW:
        points.popleft()
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        return 0.0
    return max(points[-1][1] - points[0][1], 0) / (points[-1][0] - points[0][0])


class Monitor:
    """Reads the feed and the counters, and keeps everything the page shows: the latest state with its rates, a
    time series of one point per snapshot, the history of ended turns and the log tail."""

    def __init__(self, prefix: str, keep_seconds: float):
        self.snapshot_path = snapshot_path(prefix)
        self.counters = _Counters(counters_path(prefix))
        self.keep_seconds = keep_seconds
        self.lock = threading.Lock()
        self._reset(None)

    def _reset(self, started) -> None:
        self.started = started
        self.series: deque[dict] = deque()
        self.history: deque[dict] = deque(maxlen=5000)
        self.log: deque[list] = deque(maxlen=5000)
        self.images_done: deque[dict] = deque(maxlen=1000)  # the server keeps only the last few
        self.state: dict | None = None
        self._prev = None  # (snapshot, its live totals, counters, monitor time of the counters)
        self._turn_points: dict[int, tuple[deque, deque]] = {}  # serial -> ((t, generated), (t, prefilled))
        self._seen_serials: set[int] = set()
        self._log_seq = 0
        self._mtime = None

    # -- sampling --------------------------------------------------------------- #
    def poll(self) -> None:
        try:
            mtime = os.stat(self.snapshot_path).st_mtime_ns
        except FileNotFoundError:
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            with open(self.snapshot_path) as f:
                snap = json.load(f)
        except (OSError, ValueError):
            return  # replaced while being opened: the next poll reads the new one
        counters = self.counters.read()
        with self.lock:
            if snap.get("started") != self.started:
                self._reset(snap.get("started"))
                self._mtime = mtime
            self._ingest(snap, counters, time.monotonic())

    def _ingest(self, snap: dict, counters: dict | None, mono: float) -> None:
        t = snap["t"]
        ended = snap.get("ended_turns", [])
        ended_serials = {r["serial"] for r in ended}
        live = [turn for turn in snap.get("turns", []) if turn["serial"] not in ended_serials]
        last = ended[-1] if ended else {}
        generated = last.get("total_generated", 0) + sum(turn["generated"] for turn in live)
        prefilled = last.get("total_prefilled", 0) + sum(turn["prefilled"] for turn in live)
        # The prompts, as usage reports them: every prompt token is either a cache hit (``cached_tokens``, already in
        # the slot's KV at admission) or fed through prefill (``prompt_tokens``).
        input_tokens = last.get("total_input", 0) + sum(turn["prompt_tokens"] + turn["cached_tokens"] for turn in live)
        cached_tokens = last.get("total_cached", 0) + sum(turn["cached_tokens"] for turn in live)

        # Per turn: the live rates over the last few seconds.
        for turn in live:
            points = self._turn_points.setdefault(turn["serial"], (deque(), deque()))
            points[0].append((t, turn["generated"]))
            points[1].append((t, turn["prefilled"]))
            turn["decode_tps"] = _rate(points[0], t) if turn["phase"] == "decode" else 0.0
            if turn["phase"] == "prefill":
                turn["prefill_tps"] = _rate(points[1], t)
            else:  # the whole prompt's rate, as the history reports it
                prefill_s = turn.get("prefill_s", 0.0)
                turn["prefill_tps"] = turn["prompt_tokens"] / prefill_s if prefill_s > 0 else 0.0
        live_serials = {turn["serial"] for turn in live}
        for serial in list(self._turn_points):
            if serial not in live_serials:
                del self._turn_points[serial]

        for record in ended:
            if record["serial"] in self._seen_serials:
                continue
            self._seen_serials.add(record["serial"])
            record["prefill_tps"] = record["prompt_tokens"] / record["prefill_s"] if record["prefill_s"] > 0 else 0.0
            record["decode_tps"] = record["generated"] / record["decode_s"] if record["decode_s"] > 0 else 0.0
            record["ttft_s"] = record["queued_s"] + record["prefill_s"]
            self.history.append(record)

        images = snap.get("images") or {}
        for entry in images.get("done", []):
            if entry["at"] > (self.images_done[-1]["at"] if self.images_done else 0):
                self.images_done.append(entry)

        for seq, stamp, level, message in snap.get("log", []):
            if seq > self._log_seq:
                self._log_seq = seq
                if len(message) > _LOG_MESSAGE_CHARS:  # the feed may predate this cut (a server not restarted)
                    message = f"{message[:_LOG_MESSAGE_CHARS]}... [+{len(message) - _LOG_MESSAGE_CHARS} chars]"
                self.log.append([seq, stamp, level, message])

        point = {"t": t, "decode": 0.0, "prefill": 0.0, "users": {}}
        runtime = None
        if self._prev is not None:
            prev, (prev_generated, prev_prefilled), prev_counters, prev_mono = self._prev
            dt = t - prev["t"]
            if dt > 0:
                point["decode"] = max(generated - prev_generated, 0) / dt
                point["prefill"] = max(prefilled - prev_prefilled, 0) / dt
                before = {turn["serial"]: turn for turn in prev.get("turns", [])}
                finals = {r["serial"]: r["generated"] for r in ended}
                for serial, old in before.items():
                    now_generated = finals.get(serial)
                    if now_generated is None:
                        current = next((turn for turn in live if turn["serial"] == serial), None)
                        if current is None or current["phase"] != "decode":
                            continue
                        now_generated = current["generated"]
                    label = old["user"] or f"slot {old['slot']}"
                    rate = max(now_generated - old["generated"], 0) / dt
                    point["users"][label] = point["users"].get(label, 0.0) + rate
            if counters is not None and prev_counters is not None and counters["pid"] == prev_counters["pid"]:
                cdt = mono - prev_mono
                runtime = {"steps_ps": (counters["outputs_read"] - prev_counters["outputs_read"]) / cdt if cdt else 0}
                for timer in TIMERS:
                    count = counters[f"{timer}_count"] - prev_counters[f"{timer}_count"]
                    nanos = counters[f"{timer}_ns"] - prev_counters[f"{timer}_ns"]
                    runtime[f"{timer}_us"] = nanos / count / 1e3 if count > 0 else None
        if runtime is not None:
            point["steps"] = runtime["steps_ps"]
            point["timers"] = {timer: runtime[f"{timer}_us"] for timer in TIMERS}
        states = snap.get("slot_states", {})
        for key in ("running", "queued", "parked", "empty"):
            point[key] = states.get(key, 0)
        point["host"] = snap.get("host", {}).get("users", 0)
        point["kv"] = {name: used / total if total else 0.0 for name, (used, total) in snap.get("pool", {}).items()}
        point["tokens_left"] = snap.get("tokens_left", 0)
        if snap.get("dspark") is not None:
            point["drafts"] = snap["dspark"]["drafts"]
        if images:
            point["images_live"] = len(images.get("live", []))
            point["image_rows_free"] = images.get("rows_free", 0)
        self.series.append(point)
        while self.series and t - self.series[0]["t"] > self.keep_seconds:
            self.series.popleft()

        self._prev = (snap, (generated, prefilled), counters, mono)
        snap.pop("log", None)
        snap["turns"] = live
        snap["totals"] = {
            "generated": generated,
            "prefilled": prefilled,
            "input": input_tokens,
            "cached": cached_tokens,
            "turns": last.get("total_turns", 0),
        }
        snap["runtime"] = runtime
        snap["counters"] = counters
        self.state = snap

    def run(self, period: float) -> None:
        while True:
            try:
                self.poll()
            except Exception as e:  # noqa: BLE001 - keep watching
                print(f"[monitor: {e}]", file=sys.stderr, flush=True)
            time.sleep(period)

    # -- serving ---------------------------------------------------------------- #
    def view(self, since: float, log_since: int, history_since: float, images_since: float) -> dict:
        with self.lock:
            state = self.state
            return {
                "now": time.time(),
                "started": self.started,
                "state": state,
                "series": [p for p in self.series if p["t"] > since],
                "history": [r for r in self.history if r["ended"] > history_since],
                "images_done": [e for e in self.images_done if e["at"] > images_since],
                "log": [line for line in self.log if line[0] > log_since],
            }


class _Handler(BaseHTTPRequestHandler):
    monitor: Monitor

    def log_message(self, *args) -> None:
        pass

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            query = parse_qs(url.query)

            def number(name, cast):
                try:
                    return cast(query.get(name, ["0"])[0])
                except ValueError:
                    return cast(0)

            view = self.monitor.view(
                number("since", float),
                number("log_since", int),
                number("history_since", float),
                number("images_since", float),
            )
            self._send(json.dumps(view, default=str).encode(), "application/json")
        else:
            self.send_error(404)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default=default_prefix(), help="the server's feed files, without .json/.counters")
    p.add_argument("--host", default="0.0.0.0", help="interface to serve the page on (default: all)")
    p.add_argument("--port", type=int, default=8090, help="port to serve the page on (default: 8090)")
    p.add_argument("--poll", type=float, default=0.25, help="seconds between looks at the feed (default: 0.25)")
    p.add_argument("--keep-minutes", type=float, default=120, help="time series kept (default: 120)")
    args = p.parse_args(argv)
    monitor = Monitor(args.prefix, args.keep_minutes * 60)
    threading.Thread(target=monitor.run, args=(args.poll,), name="monitor-poll", daemon=True).start()
    _Handler.monitor = monitor
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"monitoring {args.prefix}.json / .counters on http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
