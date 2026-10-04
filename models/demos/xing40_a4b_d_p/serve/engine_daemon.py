# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-d-gen's real engine as a long-lived prefill scheduler for the Xing serve stack (serve/README.md).

    <tt-d-gen python 3.12> engine_daemon.py <config.json>

Runs under tt-d-gen's interpreter with only the standard library and `tt_engine` (as
models/demos/common/bringup/testing/dgen_prefill_driver.py, whose engine setup it shares): ttnn and the tokenizer live
in other processes. One te.BackendRuntime (PREFILL role, te.device_prefill_pipeline on the runner's H2D service and
layer-ack channel) for the whole life of the server; the engine decides slots, chunks, prefix reuse (remounts) and the
interleave of concurrent requests, exactly as in production.

Clients (serve/server.py) talk JSON lines over the Unix socket config["socket"]:
    {"op": "admit", "token_ids": [int, ...]}
        -> once the request's PREFILL_DONE arrives (every chunk retired on its layer acks):
           {"ok": true, "rid", "slot", "resident", "reused", "done_position", "prompt_len", "admit_s", "done_s"}
           or {"ok": false, "error": "..."} (REJECTED / ABORTED / a short PREFILL_DONE / a bad request)
    {"op": "stats"} -> {"ok": true, "telemetry": {every scalar of te.TelemetrySnapshot, refreshed each second},
                        "inflight": n, "served": n, "errors": n}
One request per connection line; a connection may send several lines (answered in order).

config.json: socket, service_id, ack_shm_name, chunk_size, layers_per_chunk, sp_factor, max_slots, max_seq_len,
kv_block_size, connect_timeout_ms, mock (true: te.mock_prefill_pipeline, no device, for a dry run), pidfile (written
once ready: the supervisor's SIGTERM goes to this pid, not to the sh wrapper that sends the sentinel), stall_s (an
admitted request with no engine progress for this long = the server is stuck: the daemon exits 3 and the supervisor
stops the stack), status (path: a JSON status file rewritten every second).
Exit: 0 on SIGTERM / SIGINT (rt.stop() drains the H2D stream and releases the ack channel; the caller then sends the
runner's shutdown sentinel), 3 on a stall, 5 on a bad config / build / attach.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import socketserver
import sys
import threading
import time

TELEMETRY_SKIP = ("HIST_BOUNDS", "hist", "beat_us")  # arrays; everything else in the snapshot is a scalar


def telemetry_dict(tel) -> dict:
    """Every scalar field of tt-d-gen's TelemetrySnapshot (engine/include/engine/runtime/telemetry.hpp), as is."""
    out = {}
    for k in dir(tel):
        if k.startswith("_") or k in TELEMETRY_SKIP:
            continue
        v = getattr(tel, k)
        if isinstance(v, (int, float)):
            out[k] = v
    return out


def log(msg: str) -> None:
    print(f"[engine daemon] {time.strftime('%H:%M:%S')} {msg}", flush=True)


class Pending:
    """One admit waiting for its PREFILL_DONE."""

    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids
        self.reply: dict | None = None
        self.done = threading.Event()
        self.info: dict = {"prompt_len": len(token_ids)}


def main(cfg_path: str) -> int:
    cfg = json.loads(open(cfg_path).read())
    try:
        import tt_engine as te
    except Exception as e:
        log(f"import tt_engine: {type(e).__name__}: {e}")
        return 5

    rc = te.RuntimeConfig()
    rc.role = te.RuntimeRole.PREFILL
    rc.max_slots = int(cfg["max_slots"])
    rc.max_seq_len = int(cfg["max_seq_len"])
    rc.chunk_size = int(cfg["chunk_size"])
    rc.kv_block_size = int(cfg["kv_block_size"])
    if cfg.get("mock"):  # host only (te.mock_prefill_pipeline acks every chunk itself): a dry run of the protocol
        pipe = te.mock_prefill_pipeline(int(cfg["layers_per_chunk"]), auto_ack=True)
    else:
        pipe = te.device_prefill_pipeline(
            cfg["service_id"],
            cfg["ack_shm_name"],
            rc.chunk_size,
            int(cfg["layers_per_chunk"]),
            int(cfg.get("connect_timeout_ms", 600000)),
            int(cfg["sp_factor"]),
        )
    log(
        f"PREFILL runtime: max_slots {rc.max_slots}, max_seq_len {rc.max_seq_len}, chunk {rc.chunk_size}, "
        f"kv_block_size {rc.kv_block_size}, evict {rc.evict_enabled}; service {cfg['service_id']!r} ({te.__file__})"
    )
    t0 = time.monotonic()
    try:
        rt = te.BackendRuntime(rc, pipe)
    except Exception as e:
        log(f"BackendRuntime: {type(e).__name__}: {e}")
        return 5
    rt.event_handle()  # required before any admit
    log(f"attached in {time.monotonic() - t0:.1f} s")

    inbox: queue.Queue[Pending] = queue.Queue()
    stop = threading.Event()
    counters = {"served": 0, "errors": 0}
    telemetry: dict = {}
    by_rid: dict[int, Pending] = {}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                try:
                    req = json.loads(line)
                    op = req.get("op")
                    if op == "admit":
                        ids = [int(t) for t in req["token_ids"]]
                        if not 0 < len(ids) <= rc.max_seq_len:
                            reply = {"ok": False, "error": f"prompt of {len(ids)} tokens (1..{rc.max_seq_len})"}
                        else:
                            p = Pending(ids)
                            inbox.put(p)
                            while not p.done.wait(1.0):
                                if stop.is_set():
                                    p.reply = {"ok": False, "error": "engine daemon stopping"}
                                    break
                            reply = p.reply
                    elif op == "stats":
                        reply = {"ok": True, "telemetry": telemetry, "inflight": len(by_rid), **counters}
                    else:
                        reply = {"ok": False, "error": f"unknown op {op!r}"}
                except Exception as e:  # a bad line answers with an error, the connection stays usable
                    reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.wfile.write((json.dumps(reply) + "\n").encode())
                self.wfile.flush()

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    sock = cfg["socket"]
    if os.path.exists(sock):
        os.unlink(sock)
    srv = Server(sock, Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stop.set())
    if cfg.get("pidfile"):
        with open(cfg["pidfile"], "w") as f:
            f.write(str(os.getpid()))
    log(f"ENGINE_READY on {sock}")

    stall_s = float(cfg.get("stall_s", 600))
    status_path = cfg.get("status")
    code = 0
    last_progress = time.monotonic()
    last_status = 0.0

    def finish(p: Pending, reply: dict) -> None:
        p.reply = {**p.info, **reply}
        p.done.set()
        counters["served" if reply.get("ok") else "errors"] += 1

    try:
        while not stop.is_set():
            while True:  # admits happen on this thread only: the control plane has one writer
                try:
                    p = inbox.get_nowait()
                except queue.Empty:
                    break
                req = te.InferenceRequest()
                req.token_ids = p.token_ids
                try:
                    rid = rt.admit(req)
                except Exception as e:
                    finish(p, {"ok": False, "error": f"admit: {type(e).__name__}: {e}"})
                    continue
                p.info.update(rid=rid, admit_s=round(time.monotonic() - t0, 3))
                by_rid[rid] = p
                last_progress = time.monotonic()
            worked = rt.tick()
            evs = rt.drain()
            now = time.monotonic()
            if worked or evs or not by_rid:
                last_progress = now
            for ev in evs:
                p = by_rid.get(ev.request_id)
                if p is None:
                    continue
                if ev.kind == te.EventKind.ADMITTED:
                    p.info.update(slot=ev.slot_id, resident=ev.position_id, reused=ev.reused_tokens)
                elif ev.kind == te.EventKind.PREFILL_DONE:
                    del by_rid[ev.request_id]
                    p.info.update(done_position=ev.position_id, done_s=round(now - t0, 3))
                    if ev.position_id != p.info["prompt_len"]:
                        finish(
                            p,
                            {"ok": False, "error": f"PREFILL_DONE at {ev.position_id}, prompt {p.info['prompt_len']}"},
                        )
                    else:
                        finish(p, {"ok": True})
                elif ev.kind in (te.EventKind.REJECTED, te.EventKind.ABORTED):
                    del by_rid[ev.request_id]
                    finish(p, {"ok": False, "error": repr(ev)})
                    log(f"{ev!r} (rid {ev.request_id})")
            if by_rid and now - last_progress > stall_s:
                log(f"STALL: no engine progress for {stall_s:.0f} s with {len(by_rid)} request(s) in flight")
                code = 3
                break
            if now - last_status > 1.0:
                last_status = now
                telemetry.update(telemetry_dict(rt.telemetry_snapshot()), t=time.time())
                if status_path:
                    tmp = status_path + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump({"t": time.time(), "inflight": len(by_rid), **counters, "telemetry": telemetry}, f)
                    os.replace(tmp, status_path)
            if not worked and not evs:
                time.sleep(0.001)
    except Exception as e:
        log(f"engine loop: {type(e).__name__}: {e}")
        code = 2
    finally:
        stop.set()
        for p in list(by_rid.values()):
            finish(p, {"ok": False, "error": "engine daemon stopping"})
        srv.shutdown()
        log(f"stop: draining the H2D stream, releasing the ack channel ({counters})")
        try:
            rt.stop()
        except Exception as e:
            log(f"stop: {type(e).__name__}: {e}")
            code = code or 2
        if os.path.exists(sock):
            os.unlink(sock)
    log(f"exit {code}")
    return code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(5)
    sys.exit(main(sys.argv[1]))
