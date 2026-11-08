# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Producer side of the async-export device check (tt-blaze G53F-0309). Drives ``runner_harness.py`` the way tt-d-gen's
prefill worker drives the live runner: a prompt's chunks go into the H2D service back to back (12-byte header: slot,
start, end), and the next prompt only after EVERY layer ack of this one (46 per chunk) has arrived on the scheduler
channel. Per prompt it records the time each chunk's acks completed, and the wall from the first push to the last ack.

Phases (one runner process; the harness switches the export mode per prompt):
  T  timing: sync vs async on the same prompts, interleaved
  D  digest: GLM53F_EXPORT_DIGEST=1, sync vs async (and sync twice: run-to-run determinism)
  U  the pre-fix path (guard bypassed): a multi-chunk prompt must HANG at its last chunk; the driver then dumps the
     runner's Python (SIGUSR1) + native (gdb) stacks, switches the mode back to async and pushes one more prompt; the
     stuck acks arriving only after that push is the causal check
  A2 async again after the recovery, then the SHUTDOWN sentinel

    PREFILL_* env as the runner's (PREFILL_CHUNK_SIZE, PREFILL_H2D_SERVICE_ID, ...), EXPORT_CHECK_MODE_FILE,
    EXPORT_CHECK_RUNNER_PID, EXPORT_CHECK_RUNNER_LOG, EXPORT_CHECK_OUT (dir) -> python driver.py
"""

from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
import time

import numpy as np

import ttnn
from models.demos.common.prefill.runners import prefill_producer as pp

ACKS_PER_CHUNK = 46  # 45 decoder layers + the MTP row
CHUNK = pp.CHUNK_SIZE
OUT = os.environ.get("EXPORT_CHECK_OUT", ".")
MODE_FILE = os.environ["EXPORT_CHECK_MODE_FILE"]
RUNNER_LOG = os.environ["EXPORT_CHECK_RUNNER_LOG"]


def _runner_pid() -> int:
    """The harness's own pid (its first log line): the launcher's pid file holds the `timeout` wrapper, which neither
    forwards SIGUSR1 nor is the process to attach gdb to."""
    with open(RUNNER_LOG, errors="replace") as f:
        for line in f:
            if "[export-check] harness up: pid " in line:
                return int(line.split("harness up: pid ")[1].split(",")[0])
    return int(os.environ["EXPORT_CHECK_RUNNER_PID"])


RUNNER_PID = _runner_pid()
STACKS = os.environ.get("EXPORT_CHECK_STACKS", "/tmp/export_check_stacks.txt")

PROMPTS = {
    "P4": 8 * CHUNK,  # 8 full chunks, the last one FULL
    "P1": 3 * CHUNK + 5000,  # 3 full chunks + a ragged tail
    "P2": 3 * CHUNK,  # 3 full chunks, the last one FULL (a "short chunk = last" rule would miss it)
    "P3": 4092,  # one short chunk (the 22:02 hang's shape)
    "E": 0,  # one EMPTY chunk (a burst declined before compute: acks only)
}
POOL = np.random.default_rng(53).integers(1000, 150000, size=max(PROMPTS.values()) + CHUNK).tolist()


def log(msg: str) -> None:
    print(f"[driver] {time.strftime('%H:%M:%S')} {msg}", flush=True)


service = ttnn.H2DStreamService.connect(os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill"), timeout_ms=120000)
PAYLOAD = service.payload_size_bytes()
ack_channel = pp._connect_layer_ack_channel(120)
assert ack_channel is not None, "no layer-ack channel"
acked = 0


def set_mode(mode: str) -> None:
    with open(MODE_FILE + ".tmp", "w") as f:
        f.write(mode + "\n")
    os.replace(MODE_FILE + ".tmp", MODE_FILE)


def push(start: int, end: int, header=None) -> None:
    toks = POOL[start:end] + [1] * (CHUNK - (end - start))
    header = header or pp._pack_metadata(0, start, end)
    pp._push(service, PAYLOAD, pp._h2d_rows(toks, start if end > start else 0), None, header)


def wait_acks(target: int, timeout_s: float, marks=None):
    """Consume acks until ``acked >= target``; ``marks`` = {count: None} records when each count was reached."""
    global acked
    t0 = time.perf_counter()
    while acked < target:
        n = ack_channel.try_consume_all()
        if n:
            acked += n
            now = time.perf_counter()
            for k in list(marks or {}):
                if marks[k] is None and acked >= k:
                    marks[k] = now
        elif time.perf_counter() - t0 > timeout_s:
            return False
        else:
            time.sleep(0.0005)
    return True


def run_prompt(tag: str, name: str, mode: str, timeout_s: float = 180.0) -> dict:
    S = PROMPTS[name]
    chunks = [(s, min(S, s + CHUNK)) for s in range(0, S, CHUNK)] or [(0, 0)]
    set_mode(mode)
    base = acked
    marks = {base + (k + 1) * ACKS_PER_CHUNK: None for k in range(len(chunks))}
    t0 = time.perf_counter()
    for s, e in chunks:
        push(s, e)
    ok = wait_acks(base + len(chunks) * ACKS_PER_CHUNK, timeout_s, marks)
    done = [None if v is None else round(v - t0, 4) for v in marks.values()]
    r = {
        "tag": tag,
        "prompt": name,
        "tokens": S,
        "mode": mode,
        "chunks": len(chunks),
        "ok": ok,
        "acks": acked - base,
        "acks_expected": len(chunks) * ACKS_PER_CHUNK,
        "chunk_acked_s": done,
        "wall_to_last_ack_s": done[-1],
    }
    log(
        f"{tag} {name} ({S} tok, {len(chunks)} chunk(s)) {mode}: {'ok' if ok else 'HUNG'} acks {r['acks']}/"
        f"{r['acks_expected']} wall {r['wall_to_last_ack_s']} s, chunk acks at {done}"
    )
    return r


def stacks(tag: str) -> dict:
    """Python stacks (faulthandler, SIGUSR1) and native stacks (gdb) of the stuck runner."""
    size0 = os.path.getsize(STACKS) if os.path.exists(STACKS) else 0
    os.kill(RUNNER_PID, signal.SIGUSR1)
    time.sleep(2)
    with open(STACKS) as f:
        f.seek(size0)
        py = f.read()
    gdb = subprocess.run(
        [
            "sudo",
            "-n",
            "timeout",
            "300",
            "gdb",
            "-p",
            str(RUNNER_PID),
            "-batch",
            "-ex",
            "set pagination off",
            "-ex",
            "thread apply all bt 40",
        ],
        capture_output=True,
        text=True,
    )
    path = os.path.join(OUT, f"stacks_{tag}.txt")
    with open(path, "w") as f:
        f.write("=== python (faulthandler) ===\n" + py + "\n=== native (gdb) ===\n" + gdb.stdout + gdb.stderr)
    log(f"stacks -> {path} (python {len(py)} B, gdb rc {gdb.returncode}, {len(gdb.stdout)} B)")
    return {"stacks_file": path, "python_stacks": py}


def runner_lines() -> list:
    with open(RUNNER_LOG, errors="replace") as f:
        return [l.rstrip("\n") for l in f if "[glm53f-" in l or "[export-check]" in l or "CHUNK_START" in l]


def main() -> None:
    results = []
    log(f"attached: payload {PAYLOAD} B, chunk {CHUNK}, prompts {PROMPTS}, runner pid {RUNNER_PID}")
    results.append(run_prompt("warm", "P3", "sync"))
    results.append(run_prompt("warm", "P3", "async"))
    # T: timing, sync vs async interleaved
    for rep in range(2):
        for name in ("P4", "P1", "P2", "P3", "E"):
            for mode in ("sync", "async"):
                results.append(run_prompt(f"T{rep}", name, mode))
    # D: bytes
    for name, mode in (
        ("P4", "sync digest"),
        ("P4", "async digest"),
        ("P1", "sync digest"),
        ("P1", "async digest"),
        ("P1", "sync digest"),
    ):
        results.append(run_prompt("D", name, mode))
    # U: the pre-fix path must hang at the last chunk; then the causal check (stuck acks drain only after a push)
    u = run_prompt("U", "P1", "unguarded", timeout_s=60.0)
    results.append(u)
    if not u["ok"]:
        u.update(stacks("U"))
        set_mode("async")
        stuck_target = acked + (u["acks_expected"] - u["acks"])
        t_kick = time.perf_counter()
        push(0, PROMPTS["P3"])
        late = {stuck_target: None}
        ok = wait_acks(stuck_target, 120.0, late)
        u["stuck_acks_after_kick_s"] = None if late[stuck_target] is None else round(late[stuck_target] - t_kick, 4)
        ok2 = wait_acks(stuck_target + ACKS_PER_CHUNK, 120.0)
        u["recovered"] = bool(ok and ok2)
        log(f"U: stuck acks arrived {u['stuck_acks_after_kick_s']} s after the kick push; kick acked {ok2}")
    # A2: async after the recovery (a FULL last chunk), then shut the runner down
    results.append(run_prompt("A2", "P2", "async"))
    results.append(run_prompt("A2", "E", "async"))
    set_mode("async")
    push(0, 0, header=struct.pack("<3i", -1, -1, -1))
    log("SHUTDOWN sentinel sent")
    time.sleep(10)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"results": results, "runner_lines": runner_lines()}, f, indent=1)
    log(f"results -> {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    sys.exit(main())
