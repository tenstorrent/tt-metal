# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Producer side of the prefix-reuse device check (tt-blaze G53F-0311). Drives ``runner_harness.py`` the way tt-d-gen's
prefill worker drives the live runner: a prompt's chunks go into the H2D service back to back (12-byte header: slot,
start, end; tt-d-gen ``device.prefill.mark_last_chunk``: bit 29 of the slot word on every chunk, bit 30 on the prompt's
last chunk), the next
prompt only after EVERY layer ack (46 per chunk). After a prompt whose arm asks for a digest, one EMPTY chunk at 0 makes
the harness digest the export slabs [0, S) (what the KV Manager would ship).

Prompts (C = PREFILL_CHUNK_SIZE; token pools from fixed seeds):
    X    2C unrelated tokens (dirties the caches between arms)
    A    3C + 5000                       a first turn (ragged last chunk)
    A2   A + 12000                        the next turn: resumes at 3C
    A3   A2 + 30000                       the turn after: resumes at 4C; then sent again unchanged (every chunk skipped)
    B    A[:C + 7000] + 20000             a sibling sharing 1.57 chunks: resumes at C
    Ap   A[:2C]                           a strict prefix ending ON a boundary (its full last chunk is the burst's last)
Phases: R = cold references (arm cold, after X), S = the reuse sequence (each prompt's digest must equal its R digest),
T = timing (arm off vs cold vs reuse on the same prompts), U = the reuse sequence again with UNMARKED headers (an
engine without mark_last_chunk: the runner skips a full chunk only where it holds the end carries).

    PREFILL_* env as the runner's, PREFIX_CHECK_CTL, PREFIX_CHECK_RUNNER_LOG, PREFIX_CHECK_OUT -> python driver.py
"""

from __future__ import annotations

import json
import os
import re
import struct
import sys
import time

import numpy as np

import ttnn
from models.demos.common.prefill.runners import prefill_producer as pp

ACKS_PER_CHUNK = 46
CHUNK = pp.CHUNK_SIZE
OUT = os.environ.get("PREFIX_CHECK_OUT", ".")
CTL = os.environ["PREFIX_CHECK_CTL"]
RUNNER_LOG = os.environ["PREFIX_CHECK_RUNNER_LOG"]
MARKED, LAST = 1 << 29, 1 << 30

_rng = lambda seed, n: np.random.default_rng(seed).integers(1000, 150000, size=n).tolist()  # noqa: E731
A = _rng(53, 3 * CHUNK + 5000)
X = _rng(7, 2 * CHUNK)
A2 = A + _rng(11, 12000)
A3 = A2 + _rng(13, 30000)
B = A[: CHUNK + 7000] + _rng(17, 20000)
Ap = A[: 2 * CHUNK]
# phase 2 (several retained prompts, GLM53F_PREFIX_HOST_GB > 0): two conversations and two siblings, interleaved
M_A1 = _rng(101, 4 * CHUNK + 3000)
M_B1 = _rng(103, 3 * CHUNK + 2000)
M_A2 = M_A1 + _rng(105, 2 * CHUNK)  # 6C + 3000: A1's 4 full chunks stored, resume at 4C on A1's spilled tip
M_B2 = M_B1 + _rng(107, CHUNK + 5000)  # 4C + 7000: resume at 3C on B1's spilled tip
M_SYS = _rng(109, 2 * CHUNK + 500)
M_S1 = M_SYS + _rng(111, CHUNK)
M_S2 = M_SYS + _rng(113, CHUNK + 4000)  # 2 chunks restored from S1's, carries rebuilt (no boundary of S1 spilled there)
PROMPTS = {
    "X": X,
    "A": A,
    "A2": A2,
    "A3": A3,
    "B": B,
    "Ap": Ap,
    "MA1": M_A1,
    "MB1": M_B1,
    "MA2": M_A2,
    "MB2": M_B2,
    "MS1": M_S1,
    "MS2": M_S2,
}
PHASES = os.environ.get("PREFIX_CHECK_PHASES", "R,S,T,U").split(",")


def log(msg: str) -> None:
    print(f"[driver] {time.strftime('%H:%M:%S')} {msg}", flush=True)


service = ttnn.H2DStreamService.connect(os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill"), timeout_ms=120000)
PAYLOAD = service.payload_size_bytes()
ack_channel = pp._connect_layer_ack_channel(120)
assert ack_channel is not None, "no layer-ack channel"
acked = 0


def set_ctl(words: str) -> None:
    with open(CTL + ".tmp", "w") as f:
        f.write(words + "\n")
    os.replace(CTL + ".tmp", CTL)


def push(toks: list, start: int, end: int, last: bool, mark: bool = True) -> None:
    body = toks[start:end] + [1] * (CHUNK - (end - start))
    word = (MARKED | (LAST if last else 0)) if mark else 0
    header = struct.pack("<3i", word, start, end)
    pp._push(service, PAYLOAD, pp._h2d_rows(body, start if end > start else 0), None, header)


def wait_acks(target: int, timeout_s: float, marks=None) -> bool:
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


_seq = [0]


def run_prompt(phase: str, name: str, arm: str, *, digest=False, mark=True, timeout_s=300.0) -> dict:
    toks = PROMPTS[name]
    S = len(toks)
    chunks = [(s, min(S, s + CHUNK)) for s in range(0, S, CHUNK)]
    _seq[0] += 1
    tag = f"{phase}.{_seq[0]}:{name}"
    set_ctl(f"{arm} {tag}{' digest' if digest else ''}")
    base = acked
    marks = {base + (k + 1) * ACKS_PER_CHUNK: None for k in range(len(chunks))}
    t0 = time.perf_counter()
    for k, (s, e) in enumerate(chunks):
        push(toks, s, e, k == len(chunks) - 1, mark)
    ok = wait_acks(base + len(chunks) * ACKS_PER_CHUNK, timeout_s, marks)
    done = [None if v is None else round(v - t0, 4) for v in marks.values()]
    r = {
        "tag": tag,
        "phase": phase,
        "prompt": name,
        "tokens": S,
        "arm": arm,
        "mark": mark,
        "chunks": len(chunks),
        "ok": ok,
        "chunk_acked_s": done,
        "wall_to_last_ack_s": done[-1],
    }
    log(
        f"{phase} {name} ({S} tok, {len(chunks)} chunk(s)) arm {arm}{' digest' if digest else ''}: "
        f"{'ok' if ok else 'HUNG'} wall {r['wall_to_last_ack_s']} s, chunk acks at {done}"
    )
    if digest and ok:
        base = acked
        push(toks, 0, 0, True, mark)  # the flush (an empty chunk, a request of its own): the harness digests [0, S)
        r["flush_ok"] = wait_acks(base + ACKS_PER_CHUNK, 300.0)
    return r


def digests() -> dict:
    out = {}  # tag -> digest, in log order
    pat = re.compile(r"\[prefix-check\] digest (\S+) S=(\d+) dsa=(\w+) kda_rec=(\S+) kda_conv=(\S+)")
    with open(RUNNER_LOG, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out[m.group(1)] = {
                    "S": int(m.group(2)),
                    "dsa": m.group(3),
                    "kda_rec": m.group(4),
                    "kda_conv": m.group(5),
                }
    return out


def runner_lines() -> list:
    with open(RUNNER_LOG, errors="replace") as f:
        return [l.rstrip("\n") for l in f if "[glm53f-" in l or "[prefix-check]" in l or "CHUNK_START" in l]


def main() -> None:
    res = []
    log(
        f"attached: payload {PAYLOAD} B, chunk {CHUNK}, prompts {{{', '.join(f'{k}: {len(v)}' for k, v in PROMPTS.items())}}}"
    )
    res.append(run_prompt("warm", "X", "off"))
    if "R" in PHASES:
        # R: cold references, each after an unrelated prompt (different stale rows than the reuse runs see)
        for name in ("A", "A2", "A3", "B", "Ap"):
            res.append(run_prompt("R", "X", "cold"))
            res.append(run_prompt("R", name, "cold", digest=True))
        res.append(run_prompt("R2", "X", "cold"))
        res.append(run_prompt("R2", "A3", "cold", digest=True))  # run-to-run determinism of the reference itself
    if "S" in PHASES:
        # S: the reuse sequence; every digest must equal R's
        res.append(run_prompt("S", "X", "cold"))
        for name in ("A", "A2", "A3", "A3", "B", "A", "Ap", "A3"):
            res.append(run_prompt("S", name, "reuse", digest=True))
    if "M" in PHASES:
        # M (phase 2): references with retention OFF (a cold arm would still find chunks in the host store), then
        # two conversations and two siblings interleaved, each digest equal to its reference; timing of a next turn
        # whose conversation the device left (restored from the store) against the same prompt cold
        for name in ("MA1", "MB1", "MA2", "MB2", "MS1", "MS2"):
            res.append(run_prompt("R", "X", "off"))
            res.append(run_prompt("R", name, "off", digest=True))
        res.append(run_prompt("M", "X", "off"))
        for name in ("MA1", "MB1", "MA2", "MB2", "MS1", "X", "MS2", "MA2", "MB2"):
            res.append(run_prompt("M", name, "off" if name == "X" else "reuse", digest=name != "X"))
        for rep in range(2):
            res.append(run_prompt(f"MT{rep}", "X", "off"))
            res.append(run_prompt(f"MT{rep}", "MA2", "off"))
            res.append(run_prompt(f"MT{rep}", "MB2", "reuse"))  # the device now holds B2; A's chunks only in the store
            res.append(run_prompt(f"MT{rep}", "MA2", "reuse"))  # restored + resumed at A2's tip: the ragged tail only
    for rep in range(2 if "T" in PHASES else 0):
        res.append(run_prompt(f"T{rep}", "X", "off"))
        res.append(run_prompt(f"T{rep}", "A3", "off"))
        res.append(run_prompt(f"T{rep}", "X", "cold"))
        res.append(run_prompt(f"T{rep}", "A3", "cold"))
        res.append(run_prompt(f"T{rep}", "A2", "reuse"))
        res.append(run_prompt(f"T{rep}", "A3", "reuse"))  # extends A2: resume at 4C
        res.append(run_prompt(f"T{rep}", "A3", "reuse"))  # unchanged: every chunk skipped
    if "U" in PHASES:
        # U: no last-chunk mark from here on (the runner must notice and stay correct)
        res.append(run_prompt("U", "X", "cold", mark=False))
        for name in ("A", "A2", "A3", "B", "Ap"):
            res.append(run_prompt("U", name, "reuse", digest=True, mark=False))
    pp._push(service, PAYLOAD, pp._h2d_rows([1] * CHUNK, 0), None, struct.pack("<3i", -1, -1, -1))
    log("SHUTDOWN sentinel sent")
    time.sleep(10)
    d = digests()
    refs = {}
    for k, v in d.items():
        phase, name = k.split(".")[0], k.split(":")[1]
        if phase == "R":
            refs.setdefault(name, v)
    verdict = {}
    for k, v in d.items():
        phase, name = k.split(".")[0], k.split(":")[1]
        if phase in ("S", "U", "R2", "M"):
            ref = refs.get(name)
            verdict[k] = None if ref is None else all(v[f] == ref[f] for f in ("S", "dsa", "kda_rec", "kda_conv"))
    log(f"digest verdicts vs the cold references: {verdict}")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"results": res, "digests": d, "verdict": verdict, "runner_lines": runner_lines()}, f, indent=1)
    log(f"results -> {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    sys.exit(main())
