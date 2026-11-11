# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Producer side of the V4.1 prefix-reuse H1 device check (``run_device_check.sh`` starts it next to ``runner_harness.py``).
Drives the runner the way tt-d-gen's prefill worker does: a prompt's chunks go into the H2D service back to back (12-byte
header: slot word, start, end; ``device.prefill.mark_last_chunk``: bit 29 of the slot word on every chunk, bit 30 on the
prompt's last), the next prompt only after EVERY layer ack (40 per chunk). After a prompt whose arm asks for a digest, one
EMPTY chunk at 0 makes the harness digest the export [0, S) + the model's rows (``[prefix-check] digest`` line); the
runtime then answers it as tt-d-gen's prefill_wait_for_arm_us empty chunk, so each next prompt follows a declined request.

Prompts (C = PREFILL_CHUNK_SIZE; ids from fixed seeds; every length a multiple of 128, as the engine's floor128(S - 2541)):
    X    2C unrelated tokens (dirties the caches before every cold reference)
    A    3C + 4992                 a first turn (ragged last chunk; 4992 % 256 = 128: the other half of the window ring)
    B    A[:C + 7040] + 20096      shares A's chunk 0 (+7040 tokens of chunk 1): resumes at C        <- the coordinator's pair
    A2   A + 12032                 the next turn of A: resumes at 3C
    AP   A[:2C]                    a strict prefix ending ON a boundary: skipped through, carries from the checkpoint
    U    2C + 1280 unrelated tokens (H2: moves the device off a conversation; its own chunks are stored too)
Phases (PREFIX_CHECK_PHASES, default R,S):
    R  cold references, each after X; A twice = the DETERMINISM check (cold == cold)
    S  the reuse sequence; every digest must equal the R digest of the same prompt
    U  the reuse sequence with UNMARKED headers (an engine without mark_last_chunk): equal for these prompts too
    T  timing, wall to the last ack: A2 off / cold / reuse-after-A, A2 again (every chunk skipped)
  H2 (the host chunk store, V41_PREFILL_PREFIX_STORE_GB > 0; run_device_check_h2.sh: PHASES=R2,S2[,T2]):
    R2 references with arm OFF (today's runner; the store is neither read nor written), each after X; A twice = determinism
    S2 the store sequence (arm reuse, the store emptied first): A, U, A2, U, A, U, AP, U, B, A2, A2 -- each digest == its R2
       reference AND each prompt's per-chunk pattern (S skipped / R restored / C computed / C+Ns silent rebuild, parsed from
       the runner log) == EXPECT_S2
    T2 timing (2 reps): A off, then A reuse (its full chunks read into the store), U, A2 off, A2 reuse (3 chunks restored)
Verdict lines: ``[driver] VERDICT determinism ...``, ``[driver] VERDICT reuse ...`` and (S2) ``[driver] VERDICT h2 pattern ...``;
results.json in PREFIX_CHECK_OUT.

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

ACKS_PER_CHUNK = int(os.environ.get("PREFILL_NUM_LAYERS", "40"))
CHUNK = pp.CHUNK_SIZE
OUT = os.environ.get("PREFIX_CHECK_OUT", ".")
CTL = os.environ["PREFIX_CHECK_CTL"]
RUNNER_LOG = os.environ["PREFIX_CHECK_RUNNER_LOG"]
MARKED, LAST = 1 << 29, 1 << 30
PHASES = os.environ.get("PREFIX_CHECK_PHASES", "R,S").split(",")

_rng = lambda seed, n: np.random.default_rng(seed).integers(1000, 120000, size=n).tolist()  # noqa: E731
X = _rng(7, 2 * CHUNK)
A = _rng(53, 3 * CHUNK + 4992)
B = A[: CHUNK + 7040] + _rng(17, 20096)
A2 = A + _rng(11, 12032)
AP = A[: 2 * CHUNK]
U = _rng(29, 2 * CHUNK + 1280)
PROMPTS = {"X": X, "A": A, "B": B, "A2": A2, "AP": AP, "U": U}
# H2's S2 sequence and each prompt's expected per-chunk pattern (chunk 8192: A 4 chunks, A2 6, U 3, AP 2, B 5)
S2 = ["A", "U", "A2", "U", "A", "U", "AP", "U", "B", "A2", "A2"]
EXPECT_S2 = [
    "CCCC",  # A: computed, its 3 full chunks stored
    "CCC",  # U: the device leaves A -> A's tips (3C, 2C) are read into the store after U's last chunk
    "RRRCCC",  # A2 = A + 12032: A's chunks restored, resumed on A's tip at 3C
    "RRC",  # U again: restored, its partial last chunk computed on U's tip at 2C
    "RRRC",  # A again (an exact repeat after another prompt)
    "RRC",
    "RR",  # AP = A[:2C]: its last chunk restored with the stored carries at 2C (export cleared past it)
    "RRC",
    "R,C+1s,C,C,C",  # B shares A's chunk 0 only: restored, no tip at C -> rebuilt silently from 0
    "SRRRRC",  # A2 after B: chunk 0 held on the device (H1), 1..4 from the store, resumed on A2's tip at 5C
    "SSSSSC",  # A2 again: held on the device, the partial last chunk recomputed
]
assert all(len(v) % 128 == 0 for v in PROMPTS.values()), {k: len(v) for k, v in PROMPTS.items()}
assert len(A2) <= int(os.environ.get("PREFILL_MAX_SEQ_LEN", 1 << 20)), "prompts exceed PREFILL_MAX_SEQ_LEN"


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


def run_prompt(phase: str, name: str, arm: str, *, digest=False, mark=True, timeout_s=900.0) -> dict:
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
    }
    r["chunk_acked_s"], r["wall_to_last_ack_s"] = done, done[-1]
    log(
        f"{phase} {name} ({S} tok, {len(chunks)} chunk(s)) arm {arm}{' digest' if digest else ''}"
        f"{'' if mark else ' UNMARKED'}: {'ok' if ok else 'HUNG'} wall {r['wall_to_last_ack_s']} s"
    )
    if not ok:
        raise SystemExit(
            f"prompt {tag} did not complete in {timeout_s} s (acks {acked - base} of {len(chunks) * ACKS_PER_CHUNK})"
        )
    if digest:
        base = acked
        push(toks, 0, 0, True, mark)  # the flush: the harness digests [0, S) and acks every layer
        r["flush_ok"] = wait_acks(base + ACKS_PER_CHUNK, 900.0)
    return r


def digests() -> dict:
    pat = re.compile(
        r"\[prefix-check\] digest (\S+) S=(\d+) export=(\w+) rows=(\w+) (?:slab=(\w+) engram=(\w+) )?.*tensors=(\S+)"
    )
    out = {}
    with open(RUNNER_LOG, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out[m.group(1)] = {
                    "S": int(m.group(2)),
                    "export": m.group(3),
                    "rows": m.group(4),
                    "slab": m.group(5),
                    "engram": m.group(6),
                    "tensors": m.group(7),
                }
    return out


def patterns() -> dict:
    """tag -> the prompt's per-chunk pattern from the runner log: S skipped, R restored (H2), C computed, C+Ns a computed chunk
    after N silently rebuilt ones."""
    tag_re = re.compile(r"\[prefix-check\] chunk \d+: prompt (\S+) arm")
    chunk_re = re.compile(r"\[v41 runtime\] chunk \[\d+, \d+\) slot \d+ (SKIPPED|RESTORED|in )")
    silent_re = re.compile(r"rebuilt the carries at \d+ from \d+ \((\d+) chunk")
    out, tag, silent = {}, None, 0
    with open(RUNNER_LOG, errors="replace") as f:
        for line in f:
            m = tag_re.search(line)
            if m:
                tag, silent = m.group(1), 0
                out[tag] = []
                continue
            m = silent_re.search(line)
            if m and tag is not None:
                silent = int(m.group(1))
                continue
            m = chunk_re.search(line)
            if m and tag is not None and "EMPTY" not in line:
                k = {"SKIPPED": "S", "RESTORED": "R"}.get(m.group(1), "C")
                out[tag].append(f"C+{silent}s" if k == "C" and silent else k)
                silent = 0
    return {t: ("".join(v) if all(len(x) == 1 for x in v) else ",".join(v)) for t, v in out.items()}


def runner_lines() -> list:
    keep = ("[v41-prefix", "[prefix-check]", "SKIPPED", "RESTORED", "[v41 runtime] chunk")
    with open(RUNNER_LOG, errors="replace") as f:
        return [l.rstrip("\n") for l in f if any(k in l for k in keep)]


def main() -> None:
    res = []
    log(
        f"attached: payload {PAYLOAD} B, chunk {CHUNK}, acks/chunk {ACKS_PER_CHUNK}, prompts "
        f"{{{', '.join(f'{k}: {len(v)}' for k, v in PROMPTS.items())}}}, phases {PHASES}"
    )
    res.append(run_prompt("warm", "X", "off"))
    if "R" in PHASES:
        for name in ("A", "A", "B", "A2", "AP"):  # A twice: run-to-run determinism of the cold prefill itself
            res.append(run_prompt("R", "X", "cold"))
            res.append(run_prompt("R", name, "cold", digest=True))
    if "S" in PHASES:
        res.append(run_prompt("S", "X", "cold"))
        for name in ("A", "B", "A", "A2", "A2", "AP", "A"):
            res.append(run_prompt("S", name, "reuse", digest=True))
    if "U" in PHASES:
        res.append(run_prompt("U", "X", "cold", mark=False))
        for name in ("A", "A2", "B"):
            res.append(run_prompt("U", name, "reuse", digest=True, mark=False))
    if "R2" in PHASES:  # H2 references: arm off = today's runner, the store untouched
        for name in ("A", "A", "U", "A2", "AP", "B"):
            res.append(run_prompt("R2", "X", "off"))
            res.append(run_prompt("R2", name, "off", digest=True))
    if "S2" in PHASES:
        res.append(run_prompt("S2", "X", "clear"))  # the store emptied, X computed (and stored)
        for name in S2:
            res.append(run_prompt("S2", name, "reuse", digest=True))
    for rep in range(2 if "T2" in PHASES else 0):
        res.append(run_prompt(f"T2{rep}", "X", "clear"))
        res.append(run_prompt(f"T2{rep}", "A", "off"))
        res.append(run_prompt(f"T2{rep}", "X", "off"))
        res.append(run_prompt(f"T2{rep}", "A", "reuse"))  # computed; its 3 full chunks read into the store
        res.append(run_prompt(f"T2{rep}", "U", "reuse"))  # the device leaves A: A's tips spilled
        res.append(run_prompt(f"T2{rep}", "A2", "off"))  # today's runner
        res.append(run_prompt(f"T2{rep}", "A2", "reuse"))  # 3 chunks restored, 3 computed (on A's tip at 3C)
    for rep in range(2 if "T" in PHASES else 0):
        res.append(run_prompt(f"T{rep}", "X", "off"))
        res.append(run_prompt(f"T{rep}", "A2", "off"))
        res.append(run_prompt(f"T{rep}", "X", "cold"))
        res.append(run_prompt(f"T{rep}", "A2", "cold"))
        res.append(run_prompt(f"T{rep}", "A", "reuse"))
        res.append(run_prompt(f"T{rep}", "A2", "reuse"))  # resumes at 3C
        res.append(run_prompt(f"T{rep}", "A2", "reuse"))  # every chunk skipped
    pp._push(service, PAYLOAD, pp._h2d_rows([1] * CHUNK, 0), None, struct.pack("<3i", -1, -1, -1))
    log("SHUTDOWN sentinel sent")
    time.sleep(10)
    d = digests()
    refs, ref_all = {}, {}
    ref_phase = {"S": "R", "U": "R", "S2": "R2"}
    for k, v in d.items():
        phase, name = k.split(".")[0], k.split(":")[1]
        if phase in ("R", "R2"):
            refs.setdefault((phase, name), v)
            ref_all.setdefault((phase, name), []).append(v)

    def same(a, b):
        keys = ["S", "export", "rows"] + [
            k for k in ("slab", "engram") if a.get(k) is not None and b.get(k) is not None
        ]
        return all(a[k] == b[k] for k in keys)

    det = ref_all.get(("R2", "A"), ref_all.get(("R", "A"), []))
    determinism = len(det) == 2 and same(det[0], det[1])
    log(
        f"VERDICT determinism (cold A twice, export + model rows (+ slab + Engram) byte-identical): "
        f"{'PASS' if determinism else 'FAIL'} {det}"
    )
    verdict = {}
    for k, v in d.items():
        phase, name = k.split(".")[0], k.split(":")[1]
        if phase in ref_phase:
            ref = refs.get((ref_phase[phase], name))
            verdict[k] = None if ref is None else same(v, ref)
    ok = bool(verdict) and all(verdict.values())
    log(
        f"VERDICT reuse (every S/U/S2 digest == the cold reference of the same prompt): {'PASS' if ok else 'FAIL'} {verdict}"
    )
    pats = patterns()
    if "S2" in PHASES:
        got = [pats.get(r["tag"]) for r in res if r["phase"] == "S2" and r["prompt"] != "X"]
        pat_ok = got == EXPECT_S2
        log(
            f"VERDICT h2 pattern (S2 per-chunk S/R/C == expected): {'PASS' if pat_ok else 'FAIL'} "
            f"{list(zip(S2, got, EXPECT_S2))}"
        )
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(
            {
                "results": res,
                "digests": d,
                "determinism": determinism,
                "verdict": verdict,
                "patterns": pats,
                "runner_lines": runner_lines(),
            },
            f,
            indent=1,
        )
    log(f"results -> {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    sys.exit(main())
