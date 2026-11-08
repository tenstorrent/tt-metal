# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Producer side of the fixed-shapes device check (tt-blaze G53F-0313): drives ``runner_harness.py`` the way tt-d-gen's
prefill worker drives the live runner (a prompt's chunks into the H2D service back to back, 12-byte header slot / start /
end, the next prompt only after every layer ack), with the arm of each prompt in ``FS_CHECK_CTL`` (``<0|1> <tag> <S>``).

Phases (token ids from one fixed-seed pool; C = PREFILL_CHUNK_SIZE = 12288 in the serving recipe):
  G  growth, the 11:14 probe pattern: SAME chunk start and padded bucket, ends 100 tokens apart -- fixed ON first (fresh
     ends), then the per-length path at other fresh ends; at 3C + 4.2k..4.7k (bucket 5120), single-chunk 3.1k..3.4k
     (bucket 3584), and 8C + 10.0k..10.2k (bucket 10240);
  D  bit identity: every length once per arm (order alternating), digests compared; one OFF repeat (determinism);
  T  warmed timing: lengths both arms have seen, two reps.
    PREFILL_* env as the runner's, FS_CHECK_CTL, FS_CHECK_OUT -> python driver.py
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
C = pp.CHUNK_SIZE
OUT = os.environ.get("FS_CHECK_OUT", ".")
CTL = os.environ["FS_CHECK_CTL"]
RUNNER_LOG = os.environ["FS_CHECK_RUNNER_LOG"]
POOL = np.random.default_rng(53).integers(1000, 150000, size=120000).tolist()
X = np.random.default_rng(7).integers(1000, 150000, size=2 * C).tolist()


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


def wait_acks(target: int, timeout_s: float) -> bool:
    global acked
    t0 = time.perf_counter()
    while acked < target:
        n = ack_channel.try_consume_all()
        if n:
            acked += n
        elif time.perf_counter() - t0 > timeout_s:
            return False
        else:
            time.sleep(0.0005)
    return True


_seq = [0]


def run(phase: str, n: int, fixed: int, toks=None) -> dict:
    toks = POOL[:n] if toks is None else toks
    S = len(toks)
    chunks = [(s, min(S, s + C)) for s in range(0, S, C)]
    _seq[0] += 1
    tag = f"{phase}.{_seq[0]}:{S}:{'on' if fixed else 'off'}"
    set_ctl(f"{fixed} {tag} {S}")
    base = acked
    t0 = time.perf_counter()
    for s, e in chunks:
        body = toks[s:e] + [1] * (C - (e - s))
        pp._push(service, PAYLOAD, pp._h2d_rows(body, s), None, struct.pack("<3i", 0, s, e))
    ok = wait_acks(base + len(chunks) * ACKS_PER_CHUNK, 900.0)
    wall = round(time.perf_counter() - t0, 3)
    log(f"{tag}: {len(chunks)} chunk(s) {'ok' if ok else 'HUNG'} wall {wall} s")
    if not ok:
        raise SystemExit(f"prompt {tag} hung")
    return {"tag": tag, "phase": phase, "S": S, "fixed": fixed, "chunks": len(chunks), "wall_s": wall}


def digests() -> dict:
    pat = re.compile(
        r"\[fs-check\] digest (\S+) S=(\d+) fixed=(\d) dsa=(\w+) kda_rec=(\w+) kda_conv=(\w+) programs_added=(-?\d+) "
        r"chunk_wall=\[([^\]]*)\]"
    )
    out = {}
    with open(RUNNER_LOG, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out[m.group(1)] = {
                    "S": int(m.group(2)),
                    "fixed": int(m.group(3)),
                    "dsa": m.group(4),
                    "kda_rec": m.group(5),
                    "kda_conv": m.group(6),
                    "programs_added": int(m.group(7)),
                    "chunk_wall": [float(x) for x in m.group(8).split(",") if x.strip()],
                }
    return out


def main() -> None:
    res = []
    phases = os.environ.get("FS_CHECK_PHASES", "G,D,T").split(",")
    log(f"attached: payload {PAYLOAD} B, chunk {C}, phases {phases}")
    res.append(run("warm", 2 * C, 0, X))
    if "G" in phases:
        for base, on_ends, off_ends in (
            (3 * C, range(4199, 4700, 100), range(4149, 4650, 100)),
            (0, range(3101, 3402, 100), range(3151, 3452, 100)),
            (8 * C, range(10001, 10202, 100), range(10051, 10152, 100)),
        ):
            for e in on_ends:
                res.append(run("G", base + e, 1))
            for e in off_ends:
                res.append(run("G", base + e, 0))
    lens = [1001, 1003, 2047, 2051, 2501, 4093, 7001, 9001, 12287, 24576, 41863, 41857, 41858, 102401, 110003]
    if "D" in phases:
        for i, n in enumerate(lens):
            for fixed in (0, 1) if i % 2 == 0 else (1, 0):
                res.append(run("D", n, fixed))
        res.append(run("D", 41863, 0))  # run-to-run determinism of the per-length path itself
    if "T" in phases:
        for rep in range(2):
            for n in (3101, 3151, 41863, 110003):
                for fixed in (0, 1):
                    res.append(run(f"T{rep}", n, fixed))
    pp._push(service, PAYLOAD, pp._h2d_rows([1] * C, 0), None, struct.pack("<3i", -1, -1, -1))
    log("SHUTDOWN sentinel sent")
    time.sleep(10)
    d = digests()
    by_len: dict = {}
    for tag, v in d.items():
        if tag.startswith("D."):
            by_len.setdefault(v["S"], []).append((tag, v))
    verdict = {}
    for S, items in sorted(by_len.items()):
        keys = {(v["dsa"], v["kda_rec"], v["kda_conv"]) for _, v in items}
        verdict[S] = {"runs": [t for t, _ in items], "identical": len(keys) == 1, "distinct": len(keys)}
    log(f"bit identity per length (D phase): {json.dumps({S: v['identical'] for S, v in verdict.items()})}")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"results": res, "digests": d, "verdict": verdict}, f, indent=1)
    log(f"results -> {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    sys.exit(main())
