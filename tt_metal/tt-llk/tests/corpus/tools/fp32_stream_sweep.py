#!/usr/bin/env python3
"""laneMK — orchestrator for the sound 2^32 sem-vs-hand streaming sweep (one op).

Drives the certified corpus kernel's persistent-session streamer (the LANEMK_STREAM
hook in test_sfpu_unary.py) over [0, 2^32) in resume-safe bands, one band-leg per
pytest invocation, on a flocked device. Per band: run sem + hand, compare output SHA.
Verdict = BIT-EXACT-ALL-INPUTS iff every band's sem_sha == hand_sha (full contiguous
cover, checked); otherwise DIVERGENT (bands that differ are the witness bands to bisect).
Object identity is enforced by construction (same certified ELF) + the .text gate.

Usage:
  fp32_stream_sweep.py --op sign --sem-node '<pytest sem node>' --hand-node '<hand node>' \
      --farm <tests/python_tests> --venv <python> --tile-dim 256,256 \
      --band-bits 29 --chip 0 --out <dir>
"""
import argparse
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

TWO32 = 1 << 32
_SHA_RE = re.compile(r"output_sha256=([0-9a-f]{64})")
_RUNS_RE = re.compile(r"runs=(\d+)")


def run_band_leg(args, node, start, count, out_sha_file, log_file):
    """One pytest invocation: stream [start,start+count) for one leg. Returns (sha, wall, runs)."""
    if Path(out_sha_file).exists():
        txt = Path(out_sha_file).read_text()
        m = _SHA_RE.search(txt)
        if m:
            return m.group(1), 0.0, 0  # resumed
    env = dict(os.environ)
    env.update(
        CHIP_ARCH="blackhole",
        SHORT_ARCH="bh",
        LLK_HOME=args.llk_home,
        RUNNER_TEMP=args.runner_temp,
        PYTHONUNBUFFERED="1",
        LANEMK_TILE_DIM=args.tile_dim,
        LANEMK_STREAM=f"{start},{count},{out_sha_file}",
    )
    inner = (
        f"{shlex.quote(args.venv)} -m pytest -o addopts= -q -s "
        f"{shlex.quote(node)} > {shlex.quote(str(log_file))} 2>&1"
    )
    cmd = ["flock", "-x", f"/tmp/tt-dev-{args.chip}.lock", "-c", inner]
    t0 = time.time()
    subprocess.run(cmd, cwd=args.farm, env=env, timeout=args.timeout)
    dt = time.time() - t0
    txt = Path(out_sha_file).read_text() if Path(out_sha_file).exists() else ""
    m = _SHA_RE.search(txt)
    r = _RUNS_RE.search(txt)
    if not m:
        raise RuntimeError(
            f"band [{start},{start+count}) leg {node} produced no SHA; see {log_file}"
        )
    return m.group(1), dt, int(r.group(1)) if r else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--sem-node", required=True)
    ap.add_argument("--hand-node", required=True)
    ap.add_argument("--farm", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--llk-home", required=True)
    ap.add_argument("--runner-temp", required=True)
    ap.add_argument("--tile-dim", default="256,256")
    ap.add_argument("--band-bits", type=int, default=29)
    ap.add_argument("--chip", default="0")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-bit", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--total", type=lambda s: int(s, 0), default=TWO32)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "bands").mkdir(parents=True, exist_ok=True)
    band = 1 << args.band_bits
    n_bands = (args.total + band - 1) // band
    ledger = out / f"{args.op}-STREAM-LEDGER.tsv"

    rows = []
    covered = 0
    t_all = time.time()
    all_equal = True
    witness_bands = []
    for k in range(n_bands):
        s = args.start_bit + k * band
        c = min(band, args.start_bit + args.total - s)
        sem_f = out / "bands" / f"b{k:04d}-sem.txt"
        hand_f = out / "bands" / f"b{k:04d}-hand.txt"
        sem_sha, sem_dt, sem_runs = run_band_leg(
            args, args.sem_node, s, c, sem_f, out / "bands" / f"b{k:04d}-sem.log"
        )
        hand_sha, hand_dt, hand_runs = run_band_leg(
            args, args.hand_node, s, c, hand_f, out / "bands" / f"b{k:04d}-hand.log"
        )
        eq = sem_sha == hand_sha
        all_equal &= eq
        if not eq:
            witness_bands.append((k, s, c))
        covered += c
        rows.append(
            (
                k,
                s,
                c,
                sem_sha,
                hand_sha,
                "EQ" if eq else "DIFF",
                f"{sem_dt:.1f}",
                f"{hand_dt:.1f}",
            )
        )
        print(
            f"band {k+1}/{n_bands} [{s},{s+c}) sem={sem_sha[:12]} hand={hand_sha[:12]} "
            f"{'EQ' if eq else 'DIFF'} ({sem_dt:.0f}+{hand_dt:.0f}s)",
            flush=True,
        )
        with open(ledger, "w") as fh:
            fh.write(
                f"# {args.op} stream sweep; tile_dim={args.tile_dim}; band_bits={args.band_bits}; chip={args.chip}\n"
            )
            fh.write(
                "band\tstart\tcount\tsem_sha256\thand_sha256\tverdict\tsem_s\thand_s\n"
            )
            for row in rows:
                fh.write("\t".join(str(x) for x in row) + "\n")

    wall = time.time() - t_all
    assert covered == args.total, f"coverage gap: {covered} != {args.total}"
    verdict = "BIT-EXACT-ALL-INPUTS" if all_equal else "DIVERGENT"
    summary = (
        f"OP={args.op} VERDICT={verdict} bands={n_bands} covered={covered} "
        f"(full 2^32={covered==TWO32}) wall_s={wall:.1f} witness_bands={witness_bands}"
    )
    print(summary, flush=True)
    (out / f"{args.op}-VERDICT.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
