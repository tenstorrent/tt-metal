#!/usr/bin/env python3
"""sweeps/record.py — run an op-sweep and write a JSON record to sweeps/runs/.

Usage:
  sweeps/record.py <op> [-n N | --scaling "1 2 4 8"] [pytest_args...]

Examples:
  sweeps/record.py sdpa -n 4 -m "not slow" -k decode_sweep
  sweeps/record.py sdpa --scaling "1 2 4 8" -m "not slow" \\
      -k "decode_sweep or (prefill_sweep and (1536 or mqa_2k))"
"""

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(__file__).resolve().parent / "runs"

OPS = {
    "sdpa": "tests/ttnn/unit_tests/operations/sdpa/test_sdpa_sweep.py",
    "conv": "tests/ttnn/unit_tests/operations/conv/test_conv2d_sweep.py",
    "layernorm": "tests/ttnn/unit_tests/operations/fused/test_layer_norm_sweep.py",
}


def detect_arch(env=None):
    env = env if env is not None else os.environ
    sim = env.get("TT_METAL_SIMULATOR", "")
    if not sim:
        # Default-detect: if ~/sim/bh/libttsim.so exists, prefer bh.
        for arch, sub in (("blackhole", "bh"), ("wormhole_b0", "wh")):
            if (Path.home() / f"sim/{sub}/libttsim.so").exists():
                sim = str(Path.home() / f"sim/{sub}/libttsim.so")
                break
    if "/bh/" in sim or sim.endswith("_bh.so"):
        return "blackhole"
    if "/wh/" in sim or sim.endswith("_wh.so"):
        return "wormhole_b0"
    return env.get("ARCH_NAME", "unknown")


def host_info():
    return {"cores": os.cpu_count(), "hostname": socket.gethostname()}


def parse_pytest(text):
    """Pull per-test call durations + final summary from pytest output."""
    per_test = []
    for m in re.finditer(r"^\s*([\d.]+)s\s+call\s+(\S+::\S.+)$", text, re.MULTILINE):
        node = m.group(2)
        per_test.append(
            {
                "id": node.split("::", 1)[1] if "::" in node else node,
                "call_s": float(m.group(1)),
            }
        )

    counts = {}
    for what in ("passed", "failed", "skipped", "error"):
        m = re.search(rf"(\d+)\s+{what}", text)
        if m:
            counts[what] = int(m.group(1))

    m = re.search(r"(?:passed|failed|error|skipped).* in ([\d.]+)s", text)
    wall = float(m.group(1)) if m else None
    return per_test, counts, wall


def run_pytest(test_path, workers, pytest_extra):
    env = os.environ.copy()
    sim_default = str(Path.home() / "sim/bh/libttsim.so")
    env.setdefault("TT_METAL_SIMULATOR", sim_default)
    if Path(env["TT_METAL_SIMULATOR"]).exists():
        env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
        env["TT_METAL_DISABLE_SFPLOADMACRO"] = "1"
    env.setdefault("ARCH_NAME", detect_arch() or "blackhole")
    env.setdefault("TT_METAL_HOME", str(REPO))

    cmd = [
        "pytest",
        test_path,
        "-n",
        str(workers),
        "--dist",
        "worksteal",
        "-p",
        "no:cacheprovider",
        "--timeout=1800",
        "-o",
        "addopts=--durations=0 --tb=short -rN",
    ] + pytest_extra

    print(f"[record] pytest -n {workers} {' '.join(pytest_extra)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    return proc.stdout + "\n" + proc.stderr


def record_one(op, workers, pytest_extra):
    test_path = OPS[op]
    text = run_pytest(test_path, workers, pytest_extra)
    per_test, counts, wall = parse_pytest(text)

    ts = datetime.datetime.utcnow()
    rec = {
        "id": f"{ts.strftime('%Y%m%dT%H%M%S')}_{op}_{detect_arch()}_n{workers}",
        "ts": ts.isoformat(timespec="seconds") + "Z",
        "op": op,
        "arch": detect_arch(),
        "workers": workers,
        "dist": "worksteal",
        "test_path": test_path,
        "pytest_args": pytest_extra,
        "host": host_info(),
        "result": {"wall_s": wall, **counts},
        "per_test": per_test,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{rec['id']}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"[record] wrote {out.relative_to(REPO)}  ({counts.get('passed', 0)} passed in {wall}s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("op", choices=list(OPS))
    g = p.add_mutually_exclusive_group()
    g.add_argument("-n", "--workers", type=int, default=1)
    g.add_argument("--scaling", help='space-separated worker counts, e.g. "1 2 4 8"')
    args, pytest_extra = p.parse_known_args()

    counts = [int(x) for x in args.scaling.split()] if args.scaling else [args.workers]
    for n in counts:
        record_one(args.op, n, pytest_extra)


if __name__ == "__main__":
    main()
