"""Hang-rate bisection probe for intermittent device deadlocks (e.g. multi-chip CCL).

Runs a pytest node N times under a per-trial timeout and classifies each trial by a
POSITIVE completion sentinel — never by absence of error. A killed / timed-out / sentinel-
less run is a HANG, never "pass". Point it at one config, record the rate; flip a knob
(num_links, topology, tracy on/off), record again; the knob that drives the rate to zero
is the fix — found by experiment, without needing the device-level root cause.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .probes import _device_reset, _kill_tree

_TIMEOUT_CODES = {124, 137, 143, -9, -15}


def classify_trial(stdout: str, exit_code: int, sentinel: str) -> str:
    if exit_code == 0 and sentinel and sentinel in (stdout or ""):
        return "pass"
    if exit_code in _TIMEOUT_CODES:
        return "hang"
    return "fail"


def run_trial(cmd: list[str], cwd, env: dict, timeout_s: int) -> tuple[str, int]:
    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as fh:
        log_path = fh.name
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            proc.wait(timeout=timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            code = 124
    try:
        out = Path(log_path).read_text(errors="ignore")
    except OSError:
        out = ""
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass
    return out, code


def _build_cmd(node: str, under_tracy: bool, out_dir: str) -> list[str]:
    py = sys.executable
    if under_tracy:
        return [py, "-m", "tracy", "-p", "-r", "-o", out_dir, "-m", "pytest", node, "-sv"]
    return [py, "-m", "pytest", node, "-sv"]


def measure_hang_rate(
    node: str,
    trials: int,
    timeout_s: int,
    sentinel: str,
    cwd,
    env: dict | None = None,
    under_tracy: bool = True,
    reset_between: bool = True,
    reset=_device_reset,
) -> dict:
    env = dict(env or os.environ)
    out_dir = str(Path(cwd) / "_hangprobe_out")
    results = []
    for _i in range(trials):
        if reset_between:
            reset()
        out, code = run_trial(_build_cmd(node, under_tracy, out_dir), cwd, env, timeout_s)
        results.append(classify_trial(out, code, sentinel))
    n = len(results) or 1
    return {
        "node": node,
        "trials": len(results),
        "passed": results.count("pass"),
        "hung": results.count("hang"),
        "failed": results.count("fail"),
        "pass_rate": results.count("pass") / n,
        "hang_rate": results.count("hang") / n,
        "results": results,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure intermittent hang rate of a pytest node (positive-sentinel gated)."
    )
    ap.add_argument("node")
    ap.add_argument("--sentinel", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--no-tracy", action="store_true")
    ap.add_argument("--no-reset", action="store_true")
    args = ap.parse_args(argv)
    summary = measure_hang_rate(
        args.node,
        args.trials,
        args.timeout,
        args.sentinel,
        args.cwd,
        under_tracy=not args.no_tracy,
        reset_between=not args.no_reset,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
