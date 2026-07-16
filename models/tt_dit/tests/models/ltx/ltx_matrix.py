#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Drive the LTX coverage matrix: t2v + i2v across every model x tier x resolution.

Runs `test_ltx_matrix_cell` once per cell, because a cell is a process: the pipeline reads its sigma
schedule into module constants at import, so LTX_QUALITY can only be chosen before the process
starts (see the test's own note).

Run this ON THE HOST, not inside a device reservation: it dispatches each cell as its own broker job
and holds nothing in between. The sweep is hours, so a driver that took the device up front would sit
on a shared board for all of it (and be killed by the broker's window anyway) — the device is held by
the cell, for the length of one cell. --direct skips the broker for a box you already own.

    python3 models/tt_dit/tests/models/ltx/ltx_matrix.py                       # everything
    python3 models/tt_dit/tests/models/ltx/ltx_matrix.py --models ltx,sulphur --res 720p
    python3 models/tt_dit/tests/models/ltx/ltx_matrix.py --tiers fast --dry-run

Prints a PASS/FAIL table and exits non-zero if any cell failed. Results stream to --out as JSON, so
a killed sweep can be read back rather than re-run.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TEST = os.path.join(HERE, "test_pipeline_ltx_distilled.py")

MODELS = ["ltx", "sulphur", "sulphur-lora", "10eros-lora", "lora1.1-cond72", "lora1.1-cond32"]
TIERS = ["high", "medium", "fast"]
RESOLUTIONS = ["1080p", "720p"]
MODES = ["t2v", "i2v"]

# What ltx_server's keyframe worker sets (config.build_worker_env for a "-kf" mode). A keyframe job is
# served by its own worker so that warmup captures traces at the keyframe schedule and sequence
# length; a cell that skipped this would render grey and blame the model.
KF_ENV = {
    "LTX_KF_APPEND_TOKEN": "1",
    "LTX_KF_TRACE_ANCHORS": "1,2",
    "LTX_KF_TRACE_PAD": "1",
    "LTX_ITER_DIT_RESIDENT": "0",
}


def cells(models, tiers, resolutions, modes):
    for m in models:
        for t in tiers:
            for r in resolutions:
                for mode in modes:
                    yield m, t, r, mode


def run_cell(model, tier, res, mode, timeout, extra_env, direct=False):
    cell_env = {
        "LTX_MATRIX": "1",
        "LTX_MATRIX_MODEL": model,
        "LTX_MATRIX_RES": res,
        "LTX_MATRIX_MODE": mode,
        "LTX_QUALITY": tier,
        "TT_METAL_HOME": REPO,
        "PYTHONPATH": REPO,
        # Keyed per tree: build_key ignores the source tree, so sharing one cache between worktrees
        # lets a divergent tree's prewarm manifest poison this one. Pinning it also means only the
        # first cell pays the cold JIT compile — the other 35 run warm.
        "TT_METAL_CACHE": os.environ.get("TT_METAL_CACHE", os.path.join(REPO, ".jit-cache")),
        **(KF_ENV if mode == "i2v" else {}),
        **extra_env,
    }
    # The tier expands into quant + sigmas at import; a stale explicit value from the caller's shell
    # would silently outrank it and the cell would report a tier it never ran.
    env = {**os.environ, **cell_env}
    for k in ("LTX_QUANT", "LTX_S1_SIGMAS", "LTX_S2_SIGMAS", "LTX_FAST"):
        env.pop(k, None)
        cell_env.pop(k, None)
    pytest_cmd = f"python -m pytest '{TEST}::test_ltx_matrix_cell' -q -s --no-header -p no:cacheprovider"
    if direct:
        cmd = ["bash", "-lc", f"source python_env/bin/activate && {pytest_cmd}"]
    else:
        # Each cell is its own broker job: it queues behind other tenants, gets its own window, and
        # releases the device the moment it is done. Cell env rides in the command itself — the
        # broker's -e takes a file, and these are per-cell.
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(cell_env.items()))
        cmd = [
            "tt-device-mcp",
            "run",
            "-w",
            REPO,
            "-t",
            str(timeout),
            "-o",
            "400",
            f"source python_env/bin/activate && {exports} {pytest_cmd}",
        ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=timeout + 300)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode(errors="replace") + f"\nTIMEOUT after {timeout}s", 124
    dt = time.time() - t0
    status = {0: "PASS", 5: "SKIP"}.get(rc, "SKIP" if " skipped" in out and " failed" not in out else "FAIL")
    metrics = re.findall(r"MATRIX \S+ (?:t2v|i2v pin f\d+): .*", out)
    return {
        "model": model,
        "tier": tier,
        "res": res,
        "mode": mode,
        "status": status,
        "rc": rc,
        "seconds": round(dt, 1),
        "metrics": metrics,
        # Only the failing tail is kept: a passing cell's log is noise, a failing one's is the report.
        "log_tail": None if status == "PASS" else out[-4000:],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--res", default=",".join(RESOLUTIONS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--timeout", type=int, default=1500, help="per cell; the broker caps a job at 1500s")
    ap.add_argument("--out", default=os.path.join(HERE, "ltx_matrix_results.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--direct", action="store_true", help="run cells here instead of via the broker")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for every cell")
    a = ap.parse_args()

    models, tiers, res, modes = a.models.split(","), a.tiers.split(","), a.res.split(","), a.modes.split(",")
    extra = dict(kv.split("=", 1) for kv in a.env)
    todo = list(cells(models, tiers, res, modes))
    print(
        f"matrix: {len(todo)} cells ({len(models)} models x {len(tiers)} tiers x "
        f"{len(res)} resolutions x {len(modes)} modes)"
    )
    if a.dry_run:
        for m, t, r, mode in todo:
            print(f"  would run {m}/{t}/{r}/{mode}")
        return 0

    results = []
    for i, (m, t, r, mode) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {m}/{t}/{r}/{mode} ...", flush=True)
        rec = run_cell(m, t, r, mode, a.timeout, extra, direct=a.direct)
        results.append(rec)
        print(f"    {rec['status']} ({rec['seconds']}s)", flush=True)
        for line in rec["metrics"]:
            print(f"      {line}", flush=True)
        with open(a.out, "w") as f:  # stream, so a killed sweep is still readable
            json.dump(results, f, indent=1)

    print(f"\n{'model':16} {'tier':7} {'res':6} {'mode':5} {'result':6} {'sec':>6}")
    for x in results:
        print(f"{x['model']:16} {x['tier']:7} {x['res']:6} {x['mode']:5} {x['status']:6} {x['seconds']:>6}")
    bad = [x for x in results if x["status"] == "FAIL"]
    skipped = [x for x in results if x["status"] == "SKIP"]
    print(
        f"\n{sum(1 for x in results if x['status'] == 'PASS')}/{len(results)} pass, {len(bad)} fail, {len(skipped)} skip"
    )
    for x in bad:
        print(f"\n=== FAIL {x['model']}/{x['tier']}/{x['res']}/{x['mode']} (rc={x['rc']}) ===\n{x['log_tail']}")
    print(f"\nresults: {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
