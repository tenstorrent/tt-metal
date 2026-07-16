#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Drive the LTX coverage matrix: t2v + i2v across every model x tier x resolution.

Runs `test_ltx_matrix_cell` once per cell, because a cell is a process: the pipeline reads its sigma
schedule into module constants at import, so LTX_QUALITY can only be chosen before the process
starts (see the test's own note). One cell per job is also what keeps each run inside the device
broker's window — the whole sweep is hours and would never fit in one.

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
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TEST = os.path.join(HERE, "test_pipeline_ltx_distilled.py")

MODELS = ["ltx", "sulphur", "sulphur-lora", "10eros-lora", "lora1.1-cond72", "lora1.1-cond32"]
TIERS = ["high", "medium", "fast"]
RESOLUTIONS = ["1080p", "720p"]


def cells(models, tiers, resolutions):
    for m in models:
        for t in tiers:
            for r in resolutions:
                yield m, t, r


def run_cell(model, tier, res, timeout, extra_env):
    env = {
        **os.environ,
        "LTX_MATRIX": "1",
        "LTX_MATRIX_MODEL": model,
        "LTX_MATRIX_RES": res,
        "LTX_QUALITY": tier,
        "TT_METAL_HOME": REPO,
        "PYTHONPATH": REPO,
        **extra_env,
    }
    # The tier expands into quant + sigmas at import; a stale explicit value from the caller's shell
    # would silently outrank it and the cell would report a tier it never ran.
    for k in ("LTX_QUANT", "LTX_S1_SIGMAS", "LTX_S2_SIGMAS", "LTX_FAST"):
        env.pop(k, None)
    cmd = [sys.executable, "-m", "pytest", f"{TEST}::test_ltx_matrix_cell", "-q", "-s", "--no-header"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=timeout)
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
    ap.add_argument("--timeout", type=int, default=1800, help="per cell (a cold cell loads 22B weights)")
    ap.add_argument("--out", default=os.path.join(HERE, "ltx_matrix_results.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env for every cell")
    a = ap.parse_args()

    models, tiers, res = a.models.split(","), a.tiers.split(","), a.res.split(",")
    extra = dict(kv.split("=", 1) for kv in a.env)
    todo = list(cells(models, tiers, res))
    print(f"matrix: {len(todo)} cells ({len(models)} models x {len(tiers)} tiers x {len(res)} resolutions)")
    if a.dry_run:
        for m, t, r in todo:
            print(f"  would run {m}/{t}/{r}")
        return 0

    results = []
    for i, (m, t, r) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {m}/{t}/{r} ...", flush=True)
        rec = run_cell(m, t, r, a.timeout, extra)
        results.append(rec)
        print(f"    {rec['status']} ({rec['seconds']}s)", flush=True)
        for line in rec["metrics"]:
            print(f"      {line}", flush=True)
        with open(a.out, "w") as f:  # stream, so a killed sweep is still readable
            json.dump(results, f, indent=1)

    print(f"\n{'model':16} {'tier':7} {'res':6} {'result':6} {'sec':>6}")
    for x in results:
        print(f"{x['model']:16} {x['tier']:7} {x['res']:6} {x['status']:6} {x['seconds']:>6}")
    bad = [x for x in results if x["status"] == "FAIL"]
    skipped = [x for x in results if x["status"] == "SKIP"]
    print(
        f"\n{sum(1 for x in results if x['status'] == 'PASS')}/{len(results)} pass, {len(bad)} fail, {len(skipped)} skip"
    )
    for x in bad:
        print(f"\n=== FAIL {x['model']}/{x['tier']}/{x['res']} (rc={x['rc']}) ===\n{x['log_tail']}")
    print(f"\nresults: {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
