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
import glob
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
# Artifacts live outside the source tree, keyed by worktree so parallel trees do not collide.
_TREE = os.path.basename(REPO)
DEFAULT_CACHE = os.path.expanduser(f"~/.cache/tt-metal-cache/{_TREE}")
DEFAULT_OUT = os.path.expanduser(f"~/.cache/ltx-matrix/{_TREE}-results.json")

# Resolved per cell and handed to the cell as LTX_CHECKPOINT, the way ltx_server's build_worker_env
# does it. Left to default_ltx_checkpoint, an unset LTX_CHECKPOINT resolves to a bare HF repo ref that
# is not on disk, and the cell SKIPS — which pytest exits 0 for, i.e. a green cell that rendered
# nothing.
LTX_ROOT = os.environ.get("LTX_MODEL_ROOT", "/home/sulphur")
MODELS = {
    "ltx": f"{LTX_ROOT}/hf/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-distilled-1.1.safetensors",
    "sulphur": f"{LTX_ROOT}/models/sulphur_distil_bf16.safetensors",
    "sulphur-lora": f"{LTX_ROOT}/models/sulphur_lora_fused_distil.safetensors",
    "10eros-lora": f"{LTX_ROOT}/models/10eros_distil_fused.safetensors",
    "lora1.1-cond72": f"{LTX_ROOT}/models/ltx11_cond72_fused_distil.safetensors",
    "lora1.1-cond32": f"{LTX_ROOT}/models/ltx11_cond32_fused_distil.safetensors",
}


def resolve_checkpoint(model):
    hits = sorted(glob.glob(MODELS[model]))
    return hits[0] if hits else None


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


# The broker restarts under its own auto-update, and an in-flight `tt-device-mcp run` just loses its
# socket. That says nothing about the cell, so it must never be recorded as one — retry instead.
BROKER_GONE = ("Connection refused", "no tt-device-mcp server reachable", "Connection failed")


def run_cell(model, tier, res, mode, timeout, extra_env, direct=False, attempts=3):
    ckpt = resolve_checkpoint(model)
    if ckpt is None:
        return {
            "model": model,
            "tier": tier,
            "res": res,
            "mode": mode,
            "status": "SKIP",
            "rc": None,
            "seconds": 0.0,
            "metrics": [],
            "log_tail": f"checkpoint absent: {MODELS[model]}",
        }
    cell_env = {
        "LTX_MATRIX": "1",
        "LTX_MATRIX_MODEL": model,
        "LTX_MATRIX_RES": res,
        "LTX_MATRIX_MODE": mode,
        "LTX_QUALITY": tier,
        "LTX_CHECKPOINT": ckpt,
        "TT_METAL_HOME": REPO,
        "PYTHONPATH": REPO,
        # Keyed per tree, and OUTSIDE it: build_key ignores the source tree, so one shared cache lets
        # a divergent worktree's prewarm manifest poison this one. Pinning it also means only the
        # first cell pays the cold JIT compile. Kept out of the repo — a build artifact is not source.
        "TT_METAL_CACHE": os.environ.get("TT_METAL_CACHE", DEFAULT_CACHE),
        **(KF_ENV if mode == "i2v" else {}),
        **extra_env,
    }
    # The tier expands into quant + sigmas at import; a stale explicit value from the caller's shell
    # would silently outrank it and the cell would report a tier it never ran.
    for k in ("LTX_QUANT", "LTX_S1_SIGMAS", "LTX_S2_SIGMAS", "LTX_FAST"):
        cell_env.pop(k, None)
    for attempt in range(1, attempts + 1):
        rec = _run_once(model, tier, res, mode, timeout, cell_env, direct)
        if not (rec["status"] == "FAIL" and any(b in (rec["log_tail"] or "") for b in BROKER_GONE)):
            return rec
        if attempt < attempts:
            print(f"    broker unreachable (restart?) — retry {attempt}/{attempts - 1}", flush=True)
            time.sleep(30)
    rec["status"] = "SKIP"  # never ran: an infra outage is not a verdict on the cell
    return rec


def _run_once(model, tier, res, mode, timeout, cell_env, direct):
    env = {**os.environ, **cell_env}
    for k in ("LTX_QUANT", "LTX_S1_SIGMAS", "LTX_S2_SIGMAS", "LTX_FAST"):
        env.pop(k, None)
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
    # `tt-device-mcp run` exits 0 for "the job ran to completion", NOT for "your command passed" — it
    # reports the real code as "EXIT CODE: N" in its output. Reading its own rc marks every cell PASS.
    if not direct:
        m = re.search(r"^EXIT CODE:\s+(\d+|None)", out, re.M)
        rc = int(m.group(1)) if m and m.group(1) != "None" else (rc if m is None else 1)
    metrics = re.findall(r"MATRIX \S+ (?:t2v|i2v pin f\d+): .*", out)
    # pytest exits 0 on a SKIP, so rc alone cannot tell a rendered cell from one that never ran. A
    # pass has to show its work: no metric lines means nothing was rendered, whatever the code says.
    if re.search(r"\d+ skipped", out) and not re.search(r"\d+ (passed|failed)", out):
        status = "SKIP"
    elif rc == 0 and metrics:
        status = "PASS"
    elif rc == 0:
        status = "FAIL"  # green code, zero evidence
    else:
        status = "FAIL"
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
    ap.add_argument("--out", default=DEFAULT_OUT)
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

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
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
