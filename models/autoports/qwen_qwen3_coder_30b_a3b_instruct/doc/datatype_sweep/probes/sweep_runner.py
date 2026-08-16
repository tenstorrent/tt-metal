# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tier B of the sweep: one full 48-layer teacher-forcing row per candidate.

Why teacher forcing is the whole measurement
--------------------------------------------

``models.common.readiness_check.run_teacher_forcing`` returns, from a single
~3-minute run, every number a sweep row needs:

* ``top1`` / ``top5`` / ``top100`` against the AIME24 chat reference -- the
  accuracy gate;
* ``TTFT`` for the 158-token gate prompt;
* ``decode t/s/u`` -- and critically, that runner **always passes
  ``enable_trace=True``** (its docstring makes tracing a requirement of the
  ``generate()`` it accepts). So its decode figure is the traced teacher-forcing
  number the goal names as the ranking metric. No eager number is produced here
  at all, which is the safest way to satisfy "eager or untraced decode numbers
  are not valid for Pareto ranking or final selection" -- there is nothing to
  mix up.

One process per row, deliberately
---------------------------------

Each candidate is a separate subprocess with ``QWEN3_PRECISION_CONFIG`` pointing
at its own JSON file. Three reasons:

1. ``ccl_dtype`` keys the persistent CCL buffer cache, so a process that visits
   two CCL dtypes pays DRAM for both and the second row measures a differently
   loaded device. One value per process is the documented requirement, and one
   row per process satisfies it without special-casing.
2. Expert dtype changes cannot be applied to a live model -- the weights are
   quantised at upload -- so those rows need a fresh load regardless.
3. It routes every row through **the same public entry point** the readiness
   gates use (``build_generator`` reading the env var), rather than through a
   sweep-private construction path that might diverge from what ships.

Resumable: a row whose result is already in the output file is skipped, so an
interrupted sweep continues rather than restarting an hour of device time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.doc.datatype_sweep.probes.candidates import (  # noqa: E402
    CANDIDATES,
    STACKED,
    Candidate,
)

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent
MODEL_DIR = SWEEP_DIR.parent.parent
REPO = MODEL_DIR.parents[2]
CONFIG_DIR = SWEEP_DIR / "configs"
ROW_LOG_DIR = SWEEP_DIR / "logs" / "rows"

# --- the gate -----------------------------------------------------------------
#: top-5 >= 98% and top-100 == 100%. This is what the stage contract requires;
#: top-1 is reported alongside but is NOT a gate, which is the whole point of the
#: sweep -- top-5 and top-100 are pinned at 1.000 with maximum margin, so there
#: is top-1 headroom to spend on speed.
MIN_TOP5 = 0.98
MIN_TOP100 = 1.0

HARDWARE = "1x4 Blackhole P300_X2"
MESH = "MeshShape(1,4), FABRIC_1D_RING, 4 dies"
REGIME = (
    "traced teacher-forcing decode (enable_trace=True), 48 layers, batch 1, "
    "AIME24 chat reference, 158 prompt tokens / 100 generated tokens"
)

AGG = re.compile(
    r"AGGREGATE\s+top1=([\d.]+)\s+\((\d+)/(\d+)\)\s+top5=([\d.]+)\s+\((\d+)/(\d+)\)\s+"
    r"top(\d+)=([\d.]+)\s+\((\d+)/(\d+)\)(?:\s+TTFT=([\d.]+)ms)?(?:\s+decode=([\d.]+) t/s/u)?"
    r"(?:\s+e2e=([\d.]+) t/s/u)?"
)


def command_for(cfg_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "models.common.readiness_check.run_teacher_forcing",
        "--model-dir",
        str(MODEL_DIR.relative_to(REPO)),
        "--reference",
        str((MODEL_DIR / "readiness_aime24_chat.refpt").relative_to(REPO)),
        "--mesh-device",
        "P300X2",
        "--fabric-config",
        "FABRIC_1D_RING",
        "--trace-region-size",
        "300000000",
    ]


def run_row(cand: Candidate, structural: dict) -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ROW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIG_DIR / f"{cand.cid}.json"
    cand.config.write_json(cfg_path)

    cmd = command_for(cfg_path)
    env = dict(os.environ)
    # The construction path under test: build_generator reads this and resolves
    # it to a PrecisionConfig. Passing the *path* also sidesteps the two-module
    # -copy isinstance problem entirely.
    env["QWEN3_PRECISION_CONFIG"] = str(cfg_path)

    log_path = ROW_LOG_DIR / f"{cand.cid}.log"
    header = (
        f"# config_id: {cand.cid}\n"
        f"# delta: {cand.delta}\n"
        f"# why: {cand.why}\n"
        f"# env: QWEN3_PRECISION_CONFIG={cfg_path}\n"
        f"# cmd: {' '.join(cmd)}\n"
        f"# hw: {HARDWARE} / {MESH}\n"
        f"# regime: {REGIME}\n"
        f"# date: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
    )
    t0 = time.time()
    with log_path.open("w") as fh:
        fh.write(header)
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        fh.write(proc.stdout)
    elapsed = time.time() - t0

    row: dict = {
        "config_id": cand.cid,
        "group": cand.group,
        "delta_from_default": cand.delta,
        "rationale": cand.why,
        "dtype_policy": cand.dtype_policy,
        "compute_fidelity_policy": cand.fidelity_policy,
        "precision_config": cand.config.to_dict(),
        "precision_config_path": str(cfg_path.relative_to(MODEL_DIR)),
        "measurement_regime": REGIME,
        "command": "QWEN3_PRECISION_CONFIG=" + str(cfg_path.relative_to(MODEL_DIR)) + " " + " ".join(cmd),
        "hardware": HARDWARE,
        "mesh": MESH,
        "log": str(log_path.relative_to(MODEL_DIR)),
        "wall_seconds": round(elapsed, 1),
        "exit_code": proc.returncode,
    }
    # Device readback from tier A, for the same config id -- what actually
    # reached the device, including the RESOLVED block widths.
    if structural:
        row["device_audit"] = structural.get("audit")
        row["block_width_requested"] = structural.get("block_width_requested")
        row["block_width_resolved"] = structural.get("block_width_resolved")
        row["block_width_clamped"] = structural.get("block_width_clamped")

    m = None
    for line in proc.stdout.splitlines():
        found = AGG.search(line)
        if found:
            m = found
    if m is None:
        row.update({"status": "error", "pass": False, "error": "no AGGREGATE line in output"})
        return row

    top1, top5, topk = float(m.group(1)), float(m.group(4)), float(m.group(8))
    row.update(
        {
            "status": "ok",
            "top1": top1,
            "top5": top5,
            "top100": topk,
            "top1_matches": f"{m.group(2)}/{m.group(3)}",
            "top5_matches": f"{m.group(5)}/{m.group(6)}",
            "top100_matches": f"{m.group(9)}/{m.group(10)}",
            "ttft_ms": float(m.group(11)) if m.group(11) else None,
            "decode_tps_user": float(m.group(12)) if m.group(12) else None,
            "e2e_tps_user": float(m.group(13)) if m.group(13) else None,
        }
    )
    row["pass"] = bool(top5 >= MIN_TOP5 and topk >= MIN_TOP100)
    row["gate"] = f"top5 >= {MIN_TOP5}, top100 == {MIN_TOP100}"
    return row


def blocked_row(cand: Candidate, structural: dict) -> dict:
    """A candidate TTNN refuses to build: recorded as evidence, not as a gap.

    The goal allows a BFP4 group to be represented by "an exact TTNN/runtime
    blocker with evidence" instead of a measurement. This is that record: the
    op that raised, the file and line, and the assertion text, straight from
    the runtime rather than paraphrased.
    """
    err = structural.get("error", "")
    m = re.search(r"TT_FATAL @ (\S+):(\d+): (.+?)\\n", err)
    op = m.group(1).split("/")[-1] + ":" + m.group(2) if m else "unknown"
    assertion = m.group(3) if m else err[:300]
    info = re.search(r"info:\\n(.+?)\\nbacktrace", err)
    return {
        "config_id": cand.cid,
        "group": cand.group,
        "delta_from_default": cand.delta,
        "rationale": cand.why,
        "dtype_policy": cand.dtype_policy,
        "compute_fidelity_policy": cand.fidelity_policy,
        "precision_config": cand.config.to_dict(),
        "measurement_regime": "not measured -- construction blocked by TTNN (tier A, 2 layers)",
        "command": f"python doc/datatype_sweep/probes/structural_probe.py --layers 2 --only {cand.cid}",
        "hardware": HARDWARE,
        "mesh": MESH,
        "log": "doc/datatype_sweep/logs/structural_probe.log",
        "status": "blocked",
        "pass": False,
        "gate": f"top5 >= {MIN_TOP5}, top100 == {MIN_TOP100}",
        "top1": None,
        "top5": None,
        "top100": None,
        "ttft_ms": None,
        "decode_tps_user": None,
        "blocker_op": op,
        "blocker_assertion": assertion,
        "blocker_info": info.group(1) if info else None,
        "blocker_raw": err,
    }


CSV_COLUMNS = [
    "config_id",
    "group",
    "delta_from_default",
    "dtype_policy",
    "compute_fidelity_policy",
    "top1",
    "top5",
    "top100",
    "ttft_ms",
    "decode_tps_user",
    "pass",
    "gate",
    "measurement_regime",
    "hardware",
    "mesh",
    "block_width_resolved",
    "blocker_op",
    "blocker_info",
    "command",
    "log",
    "rationale",
]


def write_outputs(rows: list[dict]) -> None:
    # allow_nan=False: a bare NaN is not valid JSON (RFC 8259 has no such
    # literal) and this file is a published artifact. Probes that legitimately
    # measure a NaN record it as an explicit flag -- see
    # ``kv_bfp8_diagnosis.record_pcc``.
    (SWEEP_DIR / "sweep_results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (SWEEP_DIR / "sweep_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated config ids")
    ap.add_argument("--include-stacked", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run rows already present")
    args = ap.parse_args()

    pool = CANDIDATES + (STACKED if args.include_stacked else [])
    wanted = [c for c in pool if not args.only or c.cid in args.only.split(",")]

    out_json = SWEEP_DIR / "sweep_results.json"
    rows = json.loads(out_json.read_text()) if out_json.exists() else []
    done = {r["config_id"] for r in rows} if not args.force else set()

    structural_path = HERE / "structural_probe.json"
    structural = {}
    if structural_path.exists():
        structural = {r["config_id"]: r for r in json.loads(structural_path.read_text())}

    for i, cand in enumerate(wanted, 1):
        if cand.cid in done:
            print(f"[{i}/{len(wanted)}] {cand.cid} -- already measured, skipping", flush=True)
            continue
        print(f"[{i}/{len(wanted)}] {cand.cid} :: {cand.delta}", flush=True)
        s = structural.get(cand.cid, {})
        if s.get("status") == "error":
            # Tier A already proved this config cannot be built. Spending three
            # minutes of 48-layer load to watch it raise the same TT_FATAL would
            # buy nothing, so it is recorded as BLOCKED with the runtime's own
            # error text -- which is what the goal asks for in place of a
            # measurement ("an exact TTNN/runtime blocker with evidence").
            row = blocked_row(cand, s)
            rows = [r for r in rows if r["config_id"] != cand.cid] + [row]
            write_outputs(rows)
            print(f"    -> BLOCKED (tier A): {row['blocker_op']}", flush=True)
            continue
        row = run_row(cand, s)
        rows = [r for r in rows if r["config_id"] != cand.cid] + [row]
        write_outputs(rows)
        if row["status"] == "ok":
            print(
                f"    -> top1={row['top1']:.3f} top5={row['top5']:.3f} top100={row['top100']:.3f} "
                f"decode={row['decode_tps_user']} t/s/u ttft={row['ttft_ms']} ms "
                f"pass={row['pass']}  ({row['wall_seconds']}s)",
                flush=True,
            )
        else:
            print(f"    -> FAILED: {row.get('error')} (exit {row['exit_code']})", flush=True)

    write_outputs(rows)
    print(f"\n{len(rows)} rows in {out_json}")


if __name__ == "__main__":
    main()
