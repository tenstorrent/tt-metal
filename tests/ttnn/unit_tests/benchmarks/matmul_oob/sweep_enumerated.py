# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Time every config the enumerating candidate source proposes for each case, plus the legacy selection's.

The configs come from the selector itself (matmul_last_enumerated_configs: EnumeratingSource's proposals, each with
the matmul op's verdict), so "legal" means the same here as in the selector, and the v2 heuristic's own choice is
always among them (origin "heuristic"). Each row is one (problem, config): the problem's fields as in run_suite.py,
the config's fields one per column, and the device kernel time over the timed calls (median, min, max). Configs the
op rejects are recorded with status "invalid" and not run. The result is the sweep format sweep_data.py reads.

  python sweep_enumerated.py --cases-csv generated/matmul_oob/refactor_check/cases.csv --out sweep.csv
  python sweep_enumerated.py --cases-csv cases.csv --out counts.csv --enumerate-only   # configs only, no timing

Rows are appended as they finish, never rewritten: --resume continues an interrupted run (after a hang, reset the device
first), and --resume --redo REGEX times matching cases again under a new run id. A (problem, origin, config) row
appearing more than once means a later run supersedes an earlier one; sweep_data.py keeps the last.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_suite  # noqa: E402  (sets the profiler env vars before ttnn is imported)
from run_suite import FIELDS, CaseRun, case_fields, git_rev  # noqa: E402
from suite import cases_from_csv  # noqa: E402

import ttnn  # noqa: E402

_matmul = ttnn._ttnn.operations.matmul

# The fields that identify a problem (everything about the matmul call, none of the run's results)
PROBLEM_FIELDS = [
    "batch",
    "M",
    "K",
    "N",
    "a_shape",
    "b_shape",
    "a_dtype",
    "b_dtype",
    "out_dtype",
    "a_mem",
    "b_mem",
    "out_mem",
    "a_shard",
    "b_shard",
    "out_shard",
    "transpose_a",
    "transpose_b",
    "op",
    "bias",
    "activation",
    "core_grid",
    "fidelity",
    "fp32_acc",
    "packer_l1_acc",
    "arch",
    "grid",
]
# A program config's fields, one column each ("" where the config type has no such field)
CONFIG_FIELDS = [
    "in0_block_w",
    "out_subblock_h",
    "out_subblock_w",
    "out_block_h",
    "out_block_w",
    "per_core_M",
    "per_core_N",
    "fuse_batch",
    "mcast_in0",
    "transpose_mcast",
]
SWEEP_FIELDS = ["run", "problem_id", "origin", "family", "grid_x", "grid_y"] + CONFIG_FIELDS + FIELDS
FAMILIES = {
    "MatmulMultiCoreReuseMultiCastProgramConfig": "2d",
    "MatmulMultiCoreReuseProgramConfig": "reuse",
    "MatmulMultiCoreProgramConfig": "multicore",
}


def problem_id(fields):
    key = json.dumps({k: str(fields[k]) for k in PROBLEM_FIELDS}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def config_columns(config):
    """The family and fields of a config as matmul formats it, e.g. 'MatmulMultiCoreReuseProgramConfig(...)'"""
    kind = config.split("(", 1)[0]
    values = dict(re.findall(r"(\w+)=([^,()]+)", config))
    row = {f: values.get(f, "") for f in CONFIG_FIELDS}
    grid = values.get("compute_with_storage_grid_size", "")
    row["grid_x"], _, row["grid_y"] = grid.partition("-")
    family = FAMILIES.get(kind, kind)
    if kind == "MatmulMultiCoreReuseMultiCast1DProgramConfig":
        family = "1d_in0" if values.get("mcast_in0") in ("1", "true") else "1d_in1"
    row["family"] = family
    return row


def status_of(error):
    if re.search(r"circular buffer|L1 buffer|out of memory|beyond max L1", error, re.IGNORECASE):
        return "l1_overflow"
    return "error"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases-csv", required=True, help="the cases, from a run_suite.py results CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--filter", default=None, help="regex on case name")
    parser.add_argument("--enumerate-only", action="store_true", help="record the configs only: no timing")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--pcc-threshold", type=float, default=0.99)
    parser.add_argument(
        "--pcc-max-flops", type=float, default=2e12, help="skip the torch golden above this many flops (0: never)"
    )
    parser.add_argument("--resume", action="store_true", help="skip (problem, origin, config) rows already in --out")
    parser.add_argument(
        "--redo",
        default=None,
        help="with --resume: time the cases matching this regex again (rows are appended; readers keep the last run)",
    )
    parser.add_argument("--run", default=time.strftime("%Y%m%dT%H%M%S"), help="run id recorded in every row")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    args.configs_only = args.enumerate_only  # CaseRun: shapes only, no host data

    cases = cases_from_csv(args.cases_csv)
    if args.filter:
        cases = [c for c in cases if re.search(args.filter, c.name)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            done = {
                (r["problem_id"], r["origin"], r["config"])
                for r in csv.DictReader(f)
                if not (args.redo and re.search(args.redo, r["case"]))
            }
    elif out_path.exists():
        sys.exit(f"{out_path} exists; pass --resume to continue it or choose another --out")

    git = git_rev()
    device = ttnn.open_device(device_id=args.device_id)
    arch = str(device.arch()).split(".")[-1].lower()
    grid = device.compute_with_storage_grid_size()
    seen_programs = set()
    write_header = not out_path.exists()
    try:
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
            if write_header:
                writer.writeheader()

            def write(fields, origin, row):
                row = {**row, **fields, "run": args.run, "origin": origin, "mode": origin}
                row.update(config_columns(row.get("config", "")))
                writer.writerow(row)
                f.flush()
                return row

            for i, case in enumerate(cases):
                t0 = time.time()
                fields = case_fields(case, arch, grid, git)
                pid = fields["problem_id"] = problem_id(fields)
                try:
                    run = CaseRun(case, device, args, seen_programs)
                except (CaseRun.Infeasible, CaseRun.SetupError) as e:
                    if (pid, "legacy", "") not in done:
                        write(fields, "legacy", {"status": "infeasible", "error": str(e)})
                    print(f"[{i + 1}/{len(cases)}] {case.name:48s} skipped: {e}", flush=True)
                    continue
                try:
                    # The legacy selection's run (no program_config), recording the enumerated configs as it goes
                    _matmul.matmul_record_enumerated_configs(True)
                    _matmul.matmul_last_enumerated_configs(reset=True)
                    legacy = run.measure("oob")
                    _matmul.matmul_record_enumerated_configs(False)
                    enumerated, unsupported = _matmul.matmul_last_enumerated_configs(reset=True)
                    if (pid, "legacy", legacy.get("config", "")) not in done:
                        write(fields, "legacy", legacy)
                    if unsupported:
                        write(fields, "unsupported", {"status": "unsupported", "error": unsupported})
                    counts = {}
                    for config, origin, error in enumerated:
                        text = repr(config)
                        key = (pid, origin, text)
                        status = "invalid" if error else "ok"
                        counts[status] = counts.get(status, 0) + 1
                        if key in done:
                            continue
                        if error or args.enumerate_only:
                            row = {"status": status, "error": error, "config": text, "config_type": text.split("(")[0]}
                        else:
                            # v2 mode: bias and activation fusion follow the config (the legacy path would apply
                            # a fused activation a second time)
                            row = run.measure("v2", program_config=config)
                            if row["status"] == "error":
                                row["status"] = status_of(row["error"])
                        write(fields, origin, row)
                finally:
                    _matmul.matmul_record_enumerated_configs(False)
                    run.close()
                    device.clear_program_cache()
                print(
                    f"[{i + 1}/{len(cases)}] {case.name:48s} {len(enumerated):3d} configs {counts} "
                    f"({time.time() - t0:.1f}s) {unsupported}",
                    flush=True,
                )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
