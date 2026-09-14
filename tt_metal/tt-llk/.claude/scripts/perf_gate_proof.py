#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the LLK perf gate time-budget proof from primary sources.

Every number this prints comes from one of four places, and each is named in the
output so a reader can check it:

  1. the budget-summary artefacts of our own measurement runs (per-phase seconds,
     written by tests/run_llk_perf_budget.sh),
  2. the GitHub jobs API (job wall clock, to the second),
  3. .github/time_budget.yaml and tests/pipeline_reorg/llk_*_gate_tests.yaml
     (what is allowed, and what the gates already declare),
  4. the per-leg ceiling each gate passes to verify_time_budget.py.

Nothing is typed in by hand. Run it from a tt-metal checkout:

    python3 tt_metal/tt-llk/.claude/scripts/perf_gate_proof.py

Options:
    --gate-run ID        our gate measurement (default: the run in GATE_RUN)
    --shard-run ID       the two-shard measurement, for the sharding argument
    --cards-run ID       the four-card measurement, for the other configurations
    --gate-history N     merge-gate runs to scan for what the gates use today
    --cache DIR          where downloaded artefacts live (default: .perf_gate_proof)
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import subprocess

REPO = "tenstorrent/tt-metal"

# Our measurements. Each is a workflow run of "LLK perf time budget".
GATE_RUN = "34369520690"  # [budget-gate]: L1_TO_L1, non-SoL, unsharded, gate SKUs
SHARD_RUN = "34220250197"  # [budget-shards=2]: the same sweep split in two
CARDS_RUN = "34126602108"  # [budget-run]: four configurations, two cards each

# Where the numbers that bound us are declared.
BUDGET_FILE = ".github/time_budget.yaml"
LANES = {
    # budget key -> (matrix yaml, workflow that passes the per-leg ceiling)
    "pr_gate": (
        "tests/pipeline_reorg/llk_pr_gate_tests.yaml",
        ".github/workflows/pr-gate.yaml",
    ),
    "merge_gate": (
        "tests/pipeline_reorg/llk_merge_gate_tests.yaml",
        ".github/workflows/merge-gate.yaml",
    ),
}
GATE_SKUS = ("wh_n150_civ2", "bh_p150b_civ2_viommu")
TEAM = "llk"

ISO = "%Y-%m-%dT%H:%M:%SZ"


def ms(seconds: float) -> str:
    """Minutes and seconds, which is the unit the argument is made in."""
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def gh_json(*args: str):
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gh {' '.join(args)} failed:\n{out.stderr.strip()}")
    return json.loads(out.stdout)


def run_jobs(run_id: str) -> list[dict]:
    return gh_json("run", "view", str(run_id), "-R", REPO, "--json", "jobs")["jobs"]


def wall_clock(job: dict) -> int | None:
    """Seconds the job held a machine. None while it is still running."""
    started, done = job.get("startedAt"), job.get("completedAt")
    if not started or not done or done.startswith("0001"):
        return None
    return int(
        (
            datetime.datetime.strptime(done, ISO)
            - datetime.datetime.strptime(started, ISO)
        ).total_seconds()
    )


def summaries(run_id: str, cache: str) -> list[dict]:
    """The per-measurement JSON our runner writes, one file per configuration."""
    out = os.path.join(cache, str(run_id))
    if not glob.glob(os.path.join(out, "*", "summary-*.json")):
        os.makedirs(out, exist_ok=True)
        got = subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run_id),
                "-R",
                REPO,
                "-D",
                out,
                "-p",
                "budget-summary-*",
            ],
            capture_output=True,
            text=True,
        )
        if got.returncode != 0:
            raise SystemExit(
                f"cannot download artefacts of run {run_id}:\n{got.stderr.strip()}"
            )
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "*", "summary-*.json"))):
        with open(path) as handle:
            row = json.load(handle)
            row["_source"] = os.path.relpath(path, cache)
            rows.append(row)
    return rows


def yaml_load(path: str):
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML is required: pip3 install PyYAML")
    with open(path) as handle:
        return yaml.safe_load(handle)


def declared(matrix_path: str) -> dict[str, int]:
    """Minutes each SKU already has declared in a gate's matrix, team llk only."""
    total: dict[str, int] = {}
    for entry in yaml_load(matrix_path) or []:
        if entry.get("team") != TEAM:
            continue
        for sku, conf in (entry.get("skus") or {}).items():
            total[sku] = total.get(sku, 0) + int(conf["timeout"])
    return total


def per_leg_ceiling(workflow_path: str) -> tuple[int | None, int | None]:
    """The --max-per-test-timeout a gate passes for the llk lane, and its line."""
    with open(workflow_path) as handle:
        lines = handle.readlines()
    # The ceiling belongs to the llk call, so take the one nearest an llk impl
    # reference rather than the first in the file.
    llk_at = [
        i
        for i, line in enumerate(lines)
        if re.search(r"llk-(smoke|unit-tests)-impl\.yaml", line)
    ]
    best = None
    for i, line in enumerate(lines):
        found = re.search(r"per-test-timeout:\s*(\d+)", line)
        if not found or not llk_at:
            continue
        distance = min(abs(i - at) for at in llk_at)
        if distance <= 20 and (best is None or distance < best[2]):
            best = (int(found.group(1)), i + 1, distance)
    return (best[0], best[1]) if best else (None, None)


def budgets() -> dict[str, dict[str, int]]:
    data = yaml_load(BUDGET_FILE)
    return {lane: (data[TEAM].get(lane) or {}) for lane in LANES}


def gate_usage(limit: int) -> dict[str, dict]:
    """What the LLK gate legs actually spend today, per SKU, from finished runs."""
    runs = gh_json(
        "run",
        "list",
        "-R",
        REPO,
        "--workflow",
        "merge-gate.yaml",
        "--limit",
        str(limit),
        "--json",
        "databaseId,status,event",
    )
    per_sku: dict[str, dict] = {}
    scanned = []
    for run in runs:
        if run["status"] != "completed" or run["event"] not in ("push", "merge_group"):
            continue
        legs = [
            j
            for j in run_jobs(run["databaseId"])
            if re.search(r"LLK (FD|SD)", j["name"])
        ]
        legs = [
            j
            for j in legs
            if j.get("conclusion") == "success" and wall_clock(j) is not None
        ]
        if not legs:
            continue
        scanned.append(str(run["databaseId"]))
        for sku in GATE_SKUS:
            mine = [
                j
                for j in legs
                if f"[{sku}]" in j["name"] or sku.replace("_civ2", "") in j["name"]
            ]
            if not mine:
                continue
            slot = per_sku.setdefault(sku, {"legs": [], "per_run_sums": [], "runs": []})
            for job in mine:
                slot["legs"].append((job["name"], wall_clock(job)))
            slot["per_run_sums"].append(sum(wall_clock(j) for j in mine))
            slot["runs"].append(str(run["databaseId"]))
        if len(scanned) >= 5:
            break
    return {"per_sku": per_sku, "scanned": scanned}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-run", default=GATE_RUN)
    ap.add_argument("--shard-run", default=SHARD_RUN)
    ap.add_argument("--cards-run", default=CARDS_RUN)
    ap.add_argument("--gate-history", type=int, default=40)
    ap.add_argument("--cache", default=".perf_gate_proof")
    ap.add_argument("--json-out", default=None, help="also write the numbers as JSON")
    args = ap.parse_args()

    proof: dict = {"runs": {}, "budget": {}, "usage": {}}

    # ---------------------------------------------------------------- 1. our gate
    print("=" * 78)
    print(f"1. WHAT OUR GATE COSTS -- run {args.gate_run}")
    print(f"   https://github.com/{REPO}/actions/runs/{args.gate_run}")
    print("=" * 78)
    rows = summaries(args.gate_run, args.cache)
    jobs = {j["name"]: j for j in run_jobs(args.gate_run)}
    print(
        f"{'arch':10} {'SKU':22} {'CPUs':>4} {'compile':>8} {'measure':>8} {'phases':>8} "
        f"{'job wall':>9} {'rc':>4}"
    )
    for row in sorted(rows, key=lambda r: r["arch"]):
        job = next(
            (j for n, j in jobs.items() if row["arch"] in n and "worker" in n), None
        )
        wall = wall_clock(job) if job else None
        sku = next((s for s in GATE_SKUS if job and f"[{s}]" in job["name"]), "?")
        print(
            f"{row['arch']:10} {sku:22} {row.get('nproc','?'):>4} "
            f"{ms(row['producer_s']):>8} {ms(row['consumer_s']):>8} {ms(row['total_s']):>8} "
            f"{(ms(wall) if wall else '-'):>9} "
            f"{str(row['producer_rc'])+'/'+str(row['consumer_rc']):>4}"
        )
        proof["runs"].setdefault(args.gate_run, []).append(
            {
                "arch": row["arch"],
                "sku": sku,
                "config": row["config"],
                "compile_s": row["producer_s"],
                "measure_s": row["consumer_s"],
                "phases_s": row["total_s"],
                "job_wall_s": wall,
                "nproc": row.get("nproc"),
                "producer_rc": row["producer_rc"],
                "consumer_rc": row["consumer_rc"],
                "runner": row.get("runner"),
                "source": row["_source"],
            }
        )
    cfgs = {r["config"] for r in rows}
    sol = {bool(r["speed_of_light"]) for r in rows}
    shards = {str(r["n_groups"]) for r in rows}
    print(
        f"\n   configuration: run types {sorted(cfgs)}, speed of light {sol}, "
        f"shards {shards}, cold build {{{', '.join(str(bool(r['cold_build'])) for r in rows[:1])}}}"
    )
    print("   'phases' is compile + measure as timed by the runner script.")
    print(
        "   'job wall' is the GitHub job, which adds checkout, SFPI setup and upload."
    )
    print("   The declared timeout must cover the JOB, not the phases.")

    # -------------------------------------------------------- 2. allowed vs used
    print()
    print("=" * 78)
    print("2. WHAT IS ALLOWED, AND WHAT THE LLK GATES ALREADY DECLARE")
    print("=" * 78)
    caps = budgets()
    for lane, (matrix, workflow) in LANES.items():
        have = declared(matrix)
        ceiling, line = per_leg_ceiling(workflow)
        print(f"\n   llk.{lane}   matrix {matrix}")
        print(
            f"   per-leg ceiling {ceiling} min, from {workflow}:{line} "
            f"(strict >, so {ceiling} passes and {ceiling + 1 if ceiling else '?'} fails)"
        )
        print(f"   {'SKU':22} {'allowed':>8} {'declared':>9} {'free':>6}")
        for sku in GATE_SKUS:
            allowed = caps[lane].get(sku)
            used = have.get(sku, 0)
            free = (allowed - used) if allowed is not None else None
            print(
                f"   {sku:22} {str(allowed if allowed is not None else 'no key'):>8} "
                f"{used:>9} {str(free if free is not None else '-'):>6}"
            )
            proof["budget"].setdefault(lane, {})[sku] = {
                "allowed_min": allowed,
                "declared_min": used,
                "free_min": free,
                "per_leg_ceiling_min": ceiling,
                "ceiling_source": f"{workflow}:{line}",
            }

    # ------------------------------------------------- 3. what the gates use now
    print()
    print("=" * 78)
    print(
        f"3. WHAT THE LLK GATE LEGS ACTUALLY SPEND (last {args.gate_history} merge-gate runs)"
    )
    print("=" * 78)
    usage = gate_usage(args.gate_history)
    if not usage["per_sku"]:
        print(
            "   No finished merge-gate run in the window had LLK legs. Widen --gate-history."
        )
    else:
        print(f"   runs scanned: {', '.join(usage['scanned'])}")
        print(
            f"\n   {'SKU':22} {'legs':>5} {'slowest leg':>12} {'per-run sum':>12} {'declared':>9}"
        )
        for sku, slot in usage["per_sku"].items():
            worst = max(slot["legs"], key=lambda leg: leg[1])
            per_run = max(slot["per_run_sums"])
            dec = declared(LANES["merge_gate"][0]).get(sku, 0)
            print(
                f"   {sku:22} {len(slot['legs']):>5} {ms(worst[1]):>12} {ms(per_run):>12} {dec:>9}"
            )
            print(f"   {'':22} slowest: {worst[0]}")
            proof["usage"][sku] = {
                "legs": len(slot["legs"]),
                "slowest_leg_s": worst[1],
                "slowest_leg_name": worst[0],
                "max_per_run_sum_s": per_run,
                "declared_min": dec,
                "runs": slot["runs"],
            }

    # ------------------------------------------------------------ 4. the verdict
    print()
    print("=" * 78)
    print("4. DOES OUR GATE FIT?")
    print("=" * 78)
    ours = {}
    for entry in proof["runs"][args.gate_run]:
        wall = entry["job_wall_s"] or entry["phases_s"]
        ours[entry["sku"]] = max(ours.get(entry["sku"], 0), wall)
    for lane in LANES:
        ceiling = next(iter(proof["budget"][lane].values()))["per_leg_ceiling_min"]
        print(f"\n   llk.{lane}")
        for sku, wall in ours.items():
            info = proof["budget"][lane].get(sku, {})
            allowed, used = info.get("allowed_min"), info.get("declared_min", 0)
            # Two declarations, because they answer different questions.
            # `floor` is the smallest honest one: whole minutes, never below the
            # single measurement, and therefore with almost no margin. `want` is
            # the one to actually declare: the measurement plus 20%, which is
            # the smallest margin that survives a slow queue or a slower card.
            floor_min = max(1, -(-wall // 60))
            want_min = max(1, -(-int(wall * 1.2) // 60))
            rows = []
            for label, need in (("floor", floor_min), ("with margin", want_min)):
                fits_leg = ceiling is None or need <= ceiling
                fits_sum = allowed is not None and used + need <= allowed
                margin = need * 60 - wall
                rows.append((label, need, fits_leg, fits_sum, margin))
                print(
                    f"   {sku:22} measured {ms(wall)}  {label:11} declare {need:>2} min "
                    f"(margin {ms(margin):>5})  per-leg {'OK  ' if fits_leg else 'FAIL'} "
                    f"(ceiling {ceiling})  sum {used}+{need}={used + need} vs {allowed} "
                    f"{'OK' if fits_sum else 'FAIL'}"
                )
            proof.setdefault("verdict", {}).setdefault(lane, {})[sku] = {
                "measured_s": wall,
                "declarations": [
                    {
                        "kind": label,
                        "declare_min": need,
                        "margin_s": margin,
                        "per_leg_ok": leg,
                        "sum_ok": total,
                    }
                    for label, need, leg, total, margin in rows
                ],
            }
            print()

    # ------------------------------------------- 5. the sharding evidence, if any
    if args.shard_run:
        print()
        print("=" * 78)
        print(f"5. WHAT SHARDING DOES -- run {args.shard_run} against {args.cards_run}")
        print("=" * 78)
        one = [r for r in summaries(args.cards_run, args.cache) if r["config"] == "l1"]
        two = [r for r in summaries(args.shard_run, args.cache) if r["config"] == "l1"]
        for arch in sorted({r["arch"] for r in one} | {r["arch"] for r in two}):
            a = [r["total_s"] for r in one if r["arch"] == arch]
            b = [r["total_s"] for r in two if r["arch"] == arch]
            if not a or not b:
                continue
            print(
                f"   {arch:10} unsharded {ms(max(a))}   two shards: "
                f"slowest {ms(max(b))}, fastest {ms(min(b))}, "
                f"imbalance {((max(b) - min(b)) / max(min(b), 1)) * 100:.0f}%, "
                f"card time {ms(sum(b))} against {ms(max(a))}"
            )
            proof.setdefault("sharding", {})[arch] = {
                "unsharded_s": max(a),
                "shard_slowest_s": max(b),
                "shard_fastest_s": min(b),
                "card_time_s": sum(b),
            }
        print("\n   pytest-split divides by test item count, not by duration, so the")
        print("   halves are unequal and the slowest half still sets the wall clock.")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(proof, handle, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
