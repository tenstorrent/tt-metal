#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Grade a codegen op port: run the bands, apply the thresholds, return a verdict.

This is the tool the porting agent calls. It owns every pass/fail decision so that the agent's only
influence on the outcome is the C++ it writes -- it cannot choose the cases, the thresholds, or the
measurement method, and it cannot edit the tests, because `check_write_paths` re-derives the diff
from the base commit on every single invocation and refuses to measure a tree that has wandered
outside the port's own files. That check is the difference between a performance claim and a
performance assertion.

Thresholds are carried over unchanged from the pipeline this replaces
(`agentic_port/skills/verify/lib/constants.py`). The tie bands are not slop: op dispatch at this
scale is a few microseconds, so a strict ratio comparison would fail on host scheduling noise. Each
band is paired with an absolute escape so that a "loss" too small to matter cannot block a port,
while a relative guard keeps the escape from waving through a genuine regression on a tiny op.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# --- thresholds (agentic_port/skills/verify/lib/constants.py) --------------------------------
WALL_TIE_BAND = 0.98
WALL_TIE_ABS_US = 3.0
DEVICE_VS_NATIVE_TIE_BAND = 1.0
DEVICE_VS_NATIVE_TIE_ABS_NS = 300.0
DEVICE_VS_NATIVE_TIE_ABS_BAND = 0.95
DEVICE_VS_GENERIC_TIE_BAND = 0.95
MIN_PAIRED_SAMPLES_FOR_CI = 5

VERDICT_WIN = "win"
VERDICT_BACK_TO_TRANSLATE = "back-to-translate"
VERDICT_NOT_A_CANDIDATE = "not-a-candidate"
VERDICT_BLOCKED = "blocked"

SCRIPTS = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------------
# Write-path guard
# --------------------------------------------------------------------------------------------


def allowed_prefixes(op: str, category: str) -> list[str]:
    base = f"ttnn/cpp/ttnn/operations/{category}"
    return [
        f"{base}/{op}/codegen/",
        f"{base}/{op}/{op}.cpp",
        f"{base}/{op}/{op}.hpp",
        f"{base}/{op}/{op}_nanobind.cpp",
        f"{base}/{op}/{op}_nanobind.hpp",
        f"{base}/sources.cmake",
        f"{base}/CMakeLists.txt",
    ]


def check_write_paths(base_sha: str, op: str, category: str, repo: Path) -> list[str]:
    """Return the changed paths that fall outside the port's own files.

    Recomputed from the base commit every call rather than trusted from a prior one, because the
    whole point is to catch a tree that changed since the last check.
    """
    out = subprocess.run(
        ["git", "diff", "--name-only", base_sha],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changed = {p for p in (out + untracked).splitlines() if p.strip()}
    prefixes = allowed_prefixes(op, category)
    return sorted(p for p in changed if not any(p == a or p.startswith(a) for a in prefixes))


# --------------------------------------------------------------------------------------------
# Band execution
# --------------------------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)


def build_ledger(manifest: str, out: Path, repo: Path) -> dict:
    proc = run([sys.executable, str(SCRIPTS / "ledger.py"), "--manifest", manifest, "--out", str(out)], repo)
    if proc.returncode != 0:
        raise RuntimeError(f"ledger failed: {proc.stderr[-2000:]}")
    return json.loads(out.read_text())


def run_measure(
    op: str, ledger: Path, band: str, out: Path, repo: Path, extra: list[str], env: dict | None = None
) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPTS / "measure.py"),
        "--op",
        op,
        "--ledger",
        str(ledger),
        "--band",
        band,
        "--out",
        str(out),
        *extra,
    ]
    proc = run(cmd, repo, env)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"measure {band} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    return json.loads(out.read_text())


def run_device_band(op: str, ledger: Path, out: Path, repo: Path, limit: int, reps: int, reports: Path) -> dict:
    """Device timings need a tracy-enabled build, so this leg runs under `python3 -m tracy`, which
    sets TT_METAL_DEVICE_PROFILER and post-processes the device logs into an ops report.

    Usage is `python3 -m tracy [options] scriptfile [args]` with the script positional, and `-o`
    pins the artifacts folder so the CSV is found by path rather than by globbing a shared
    directory that may hold reports from an earlier iteration.
    """
    proc = run(
        [
            sys.executable,
            "-m",
            "tracy",
            "-p",
            "-r",
            "-o",
            str(reports),
            str(SCRIPTS / "measure.py"),
            "--op",
            op,
            "--ledger",
            str(ledger),
            "--band",
            "device",
            "--out",
            str(out),
            "--limit",
            str(limit),
            "--reps",
            str(reps),
        ],
        repo,
        {"TRACY_NO_INVARIANT_CHECK": "1"},
    )
    if not out.is_file():
        raise RuntimeError(f"device band produced no output:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    return json.loads(out.read_text())


def latest_ops_csv(repo: Path, reports: Path) -> Path | None:
    for root in (reports, repo / "generated/profiler/reports"):
        matches = sorted(glob.glob(str(root / "**/ops_perf_results_*.csv"), recursive=True))
        if matches:
            return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))
    return None


def attribute_device_rows(csv_path: Path, order: list[dict]) -> tuple[dict, list[str]]:
    """Join profiler rows to dispatches by order, and refuse to guess if they do not line up.

    Attribution is positional because signpost columns do not reliably survive post-processing. The
    safety net is the op-code consistency check: if one leg maps to more than one op code, the
    alignment is wrong and the numbers are discarded rather than reported.
    """
    notes: list[str] = []
    with csv_path.open() as fh:
        rows = [r for r in csv.DictReader(fh)]

    def col(row, *names):
        for n in names:
            for key in row:
                if key.strip().upper() == n:
                    return row[key]
        return None

    durations = []
    for row in rows:
        value = col(row, "DEVICE KERNEL DURATION [NS]")
        code = col(row, "OP CODE")
        if value not in (None, ""):
            try:
                durations.append((code, float(value)))
            except ValueError:
                continue

    dispatched = [o for o in order if o.get("error") is None and o.get("leg") != "setup"]
    if len(durations) != len(dispatched):
        notes.append(
            f"device attribution inconclusive: {len(durations)} profiler rows vs "
            f"{len(dispatched)} recorded dispatches"
        )
        return {}, notes

    samples: dict[tuple[str, str], list[float]] = {}
    codes: dict[str, set] = {}
    for (code, ns), entry in zip(durations, dispatched):
        leg = entry["leg"]
        if leg.endswith(":warmup"):
            continue  # consumed positionally, but a first call measures compilation, not the op
        samples.setdefault((entry["case_id"], leg), []).append(ns)
        codes.setdefault(leg, set()).add(code)

    for leg, seen in codes.items():
        if len(seen) > 1:
            notes.append(f"device attribution inconclusive: leg {leg!r} mapped to op codes {sorted(seen)}")
            return {}, notes
    if codes:
        notes.append("op codes: " + ", ".join(f"{leg}={sorted(s)[0]}" for leg, s in sorted(codes.items())))
    return samples, notes


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------


def wall_passes(native_us: float, ported_us: float) -> bool:
    if ported_us <= 0:
        return False
    if native_us / ported_us >= WALL_TIE_BAND:
        return True
    return native_us > 0.0 and (ported_us - native_us) <= WALL_TIE_ABS_US


def device_vs_native_passes(native_ns: float, ported_ns: float) -> bool:
    if ported_ns <= 0:
        return False
    ratio = native_ns / ported_ns
    if ratio >= DEVICE_VS_NATIVE_TIE_BAND:
        return True
    # Absolute escape, fenced by a relative guard: a sub-300ns deficit is below the noise floor,
    # but only counts when the port is not also losing badly in proportional terms.
    return (ported_ns - native_ns) <= DEVICE_VS_NATIVE_TIE_ABS_NS and ratio >= DEVICE_VS_NATIVE_TIE_ABS_BAND


def grade(wall: dict, device_samples: dict, cases: list[dict]) -> dict:
    by_id = {c["case_id"]: c for c in cases}
    configs = []
    for record in wall.get("results", []):
        if record.get("error"):
            configs.append({"case_id": record["case_id"], "error": record["error"], "passes": False})
            continue
        case_id = record["case_id"]
        native_us, ported_us = record["native_us"], record["ported_us"]
        entry = {
            "case_id": case_id,
            "dtype": by_id.get(case_id, {}).get("dtype"),
            "layout": by_id.get(case_id, {}).get("layout"),
            "wall_native_us": native_us,
            "wall_ported_us": ported_us,
            "wall_ratio": (native_us / ported_us) if ported_us else None,
        }

        native_ns = device_samples.get((case_id, "native"))
        ported_ns = device_samples.get((case_id, "ported"))
        generic_ns = device_samples.get((case_id, "generic"))
        if native_ns and ported_ns:
            entry["device_native_ns"] = min(native_ns)
            entry["device_ported_ns"] = min(ported_ns)
            entry["device_vs_native"] = min(native_ns) / min(ported_ns)
        if generic_ns and ported_ns:
            entry["device_generic_ns"] = min(generic_ns)
            entry["device_vs_generic_op"] = min(generic_ns) / min(ported_ns)

        checks = {"wall": wall_passes(native_us, ported_us)}
        if "device_vs_native" in entry:
            checks["device_vs_native"] = device_vs_native_passes(entry["device_native_ns"], entry["device_ported_ns"])
        if "device_vs_generic_op" in entry:
            checks["device_vs_generic_op"] = entry["device_vs_generic_op"] >= DEVICE_VS_GENERIC_TIE_BAND
        entry["checks"] = checks
        entry["passes"] = all(checks.values())
        configs.append(entry)

    strict_win = any((c.get("wall_ratio") or 0) > 1.0 or (c.get("device_vs_native") or 0) > 1.0 for c in configs)
    all_pass = bool(configs) and all(c.get("passes") for c in configs)
    if not configs:
        verdict = VERDICT_BLOCKED
    elif all_pass and strict_win:
        verdict = VERDICT_WIN
    elif all_pass:
        # Every gate held but nothing actually got faster; shipping this buys complexity for free.
        verdict = VERDICT_NOT_A_CANDIDATE
    else:
        verdict = VERDICT_BACK_TO_TRANSLATE

    ratios = [c["wall_ratio"] for c in configs if c.get("wall_ratio")]
    device_ratios = [c["device_vs_native"] for c in configs if c.get("device_vs_native")]
    return {
        "verdict": verdict,
        "has_strict_win": strict_win,
        "configs": configs,
        "failing": [c["case_id"] for c in configs if not c.get("passes")],
        "summary": {
            # Worst case, not average: a port is only as good as its weakest measured configuration.
            "wall_ratio_min": min(ratios) if ratios else None,
            "wall_ratio_median": statistics.median(ratios) if ratios else None,
            "device_vs_native_min": min(device_ratios) if device_ratios else None,
        },
    }


def grade_correctness(results: list[dict]) -> dict:
    failures = [r for r in results if not r.get("equal") or r.get("error")]
    routing = [r for r in results if r.get("scope") == "out" and r.get("routing_ok") is False]
    return {
        "total": len(results),
        "failures": [
            {
                "case_id": r["case_id"],
                "scope": r.get("scope"),
                "error": r.get("error"),
                "max_abs_diff": r.get("max_abs_diff"),
            }
            for r in failures[:25]
        ],
        "failure_count": len(failures),
        "routing_violations": [r["case_id"] for r in routing],
        "passes": not failures and not routing,
    }


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--band", default="both", choices=["correctness", "performance", "both"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--category", default="data_movement")
    ap.add_argument("--base-sha", default=os.environ.get("PORT_BASE_SHA", ""))
    ap.add_argument("--work", default="/tmp/port-gate")
    ap.add_argument("--limit", type=int, default=24, help="cases per perf band")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--skip-write-check", action="store_true", help="baseline runs, before any edits")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"op": args.op, "band": args.band, "notes": []}

    if not args.skip_write_check:
        if not args.base_sha:
            print(
                json.dumps(
                    {"verdict": VERDICT_BLOCKED, "error": "no --base-sha; refusing to measure an unverifiable tree"},
                    indent=2,
                )
            )
            return 2
        stray = check_write_paths(args.base_sha, args.op, args.category, repo)
        if stray:
            print(
                json.dumps(
                    {
                        "verdict": VERDICT_BLOCKED,
                        "error": "changes outside the port's own files; tests and build scripts are off limits",
                        "unexpected_changes": stray,
                        "allowed": allowed_prefixes(args.op, args.category),
                    },
                    indent=2,
                )
            )
            return 2

    try:
        ledger_path = work / f"{args.op}_ledger.json"
        ledger = build_ledger(args.manifest, ledger_path, repo)
        report["ledger_counts"] = ledger["counts"]

        if args.band in ("correctness", "both"):
            out = work / f"{args.op}_correctness.json"
            data = run_measure(args.op, ledger_path, "correctness", out, repo, [])
            report["correctness"] = grade_correctness(data["results"])

        if args.band in ("performance", "both"):
            wall = run_measure(
                args.op,
                ledger_path,
                "wall",
                work / f"{args.op}_wall.json",
                repo,
                ["--limit", str(args.limit), "--iters", str(args.iters)],
            )
            device_samples: dict = {}
            try:
                reports = work / f"{args.op}_profiler"
                dev = run_device_band(
                    args.op, ledger_path, work / f"{args.op}_device.json", repo, args.limit, args.reps, reports
                )
                csv_path = latest_ops_csv(repo, reports)
                if csv_path is None:
                    report["notes"].append("no ops_perf_results CSV found; device band skipped")
                else:
                    device_samples, notes = attribute_device_rows(csv_path, dev["order"])
                    report["notes"].extend(notes)
            except Exception as exc:  # noqa: BLE001 - wall alone still yields a usable verdict
                report["notes"].append(f"device band unavailable: {exc}")
            report["performance"] = grade(wall, device_samples, ledger["cases"])
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"verdict": VERDICT_BLOCKED, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2

    correctness_ok = report.get("correctness", {}).get("passes", True)
    perf = report.get("performance", {})
    if not correctness_ok:
        report["verdict"] = VERDICT_BACK_TO_TRANSLATE
    else:
        report["verdict"] = perf.get("verdict", VERDICT_WIN if args.band == "correctness" else VERDICT_BLOCKED)

    Path(work / f"{args.op}_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["verdict"] == VERDICT_WIN else 1


if __name__ == "__main__":
    sys.exit(main())
