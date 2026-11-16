#!/usr/bin/env python3
"""Refresh candidate-only cap512->1024 extensions for suffix2048 controls.

The helper selects only explicit root coordinator rows adjudicated
inconclusive at exactly 512 generated IDs. It copies the saved suffix2048
fixture without prompt/template reconstruction, changes only per-trial seed
and provenance metadata, and runs CPU-only schema validation. It never imports
TTNN or launches inference, and refuses refresh once this candidate is admitted.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[6]
BASE = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
SETUP = BASE / "setup"
SOURCE = SETUP / "swe85-adaptive-suffix2048-candidate"
DEST = SETUP / "swe85-adaptive-suffix2048-cap1024-candidate"
BUILD = {
    "served": SETUP / "head_bfp8_build_kwargs.json",
    "full": SETUP / "full_precision_build_kwargs.json",
}
TRACE_LOG = {mode: SETUP / f"swe85-adaptive-suffix1024/{mode}.log" for mode in BUILD}
EXPECTED_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
TRIAL_SEEDS = (42, 43, 44)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def rel(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: pathlib.Path):
    return json.loads(path.read_text())


def write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def is_admitted() -> bool:
    return (DEST / "admission.json").exists()


def collect_root_rows() -> tuple[list[dict], dict]:
    known_cases = {}
    seed_cases = {}
    for mode in BUILD:
        manifest_path = SOURCE / f"{mode}-cases.json"
        manifest = read_json(manifest_path)
        known_cases[mode] = {entry["name"]: entry for entry in manifest}
        for entry in manifest:
            for seed in TRIAL_SEEDS:
                seed_cases[(mode, entry["name"], seed)] = entry

    snapshots = []
    adjudications: dict[tuple[str, str, int], dict] = {}
    for coordinator in sorted(SOURCE.glob("*-coordinator-*.json")):
        document = read_json(coordinator)
        coordinator_hash = sha(coordinator)
        snapshots.append({"path": rel(coordinator), "sha256": coordinator_hash})
        for row in document.get("rows", []):
            mode, job, seed = row.get("mode"), row.get("case"), row.get("seed")
            if (mode, job, seed) not in seed_cases:
                continue
            identity = (mode, job, seed)
            signature = {k: row.get(k) for k in ("verdict", "tokens", "source", "source_sha256", "reason")}
            prior = adjudications.get(identity)
            if prior and prior["signature"] != signature:
                raise ValueError(
                    f"Conflicting root coordinator rows for {identity}: {prior['signature']} vs {signature}"
                )
            if not prior:
                adjudications[identity] = {"row": row, "signature": signature, "snapshots": []}
            adjudications[identity]["snapshots"].append({"path": rel(coordinator), "sha256": coordinator_hash})

    pending = {}
    for mode in BUILD:
        pending[mode] = []
        for job in known_cases[mode]:
            for seed in TRIAL_SEEDS:
                identity = (mode, job, seed)
                entry = adjudications.get(identity)
                if entry is None:
                    pending[mode].append({"case": job, "seed": seed, "status": "no_coordinator_row"})
                elif "pending" in str(entry["row"].get("reason", "")).lower():
                    pending[mode].append(
                        {
                            "case": job,
                            "seed": seed,
                            "status": "adjudication_followup_pending",
                            "root_verdict_so_far": entry["row"].get("verdict"),
                            "reason": entry["row"].get("reason", ""),
                        }
                    )
    selected = []
    for identity, evidence in adjudications.items():
        row = evidence["row"]
        if row.get("verdict") != "inconclusive" or row.get("tokens") != 512:
            continue
        mode, job, seed = identity
        result_path = ROOT / row["source"]
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        source_hash = sha(result_path)
        if row.get("source_sha256") != source_hash:
            raise ValueError(f"Coordinator result hash mismatch: {rel(result_path)}")
        result = read_json(result_path)
        if "diagnostic_total_wall_s" not in result:
            raise ValueError(f"Selected 512-token source result is incomplete: {rel(result_path)}")
        trial = [t for t in result.get("free_running", []) if t.get("sampling", {}).get("seed") == seed]
        if len(trial) != 1:
            raise ValueError(f"Expected one source seed {seed} trial in {rel(result_path)}")
        trial = trial[0]
        if len(trial.get("token_ids", [])) != 512:
            raise ValueError(
                f"Coordinator says512 but source has{len(trial.get('token_ids', []))} IDs: {rel(result_path)}"
            )
        source_fixture = ROOT / known_cases[mode][job]["case"]
        fixture = read_json(source_fixture)
        if result.get("case") != rel(source_fixture):
            raise ValueError(f"Run result references unexpected fixture for {identity}")
        if len(fixture.get("prompt_token_ids", [])) != 2048:
            raise ValueError(f"Expected exact 2048-token saved suffix in {rel(source_fixture)}")
        if fixture.get("metadata", {}).get("sampling") != EXPECTED_SAMPLING:
            raise ValueError(f"Unexpected source sampling settings in {rel(source_fixture)}")
        if (
            trial.get("sampling", {}).get("temperature") != EXPECTED_SAMPLING["temperature"]
            or trial.get("sampling", {}).get("top_p") != EXPECTED_SAMPLING["top_p"]
            or trial.get("sampling", {}).get("top_k") != EXPECTED_SAMPLING["top_k"]
        ):
            raise ValueError(f"Source result sampling differs for {identity}")
        selected.append(
            {
                "mode": mode,
                "job": job,
                "seed": seed,
                "row": row,
                "coordinator_snapshots": evidence["snapshots"],
                "result_path": result_path,
                "result_sha256": source_hash,
                "result": result,
                "trial": trial,
                "source_fixture": source_fixture,
                "fixture": fixture,
            }
        )
    counts = collections.Counter(mode for mode, _, _ in adjudications)
    summary = {
        "source_directory": rel(SOURCE),
        "coordinator_snapshots": snapshots,
        "coordinator_rows_seen_by_mode": {mode: counts.get(mode, 0) for mode in BUILD},
        "finalized_rows_by_mode": {
            mode: counts.get(mode, 0)
            - sum(row.get("status") == "adjudication_followup_pending" for row in pending[mode])
            for mode in BUILD
        },
        "selected_inconclusive_512_by_mode": {mode: sum(x["mode"] == mode for x in selected) for mode in BUILD},
        "pending_root_review_by_mode": pending,
        "pending_counts_by_mode": {mode: len(rows) for mode, rows in pending.items()},
        "note": "Only explicit root coordinator rows are considered reviewed. Unlisted full rows remain pending; they are not inferred normal or complete.",
    }
    return selected, summary


def measured_trace_preparation(mode: str) -> dict:
    log = TRACE_LOG[mode]
    if not log.is_file():
        raise FileNotFoundError(f"Missing measured {mode} trace-preparation log: {log}")
    events = []
    for line_no, line in enumerate(log.read_text().splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "generator_trace_prepare":
            events.append({"line": line_no, "seconds": float(event["preparation_seconds"])})
    if len(events) != 1 or events[0]["seconds"] <= 0:
        raise ValueError(f"Expected exactly one positive generator_trace_prepare event in {rel(log)}; got {events}")
    return {
        "path": rel(log),
        "sha256": sha(log),
        "event": events[0],
        "proxy_note": "Measured same-mode first generator trace preparation from suffix1024 run; used once for this branchless cap1024 batch, not a measured suffix2048/cap1024 warmup.",
    }


def prepare(selected: list[dict], scan: dict, validate: bool) -> dict:
    if is_admitted():
        raise SystemExit(f"Refusing to refresh admitted destination: {rel(DEST / 'admission.json')}")
    manifests = {mode: [] for mode in BUILD}
    provenance = []
    for item in selected:
        mode, job, seed = item["mode"], item["job"], item["seed"]
        new_job = f"{job}.suffix2048.cap1024.seed{seed}"
        case = json.loads(json.dumps(item["fixture"]))
        original_seed = case.get("metadata", {}).get("seed")
        if original_seed != 42:
            raise ValueError(f"Source suffix fixture seed base is not42: {rel(item['source_fixture'])}")
        case["metadata"]["seed"] = seed
        case["metadata"]["adaptive_cap1024_provenance"] = {
            "source_fixture": rel(item["source_fixture"]),
            "source_fixture_sha256": sha(item["source_fixture"]),
            "source_result": rel(item["result_path"]),
            "source_result_sha256": item["result_sha256"],
            "coordinator_snapshots": item["coordinator_snapshots"],
            "mode": mode,
            "seed": seed,
            "previous_cap": 512,
            "new_cap": 1024,
            "verdict": "inconclusive",
            "reason": item["row"].get("reason", ""),
            "input_policy": "copy exact suffix2048 prompt IDs, chosen continuation and sampling; set trial seed only",
        }
        fixture_path = DEST / "fixtures" / mode / f"{new_job}.private.json"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
        fixture_hash = sha(fixture_path)
        manifests[mode].append(
            {"name": new_job, "case": rel(fixture_path), "branches": [], "free_run_tokens": 1024, "free_run_trials": 1}
        )
        provenance.append(
            {
                "mode": mode,
                "job": new_job,
                "source_job": job,
                "seed": seed,
                "verdict": "inconclusive",
                "reason": item["row"].get("reason", ""),
                "coordinator_snapshots": item["coordinator_snapshots"],
                "source_result": rel(item["result_path"]),
                "source_result_sha256": item["result_sha256"],
                "source_fixture": rel(item["source_fixture"]),
                "source_fixture_sha256": sha(item["source_fixture"]),
                "candidate_fixture": rel(fixture_path),
                "candidate_fixture_sha256": fixture_hash,
                "prompt_tokens": len(case["prompt_token_ids"]),
                "chosen_continuation_tokens": len(case["continuations"]["chosen"]),
                "sampling": case["metadata"]["sampling"],
                "seed": seed,
                "original_fixture_seed": original_seed,
            }
        )

    for mode, rows in manifests.items():
        rows.sort(key=lambda r: r["name"])
        write_json(DEST / f"{mode}-cases.json", rows)
    write_json(
        DEST / "provenance.json",
        {
            "status": "candidate_only_not_admitted",
            "source_directory": rel(SOURCE),
            "selection": "Fresh coordinator glob; only verdict=inconclusive and tokens=512 rows selected.",
            "deduplication": "mode+source job+seed; identical duplicate rows are deduplicated; conflicting adjudications fail closed.",
            "preservation": "Exact saved suffix2048 prompt IDs, chosen continuation, and sampling preserved; only metadata seed and new cap provenance are added. Candidate branches are empty and one free trial is requested.",
            "no_launch": True,
            "cases": provenance,
        },
    )
    write_json(DEST / "root-review-scan.json", scan)
    forecast = estimate_forecast(manifests, provenance, selected)
    write_json(DEST / "forecast.json", forecast)
    validation = []
    if validate:
        runner = ROOT / "python_env/bin/python"
        for mode in BUILD:
            manifest_path = DEST / f"{mode}-cases.json"
            command = [
                str(runner),
                "-m",
                "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
                "--cases",
                str(manifest_path),
                "--build-kwargs-file",
                str(BUILD[mode]),
                "--output-dir",
                str(DEST / "unused"),
                "--validate-only",
            ]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            validation.append(
                {
                    "mode": mode,
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                }
            )
            if result.returncode:
                raise SystemExit(result.stderr)
        write_json(
            DEST / "cpu-validation.json",
            {
                "status": "passed",
                "scope": "CPU-only precision_repro_batch --validate-only; no TTNN import or device work.",
                "checks": validation,
            },
        )
    return {
        "selected": len(provenance),
        "by_mode": {m: len(v) for m, v in manifests.items()},
        "pending": scan["pending_counts_by_mode"],
        "forecast": forecast,
        "validation": validation,
    }


def estimate_forecast(manifests, provenance, selected):
    p_by_key = {(p["mode"], p["source_job"], p["seed"]): p for p in provenance}
    s_by_key = {(s["mode"], s["job"], s["seed"]): s for s in selected}
    output = {
        "status": "candidate planning estimate, not serving performance",
        "formula": "For each explicit 512-token inconclusive source trial, preserve observed prefill/host overhead (diagnostic wall including prefill minus decode wall) and project decode as measured decode_wall_s * 1023/511. Add same-mode observed source-candidate model load once and one same-mode measured generator_trace_prepare event from the suffix1024 log. Range scales variable case work by 0.7x-1.7x, with load and preparation fixed.",
        "source_cap": 512,
        "candidate_cap": 1024,
        "source_decode_tokens": 511,
        "projected_decode_tokens": 1023,
        "range_variable_component_multipliers": [0.7, 1.7],
        "modes": {},
    }
    for mode, jobs in manifests.items():
        traceprep = measured_trace_preparation(mode)
        load_path = SOURCE / "runs" / mode / "load.json"
        loads = []
        if load_path.is_file():
            loads.append(
                {"path": rel(load_path), "wall_s": float(read_json(load_path)["wall_s"]), "sha256": sha(load_path)}
            )
        if jobs and not loads:
            raise FileNotFoundError(f"Missing measured {mode} load: {load_path}")
        load_s = max((x["wall_s"] for x in loads), default=0.0)
        work_sum = 0.0
        trials = []
        for job in jobs:
            source_job = job["name"].split(".suffix2048.cap1024.seed", 1)[0]
            seed = int(job["name"].rsplit(".seed", 1)[1])
            item = s_by_key[(mode, source_job, seed)]
            run = item["trial"]
            timing = run["decode_timing"]
            if timing.get("decode_tokens") != 511:
                raise ValueError(f"Expected 511 decode replays for selected 512-token source {job['name']}")
            decode = float(timing["decode_wall_s"])
            overhead = max(0.0, float(run["diagnostic_wall_s_including_prefill"]) - decode)
            projected_decode = decode * (1023 / 511)
            work = overhead + projected_decode
            work_sum += work
            p = p_by_key[(mode, source_job, seed)]
            trials.append(
                {
                    "job": job["name"],
                    "seed": item["seed"],
                    "source_result": p["source_result"],
                    "source_result_sha256": p["source_result_sha256"],
                    "source_prefill_host_overhead_seconds": overhead,
                    "source_decode_tokens": 511,
                    "source_decode_seconds": decode,
                    "projected_1023_decode_seconds": projected_decode,
                    "projected_case_seconds": work,
                }
            )
        fixed = load_s + traceprep["event"]["seconds"]
        output["modes"][mode] = {
            "jobs": len(jobs),
            "load_sources": loads,
            "load_seconds_used": load_s,
            "trace_preparation": traceprep,
            "case_estimates": trials,
            "estimated_total_seconds": fixed + work_sum,
            "estimated_range_seconds": [fixed + 0.7 * work_sum, fixed + 1.7 * work_sum],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    if is_admitted():
        raise SystemExit(f"Refusing to refresh admitted destination: {rel(DEST / 'admission.json')}")
    selected, scan = collect_root_rows()
    result = prepare(selected, scan, validate=not args.no_validate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
