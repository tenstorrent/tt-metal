#!/usr/bin/env python3
"""Refresh candidate-only 1024->2048 diagnostic cap extensions.

This helper reads reviewed second-cap coordinator snapshots and matched source
outputs, rebuilds exact-input 2048-token fixtures, and runs only the batch
runner's --validate-only path. It does not import TTNN or launch inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[6]
BASE = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
SETUP = BASE / "setup"
SOURCE = SETUP / "swe85-second-cap-extensions"
DEST = SETUP / "swe85-third-cap-extensions"
BUILD = {
    "served": SETUP / "head_bfp8_build_kwargs.json",
    "full": SETUP / "full_precision_build_kwargs.json",
}
FIRST_PREP = (
    SETUP / "swe85-persistent-short-controls/runs/full/"
    "anomaly.psf__requests-5414.call10.onset.optional-suffix512.json"
)
EXPECTED_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: pathlib.Path):
    return json.loads(path.read_text())


def write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def is_admitted() -> bool:
    return (DEST / "admission.json").exists()


def source_job(case_name: str, seed: int) -> str:
    suffix = f".cap1024.seed{seed}"
    if not case_name.endswith(suffix):
        raise ValueError(f"Unexpected second-cap case identity: {case_name}")
    return case_name[: -len(suffix)]


def collect_rows() -> tuple[list[dict], dict]:
    rows = []
    seen: dict[tuple[str, str, int], str] = {}
    snapshots = []
    for coordinator in sorted(SOURCE.glob("*-coordinator-*.json")):
        coordinator_hash = sha(coordinator)
        snapshots.append({"path": rel(coordinator), "sha256": coordinator_hash})
        document = read_json(coordinator)
        for row in document.get("rows", []):
            if row.get("verdict") != "inconclusive" or row.get("tokens") != 1024:
                continue
            mode, seed, name = row.get("mode"), row.get("seed"), row.get("case")
            if mode not in BUILD or not isinstance(seed, int) or not isinstance(name, str):
                raise ValueError(f"Malformed eligible row in {coordinator}: {row}")
            job = source_job(name, seed)
            source_result = ROOT / row["source"]
            if not source_result.is_file():
                raise FileNotFoundError(source_result)
            source_result_hash = sha(source_result)
            if row.get("source_sha256") != source_result_hash:
                raise ValueError(f"Coordinator source hash mismatch: {source_result}")
            result = read_json(source_result)
            trials = [t for t in result.get("free_running", []) if t.get("sampling", {}).get("seed") == seed]
            if len(trials) != 1:
                raise ValueError(f"Expected exactly one matched seed {seed} in {source_result}")
            trial = trials[0]
            if len(trial.get("token_ids", [])) != 1024:
                raise ValueError(f"Expected 1024 generated IDs in {source_result}")
            timing = trial.get("decode_timing", {})
            if timing.get("decode_tokens") != 1023:
                raise ValueError(f"Expected 1023 decode tokens after first token in {source_result}")
            source_case = ROOT / result["case"]
            if not source_case.is_file():
                raise FileNotFoundError(source_case)
            case = read_json(source_case)
            metadata = case.get("metadata", {})
            if metadata.get("seed") != seed or metadata.get("sampling") != EXPECTED_SAMPLING:
                raise ValueError(f"Seed/sampling mismatch in source fixture {source_case}")
            if (
                trial.get("sampling", {}).get("temperature") != EXPECTED_SAMPLING["temperature"]
                or trial.get("sampling", {}).get("top_p") != EXPECTED_SAMPLING["top_p"]
                or trial.get("sampling", {}).get("top_k") != EXPECTED_SAMPLING["top_k"]
            ):
                raise ValueError(f"Source result sampling differs from expected settings: {source_result}")
            previous = metadata.get("cap_extension")
            if not isinstance(previous, dict) or previous.get("previous_cap") != 512 or previous.get("new_cap") != 1024:
                raise ValueError(f"Missing preserved 512->1024 provenance in {source_case}")
            ident = (mode, job, seed)
            prior_hash = seen.get(ident)
            if prior_hash is not None:
                if prior_hash != source_result_hash:
                    raise ValueError(f"Conflicting duplicate adjudication for {ident}")
                continue
            seen[ident] = source_result_hash
            rows.append(
                {
                    "mode": mode,
                    "coordinator": coordinator,
                    "coordinator_sha256": coordinator_hash,
                    "row": row,
                    "job": job,
                    "source_result": source_result,
                    "source_result_sha256": source_result_hash,
                    "source_case": source_case,
                    "case": case,
                    "trial": trial,
                    "identity": ident,
                }
            )

    # Report all admitted second-cap rows absent from the current coordinator
    # snapshots, without treating absent full-mode review as a completed result.
    expected = {mode: [] for mode in BUILD}
    for mode in BUILD:
        manifest_path = SOURCE / f"{mode}-candidate-cases.json"
        for entry in read_json(manifest_path):
            expected[mode].append(entry["name"])
    all_row_ids = set()
    for coordinator in SOURCE.glob("*-coordinator-*.json"):
        for row in read_json(coordinator).get("rows", []):
            mode, seed, name = row.get("mode"), row.get("seed"), row.get("case")
            if mode in BUILD and isinstance(seed, int) and isinstance(name, str):
                try:
                    all_row_ids.add((mode, source_job(name, seed), seed))
                except ValueError:
                    pass
    pending = {mode: [] for mode in BUILD}
    for mode, names in expected.items():
        manifest_path = SOURCE / f"{mode}-candidate-cases.json"
        for entry in read_json(manifest_path):
            name = entry["name"]
            seed = int(name.rsplit(".seed", 1)[1])
            base = source_job(name, seed)
            if (mode, base, seed) not in all_row_ids:
                pending[mode].append(name)
    scan = {
        "source_directory": rel(SOURCE),
        "coordinator_snapshots": snapshots,
        "second_cap_manifest_counts": {mode: len(names) for mode, names in expected.items()},
        "latest_coordinator_row_counts_by_mode": {
            mode: len({(m, j, s) for (m, j, s) in all_row_ids if m == mode}) for mode in BUILD
        },
        "selected_inconclusive_1024_rows": len(rows),
        "selected_by_mode": {mode: sum(r["mode"] == mode for r in rows) for mode in BUILD},
        "pending_without_any_coordinator_row_by_mode": pending,
        "note": "Coordinator snapshots are incomplete while second-cap full review is ongoing. Missing rows are listed explicitly and are not assumed complete or normal.",
    }
    return rows, scan


def prepare(rows: list[dict], scan: dict, validate: bool) -> dict:
    if is_admitted():
        raise SystemExit(f"Refusing to refresh admitted destination: {DEST / 'admission.json'}")
    manifests = {mode: [] for mode in BUILD}
    provenance = []
    for item in rows:
        mode, job, seed = item["identity"]
        new_job = f"{job}.cap2048.seed{seed}"
        case = json.loads(json.dumps(item["case"]))
        old_extension = case["metadata"]["cap_extension"]
        case["metadata"]["cap_extension_history"] = [old_extension]
        extension = {
            "source_case": rel(item["source_case"]),
            "source_case_sha256": sha(item["source_case"]),
            "source_result": rel(item["source_result"]),
            "source_result_sha256": item["source_result_sha256"],
            "coordinator": rel(item["coordinator"]),
            "coordinator_sha256": item["coordinator_sha256"],
            "mode": mode,
            "seed": seed,
            "previous_cap": 1024,
            "new_cap": 2048,
            "verdict": "inconclusive",
            "adjudication_reason": item["row"].get("reason", ""),
            "prompt_continuations_sampling_policy": "preserve exact second-cap saved values; no prompt reconstruction",
            "previous_cap_metadata_preserved": True,
        }
        case["metadata"]["cap_extension"] = extension
        fixture = DEST / "fixtures" / mode / f"{new_job}.private.json"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
        fixture_hash = sha(fixture)
        manifests[mode].append(
            {
                "name": new_job,
                "case": rel(fixture),
                "branches": [],
                "free_run_tokens": 2048,
                "free_run_trials": 1,
            }
        )
        provenance.append(
            {
                "mode": mode,
                "job": new_job,
                "source_job": job,
                "seed": seed,
                "verdict": "inconclusive",
                "adjudication_reason": item["row"].get("reason", ""),
                "coordinator": rel(item["coordinator"]),
                "coordinator_sha256": item["coordinator_sha256"],
                "source_result": rel(item["source_result"]),
                "source_result_sha256": item["source_result_sha256"],
                "source_case": rel(item["source_case"]),
                "source_case_sha256": sha(item["source_case"]),
                "second_cap_metadata": old_extension,
                "fixture": rel(fixture),
                "fixture_sha256": fixture_hash,
                "prompt_token_count": len(case["prompt_token_ids"]),
                "continuation_lengths": {k: len(v) for k, v in case["continuations"].items()},
                "sampling": case["metadata"]["sampling"],
                "seed_policy": "one free run using the seed of the matched reviewed 1024-token source trial",
            }
        )

    for mode, manifest in manifests.items():
        manifest.sort(key=lambda entry: entry["name"])
        write_json(DEST / f"{mode}-candidate-cases.json", manifest)
    write_json(
        DEST / "provenance.json",
        {
            "status": "candidate_only_not_admitted",
            "source_directory": rel(SOURCE),
            "selection": "Fresh glob of second-cap coordinator snapshots; only verdict=inconclusive and tokens=1024 rows are selected.",
            "deduplication": "mode+source job+seed; repeated identical source result hashes deduplicate; conflicting hashes fail closed.",
            "preservation": "Each candidate copies the exact second-cap fixture, prompt IDs, every continuation, sampling metadata, and seed; the prior 512->1024 cap_extension is copied into cap_extension_history.",
            "no_launch": True,
            "cases": provenance,
        },
    )
    write_json(DEST / "coordinator-scan-summary.json", scan)
    costs = estimate_costs(manifests, provenance, rows)
    write_json(DEST / "cost-estimate.json", costs)

    validation = []
    if validate:
        runner = ROOT / "python_env/bin/python"
        for mode in BUILD:
            manifest_path = DEST / f"{mode}-candidate-cases.json"
            cmd = [
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
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            validation.append(
                {
                    "mode": mode,
                    "command": cmd,
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                }
            )
            if result.returncode:
                raise SystemExit(result.stderr)
        write_json(
            DEST / "host-validation.json",
            {
                "status": "passed",
                "scope": "CPU-only precision_repro_batch --validate-only; no TTNN import, model load, or device work.",
                "checks": validation,
            },
        )
    return {
        "selected": len(provenance),
        "by_mode": {m: len(v) for m, v in manifests.items()},
        "pending": scan["pending_without_any_coordinator_row_by_mode"],
        "validation": validation,
        "costs": costs,
    }


def estimate_costs(manifests, provenance, rows):
    by_key = {(p["mode"], p["job"]): p for p in provenance}
    row_by_key = {(r["mode"], r["job"], r["identity"][2]): r for r in rows}
    prep = read_json(FIRST_PREP)
    # The recorded 64-token chosen teacher-forcing run is stored under
    # continuations.chosen in this result schema.
    prep_trial = prep.get("continuations", {}).get("chosen")
    if prep_trial is None:
        # Historical generator result schema records TF branches by name in branches[].
        prep_trial = next((t for t in prep.get("branches", []) if t.get("name") in ("chosen64", "chosen_64")), None)
    if prep_trial is None:
        raise ValueError(f"Could not identify measured first chosen64 TF preparation in {FIRST_PREP}")
    prep_seconds = float(prep_trial.get("diagnostic_wall_s_including_prefill", prep_trial.get("wall_s", 0)))
    if prep_seconds <= 0:
        raise ValueError("Measured first-TF preparation allowance is not positive")
    output = {
        "status": "candidate planning estimate, not serving performance",
        "method": "For each matched 1024-token result, prefill/host overhead = diagnostic_wall_s_including_prefill - decode_wall_s; project decode as decode_wall_s * 2047/1023. Add max same-mode observed model-load time and one measured prior first chosen64 teacher-forcing branch as generator-preparation proxy. Range scales variable case work by 0.7x-1.7x and keeps load/preparation fixed.",
        "target_generated_ids": 2048,
        "source_generated_ids": 1024,
        "source_decode_tokens": 1023,
        "projected_decode_tokens": 2047,
        "range_variable_component_multipliers": [0.7, 1.7],
        "first_preparation_allowance_seconds": prep_seconds,
        "first_preparation_source": {
            "path": rel(FIRST_PREP),
            "sha256": sha(FIRST_PREP),
            "measurement": "first chosen64 teacher-forcing branch diagnostic wall; preparation proxy, not measured 2048-cap warmup",
        },
        "modes": {},
    }
    for mode, jobs in manifests.items():
        work_total = 0.0
        case_rows = []
        for job in jobs:
            meta = by_key[(mode, job["name"])]
            item = row_by_key[(mode, meta["source_job"], meta["seed"])]
            trial = item["trial"]
            timing = trial["decode_timing"]
            decode = float(timing["decode_wall_s"])
            overhead = max(0.0, float(trial["diagnostic_wall_s_including_prefill"]) - decode)
            projected_decode = decode * (2047 / timing["decode_tokens"])
            work = overhead + projected_decode
            work_total += work
            case_rows.append(
                {
                    "job": job["name"],
                    "seed": meta["seed"],
                    "source_prefill_host_overhead_seconds": overhead,
                    "source_decode_tokens": timing["decode_tokens"],
                    "source_decode_seconds": decode,
                    "projected_2047_decode_seconds": projected_decode,
                    "projected_case_seconds": work,
                }
            )
        source_loads = [
            SOURCE / "runs" / mode / "load.json",
            SETUP / "swe85-cap-extensions/runs" / mode / "load.json",
            SETUP / "swe85-pilot-split85/runs/rest" / mode / "load.json",
        ]
        loads = [
            {"path": rel(p), "wall_s": float(read_json(p)["wall_s"]), "sha256": sha(p)}
            for p in source_loads
            if p.is_file()
        ]
        if jobs and not loads:
            raise FileNotFoundError(f"No measured {mode} load data found: {source_loads}")
        load_s = max((x["wall_s"] for x in loads), default=0.0)
        fixed = load_s + prep_seconds
        output["modes"][mode] = {
            "jobs": len(jobs),
            "load_sources": loads,
            "load_seconds_used": load_s,
            "first_preparation_allowance_seconds": prep_seconds,
            "case_estimates": case_rows,
            "estimated_total_seconds": fixed + work_total,
            "estimated_range_seconds": [fixed + 0.7 * work_total, fixed + 1.7 * work_total],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    if is_admitted():
        raise SystemExit(f"Refusing to refresh admitted destination: {DEST / 'admission.json'}")
    rows, scan = collect_rows()
    result = prepare(rows, scan, validate=not args.no_validate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
