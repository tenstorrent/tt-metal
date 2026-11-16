#!/usr/bin/env python3
"""Refresh candidate-only 512->1024 cap extensions from reviewed coordinators.

This host-only helper never loads TTNN or runs inference. It scans the coordinator
snapshots currently present in setup/swe85-cap-extensions, selects only rows
explicitly adjudicated inconclusive at 512 generated IDs, and reconstructs a
1024-token free-run fixture from the exact original saved case. It refuses to
write if the destination has an admission.json.
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
SOURCE = SETUP / "swe85-cap-extensions"
DEST = SETUP / "swe85-second-cap-extensions"
BUILD = {
    "served": SETUP / "head_bfp8_build_kwargs.json",
    "full": SETUP / "full_precision_build_kwargs.json",
}


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


def collect_rows() -> list[dict]:
    rows = []
    seen: dict[tuple[str, str, int], str] = {}
    for coordinator in sorted(SOURCE.glob("*-coordinator-*.json")):
        document = read_json(coordinator)
        for row in document.get("rows", []):
            if row.get("verdict") != "inconclusive" or row.get("tokens") != 512:
                continue
            mode = row.get("mode")
            seed = row.get("seed")
            job = row.get("job")
            if mode not in BUILD or not isinstance(seed, int) or not isinstance(job, str):
                raise ValueError(f"Malformed eligible row in {coordinator}: {row}")
            source_result = ROOT / row["source"]
            if not source_result.is_file():
                raise FileNotFoundError(source_result)
            result = read_json(source_result)
            free = result.get("free_running", [])
            matching = [trial for trial in free if trial.get("sampling", {}).get("seed") == seed]
            if len(matching) != 1:
                raise ValueError(f"Expected one seed {seed} free-run result in {source_result}")
            if len(matching[0].get("token_ids", [])) != 512:
                raise ValueError(f"Coordinator says 512 IDs but source run disagrees: {source_result}")
            source_case = ROOT / result["case"]
            if not source_case.is_file():
                raise FileNotFoundError(source_case)
            case = read_json(source_case)
            source_seed = case.get("metadata", {}).get("seed")
            if source_seed is None:
                raise ValueError(f"Missing original seed in {source_case}")
            if source_seed != seed:
                raise ValueError(f"Fixture seed {source_seed} differs from selected result seed {seed}: {source_case}")
            if case.get("metadata", {}).get("sampling") != {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
            }:
                raise ValueError(f"Unexpected or missing sampling settings in {source_case}")
            identity = (mode, row.get("source_job", job.rsplit(".cap512.seed", 1)[0]), seed)
            signature = sha(source_result)
            previous = seen.get(identity)
            if previous is not None:
                if previous != signature:
                    raise ValueError(f"Conflicting duplicate adjudication for {identity}")
                continue
            seen[identity] = signature
            rows.append(
                {
                    "mode": mode,
                    "coordinator": coordinator,
                    "coordinator_sha256": sha(coordinator),
                    "row": row,
                    "source_result": source_result,
                    "source_case": source_case,
                    "case": case,
                    "trial": matching[0],
                    "identity": identity,
                }
            )
    return rows


def prepare(rows: list[dict], validate: bool) -> dict:
    if is_admitted():
        raise SystemExit(f"Refusing to overwrite admitted destination: {DEST / 'admission.json'}")
    manifests = {mode: [] for mode in BUILD}
    provenance = []
    fixture_hashes: dict[str, str] = {}
    for item in rows:
        mode = item["mode"]
        _, source_job, seed = item["identity"]
        job = f"{source_job}.cap1024.seed{seed}"
        case = json.loads(json.dumps(item["case"]))
        # Preserve prompt IDs, all continuations, and sampling; seed is per
        # trial and encoded only in the new cap-extension provenance.
        case["metadata"]["cap_extension"] = {
            "source_case": rel(item["source_case"]),
            "source_case_sha256": sha(item["source_case"]),
            "source_result": rel(item["source_result"]),
            "source_result_sha256": sha(item["source_result"]),
            "coordinator": rel(item["coordinator"]),
            "coordinator_sha256": item["coordinator_sha256"],
            "mode": mode,
            "seed": seed,
            "previous_cap": 512,
            "new_cap": 1024,
            "verdict": "inconclusive",
            "adjudication_reason": item["row"].get("reason", ""),
            "prompt_continuations_sampling_policy": "preserve exact original saved values",
        }
        filename = f"{job}.private.json"
        fixture = DEST / "fixtures" / mode / filename
        fixture_bytes = (json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
        existing = fixture_hashes.get(str(fixture))
        if existing is not None and existing != fixture_hash:
            raise ValueError(f"Conflicting duplicate fixture: {fixture}")
        fixture_hashes[str(fixture)] = fixture_hash
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(fixture_bytes)
        manifests[mode].append(
            {
                "name": job,
                "case": rel(fixture),
                "branches": [],
                "free_run_tokens": 1024,
                "free_run_trials": 1,
            }
        )
        provenance.append(
            {
                "mode": mode,
                "job": job,
                "seed": seed,
                "source_job": source_job,
                "verdict": "inconclusive",
                "adjudication_reason": item["row"].get("reason", ""),
                "coordinator": rel(item["coordinator"]),
                "coordinator_sha256": item["coordinator_sha256"],
                "source_result": rel(item["source_result"]),
                "source_result_sha256": sha(item["source_result"]),
                "source_case": rel(item["source_case"]),
                "source_case_sha256": sha(item["source_case"]),
                "fixture": rel(fixture),
                "fixture_sha256": fixture_hash,
                "prompt_token_count": len(case["prompt_token_ids"]),
                "continuation_lengths": {key: len(tokens) for key, tokens in case["continuations"].items()},
                "sampling": case["metadata"]["sampling"],
                "seed_policy": "one free run with seed matching this reviewed source trial",
            }
        )

    for mode, manifest in manifests.items():
        manifest.sort(key=lambda job: job["name"])
        write_json(DEST / f"{mode}-candidate-cases.json", manifest)
    write_json(
        DEST / "provenance.json",
        {
            "status": "candidate_only_not_admitted",
            "source_directory": rel(SOURCE),
            "selection": "Only coordinator rows with verdict=inconclusive and tokens=512; fresh glob on every invocation.",
            "exclusions": [
                "Do not select provisional sklearn other-anomaly rows unless a later coordinator adjudicates them inconclusive.",
                "Previously admitted swe85-cap-extensions queue is not modified.",
            ],
            "duplicate_policy": "Deduplicate by mode+source_job+seed; same result hash is a duplicate, conflicting result hashes are a hard error.",
            "cases": provenance,
        },
    )
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
            {"status": "passed", "scope": "CPU-only --validate-only; no TTNN import/device work", "checks": validation},
        )
    return {
        "selected": len(provenance),
        "by_mode": {m: len(v) for m, v in manifests.items()},
        "validation": validation,
        "costs": costs,
    }


def estimate_costs(manifests, provenance, rows):
    by_key = {(x["mode"], x["job"]): x for x in provenance}
    row_by_key = {(x["mode"], x["identity"][1], x["identity"][2]): x for x in rows}
    result = {
        "status": "planning_only_not_serving_performance",
        "method": "Per selected matched 512-token run, preserve measured prefill/host overhead (diagnostic wall minus decode wall) and scale decode by 1023/511, then add same-mode observed model-load time and a first-preparation allowance. Range is 0.7x–1.5x case work plus fixed load/preparation.",
        "first_preparation_allowance_seconds": 180,
        "modes": {},
    }
    for mode, jobs in manifests.items():
        if not jobs:
            result["modes"][mode] = {"jobs": 0, "estimated_total_seconds": 0, "range_seconds": [0, 0]}
            continue
        total = 0.0
        rows_out = []
        for job in jobs:
            meta = by_key[(mode, job["name"])]
            ident = (mode, meta["source_job"], meta["seed"])
            item = row_by_key[ident]
            trial = item["trial"]
            timing = trial["decode_timing"]
            decoded = timing["decode_tokens"]
            decode = float(timing["decode_wall_s"])
            prefill = max(0.0, float(trial["diagnostic_wall_s_including_prefill"]) - decode)
            projected_decode = decode * (1023 / max(1, decoded))
            work = prefill + projected_decode
            total += work
            rows_out.append(
                {
                    "job": job["name"],
                    "seed": meta["seed"],
                    "source_prefill_host_overhead_seconds": prefill,
                    "source_decode_tokens": decoded,
                    "source_decode_seconds": decode,
                    "projected_1023_decode_seconds": projected_decode,
                    "projected_case_seconds": work,
                }
            )
        load_paths = [
            SOURCE / "runs" / mode / "load.json",
            SETUP / "swe85-pilot-split85/runs/rest" / mode / "load.json",
        ]
        loads = [
            {"path": rel(path), "wall_s": read_json(path)["wall_s"], "sha256": sha(path)}
            for path in load_paths
            if path.exists()
        ]
        if not loads:
            raise FileNotFoundError(f"No measured {mode} model load available: {load_paths}")
        load_s = max(x["wall_s"] for x in loads)
        fixed = load_s + result["first_preparation_allowance_seconds"]
        result["modes"][mode] = {
            "jobs": len(jobs),
            "load_sources": loads,
            "load_seconds_used": load_s,
            "first_preparation_allowance_seconds": result["first_preparation_allowance_seconds"],
            "case_estimates": rows_out,
            "estimated_total_seconds": fixed + total,
            "estimated_range_seconds": [fixed + 0.7 * total, fixed + 1.5 * total],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-validate", action="store_true", help="Skip runner --validate-only checks")
    args = parser.parse_args()
    if is_admitted():
        raise SystemExit(f"Refusing to refresh admitted destination: {DEST / 'admission.json'}")
    rows = collect_rows()
    result = prepare(rows, validate=not args.no_validate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
