#!/usr/bin/env python3
"""Host-only closure checker for the admitted 512->1024 diagnostic batch.

Run only after both mode queues and all coordinator adjudications are complete.
The script fails closed on missing/partial/conflicting evidence and writes its
summary outside the admitted evidence directory. It never imports TTNN.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[6]
BASE = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
SETUP = BASE / "setup"
SOURCE = SETUP / "swe85-second-cap-extensions"
OUTPUT = SETUP / "swe85-second-cap-extensions-completion-summary.json"
MODES = ("served", "full")


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def require_hash(path: pathlib.Path, expected: str, label: str) -> None:
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch for {rel(path)}: expected {expected}, got {actual}")


def trial_by_seed(document: dict, seed: int, path: pathlib.Path) -> dict:
    matches = [x for x in document.get("free_running", []) if x.get("sampling", {}).get("seed") == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one seed {seed} free-run trial in {rel(path)}, found {len(matches)}")
    return matches[0]


def build_expected() -> dict[tuple[str, str, int], dict]:
    expected = {}
    for mode in MODES:
        manifest_path = SOURCE / f"{mode}-candidate-cases.json"
        manifest = read_json(manifest_path)
        if not isinstance(manifest, list):
            raise ValueError(f"Expected list manifest: {rel(manifest_path)}")
        for job in manifest:
            name = job["name"]
            try:
                seed = int(name.rsplit(".seed", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Cannot recover seed from job name: {name}") from exc
            key = (mode, name, seed)
            if key in expected:
                raise ValueError(f"Duplicate case in {mode} manifest: {name}")
            if job.get("free_run_tokens") != 1024 or job.get("free_run_trials") != 1 or job.get("branches") != []:
                raise ValueError(f"Unexpected cap/branch/trial settings for {name}")
            expected[key] = {"job": job, "manifest": manifest_path}
    counts = collections.Counter(mode for mode, _, _ in expected)
    if counts != {"served": 14, "full": 24}:
        raise ValueError(f"Expected frozen manifest counts served14/full24, got {dict(counts)}")
    return expected


def collect_coordinator_rows(expected: dict) -> tuple[dict, list[dict]]:
    evidence: dict[tuple[str, str, int], dict] = {}
    hashes = []
    for coordinator in sorted(SOURCE.glob("*-coordinator-*.json")):
        coordinator_doc = read_json(coordinator)
        hashes.append({"path": rel(coordinator), "sha256": sha(coordinator)})
        for row in coordinator_doc.get("rows", []):
            mode, name, seed = row.get("mode"), row.get("case"), row.get("seed")
            key = (mode, name, seed)
            if key not in expected:
                continue
            sig = {k: row.get(k) for k in ("verdict", "tokens", "source", "source_sha256", "reason")}
            prior = evidence.get(key)
            if prior and prior["signature"] != sig:
                raise ValueError(f"Conflicting coordinator adjudications for {key}: {prior['signature']} vs {sig}")
            if not prior:
                evidence[key] = {"row": row, "signature": sig, "coordinators": []}
            evidence[key]["coordinators"].append({"path": rel(coordinator), "sha256": sha(coordinator)})
    return evidence, hashes


def verify_result(mode: str, key: tuple[str, str, int], spec: dict, adjudication: dict) -> dict:
    _, name, seed = key
    result_path = SOURCE / "runs" / mode / f"{name}.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing completed result: {rel(result_path)}")
    result = read_json(result_path)
    if "diagnostic_total_wall_s" not in result:
        raise ValueError(f"Result is partial/incomplete: {rel(result_path)}")
    job = spec["job"]
    fixture_path = ROOT / job["case"]
    fixture = read_json(fixture_path)
    trial = trial_by_seed(result, seed, result_path)
    ids = trial.get("token_ids", [])
    if not ids or len(ids) > 1024:
        raise ValueError(f"Generated token count outside (0,1024]: {rel(result_path)}")
    if len(ids) != adjudication["tokens"]:
        raise ValueError(
            f"Coordinator token count {adjudication['tokens']} != result count {len(ids)}: {rel(result_path)}"
        )
    if not isinstance(adjudication.get("verdict"), str) or not adjudication["verdict"]:
        raise ValueError(f"Missing root verdict for {key}")
    if adjudication.get("mode") != mode or adjudication.get("seed") != seed or adjudication.get("case") != name:
        raise ValueError(f"Coordinator row identity mismatch for {key}")
    if adjudication.get("source") != rel(result_path):
        raise ValueError(f"Coordinator references a different result for {key}")
    result_hash = sha(result_path)
    require_hash(result_path, adjudication.get("source_sha256", ""), "Coordinator result")

    ext = fixture.get("metadata", {}).get("cap_extension", {})
    if (
        ext.get("new_cap") != 1024
        or ext.get("previous_cap") != 512
        or ext.get("seed") != seed
        or ext.get("mode") != mode
    ):
        raise ValueError(f"Invalid 512->1024 provenance for {key}")
    source_result = ROOT / ext["source_result"]
    if not source_result.is_file():
        raise FileNotFoundError(f"Missing matched cap512 source result: {rel(source_result)}")
    require_hash(source_result, ext["source_result_sha256"], "Fixture cap512 source result")
    old = read_json(source_result)
    old_trial = trial_by_seed(old, seed, source_result)
    old_ids = old_trial.get("token_ids", [])
    if len(old_ids) != 512:
        raise ValueError(f"Expected matched source to contain exactly512 IDs for {key}; found{len(old_ids)}")
    if ids[:512] != old_ids:
        raise ValueError(f"First512 generated IDs do not match matched cap512 source for {key}")

    timing = trial.get("decode_timing", {})
    generated_minus_first = len(ids) - 1
    delta = timing.get("trace_counters_delta", {})
    if timing.get("decode_tokens") != generated_minus_first:
        raise ValueError(f"decode_tokens != generated_ids-1 for {key}")
    for counter in ("model_replays", "sample_replays", "readbacks"):
        if delta.get(counter) != generated_minus_first:
            raise ValueError(f"trace delta {counter} != generated_ids-1 for {key}: {delta.get(counter)}")
    if delta.get("captures") != 0:
        raise ValueError(f"trace capture delta is not zero for {key}: {delta.get('captures')}")

    return {
        "mode": mode,
        "job": name,
        "seed": seed,
        "verdict": adjudication["verdict"],
        "reason": adjudication.get("reason", ""),
        "generated_ids": len(ids),
        "source_cap512_ids": len(old_ids),
        "prefix512_identical": True,
        "decode_tokens": timing["decode_tokens"],
        "trace_deltas": {k: delta.get(k) for k in ("model_replays", "sample_replays", "readbacks", "captures")},
        "diagnostic_total_wall_s": result["diagnostic_total_wall_s"],
        "decode_wall_s": timing.get("decode_wall_s"),
        "result": rel(result_path),
        "result_sha256": result_hash,
        "fixture": rel(fixture_path),
        "fixture_sha256": sha(fixture_path),
        "cap512_source_result": rel(source_result),
        "cap512_source_result_sha256": sha(source_result),
        "coordinator_snapshots": adjudication["coordinators"],
        "adjudication_reason": adjudication.get("reason", ""),
    }


def mode_timing(mode: str) -> dict:
    launch = SOURCE / f"{mode}-launch.json"
    completion = SOURCE / f"{mode}-completion.json"
    if not launch.is_file() or not completion.is_file():
        raise FileNotFoundError(f"Missing launch/completion timing record for {mode}: {rel(launch)}, {rel(completion)}")
    launched, completed = read_json(launch), read_json(completion)
    if completed.get("exit_code") != 0:
        raise ValueError(f"{mode} completion exit code is not zero")
    prediction = launched.get("prediction", {})
    measured = completed.get("measured_hours")
    if measured is None:
        start, end = launched.get("start_unix"), completed.get("end_unix")
        if start is None or end is None or end < start:
            raise ValueError(f"Cannot derive elapsed wall hours for {mode}")
        measured = (end - start) / 3600.0
    expected = prediction.get("predicted_hours")
    interval = prediction.get("range_hours")
    if expected is None or not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"Incomplete prediction data for {mode}")
    return {
        "mode": mode,
        "launch": rel(launch),
        "launch_sha256": sha(launch),
        "completion": rel(completion),
        "completion_sha256": sha(completion),
        "predicted_hours": expected,
        "predicted_range_hours": interval,
        "measured_hours": measured,
        "measured_minus_predicted_hours": measured - expected,
        "measured_within_predicted_range": interval[0] <= measured <= interval[1],
    }


def build_summary() -> dict:
    if not (SOURCE / "admission.json").is_file():
        raise FileNotFoundError(f"Missing admitted evidence identity: {rel(SOURCE / 'admission.json')}")
    expected = build_expected()
    coordinator_rows, coordinator_hashes = collect_coordinator_rows(expected)
    missing_rows = [key for key in expected if key not in coordinator_rows]
    if missing_rows:
        raise ValueError(f"Coordinator adjudications incomplete ({len(missing_rows)} rows); pending: {missing_rows}")

    rows = []
    for key, spec in sorted(expected.items()):
        entry = coordinator_rows[key]
        row = entry["row"]
        rows.append(verify_result(key[0], key, spec, {**row, "coordinators": entry["coordinators"]}))
    timings = [mode_timing(mode) for mode in MODES]
    by_mode = {}
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        by_mode[mode] = {
            "trials": len(mode_rows),
            "verdict_counts": dict(collections.Counter(row["verdict"] for row in mode_rows)),
            "generated_id_total": sum(row["generated_ids"] for row in mode_rows),
            "root_adjudications": mode_rows,
            "timing_vs_prediction": next(x for x in timings if x["mode"] == mode),
        }
    return {
        "status": "completed_second_cap_queue_evidence_summary",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "scope": "Host-only aggregation of all admitted served14/full24 cap1024 results and root coordinator adjudications; no score or precision-selection decision.",
        "admission": {"path": rel(SOURCE / "admission.json"), "sha256": sha(SOURCE / "admission.json")},
        "manifest_counts": {mode: sum(row["mode"] == mode for row in rows) for mode in MODES},
        "invariants": {
            "all_second_cap_results_present_and_complete": True,
            "all_root_coordinator_rows_present": True,
            "all_matched_cap512_prefixes_identical": True,
            "all_decode_tokens_equal_generated_ids_minus_one": True,
            "all_model_replay_sample_replay_readback_deltas_equal_generated_ids_minus_one": True,
            "all_trace_capture_deltas_zero": True,
        },
        "coordinator_snapshots": coordinator_hashes,
        "by_mode": by_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=pathlib.Path, default=OUTPUT, help="Summary path outside the admitted evidence directory"
    )
    args = parser.parse_args()
    try:
        summary = build_summary()
    except Exception as exc:
        print(f"NOT CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(args.output),
                "counts": summary["manifest_counts"],
                "verdicts": {m: summary["by_mode"][m]["verdict_counts"] for m in MODES},
                "timing_vs_prediction": {m: summary["by_mode"][m]["timing_vs_prediction"] for m in MODES},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
