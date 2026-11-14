#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Remeasure used B32 rows into isolated, precision-labelled stage12 evidence."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from collect_benchmark_roofline import check, collect

MODEL = Path(__file__).resolve().parents[1]
TOOLS = MODEL / "tools"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def policy_hash(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def validate_raw(raw, row, count):
    if (
        raw["failed"]
        or raw["completed"] != count
        or raw["input_lens"] != [row["input_tokens"]] * count
        or raw["output_lens"] != [row["output_tokens"]] * count
    ):
        raise ValueError("Incomplete or mismatched request population")


def measured_workload(planned, prior):
    for key in ("id", "batch", "concurrency", "input_tokens", "output_tokens", "sampling"):
        if planned[key] != prior[key]:
            raise ValueError(f"Original measured workload differs from plan: {key}")
    return dict(planned, requests=prior["requests"])


def rate_prediction(prior, original_server, candidate_server):
    ranges = {"decode_tok_s_user": (0.90, 1.05), "prefill_tok_s": (0.85, 1.05)}
    if "duration_s" in prior:
        ranges["duration_s"] = (0.90, 1.15)
    return {
        "written_unix": time.time(),
        "basis": "Head-only BFP8 short-control slowdown1.46%; decode predicted0.90–1.05x prior, prefill0.85–1.05x prior. Head fraction and workload length/concurrency can differ from short control. Matching-population measured duration predicted0.90–1.15x prior, excluding warmup.",
        "original_row": prior,
        "original_source_commits": original_server["source_commits"],
        "candidate_source_commits": candidate_server["source_commits"],
        "original_configuration_evidence_sha256": original_server.get("configuration_evidence_sha256"),
        "candidate_configuration_evidence_sha256": candidate_server["configuration_evidence_sha256"],
        "ranges": {
            key: {
                "prior": prior[key],
                "ratio_low": low,
                "ratio_high": high,
                "low": low * prior[key],
                "high": high * prior[key],
            }
            for key, (low, high) in ranges.items()
        },
    }


def compare_prediction(measured, prediction):
    comparisons = {}
    for key, expected in prediction["ranges"].items():
        ratio = measured[key] / expected["prior"]
        hit = expected["ratio_low"] <= ratio <= expected["ratio_high"]
        comparisons[key] = {
            "measured": measured[key],
            "ratio_to_prior": ratio,
            "range_hit": hit,
            "explanation_required": not hit,
        }
    return {
        "rates": comparisons,
        "explanation_required": any(item["explanation_required"] for item in comparisons.values()),
        "explanation": None,
    }


def measured_row(row, raw, directory, policy):
    validate_raw(raw, row, row["requests"])
    if raw["mean_ttft_ms"] <= 0 or raw["mean_tpot_ms"] <= 0:
        raise ValueError("Nonpositive timing")
    return dict(
        row,
        config_label=policy["config_id"],
        precision_sha256=policy_hash(policy),
        prefill_tok_s=row["input_tokens"] / (raw["mean_ttft_ms"] / 1000),
        decode_tok_s_user=1000 / raw["mean_tpot_ms"],
        ttft_s=raw["mean_ttft_ms"] / 1000,
        step_ms=raw["mean_tpot_ms"],
        duration_s=raw["duration"],
        completed=raw["completed"],
        raw=str(directory / "perf-b32.json"),
        server_identity=str(directory / "perf-b32-server.json"),
        phase_accounting=str(directory / "roofline.json"),
    )


def report(output, original, rows, server, policy, expected_ids):
    rates = dict(
        schema_version=1,
        status="completed" if len(rows) == len(expected_ids) else "incomplete",
        config_label=policy["config_id"],
        precision_sha256=policy_hash(policy),
        server_config=server,
        server=server,
        validated_batches=[32],
        rows=rows,
        missing_rows=[
            {"id": i, "reason": "Not completed in candidate run"}
            for i in expected_ids
            if i not in {r["id"] for r in rows}
        ],
        sampling_policies=original["sampling_policies"],
    )
    save(output / "serving_rates.json", rates)
    for name in ("gpqa", "swe"):
        save(
            output / f"serving_rates-{name}.json",
            dict(
                rates,
                selected_sampling_policy=name,
                rows=[r for r in rows if r["sampling"] == name],
                missing_rows=[r for r in rates["missing_rows"] if r["id"].endswith("-" + name)],
            ),
        )
    # Deliberately no mixed top-level rows: explain_hours cannot select precision.
    save(
        output / "serving_rates-by-config.json",
        {
            "schema_version": 1,
            "usage": "Select one configuration and one policy before replay.",
            "configurations": {
                original["server_config"]["precision"]["config_id"]: original,
                policy["config_id"]: rates,
            },
        },
    )


def invoke(argv, env, log, timeout):
    with log.open("w") as stream:
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Stop only this client process group, never the API server/device owner.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(
                "Client timed out; server may still drain requests. No subsequent rows or shutdown attempted."
            )
    if code:
        raise subprocess.CalledProcessError(code, argv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-precision", type=Path, required=True)
    p.add_argument(
        "--server-identity",
        type=Path,
        required=True,
        help="Fresh observed B32 identity from benchmark_server; no server launch here",
    )
    p.add_argument("--original-rates", type=Path, default=MODEL / "doc/benchmark/serving_rates.json")
    p.add_argument("--original-plan", type=Path, default=MODEL / "doc/benchmark/plan.json")
    p.add_argument("--budget-seconds", type=float, required=True)
    p.add_argument("--row-timeout-seconds", type=float, default=1800)
    p.add_argument("--stop-file", type=Path, required=True)
    p.add_argument("--row-id", help="Remeasure only this existing used row")
    p.add_argument("--dataset-seed", type=int, help="Explicit original measured seed; warmup uses seed+1")
    args = p.parse_args()
    if args.budget_seconds <= 600 or args.row_timeout_seconds <= 0:
        p.error("Budget must exceed600s closure reserve; timeout must be positive")
    policy = read(args.expected_precision)
    server = read(args.server_identity)
    original = read(args.original_rates)
    if server["precision"] != policy or server["max_num_seqs"] != 32 or server["max_model_len"] != 262144:
        raise ValueError("Fresh server identity does not match candidate B32/full-context policy")
    if policy["config_id"] == original["server_config"]["precision"]["config_id"]:
        raise ValueError("Candidate needs distinct configuration label")
    selected = [r for r in original["rows"] if r["batch"] == 32 and r["sampling"] in ("gpqa", "swe")]
    if len(selected) != 15:
        raise ValueError("Expected exactly15 previously used B32 rows")
    if args.row_id:
        selected = [r for r in selected if r["id"] == args.row_id]
        if len(selected) != 1:
            raise ValueError("Requested row is not one of the15 used rows")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "original-serving_rates.json", original)
    save(args.output / "expected-precision.json", policy)
    save(args.output / "perf-b32-server.json", server)
    plan = read(args.original_plan)["rows"]
    completed = []
    started = time.monotonic()
    deadline = started + args.budget_seconds
    record = {"status": "running", "started_unix": time.time(), "budget_seconds": args.budget_seconds, "commands": []}

    def persist():
        save(args.output / "run.json", record)
        report(args.output, original, completed, server, policy, [r["id"] for r in selected])

    persist()
    try:
        check(argparse.Namespace(run_dir=args.output, concurrency=32), server)
        for prior in selected:
            planned = next(r for r in plan if r["id"] == prior["id"])
            row = measured_workload(planned, prior)
            dataset_seed = args.dataset_seed if args.dataset_seed is not None else 7000 + plan.index(planned) * 2
            estimate = 1.5 * prior["duration_s"] * (1 + row["concurrency"] / row["requests"])
            if args.stop_file.exists() or time.monotonic() + estimate + 600 >= deadline:
                record["status"] = "stopped_between_rows"
                break
            dest = args.output / "grid" / row["id"]
            dest.mkdir(parents=True)
            prediction = rate_prediction(prior, original["server_config"], server)
            prediction.update(row=row, estimated_seconds=estimate)
            save(dest / "prediction.json", prediction)
            save(dest / "perf-b32-server.json", server)
            shutil.copy2(
                args.server_identity.parent / server["configuration_evidence"], dest / server["configuration_evidence"]
            )
            env = dict(
                os.environ,
                BENCHMARK_SERVER_BATCH="32",
                BENCHMARK_PHASE_DIR=str(Path(server["phase_log"]).parent),
                BENCHMARK_SAMPLING=row["sampling"],
            )
            row_deadline = min(deadline - 600, time.monotonic() + args.row_timeout_seconds)
            for warm in (True, False):
                count = row["concurrency"] if warm else row["requests"]
                name = "perf-b32" + ("-warmup" if warm else "")
                argv = [
                    sys.executable,
                    str(TOOLS / "benchmark_eval_client.py"),
                    "--model",
                    server["model"],
                    "--base-url",
                    server["base_url"],
                    "--result-dir",
                    str(dest),
                    "--result-filename",
                    name + ".json",
                    "--max-concurrency",
                    str(row["concurrency"]),
                    "--num-prompts",
                    str(count),
                    "--random-input-len",
                    str(row["input_tokens"]),
                    "--random-output-len",
                    str(row["output_tokens"]),
                    "--seed",
                    str(dataset_seed + int(warm)),
                ]
                record["commands"].append({"argv": argv, "sampling": row["sampling"], "started_unix": time.time()})
                persist()
                remaining = row_deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Row deadline exhausted before next client")
                invoke(argv, env, dest / (name + ".log"), remaining)
                validate_raw(read(dest / (name + ".json")), row, count)
            collect(argparse.Namespace(run_dir=dest, concurrency=32), server)
            measured = measured_row(row, read(dest / "perf-b32.json"), dest, policy)
            measured["prediction"] = str(dest / "prediction.json")
            measured["prediction_comparison"] = compare_prediction(measured, prediction)
            save(dest / "prediction_comparison.json", measured["prediction_comparison"])
            completed.append(measured)
            save(dest / "complete.json", {"row": row, "finished_unix": time.time()})
            persist()
        else:
            record["status"] = "completed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        persist()


if __name__ == "__main__":
    main()
