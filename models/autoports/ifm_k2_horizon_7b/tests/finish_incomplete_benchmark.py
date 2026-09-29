"""Preserve incomplete accuracy and measure remaining profiles on the SAME clock.

Fallback only: caller must stop its owned accuracy runner, await failed-summary
creation and confirm that the server is idle before invoking this helper. This
script sends no process signals and performs no direct hardware operation. The
original server hook owns profile replacement; the packaged command helper owns
the clients it launches and their deadline cleanup. Stage status stays failed.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def partial_accuracy(run, config, manifest):
    result = {}
    for task in config["tasks"]:
        path = run / task / "responses.jsonl"
        responses, parse_errors = [], []
        if path.exists():
            for number, line in enumerate(path.open(), 1):
                if not line.strip():
                    continue
                try:
                    responses.append(json.loads(line))
                except ValueError as exc:
                    parse_errors.append({"line": number, "error": str(exc)})
        reasons = Counter()
        empty_length = 0
        for response in responses:
            choices = response.get("choices", [])
            if choices:
                reason = choices[0].get("finish_reason", "missing")
                reasons[reason] += 1
                message = choices[0].get("message", {})
                if reason == "length" and not message.get("content"):
                    empty_length += 1
            else:
                reasons["missing_choice"] += 1
        group = manifest["groups"][task]
        result[task] = {
            "completed_raw_responses": len(responses),
            "expected_samples": group["sample_count"],
            "population": group["population"],
            "unanswered_due_to_client_interruption": group["sample_count"] - len(responses),
            "finish_reasons": dict(reasons),
            "empty_final_at_length": empty_length,
            "raw_parse_errors": parse_errors,
            "score": None,
            "scoring_status": "not completed",
            "responses_path": str(path.relative_to(run)),
            "responses_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            "generation": config["generation"][task],
            "frozen_tasks": {
                child: {
                    key: value
                    for key, value in manifest["tasks"][child].items()
                    if key not in ("indices", "document_sha256")
                }
                for child in group["tasks"]
            },
        }
    return result


def render_report(run, config, manifest, summary, started):
    from benchmark_stage.evidence import benchmark_rows
    from benchmark_stage.report import write_report

    # Do not turn a fast partial response set into a scored subset. The package
    # renders all validated performance/roofline details; explicit N/A rows
    # below replace its empty accuracy table without touching any raw result.
    rendering = copy.deepcopy(config)
    rendering.update(tasks=[], references={}, metrics={})
    reporting_summary = copy.deepcopy(summary)
    reporting_summary["accuracy"] = {}
    write_report(run, rendering, reporting_summary)
    path = run / "REPORT.md"
    text = path.read_text()
    marker = "|---|---:|---|---:|---:|---:|---|\n"
    rows = []
    for task in config["tasks"]:
        row = summary["incomplete_accuracy"][task]
        completed = summary.get("completed_accuracy", {}).get(task)
        if completed:
            result = read(run / completed["results_path"])
            for metric, score, reference in benchmark_rows(config, manifest, task, result):
                published = f"{reference['score']:.2f}" if reference else "Unavailable"
                delta = f"{score-reference['score']:+.2f}" if reference else "N/A"
                source = f"[reference]({reference['source_url']})" if reference else "No matching published figure"
                rows.append(
                    f"| {task} | {row['completed_raw_responses']} completed / {row['expected_samples']} frozen / {row['population']} full | {metric} | {score:.2f} | {published} | {delta} | {source} |"
                )
            continue
        rows.append(
            f"| {task} | {row['completed_raw_responses']} completed / {row['expected_samples']} frozen / {row['population']} full | Upstream scoring incomplete | N/A | Unavailable | N/A | No matching published figure |"
        )
    text = text.replace(marker, marker + "\n".join(rows) + "\n", 1)
    text = text.replace("Samples / full", "Completed / frozen / full", 1)
    text = text.replace(
        "Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.",
        "The overall accuracy stage is incomplete. Aggregate scores are shown only for tasks with every frozen question received and scored by upstream code. Incomplete tasks retain per-question results without a partial-sample aggregate. No accuracy verdict is made.",
    )
    text += "\n## Incomplete accuracy and preserved protocol\n\n"
    text += summary["error"] + "\n\n"
    text += "| Task | Completed / expected | Unanswered after client interruption | Stop | Token-limited | Empty final at limit |\n|---|---:|---:|---:|---:|---:|\n"
    for task, row in summary["incomplete_accuracy"].items():
        reasons = row["finish_reasons"]
        text += f"| {task} | {row['completed_raw_responses']} / {row['expected_samples']} | {row['unanswered_due_to_client_interruption']} | {reasons.get('stop', 0)} | {reasons.get('length', 0)} | {row['empty_final_at_length']} |\n"
    text += "\nUnanswered questions were interrupted by the client; they are not counted as model token-limit truncations. Stop and length counts apply only to retained complete API responses.\n"
    if summary.get("offline_accuracy"):
        offline = summary["offline_accuracy"]
        text += (
            f"\nOffline upstream scoring retained {offline['received_and_scored']} received answers across "
            f"{offline['intended_questions']} frozen questions. [Per-question scores](upstream-question-scores.jsonl), "
            "[scoring and complete-task aggregation](upstream-question-scoring.json), "
            "[actual received prompt/token verification](chat-token-verification-partial.json). "
            "Missing questions remain explicitly unanswered, with no fabricated model responses.\n"
        )
        for task, completed in summary.get("completed_accuracy", {}).items():
            text += f"\n`{task}`: [complete frozen-task upstream results]({completed['results_path']}); {completed['samples']} / {completed['samples']} received and scored.\n"
    text += "\n" + config.get("protocol_notes", "") + "\n\n" + config.get("reference_notes", "") + "\n\n"
    text += "Generation settings were preserved; no shorter thinking budget, changed question subset, or partial-score denominator was substituted. [Frozen document IDs, hashes and few-shot settings](manifest.json), [exact generation configuration](run_config.json), [partial response counts and protocol](incomplete-accuracy.json).\n\n"
    for task, row in summary["incomplete_accuracy"].items():
        text += f"- `{task}` generation: `{json.dumps(row['generation'], sort_keys=True)}`. Frozen scorer/task settings: `{json.dumps(row['frozen_tasks'], sort_keys=True)}`.\n"
    text += "\n## Implementation, setup and clock\n\n"
    text += "Full 36-layer generated autoport, model/tokenizer revision `036114ce8d46c32b24c15423211069abb9c5d25e`, selected precision `down4_l24to34_head8_lofi`, four Blackhole chips, TP4/DP1, and context capacity 524288. [Identity](../identity.json), [setup notes](../RUN_NOTES.md), [preserved interruption summary](accuracy-interrupted-summary.json).\n\n"
    starting = summary["original_invocation"]["starting_server"]
    if starting["running"]:
        prior = starting["recorded_process"]
        text += f"Supplied server: {prior['max_num_seqs']} slots, PID {prior['pid']}; startup before this invocation: {prior['startup_seconds']:.3f} seconds. Setup is separate from the continuous client-stage clock.\n\n"
    else:
        text += "No live server was supplied; the initial 32-slot launch is included in this invocation's clock.\n\n"
    if config.get("setup_record"):
        text += f"[This invocation's pre-run setup record](../{config['setup_record']}) preserves previous incomplete attempts separately.\n\n"
    text += "The clock starts at the original invocation, before its first selection hook. This continuation does not reset it. Verification, all inference, profile switches/reloads, compilation, warmups, phase collection, checks and this report remain on that clock.\n\n"
    for check, result in summary.get("checks", {}).items():
        text += f"- [{check}]({check}.log): exit code `{result.get('exit_code')}`, {result.get('status')}.\n"
    elapsed = time.monotonic() - started
    text += f"\nTotal original client-stage elapsed wall time through this report: {elapsed:.3f} seconds. Deadline: {summary['budget_seconds']:.0f} seconds. Final status: failed; accuracy incomplete.\n"
    path.write_text(text)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["within_original_budget"] = summary["elapsed_seconds"] < summary["budget_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=MODEL_DIR / "doc/benchmark/run")
    parser.add_argument(
        "--accuracy-stopped-server-idle",
        action="store_true",
        required=True,
        help="Caller confirms owned accuracy process exited and server has drained normally",
    )
    parser.add_argument(
        "--reason", required=True, help="Observed reason for incomplete accuracy; do not state an accuracy verdict"
    )
    args = parser.parse_args()
    from benchmark_stage.evidence import validate_performance, validate_server
    from benchmark_stage.roofline import load_roofline
    from benchmark_stage.run import command
    from benchmark_stage.subsets import digest

    run = args.run_dir.resolve()
    config = read(run / "run_config.json")
    manifest = read(run / "manifest.json")
    invocation = read(run.parent / "invocation.json")
    original = read(run / "summary.json")
    if original.get("status") != "failed":
        raise RuntimeError("Require the original runner's final failed summary before fallback")
    started = float(invocation["started_monotonic"])
    budget = min(float(config.get("budget_seconds", 3600)), 3600)
    deadline = started + budget
    if not 0 <= time.monotonic() - started < budget:
        raise RuntimeError("Original invocation deadline expired or monotonic clock changed; no new run is allowed")
    for source, target in (
        ("summary.json", "accuracy-interrupted-summary.json"),
        ("REPORT.md", "accuracy-interrupted-REPORT.md"),
    ):
        if (run / target).exists():
            raise FileExistsError(f"Refusing to overwrite preserved interruption evidence: {target}")
        if (run / source).exists():
            shutil.copyfile(run / source, run / target)
    summary = copy.deepcopy(original)
    summary.update(
        status="failed",
        error="Accuracy incomplete: " + args.reason,
        accuracy={},
        accuracy_execution=config.get("accuracy_execution", "sequential"),
        performance=dict(original.get("performance", {})),
        original_invocation=invocation,
        original_packaged_elapsed_seconds=original.get("elapsed_seconds"),
        budget_seconds=budget,
        original_deadline_monotonic=deadline,
        clock_scope="original invocation through continuation reporting and evidence/context checks",
        checks={},
        continuation_started_monotonic=time.monotonic(),
    )
    summary["incomplete_accuracy"] = partial_accuracy(run, config, manifest)
    write(run / "incomplete-accuracy.json", summary["incomplete_accuracy"])
    baseline = read(run / "perf-b32-server.json")

    def phase(capacity, action):
        command(
            [*config["roofline_command"], "--run-dir", str(run), "--concurrency", str(capacity), "--action", action],
            run / f"roofline-b{capacity}-{action}-continuation.log",
            deadline,
        )
        if action == "collect":
            load_roofline(run, required=(str(capacity),))

    try:
        validate_server(baseline, 32, config["model"], config["base_url"], output=run)
        phase(32, "check")
        for concurrency in (32, 1):
            server = baseline
            if concurrency == 1:
                server_path = run / "perf-b1-server.json"
                command(
                    [
                        *config["performance_server_command"],
                        "--max-num-seqs",
                        "1",
                        "--base-url",
                        config["base_url"],
                        "--output",
                        str(server_path),
                    ],
                    run / "perf-b1-server.log",
                    deadline,
                )
                server = read(server_path)
                validate_server(server, 1, config["model"], config["base_url"], baseline=baseline, output=run)
                phase(1, "check")
            for warmup in (True, False):
                name = f"perf-b{concurrency}" + ("-warmup" if warmup else "")
                requests = concurrency if warmup else max(8, concurrency * 3)
                if (run / f"{name}.json").exists():
                    raise FileExistsError(f"Refusing to overwrite existing performance evidence: {name}")
                argv = [
                    *config.get("benchmark_command", [config.get("vllm_cli", "vllm"), "bench", "serve"]),
                    "--backend",
                    "vllm",
                    "--model",
                    config["model"],
                    "--base-url",
                    config["base_url"],
                    "--endpoint",
                    "/v1/completions",
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    "4096",
                    "--random-output-len",
                    str(config.get("output_tokens", 128)),
                    "--random-range-ratio",
                    "0.0",
                    "--num-prompts",
                    str(requests),
                    "--max-concurrency",
                    str(concurrency),
                    "--request-rate",
                    "inf",
                    "--ignore-eos",
                    "--temperature",
                    "0",
                    "--seed",
                    str(4100 + concurrency + int(warmup)),
                    "--percentile-metrics",
                    "ttft,tpot,itl,e2el",
                    "--metric-percentiles",
                    "50,95,99",
                    "--save-result",
                    "--save-detailed",
                    "--result-dir",
                    str(run),
                    "--result-filename",
                    f"{name}.json",
                ]
                command(argv, run / f"{name}.log", deadline)
                raw = read(run / f"{name}.json")
                validate_performance(raw, requests, config.get("output_tokens", 128))
                if raw.get("model_id") != config["model"] or raw.get("max_concurrency") != concurrency:
                    raise ValueError(f"{name}: raw performance model/concurrency mismatch")
                if not warmup:
                    summary["performance"][str(concurrency)] = {
                        "concurrency": concurrency,
                        "requested_input_tokens": 4096,
                        "server_max_num_seqs": server["max_num_seqs"],
                        "server_identity_sha256": digest(server),
                        "requested_output_tokens": config.get("output_tokens", 128),
                        "requests": requests,
                        "completed": raw["completed"],
                        **{
                            key: value
                            for key, value in raw.items()
                            if key.startswith(("mean_", "median_", "p95_", "p99_"))
                            or key
                            in (
                                "duration",
                                "request_throughput",
                                "output_throughput",
                                "total_token_throughput",
                                "total_input_tokens",
                                "total_output_tokens",
                            )
                        },
                    }
            # This collector finishes while this exact profile is still live.
            phase(concurrency, "collect")
            summary["elapsed_seconds"] = time.monotonic() - started
            write(run / "summary.json", summary)
        load_roofline(run, required=("1", "32"))
        summary["performance_status"] = "both profiles and phase accounting valid"
    except BaseException as exc:  # noqa: BLE001 - retain evidence on client interruption too
        summary["performance_status"] = "incomplete"
        summary["continuation_error"] = f"{type(exc).__name__}: {exc}"
        summary["error"] += "; performance continuation: " + summary["continuation_error"]
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write(run / "summary.json", summary)
        try:
            command(
                [
                    sys.executable,
                    str(MODEL_DIR / "tests/finalize_partial_benchmark_accuracy.py"),
                    "--run-dir",
                    str(run),
                ],
                run / "finalize-partial-accuracy.log",
                deadline,
            )
            summary = read(run / "summary.json")
        except BaseException as exc:  # noqa: BLE001 - preserve failed offline evidence before closure
            summary["offline_accuracy_error"] = f"{type(exc).__name__}: {exc}"
            summary["error"] += "; offline evidence: " + summary["offline_accuracy_error"]
        # Prompt reconstruction is offline and preserves actual 4096-token
        # warmup/measured inputs. Its work stays on the same original clock.
        if summary.get("performance_status") == "both profiles and phase accounting valid":
            try:
                environment = dict(os.environ)
                environment.pop("PYTHONPATH", None)
                environment.update(VLLM_PLUGINS="", VLLM_TARGET_DEVICE="empty")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Original deadline expired before performance prompt preservation")
                with (run / "preserve_performance_prompts.log").open("w") as log:
                    subprocess.run(
                        [
                            str(Path(config["vllm_cli"]).with_name("python")),
                            str(MODEL_DIR / "tests/preserve_performance_prompts.py"),
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=remaining,
                        check=True,
                    )
            except BaseException as exc:  # noqa: BLE001 - finish the original-clock failure report
                summary["performance_input_preservation_error"] = f"{type(exc).__name__}: {exc}"
                summary["error"] += (
                    "; performance input preservation: " + summary["performance_input_preservation_error"]
                )
        render_report(run, config, manifest, summary, started)
        write(run / "summary.json", summary)
        plugin = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])
        checks = {
            "evidence-check": [
                sys.executable,
                "-m",
                "benchmark_stage.check",
                "--model-dir",
                str(MODEL_DIR),
                "--hf-model",
                config["model"],
            ],
            "context-check": [
                sys.executable,
                str(plugin / "scripts/check_context_contract.py"),
                "--model-dir",
                str(MODEL_DIR),
                "--hf-model",
                config["model"],
                "--stage",
                "benchmark",
                "--require-contract",
            ],
        }
        for label, argv in checks.items():
            remaining = deadline - time.monotonic()
            result = {"command": argv, "exit_code": None, "status": "not run: original deadline exhausted"}
            if remaining > 0:
                try:
                    with (run / f"{label}.log").open("w") as log:
                        log.write(json.dumps(argv) + "\n")
                        log.flush()
                        proc = subprocess.run(
                            argv, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=False
                        )
                    result.update(exit_code=proc.returncode, status="passed" if proc.returncode == 0 else "failed")
                except subprocess.TimeoutExpired:
                    result["status"] = "timed out at original deadline"
            summary["checks"][label] = result
        original_hash = (run.parent / "setup/context-contract-original.sha256").read_text().strip()
        actual_hash = hashlib.sha256((MODEL_DIR / "doc/context_contract.json").read_bytes()).hexdigest()
        summary["context_contract_bytes_unchanged"] = actual_hash == original_hash
        summary["checks"]["evidence-check"]["expected_failure"] = "stage status is failed and accuracy is incomplete"
        render_report(run, config, manifest, summary, started)
        write(run / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "failed",
                "accuracy": "incomplete",
                "performance": summary["performance_status"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "within_original_budget": summary["within_original_budget"],
            }
        )
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
