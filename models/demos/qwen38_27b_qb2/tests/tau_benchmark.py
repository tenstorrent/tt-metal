# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded official Tau3 pilot; raw conversations remain private host artifacts."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REVISION = "17e07b1da2bbc0cadfddeea36412686e0604127b"
DATASET_SHA256 = "3c6e2123a7290c4a3d94a234f3adba1cf4a9883b1323291bfd29ae14f65199ac"
TASK_IDS = (
    "task_032",
    "task_050",
    "task_019",
    "task_016",
    "task_057",
    "task_047",
    "task_060",
    "task_026",
    "task_051",
    "task_102",
    "task_066",
    "task_097",
)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def validate_source(source):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Tau3 source differs from the frozen revision")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("Tracked Tau3 source has modifications")
    dataset = source / "data/tau2/domains/banking_knowledge/tasks.json"
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Tau3 banking dataset differs from the frozen sample")
    ids = {row["id"] for row in json.loads(dataset.read_text())}
    if not set(TASK_IDS) <= ids:
        raise ValueError("Frozen tasks missing from the dataset")
    if not (source / "data/tau2/user_simulator/simulation_guidelines.md").is_file():
        raise ValueError("Tau3 global user-simulation guidelines are missing from the checkout")


def preflight():
    """Construct actual upstream task metadata without making model calls."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.helpers import get_info, get_tasks

    config = TextRunConfig(domain="banking_knowledge", task_ids=list(TASK_IDS), retrieval_config="bm25")
    tasks = get_tasks("banking_knowledge", task_ids=list(TASK_IDS))
    if len(tasks) != len(TASK_IDS):
        raise ValueError("Upstream runner did not load every selected task")
    get_info(config)


def trial(args):
    """Run upstream simulation/evaluation, recording calls before its tool parser."""
    os.umask(0o077)
    os.environ.update(
        OPENAI_API_KEY="EMPTY",
        OPENAI_BASE_URL=args.base_url + "/v1",
        OPENAI_API_BASE=args.base_url + "/v1",
        TAU2_DATA_DIR=str(args.source / "data"),
    )
    import tau2.evaluator.evaluator_nl_assertions as assertions
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain
    from tau2.utils import llm_utils

    args.output.mkdir(parents=True, exist_ok=False)
    original = llm_utils.completion

    # Observe, do not repair or reinterpret malformed tool-call arguments.
    def record_completion(*positional, **kwargs):
        started = time.monotonic()
        request = dict(kwargs)
        request.pop("api_key", None)
        row = {"request": request, "started_at": time.time()}
        try:
            response = original(*positional, **kwargs)
            row["response"] = response.to_dict()
            return response
        except Exception as error:
            row["error_type"] = type(error).__name__
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            with (args.output / "raw-calls.jsonl").open("a") as log:
                log.write(json.dumps(row, default=str) + "\n")

    llm_utils.completion = record_completion
    model = "openai/Qwen/Qwen3.8-27B"
    common = dict(seed=70914841, timeout=300, num_retries=0, api_base=args.base_url + "/v1", api_key="EMPTY")
    agent = dict(
        common,
        temperature=1.0,
        top_p=0.95,
        max_tokens=8192,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    )
    user = dict(
        common,
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assertions.DEFAULT_LLM_NL_ASSERTIONS = model
    assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = user
    llm_utils.set_llm_log_mode("all")
    cfg = TextRunConfig(
        domain="banking_knowledge",
        task_split_name="base",
        task_ids=[args.task_id],
        llm_agent=model,
        llm_user=model,
        llm_args_agent=agent,
        llm_args_user=user,
        num_trials=1,
        max_steps=60,
        max_errors=10,
        timeout=1200,
        max_concurrency=1,
        seed=70914841,
        max_retries=0,
        hallucination_retries=0,
        auto_review=False,
        retrieval_config="bm25",
        verbose_logs=True,
        log_level="INFO",
        save_to=str(args.output / "official"),
    )
    (args.output / "config.json").write_text(cfg.model_dump_json(indent=2))
    result = run_domain(cfg)
    (args.output / "completed-result.json").write_text(result.model_dump_json(indent=2))


def call_statistics(path):
    stats = dict(calls=0, truncated_calls=0, tool_calls=0, malformed_tool_calls=0, call_errors=0)
    if not path.exists():
        return stats
    for line in path.read_text().splitlines():
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            # A killed call writer may leave a partial final line. Count it.
            stats["call_errors"] += 1
            continue
        stats["calls"] += 1
        stats["call_errors"] += int("error_type" in call)
        for choice in call.get("response", {}).get("choices", []):
            stats["truncated_calls"] += int(choice.get("finish_reason") == "length")
            for tool in choice.get("message", {}).get("tool_calls") or []:
                stats["tool_calls"] += 1
                try:
                    arguments = json.loads(tool["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object")
                except (ValueError, TypeError, KeyError):
                    stats["malformed_tool_calls"] += 1
    return stats


def summarize(output, outcomes, *, final=False):
    rows = []
    for task_id in TASK_IDS:
        process = outcomes.get(task_id)
        row = dict(task_id=task_id, attempted=process is not None, passed=False, process=process)
        result_path = output / task_id / "completed-result.json"
        row.update(call_statistics(output / task_id / "raw-calls.jsonl"))
        if result_path.exists():
            try:
                simulations = json.loads(result_path.read_text())["simulations"]
                if len(simulations) != 1 or simulations[0]["task_id"] != task_id:
                    raise ValueError("Mismatched or duplicate trial")
                simulation = simulations[0]
                row.update(
                    reward=(simulation.get("reward_info") or {}).get("reward"),
                    termination=simulation.get("termination_reason"),
                )
                row["passed"] = bool(
                    row["reward"] == 1.0
                    and process
                    and process["returncode"] == 0
                    and not process["timed_out"]
                    and row["termination"] != "timeout"
                )
            except (ValueError, KeyError, TypeError) as error:
                row["artifact_error"] = str(error)
        rows.append(row)
    passed = sum(row["passed"] for row in rows)
    report = dict(
        state="completed" if final else "running",
        selected=len(TASK_IDS),
        attempted=len(outcomes),
        passed=passed,
        accuracy=passed / len(TASK_IDS) if final else None,
        all_selected_denominator=len(TASK_IDS),
        all_tasks_attempted=len(outcomes) == len(TASK_IDS),
        not_a_matched_published_reference=True,
        manual_review_pending=True,
        trials=rows,
        **{
            key: sum(row[key] for row in rows)
            for key in ("calls", "truncated_calls", "tool_calls", "malformed_tool_calls", "call_errors")
        },
    )
    # A harness failure before inference is not a measured model accuracy.
    if final and report["calls"] == 0:
        report.update(state="setup_failed", accuracy=None)
    return report


def stop(process):
    for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 10)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


def run(args):
    os.umask(0o077)
    validate_source(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        revision=REVISION,
        dataset_sha256=DATASET_SHA256,
        task_ids=TASK_IDS,
        selection="Frozen prior stratified diagnostic sample: 1 action, 4 small, 4 medium, 3 large document sets",
        concurrency=8,
        trials=1,
        retries=0,
        per_task_seconds=1200,
        run_seconds=2700,
        max_steps=60,
        agent_max_tokens=8192,
        user_and_judge_max_tokens=2048,
        model="Qwen/Qwen3.8-27B",
        simulator_and_judge="same local Qwen, greedy, no thinking",
        agent_sampling={"temperature": 1, "top_p": 0.95, "top_k": 20, "thinking": True},
        base_url=args.base_url,
        started_at=time.time(),
    )
    save(args.output / "protocol.json", manifest)
    deadline = time.monotonic() + 2700
    outcomes = {}
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())

    def worker(task_id):
        started = time.monotonic()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(args.source),
            "--base-url",
            args.base_url,
            "--output",
            str(args.output / task_id),
            "--task-id",
            task_id,
        ]
        process = None
        timed_out = False
        try:
            with (args.output / f"{task_id}.log").open("x") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                task_deadline = min(started + 1200, deadline)
                while process.poll() is None and not stopping.is_set():
                    if time.monotonic() >= task_deadline:
                        timed_out = True
                        break
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                timed_out = timed_out or stopping.is_set()
                stop(process)
            return dict(returncode=process.returncode, timed_out=timed_out, elapsed_seconds=time.monotonic() - started)
        finally:
            if process is not None:
                stop(process)

    tasks = iter(TASK_IDS)
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {}
        while True:
            while len(pending) < 8 and time.monotonic() < deadline and not stopping.is_set():
                task_id = next(tasks, None)
                if task_id is None:
                    break
                pending[pool.submit(worker, task_id)] = task_id
            if not pending:
                break
            finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            if not finished:
                continue
            for future in finished:
                task_id = pending.pop(future)
                try:
                    outcomes[task_id] = future.result()
                except Exception as error:
                    outcomes[task_id] = dict(returncode=None, timed_out=False, controller_error=type(error).__name__)
            report = summarize(args.output, outcomes)
            save(args.output / "progress.json", report)
            print(
                json.dumps(
                    {k: report[k] for k in ("attempted", "selected", "passed", "tool_calls", "malformed_tool_calls")}
                ),
                flush=True,
            )
    report = summarize(args.output, outcomes, final=True)
    report.update(elapsed_seconds=time.time() - manifest["started_at"])
    if stopping.is_set():
        report.update(state="interrupted", accuracy=None)
    save(args.output / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task-id", choices=TASK_IDS)
    arguments = parser.parse_args()
    trial(arguments) if arguments.task_id else run(arguments)
