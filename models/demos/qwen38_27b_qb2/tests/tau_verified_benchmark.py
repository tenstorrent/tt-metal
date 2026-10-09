# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned, bounded Tau-Verified airline evaluation with an independent simulator."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

REVISION = "864350a8971a8f8ee9e7b8472e2edc380a806b0c"
DATA_HASHES = {
    "db.json": "1af9fea6e03ca7ca15a22bb3fcaf3e351393e3fc9070b6777947da8996f7531b",
    "policy.md": "10dc0525421521208be39cee235bba84a16e2bcba9899eb93d92cd81d2f62fc4",
    "split_tasks.json": "b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b",
    "tasks.json": "202f0cdb1cf4bc20b8a3284f00e212882a8dda8eaab81d1d69a5dd9676911ebe",
}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_source(source):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Tau-Verified revision differs from the pinned reference")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("Tau-Verified tracked source was modified")
    data = source / "data/tau2/domains/airline"
    for name, expected in DATA_HASHES.items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Tau-Verified dataset changed: " + name)
    tasks = json.loads((data / "tasks.json").read_text())
    ids = [task["id"] for task in tasks]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("Expected all 50 distinct verified airline tasks")
    return ids


def validate_endpoint(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Endpoint must be an HTTP(S) API base URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not embed credentials or query parameters")
    return value.rstrip("/")


def preflight(source):
    os.environ["TAU2_DATA_DIR"] = str(source / "data")
    import tau2
    from tau2.run import get_environment_info, get_tasks

    if not Path(tau2.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError("Imported tau2 is not the pinned verified checkout")
    tasks = get_tasks("airline", task_split_name="base")
    if {task.id for task in tasks} != set(validate_source(source)):
        raise ValueError("Upstream task loader did not load all verified tasks")
    get_environment_info("airline", include_tool_info=True)
    return dict(passed=True, tasks=len(tasks), hardware_opened=False, model_calls=0, revision=REVISION)


def protocol(args, ids):
    return dict(
        revision=REVISION,
        dataset_hashes=DATA_HASHES,
        domain="airline",
        task_ids=ids,
        selected=50,
        trials=1,
        seed=300,
        concurrency=8,
        max_steps=200,
        agent_model="openai/Qwen/Qwen3.8-27B",
        simulator_model=args.simulator_model,
        judge_model=args.judge_model,
        agent_base_url=args.base_url,
        simulator_base_url=args.simulator_base_url,
        simulator_key_env=args.simulator_key_env,
        temperature=0,
        agent_max_tokens=16384,
        request_timeout_seconds=900,
        wall_timeout_seconds=args.run_seconds,
        request_retries=0,
        exact_openrouter_parity=False,
        differences=[
            "Official pinned Tau-Verified runner, not OpenRouter's private Inspect configuration",
            "One trial, 16384-token agent output bound, 900-second request deadline and bounded total runtime",
            "No request retries; independent simulator and upstream gpt-4o-mini assertion judge",
        ],
    )


def child(args):
    # Only the independent simulator/judge use this credential. The local agent
    # has an explicit dummy key and endpoint on every call. Never save secrets.
    os.environ.update(
        OPENAI_API_KEY=os.environ[args.simulator_key_env],
        OPENAI_API_BASE=args.simulator_base_url,
        OPENAI_BASE_URL=args.simulator_base_url,
        TAU2_DATA_DIR=str(args.source / "data"),
    )
    from tau2.data_model.simulation import RunConfig
    from tau2.evaluator import evaluator_nl_assertions
    from tau2.run import run_domain
    from tau2.utils import llm_utils

    original = llm_utils.completion
    calls = args.output / "calls"
    calls.mkdir()

    def observe(*positional, **kwargs):
        started = time.monotonic()
        record = dict(
            role="agent" if kwargs.get("model") == "openai/Qwen/Qwen3.8-27B" else "simulator_or_judge",
            request={
                key: kwargs[key] for key in ("model", "messages", "tools", "temperature", "max_tokens") if key in kwargs
            },
        )
        try:
            response = original(*positional, **kwargs)
            record["response"] = response.to_dict()
            return response
        except Exception as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            save(calls / (uuid.uuid4().hex + ".json"), record)

    llm_utils.completion = observe
    common = dict(timeout=900, num_retries=0, max_retries=0)
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = args.judge_model
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        **common,
        api_base=args.simulator_base_url,
    )
    cfg = RunConfig(
        domain="airline",
        task_split_name="base",
        llm_agent="openai/Qwen/Qwen3.8-27B",
        llm_user=args.simulator_model,
        llm_args_agent=dict(
            common,
            api_base=args.base_url,
            api_key="EMPTY",
            temperature=0,
            max_tokens=16384,
            extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": True}},
        ),
        llm_args_user=dict(common, api_base=args.simulator_base_url, temperature=0),
        num_trials=1,
        max_steps=200,
        max_errors=10,
        max_concurrency=8,
        seed=300,
        save_to=str((args.output / "official").resolve()),
    )
    save(args.output / "config.json", cfg.model_dump(mode="json"))
    result = run_domain(cfg)
    save(args.output / "completed-result.json", result.model_dump(mode="json"))


def summarize(output, ids, returncode, timed_out):
    path = output / "completed-result.json"
    if not path.exists():
        path = output / "official.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    simulations = data.get("simulations", [])
    observed = [s["task_id"] for s in simulations]
    if len(observed) != len(set(observed)) or not set(observed) <= set(ids):
        raise ValueError("Duplicate or unexpected task results")
    rows = []
    for task_id in ids:
        s = next((s for s in simulations if s["task_id"] == task_id), None)
        rows.append(
            dict(
                task_id=task_id,
                completed=s is not None,
                reward=(s.get("reward_info") or {}).get("reward") if s else None,
            )
        )
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    truncated = sum(
        choice.get("finish_reason") == "length"
        for call in calls
        for choice in call.get("response", {}).get("choices", [])
    )
    errors = sum("error_type" in call for call in calls)
    correct = sum(row["reward"] == 1 for row in rows)
    complete = len(simulations) == len(ids) and returncode == 0 and not timed_out and not errors and bool(calls)
    return dict(
        complete=complete,
        selected=len(ids),
        completed=len(simulations),
        correct=correct,
        accuracy=correct / len(ids) if complete else None,
        lower_bound_counting_missing_as_incorrect=correct / len(ids) if calls else None,
        call_errors=errors,
        truncated_calls=truncated,
        calls=len(calls),
        returncode=returncode,
        timed_out=timed_out,
        trials=rows,
        exact_openrouter_parity=False,
        agentic_release_qualified=False,
    )


def main(args):
    os.umask(0o077)
    ids = validate_source(args.source)
    if args.prepare:
        result = preflight(args.source)
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "preflight.json", result)
        print(json.dumps(result))
        return
    args.base_url = validate_endpoint(args.base_url)
    args.simulator_base_url = validate_endpoint(args.simulator_base_url)
    if args.base_url == args.simulator_base_url or "qwen3.8-27b" in args.simulator_model.lower():
        raise ValueError("Reference evaluation requires an independent simulator")
    if not os.environ.get(args.simulator_key_env):
        raise ValueError("Independent simulator credential is missing: " + args.simulator_key_env)
    if args.worker:
        child(args)
        return
    preflight(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "protocol.json", protocol(args, ids))
    process = None
    timed_out = False

    def interrupted(*_):
        raise InterruptedError("Tau supervisor interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (args.output / "run.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, *sys.argv, "--worker"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                process.wait(timeout=args.run_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        result = summarize(args.output, ids, process.returncode if process else None, timed_out)
        save(args.output / "summary.json", result)
    print(json.dumps(result))
    if not result["complete"]:
        raise RuntimeError("Tau-Verified did not complete all selected tasks cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8079/v1")
    parser.add_argument("--simulator-base-url", default="https://openrouter.ai/api/v1")
    # First openai/ selects LiteLLM's compatible transport; the remaining
    # openai/model is the model ID expected by OpenRouter.
    parser.add_argument("--simulator-model", default="openai/openai/gpt-5.1")
    parser.add_argument("--judge-model", default="openai/openai/gpt-4o-mini")
    parser.add_argument("--simulator-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--run-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    main(parser.parse_args())
