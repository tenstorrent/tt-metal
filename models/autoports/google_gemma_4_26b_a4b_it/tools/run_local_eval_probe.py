# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Run one capped real Harbor trial against an exclusively owned existing server."""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tti-root", type=Path, required=True)
    parser.add_argument("--harbor-python", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--repetition-detection", type=json.loads)
    parser.add_argument("--disable-thinking", action="store_true", help="Separate, explicit agent-policy experiment")
    parser.add_argument("--agent-system-template", type=Path, help="Explicit task-independent agent-policy control")
    parser.add_argument(
        "--normalize-submission-marker", action="store_true", help="Opt-in audited harness policy change"
    )
    parser.add_argument(
        "--request-seed", type=int, help="Explicit paired-diagnostic policy change, not a release default"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sys.path.insert(0, str(args.tti_root.resolve()))
    from llm_module.agentic.harbor import HarborRunConfig, run

    saved = json.loads(args.source_config.read_text())
    kwargs = copy.deepcopy(saved["agents"][0]["kwargs"])
    if args.agent_system_template:
        kwargs["config"].setdefault("agent", {})["system_template"] = args.agent_system_template.read_text()
    if args.disable_thinking:
        kwargs["config"]["model"]["model_kwargs"].setdefault("extra_body", {})["chat_template_kwargs"] = {
            "enable_thinking": False
        }
    if args.request_seed is not None:
        kwargs["config"]["model"]["model_kwargs"]["seed"] = args.request_seed
    if args.repetition_detection:
        kwargs["config"]["model"]["model_kwargs"]["extra_body"]["repetition_detection"] = args.repetition_detection
    # The local server has no API authentication; do not reuse CI credentials.
    config = HarborRunConfig(
        task_name=args.output.name,
        dataset="swebench-verified",
        agent="mini-swe-agent",
        model_name="openai/google/gemma-4-26B-A4B-it",
        jobs_dir=args.output,
        api_base=args.server.rstrip("/") + "/v1",
        n_concurrent_trials=1,
        n_attempts=1,
        environment_type="docker",
        agent_kwargs=kwargs,
        n_tasks=None,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=args.seconds,
        task_names=[args.task],
        llm_timeout_sec=1800,
        request_telemetry=True,
        normalize_submission_marker=args.normalize_submission_marker,
        agent_env={"OPENAI_API_KEY": "local-diagnostic"},
        venv_python=args.harbor_python,
    )
    for attempt in range(120):
        try:
            with urlopen(args.server.rstrip("/") + "/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            if attempt % 6 == 0:
                logging.info("Waiting for existing local server readiness")
        time.sleep(5)
    else:
        raise TimeoutError("Existing server did not become healthy within ten minutes")
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
