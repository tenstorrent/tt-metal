# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Launch the inherited full-context server with isolated TTFT artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from benchmark_server_config import profile_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-async-scheduling", action="store_true", help="Explicit scheduler comparison experiment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = profile_plan(32)
    plan.pop("concurrent_requests", None)  # Request concurrency belongs to each client run, not the launch.
    if args.no_async_scheduling:
        plan["command"] = [
            "--no-async-scheduling" if value == "--async-scheduling" else value for value in plan["command"]
        ]
    plan["status"] = "launch_intent_execve_pending_not_running_identity"
    plan["required_before_launch"] = [
        "Verify exclusive device ownership before invoking this launcher.",
        "Use server startup logs and server_info.json to attest the actual running configuration.",
    ]
    environment = {**os.environ, **plan["environment_overrides"]}
    environment.update(
        TT_METAL_LOGS_PATH=str(args.output / "runtime"),
        VLLM_CACHE_ROOT=str(args.output / "vllm_cache"),
    )
    plan["environment_overrides"] = {
        key: environment[key] for key in (*plan["environment_overrides"], "TT_METAL_LOGS_PATH", "VLLM_CACHE_ROOT")
    }
    plan["environment_overrides"].update(
        {key: value for key, value in environment.items() if key.startswith("GEMMA4_")}
    )
    model_root = Path(__file__).resolve().parent.parent
    plan["implementation_sha256"] = {
        str(path.relative_to(model_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((model_root / "tt").glob("*.py"))
    }
    (args.output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    os.execve(plan["command"][0], plan["command"], environment)


if __name__ == "__main__":
    main()
