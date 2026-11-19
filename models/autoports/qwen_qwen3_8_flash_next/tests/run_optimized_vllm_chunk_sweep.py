# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial real-server chunk sweep, retaining owned servers on diagnostic failure."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def get(path):
    with urllib.request.urlopen("http://localhost:8000" + path, timeout=5) as response:
        return response.read().decode()


def idle_metrics():
    values = {}
    for line in get("/metrics").splitlines():
        for name in ("num_requests_running", "num_requests_waiting"):
            if line.startswith("vllm:" + name + "{") or line.startswith("vllm:" + name + " "):
                values[name] = values.get(name, 0) + float(line.split()[-1])
    return values if len(values) == 2 and all(value == 0 for value in values.values()) else None


def devices_free():
    result = subprocess.run(["fuser", *[f"/dev/tenstorrent/{i}" for i in range(4)]], capture_output=True)
    if result.returncode == 0:
        return False
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise RuntimeError(f"Device ownership check failed: {result}")
    return True


def process_snapshot():
    processes = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            processes[int(path.parent.name)] = (int(fields[1]), int(fields[19]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return processes


def owned_processes(root):
    processes = process_snapshot()
    owned = {root}
    while True:
        expanded = owned | {pid for pid, (parent, _) in processes.items() if parent in owned}
        if expanded == owned:
            return {pid: processes[pid][1] for pid in owned if pid in processes}
        owned = expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default="256,33,17")
    parser.add_argument("--evidence-prefix", default="stage10_chunk")
    args = parser.parse_args()
    if not args.evidence_prefix.replace("_", "").isalnum():
        parser.error("Evidence prefix must use letters, digits and underscores")
    model = Path(__file__).resolve().parents[1]
    stage = model / "doc/optimized_vllm"
    summary = dict(
        passed=False,
        context=262144,
        max_num_seqs=2,
        decode_users=1,
        runs=[],
        prefill_chunk_alignment=8,
        decode_interleave_prefill_steps=1,
        decode_interleave_decode_steps=2,
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                model / "tt/generator.py",
                model / "tt/generator_vllm.py",
                stage / "run_server.sh",
                Path("models/common/sampling/generator.py"),
                Path("models/common/sampling/tt_sampling.py"),
            )
        },
    )
    manifest = stage / "chunk_sweep_execution.json"
    if manifest.exists():
        parser.error("Preserve previous execution manifest before a new sweep")
    try:
        for budget in map(int, args.budgets.split(",")):
            if not devices_free():
                raise RuntimeError("Devices occupied; no server launched")
            evidence = model / f"readiness_vllm/{args.evidence_prefix}_{budget}"
            result_path = model / f"doc/eval_readiness/long_context_timing/chunk_{budget}.json"
            if result_path.exists():
                raise RuntimeError(f"Preserve previous result: {result_path}")
            if evidence.exists():
                raise RuntimeError(f"Preserve existing evidence: {evidence}")
            evidence.mkdir(parents=True)
            server_log = evidence / "server.log"
            launch = ["bash", str(stage / "run_server.sh"), str(evidence)]
            env = dict(os.environ, TOKEN_BUDGET=str(budget), TRACKING="0")
            with (evidence / "runner.log").open("w") as output:
                server = subprocess.Popen(launch, env=env, stdout=output, stderr=subprocess.STDOUT)
            row = dict(budget=budget, runner_pid=server.pid, server_command=launch, evidence=str(evidence))
            summary["runs"].append(row)
            manifest.write_text(json.dumps(summary, indent=2) + "\n")
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Owned server exited during startup: {server.returncode}")
                try:
                    get("/health")
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Server health deadline; retained for diagnosis")
                    time.sleep(2)
            owned = owned_processes(server.pid)
            command = [
                sys.executable,
                "-m",
                "readiness_check.run_long_context",
                "--part",
                "chunk",
                "--server-url",
                "http://localhost:8000",
                "--model-dir",
                str(model),
                "--server-log",
                str(server_log),
                "--max-num-batched-tokens",
                str(budget),
                "--decode-users",
                "1",
                "--chunk-prompts",
                "4096,32768",
                "--decoder-max-tokens",
                "65536",
                "--request-timeout",
                "7200",
                "--itl-bound-ms",
                "1000",
            ]
            row["probe_command"] = command
            print(f"CHUNK_SWEEP_START budget={budget} runner={server.pid}", flush=True)
            with (evidence / "chunk_probe.log").open("w") as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=7500)
            row["probe_exit"] = result.returncode
            row["result"] = json.loads(result_path.read_text()) if result_path.exists() else None
            if result.returncode or row["result"] is None or row["result"]["errors"]:
                raise RuntimeError("Chunk probe failed; owned server retained for diagnosis")
            if sorted(p["prompt_tokens"] for p in row["result"]["per_prompt"]) != [4096, 32768]:
                raise RuntimeError("Chunk probe did not measure both requested prompt lengths")
            deadline = time.monotonic() + 120
            while True:
                idle = idle_metrics()
                if idle is not None:
                    row["idle_before_shutdown"] = idle
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Requests did not drain; owned server retained for diagnosis")
                time.sleep(2)
            owned.update(owned_processes(server.pid))
            row["owned_process_start_ticks"] = owned
            server.send_signal(signal.SIGTERM)
            row["runner_shutdown_exit"] = server.wait(timeout=120)
            deadline = time.monotonic() + 30
            while True:
                remaining = process_snapshot()
                row["remaining_owned_pids"] = [
                    pid for pid, started in owned.items() if pid in remaining and remaining[pid][1] == started
                ]
                if not row["remaining_owned_pids"] or time.monotonic() >= deadline:
                    break
                time.sleep(1)
            if row["remaining_owned_pids"]:
                raise RuntimeError("Owned server processes remain after shutdown")
            row["devices_free"] = devices_free()
            if not row["devices_free"]:
                raise RuntimeError("Owned server cleanup left device handles; inspect before continuing")
            print(f"CHUNK_SWEEP_DONE budget={budget} result={row['result']}", flush=True)
        eligible = [
            row["budget"]
            for row in summary["runs"]
            if all(isinstance(p["worst_itl_ms"], (int, float)) for p in row["result"]["per_prompt"])
            and row["result"]["worst_itl_ms"] <= 1000
        ]
        summary["picked_budget"] = max(eligible) if eligible else None
        summary["passed"] = bool(eligible)
        if not eligible:
            raise RuntimeError("No tested budget met the mixed-traffic ITL bound")
    finally:
        manifest.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
