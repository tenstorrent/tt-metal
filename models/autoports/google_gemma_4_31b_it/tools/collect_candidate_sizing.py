#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect candidate-specific launch sizing without touching a live device."""
import argparse
import json
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-count", type=positive_int, default=10)
    parser.add_argument("--host-slots", type=positive_int, default=10)
    args = parser.parse_args()
    identity = json.loads(args.identity.read_text())
    if identity["precision"]["weight_groups"]["lm_head"] not in ("bfp8", "bfp8_lofi"):
        raise ValueError("Expected candidate BFP8 language-model head")
    if (
        identity["precision"]["kv_cache_dtype"] != "bfloat8_b"
        or identity["max_model_len"] != 262144
        or identity["max_num_seqs"] != 32
    ):
        raise ValueError("Unexpected candidate cache/context/batch contract")
    log = Path(identity["server_log"])
    events = []
    for line in log.read_text().splitlines():
        if '"event": "serving_pool_launch_validation"' in line:
            events.append(json.loads(line[line.index("{") :]))
    if len(events) != 1 or not events[0]["fits"]:
        raise ValueError("Need exactly one successful actual launch pool validation")
    measurement = events[0]
    devices = measurement["per_device"]
    if sorted(d["device_id"] for d in devices) != sorted(identity["device_ids"]):
        raise ValueError("Memory and identity device sets differ")
    pools = {d["pool_bytes"] for d in devices}
    if len(pools) != 1 or next(iter(pools)) % 696320:
        raise ValueError("Unexpected normalized hybrid pool geometry")
    pool_ids = next(iter(pools)) // 696320
    capacity = (pool_ids - 1) // 2854
    agents = min(args.task_count, args.host_slots, capacity)
    if agents < 1:
        raise ValueError("Measured pool cannot hold one planning working set")
    result = {
        "precision": identity["precision"],
        "source_identity": str(args.identity),
        "server_pid": identity["pid"],
        "launch_evidence": str(log),
        "measurement": measurement,
        "free_gb": [d["free_bytes"] / 1e9 for d in devices],
        "pool_gb": [d["pool_bytes"] / 1e9 for d in devices],
        "working_set_tokens": 131072,
        "pool_ids": pool_ids,
        "ids_per_agent": 2854,
        "capacity_agents": (pool_ids - 1) // 2854,
        "pool_tokens": ((pool_ids - 1) // 2854) * 131072,
        "agents": agents,
        "selection_bounds": {
            "task_count": args.task_count,
            "host_slots": args.host_slots,
            "pool_capacity_agents": capacity,
        },
        "batch": 32,
        "max_num_seqs": 32,
        "served_max_model_len": 262144,
        "full_context_capacity_agents": (pool_ids - 1) // 4902,
        "reason": measurement["pool_sizing_reason"],
        "suites": ["r1_gpqa_diamond", "swe_bench_verified"],
        "feasible_ten_agents": (pool_ids - 1) // 2854 >= 10,
        "working_set_basis": "GPQA main131072 request envelope; SWE conservative131072 planning estimate from current SWE phase2 measured48361 finalcontext plus32768 output allowance rounded up, not a conversation cap. Full262144 remains served; simultaneous full-context capacity recorded separately.",
        "selection": f"Selected {agents}=min(tasks {args.task_count}, host slots {args.host_slots}, measured capacity {capacity}). This collector validates only the supplied B32 candidate launch; it does not select a smaller server batch. For12 agents B32 is the smallest validated1/8/32 batch. Host bounds require separate resource evidence.",
        "working_set_evidence": "doc/agentic_evals/current/swe-phase2-ready/sampling_counts_completed.json; main SWE output cap32768; sum81129 rounded conservatively to131072.",
        "validity": "Actual candidate launch only. Restart or precision/pool change requires new measurement; baseline sizing artifacts remain untouched.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for suite in result["suites"]:
        name = "gpqa" if suite == "r1_gpqa_diamond" else "swe"
        (args.output / f"{name}_server_size_head_bfp8.json").write_text(
            json.dumps({**result, "suite": suite}, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                "pool_ids": pool_ids,
                "capacity_agents": result["capacity_agents"],
                "feasible_ten_agents": result["feasible_ten_agents"],
                "remaining_bytes_after_reserves": measurement["remaining_bytes_after_reserves"],
            }
        )
    )


if __name__ == "__main__":
    main()
