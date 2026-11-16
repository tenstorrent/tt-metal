# SPDX-License-Identifier: Apache-2.0
"""Prospective replay of the retained85 histories on three measured-memory slots."""
import argparse
import copy
import hashlib
import heapq
import importlib.util
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    source = base / "current/swe-extended85-head-bfp8/collected-final/calls.json"
    oracle = Path(
        "/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/scripts/explain_hours.py"
    )
    spec = importlib.util.spec_from_file_location("installed_explain_hours", oracle)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original = json.loads(source.read_text())
    rates = json.loads(args.rates.read_text())
    assert len(original["tasks"]) == 85 and rates["status"] == "completed"
    tasks = sorted(original["tasks"], key=lambda t: t["start"])
    scenarios = {}
    for mode in ["observed_lifecycle_three_slots", "single_request_rates_optimistic"]:
        population = copy.deepcopy(original)
        population["tasks"] = copy.deepcopy(tasks)
        population["agents"] = 3
        population["server_batch"] = 8
        slots = [0.0] * 3
        cursor = 0.0
        for task in population["tasks"]:
            start = heapq.heappop(slots)
            duration = task["end"] - task["start"]
            delta = start - task["start"]
            task["start"], task["end"] = start, start + duration
            heapq.heappush(slots, start + duration)
            for call in task["calls"]:
                if mode == "observed_lifecycle_three_slots":
                    call["start"] += delta
                    call["end"] += delta
                else:
                    # Synthetic non-overlap requests deliberately price every call
                    # at measured C1 rates; these are not observed timestamps.
                    call["start"], call["end"] = cursor, cursor + 1
                    cursor += 2
        events = sorted(
            (point, direction)
            for t in population["tasks"]
            for c in t["calls"]
            for point, direction in [(c["start"], 1), (c["end"], -1)]
        )
        active = peak = 0
        for _, direction in events:
            active += direction
            peak = max(peak, active)
        assert peak <= (3 if mode == "observed_lifecycle_three_slots" else 1), peak
        result = mod.explain(rates, population, agents=3, server_batch=8)
        result["modelled_peak_request_concurrency"] = peak
        result["global_outside_trials_s"] = original["global_outside_trials_s"]
        result["projected_total_hours"] = result["explained_hours"] + original["global_outside_trials_s"] / 3600
        scenarios[mode] = result
    now = time.time()
    remaining = (1794806125 - now) / 3600
    forecast = scenarios["observed_lifecycle_three_slots"]["projected_total_hours"]
    output = {
        "status": "prospective_projection_not_scored_run",
        "recorded_unix": now,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rates": str(args.rates),
        "rates_sha256": hashlib.sha256(args.rates.read_bytes()).hexdigest(),
        "oracle": str(oracle),
        "oracle_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
        "tasks": 85,
        "agents": 3,
        "server_batch": 8,
        "output_tokens": sum(c["output"] for t in tasks for c in t["calls"]),
        "output_only_hours_at_best_measured_rate": sum(c["output"] for t in tasks for c in t["calls"])
        / (3 * max(r["decode_tok_s_user"] for r in rates["rows"]))
        / 3600,
        "output_only_basis": "Optimistic arithmetic under fastest measured grid rate, ignoring every input token and all outside time; not an observed run or guarantee outside measured grid.",
        "scenarios": scenarios,
        "remaining_stage_hours": remaining,
        "run_timeout_hours": 12,
        "planning_margin": 1.25,
        "admit": forecast * 1.25 <= min(remaining, 12),
        "limitations": [
            "Retains all85 task histories, including anomalies and fetched solutions; a new precision could change trajectories and scores.",
            "Observed-lifecycle scenario places original task durations and call offsets on3 slots before repricing. It is not iterated to concurrency self-consistency.",
            "Optimistic scenario prices every call at measured C1 rates while scheduling task costs on3 slots; synthetic request timestamps are not observed execution evidence.",
            "Input-length/concurrency extrapolation is reported by the installed oracle. No new eval was launched by this script.",
            "Global outside time is added once. New setup/verifier recovery or longer successful trajectories could add cost.",
        ],
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ["tasks", "agents", "remaining_stage_hours", "admit"]}))
    print({k: v["projected_total_hours"] for k, v in scenarios.items()})


if __name__ == "__main__":
    main()
