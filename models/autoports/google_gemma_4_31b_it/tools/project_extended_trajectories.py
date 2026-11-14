#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prospective synthetic traffic only; never a scored-run timing explanation."""
import bisect
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path

BASE = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
SOURCE = BASE / "current/swe-phase2-ready/collected/calls.json"
RATES = BASE / "current/head-bfp8-rates/measurement/serving_rates-swe.json"
ORACLE = Path(
    "/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/scripts/explain_hours.py"
)
spec = importlib.util.spec_from_file_location("installed_explain_hours", ORACLE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def population(original, count):
    slots = [(0.0, slot) for slot in range(12)]
    heapq.heapify(slots)
    tasks = []
    # Prefixes are nested. Each consecutive ten includes every phase archetype,
    # including the long Astropy run; no outcome-dependent sampling.
    for index in range(count):
        source = original["tasks"][index % 10]
        start, slot = heapq.heappop(slots)
        duration = source["end"] - source["start"]
        task = {
            "task": f'synthetic-{index:04d}:{source["task"]}',
            "archetype": source["task"],
            "start": start,
            "end": start + duration,
            "outside_s": source["outside_s"],
            "calls": [],
        }
        for call in source["calls"]:
            copied = {k: call[k] for k in ("uncached_input", "output", "context")}
            copied.update(start=start + call["start"] - source["start"], end=start + call["end"] - source["start"])
            task["calls"].append(copied)
        tasks.append(task)
        heapq.heappush(slots, (start + duration, slot))
    return {"agents": 12, "server_batch": 32, "tasks": tasks}


def fast_explain(rates, calls):
    table = module.Rates(rates)
    starts = sorted(c["start"] for t in calls["tasks"] for c in t["calls"])
    ends = sorted(c["end"] for t in calls["tasks"] for c in t["calls"])
    reasons = set()
    per_task = []
    active_hist = {}
    for task in calls["tasks"]:
        server = 0.0
        for c in task["calls"]:
            mid = (c["start"] + c["end"]) / 2
            # Identical inclusive endpoints to oracle s<=mid and e>=mid.
            active = bisect.bisect_right(starts, mid) - bisect.bisect_left(ends, mid)
            active = active or 1
            active_hist[active] = active_hist.get(active, 0) + 1
            prompt = max(1, c["context"] - c["output"])
            server += c["uncached_input"] * table.on_server(32, active, prompt, 1, reasons)
            server += c["output"] * table.on_server(32, active, prompt + c["output"] / 2, 2, reasons)
        per_task.append(
            {
                "task": task["task"],
                "start": task["start"],
                "server_s": round(server, 1),
                "outside_s": round(task["outside_s"], 1),
            }
        )
    slots = [0.0] * 12
    for task in sorted(per_task, key=lambda t: t["start"] or 0):
        begin = heapq.heappop(slots)
        heapq.heappush(slots, begin + task["server_s"] + task["outside_s"])
    return {
        "explained_hours": round(max(slots) / 3600, 4),
        "approximate": bool(reasons),
        "reasons": sorted(reasons),
        "active_histogram": active_hist,
    }


def select_with_margin(rows, run_timeout_hours, margin=1.25):
    """Apply the planning margin once against the actual run deadline."""
    if run_timeout_hours <= 0 or margin < 1:
        raise ValueError("Need a positive timeout and margin >= 1")
    return max(
        (row for row in rows if row["hours"] * margin <= run_timeout_hours),
        key=lambda row: row["count"],
        default=None,
    )


def main():
    original = json.loads(SOURCE.read_text())
    rates = json.loads(RATES.read_text())
    overhead = original["global_outside_trials_s"]
    validations = []
    for count in (10, 12, 20):
        synthetic = population(original, count)
        oracle = module.explain(rates, synthetic)
        fast = fast_explain(rates, synthetic)
        assert all(oracle[key] == fast[key] for key in ("explained_hours", "approximate", "reasons"))
        validations.append(
            {
                "count": count,
                "oracle_hours": oracle["explained_hours"],
                "fast_hours": fast["explained_hours"],
                "exact_agreement": True,
            }
        )
    rows = []
    for count in range(10, 501):
        result = fast_explain(rates, population(original, count))
        projected = result["explained_hours"] + overhead / 3600
        rows.append({"count": count, "hours": projected, "hours_with_1_25_margin": projected * 1.25})
    plain = max((r for r in rows if r["hours"] <= 10), key=lambda r: r["count"])
    timeout = json.loads(Path("bringup/agentic_evals/inputs.json").read_text())["run_timeout_hours"]
    margin = select_with_margin(rows, timeout, 1.25)
    out = {
        "status": "prospective_synthetic_projection_not_admission_not_scored_run",
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "rates": str(RATES),
        "oracle": str(ORACLE),
        "oracle_sha256": hashlib.sha256(ORACLE.read_bytes()).hexdigest(),
        "algorithm": "Deterministic balanced cyclic resampling in original10-task list order; all ten archetypes preserved including Astropy1.403h. Place task lifecycles on12 earliest-free slots using actual measured task durations; shift every exact call interval by its task start and preserve tool/outside gaps. Price synthetic call spans by unchanged installed rate interpolation and midpoint-active semantics; replay priced tasks on12 slots as oracle does. Efficient active count uses bisect starts<=mid minus ends<mid, with exact small-case unchanged-oracle validation. Source files unchanged.",
        "smallcase_oracle_validation": validations,
        "agents": 12,
        "batch": 32,
        "working_set_tokens": 131072,
        "pool_ids": 36896,
        "context": 262144,
        "global_overhead_once_seconds": overhead,
        "full500": rows[-1],
        "largest_count_under10h": plain,
        "largest_count_with_1_25_margin_within_run_timeout": margin,
        "run_timeout_hours": timeout,
        "planning_margin": 1.25,
        "all_counts": rows,
        "limitations": [
            "Synthetic active concurrency uses observed-duration schedule and is not iterated to self-consistency after rate pricing.",
            "All10 archetypes repeated evenly; unseen repositories, task difficulty, anomaly frequency and larger tails not sampled or guaranteed. Astropy anomaly tail is retained unchanged.",
            "1.25 margin multiplies predicted total once; it is a sensitivity allowance, not empirical confidence interval.",
            "Image acquisition, dependency setup, additional verifier recovery and growing disk costs are not priced. See setup/extended_official_image_availability.json and completed-phase projection host constraints.",
            "Only10 official images cached; current~450GiB free minus120GiB sandbox provision leaves~330GiB for additional image layers, but shared-layer sizes unknown.",
            "No real task IDs selected; nested synthetic prefixes are not a proposed final list. Phase2 anomaly controls must finish and root must approve admission.",
        ],
        "previous_comparisons": "setup/extended_projection_completed_phase.json preserves observed-duration, scaled-duration and constant12-active bounds.",
    }
    out[
        "precision_condition"
    ] = "Baseline only, conditional on unchanged stage12_head_bfp8_lofi; a precision fix requires remeasured rows and regenerated projection before admission."
    (BASE / "setup/extended_projection_trajectory_replay_corrected_margin.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "smallcase_oracle_validation",
                    "full500",
                    "largest_count_under10h",
                    "largest_count_with_1_25_margin_within_run_timeout",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
