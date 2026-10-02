# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Summarize serial SWE suite phases, rewards, and payload-free request counters."""

import argparse
import json
from pathlib import Path

from analyze_eval_timing import analyze, epoch

ORIGINAL_TASKS = frozenset(
    {
        "astropy__astropy-14096",
        "django__django-11299",
        "matplotlib__matplotlib-25332",
        "scikit-learn__scikit-learn-14629",
        "sympy__sympy-13551",
    }
)


def duration(phase):
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return None
    return epoch(phase["finished_at"]) - epoch(phase["started_at"])


def counter_delta(events, responses):
    snapshots = [event for event in events if event.get("event") == "server_metrics" and event.get("counters")]
    if len(snapshots) < 2:
        return {"valid": False, "reason": "missing_boundary_snapshots"}
    first, last = snapshots[0], snapshots[-1]
    delta = {key: value - first["counters"][key] for key, value in last["counters"].items() if key in first["counters"]}
    expected_tokens = sum((event.get("usage") or {}).get("completion_tokens", 0) for event in responses)
    expected_prompt = sum((event.get("usage") or {}).get("prompt_tokens", 0) for event in responses)
    valid = (
        first.get("phase") == "before_request"
        and last.get("phase") == "after_response"
        and all(value >= 0 for value in delta.values())
        and delta.get("time_to_first_token_seconds_count") == len(responses)
        and delta.get("request_success_total") == len(responses)
        and delta.get("generation_tokens_total") == expected_tokens
        and delta.get("prompt_tokens_total") == expected_prompt
        and "e2e_request_latency_seconds_sum" in delta
        and "time_to_first_token_seconds_sum" in delta
    )
    return {
        "valid": valid,
        "reason": "exclusive_C1_counts_and_tokens_match" if valid else "lag_reset_overlap_or_missing_counter",
        "raw_delta": delta,
        "ttft_s": delta["time_to_first_token_seconds_sum"] if valid else None,
        "post_first_token_s": (
            delta["e2e_request_latency_seconds_sum"] - delta["time_to_first_token_seconds_sum"] if valid else None
        ),
    }


def completed_response_counters(events, responses):
    """Attribute saved responses only, excluding a trailing uncompleted request."""
    end = next(
        (
            i + 1
            for i in range(len(events) - 1, -1, -1)
            if events[i].get("event") == "server_metrics" and events[i].get("phase") == "after_response"
        ),
        0,
    )
    result = counter_delta(events[:end], responses)
    result["scope"] = "saved completed API responses, including any late response; not clipped agent time"
    return result


def summarize(root):
    events = []
    for path in root.glob("**/*requests.jsonl"):
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows = []
    for path in root.glob("**/agent/mini-swe-agent.trajectory.json"):
        result = json.loads((path.parent.parent / "result.json").read_text())
        row = analyze(path, [])
        for key in (
            "requests",
            "top_repeats",
            "server_generated_tokens_10s_approx",
            "server_prompt_tokens_10s_approx",
            "server_decode_active_s_10s_approx",
        ):
            row.pop(key, None)
        trial = result["trial_name"]
        trial_events = sorted(
            (event for event in events if (event.get("session") or "").startswith(trial + "__")),
            key=lambda event: event["unix_s"],
        )
        responses = [event for event in trial_events if event["event"] == "response"]
        row.update(
            trial=trial,
            started_at=result["started_at"],
            finished_at=result["finished_at"],
            trial_s=duration(result),
            phase_s={
                key: duration(result.get(key))
                for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
            },
            proxy_response_count=len(responses),
            proxy_request_s=sum(event["elapsed_s"] for event in responses),
            metrics_collection_s=sum(event.get("collection_s", 0) for event in trial_events),
            submission_normalizations=sum(event["event"] == "submission_marker_normalized" for event in trial_events),
            server=counter_delta(trial_events, responses),
            completed_response_server=completed_response_counters(trial_events, responses),
            repeated_tool_advisories=sum(event["event"] == "repeated_tool_feedback" for event in trial_events),
            reasoning_history_interventions=sum(
                event["event"] == "reasoning_history_limited" for event in trial_events
            ),
        )
        verifier = path.parent.parent / "verifier/report.json"
        if verifier.exists():
            report = json.loads(verifier.read_text()).get(row["task"], {})
            row["resolved"] = report.get("resolved")
            row["tests"] = {
                key: {state: len(names) for state, names in values.items()}
                for key, values in report.get("tests_status", {}).items()
            }
        rows.append(row)
    rows.sort(key=lambda row: row["started_at"])
    tasks = [row["task"] for row in rows]
    return {
        "exact_original_five": len(tasks) == 5 and set(tasks) == ORIGINAL_TASKS,
        "observed_order": tasks,
        "trials_span_s": epoch(rows[-1]["finished_at"]) - epoch(rows[0]["started_at"]) if rows else None,
        "sum_agent_s": sum(row["wall_s"] for row in rows),
        "sum_trial_s": sum(row["trial_s"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--dispatch-at")
    parser.add_argument("--finish-at")
    args = parser.parse_args()
    summary = summarize(args.artifact_root)
    if args.dispatch_at and args.finish_at:
        summary["dispatch_to_finish_s"] = epoch(args.finish_at) - epoch(args.dispatch_at)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
