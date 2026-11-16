"""Summarize the completed list's native accounting by adjudicated primary cause."""
import collections
import datetime
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
C = D / "current/swe-extended85-head-bfp8/collected-final"
accounting = json.loads((C / "calls.json").read_text())
score = json.loads((C / "final-score-with-adjudication.json").read_text())
grades = {r["task"]: r for r in score["rows"]}
assert len(grades) == len(accounting["tasks"]) == 85
groups = collections.defaultdict(lambda: collections.defaultdict(float))
rows = []
for task in accounting["tasks"]:
    grade = grades[task["task"]]
    cause = "pass" if grade["counted_reward"] == 1 else grade["cause"]
    assert cause is not None
    group = groups[cause]
    group["tasks"] += 1
    group["confirmed_passes"] += grade["counted_reward"] == 1
    group["unknown_outcomes"] += grade["counted_reward"] is None
    group["native_passes_before_adjudication"] += grade["native_reward"] == 1
    group["calls"] += len(task["calls"])
    group["task_seconds"] += task["end"] - task["start"]
    group["outside_model_seconds"] += task["outside_s"]
    for call in task["calls"]:
        group["prompt_tokens"] += call["prompt_tokens"]
        group["cached_tokens"] += call["cached_tokens"]
        group["output_tokens"] += call["output"]
        group["model_call_wall_seconds"] += call["end"] - call["start"]
    rows.append(
        {"task": task["task"], "cause": cause, "counted_reward": grade["counted_reward"], "calls": len(task["calls"])}
    )
assert sum(g["calls"] for g in groups.values()) == 4427
out = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "All85 native first-attempt trajectories. Summed agent/call times overlap; these are not list elapsed times. Infrastructure recovery costs are separate evidence, not re-added here.",
    "sources": ["collected-final/calls.json", "collected-final/final-score-with-adjudication.json"],
    "groups": dict(groups),
    "tasks": rows,
    "measured_list_hours": score["measured_hours"],
    "explained_list_hours": score["explained_hours"],
}
(C / "cost-by-cause.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"tasks": len(rows), "groups": dict(groups)}))
