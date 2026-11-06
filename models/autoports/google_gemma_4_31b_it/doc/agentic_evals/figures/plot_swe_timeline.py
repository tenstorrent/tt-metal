"""Render existing valid-smoke timing evidence without reinterpreting query time as device time."""
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root = Path(__file__).resolve().parent.parent
calls = json.loads((root / "runs/swe-valid-runtime-analysis/calls.json").read_text())
summary = json.loads((root / "runs/swe-valid-runtime-analysis/result.json").read_text())
audit = json.loads((root / "runs/smoke2-swe-activation-repair/audit.json").read_text())
starts = {}
for task in audit["tasks"]:
    trajectory = root / "runs/smoke2-swe-activation-repair" / task["source"]
    result = json.loads((trajectory.parent.parent / "result.json").read_text())
    starts[task["task"]] = datetime.fromisoformat(
        result["agent_execution"]["started_at"].replace("Z", "+00:00")
    ).timestamp()
origin = min(starts.values())
fig, (ax, sizes) = plt.subplots(2, 1, figsize=(11, 5.7), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
colors = ["#3478b8", "#a34893"]
for lane, task in enumerate(summary["tasks"]):
    name = task["task"]
    t = starts[name] - origin
    times, prompts = [], []
    for call in [x for x in calls if x["task"] == name]:
        query = call["query_phase_window_seconds"]
        tool = call["tool_phase_window_seconds"]
        ax.broken_barh([(t, query)], (lane - 0.22, 0.44), facecolors=colors[lane])
        times.append(t + query / 2)
        prompts.append(call["prompt_tokens"])
        t += query
        ax.broken_barh([(t, tool)], (lane - 0.22, 0.44), facecolors="#e6a23c")
        t += tool
    sizes.plot(times, prompts, ".-", color=colors[lane], label=name, linewidth=1, markersize=3)
    assert abs(t - (starts[name] - origin) + task["post_submission_tail_seconds"] - task["agent_seconds"]) < 1e-5
ax.set_yticks(range(2), [x["task"] for x in summary["tasks"]])
ax.set_title("Valid SWE smoke: overlapping agents and growing prompts")
ax.legend(
    handles=[
        Patch(facecolor="#3478b8", label="Query window: API plus client overhead"),
        Patch(facecolor="#e6a23c", label="Tool window: execution plus formatting"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    fontsize=9,
    ncol=2,
)
sizes.set_ylabel("Input tokens per call")
sizes.set_xlabel("Seconds since the first agent started")
sizes.legend(fontsize=8, loc="upper left")
for item in (ax, sizes):
    item.grid(axis="x", alpha=0.18)
    item.set_xlim(0, summary["agent_union_seconds"])
fig.text(
    0.02,
    0.015,
    "91 calls; suite wall 374.988 s, agent union 342.938 s. Query windows are not device or pure generation time.\nSource: runs/swe-valid-runtime-analysis/; both valid tasks scored 0. No phase-cost or speedup inference.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=2)
fig.savefig(root / "figures/swe_valid_timeline.svg")
