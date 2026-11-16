import collections
import json
import pathlib
import re
import time

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
L = D / "current/swe-extended85-head-bfp8/live-tasks"
J = D / "setup/swe-jobs/swe-extended85-head-bfp8"
rows = []
cs = []
for f in sorted(L.glob("*/coordinator.json")):
    c = json.loads(f.read_text())
    cs.append(c)
    rows.append({**{k: c.get(k) for k in ["task", "counted_reward", "native_reward", "cause"]}, "evidence": str(f)})
s = {
    "status": "running",
    "at_unix": time.time(),
    "planned_tasks": 85,
    "reviewed_count": len(rows),
    "reviewed_tasks": rows,
    "not_final_score": True,
}
(L.parent / "live_review_summary.json").write_text(json.dumps(s, indent=2) + "\n")
p = D / "results.json"
j = json.loads(p.read_text())
r = next(x for x in j["runs"] if x["id"] == "swe-extended85-head-bfp8")
r["live_collection"].update(
    at_unix=time.time(),
    reviewed_count=len(rows),
    completed_tasks=len(list(J.glob("*/result.json"))),
    collected_tasks=sorted(x.parent.name.removesuffix("-first") for x in L.glob("*/collected.json")),
    coordinator_completed=[x["task"] for x in rows],
    pending_review=sorted(
        x.parent.name.removesuffix("-first")
        for x in L.glob("*/collected.json")
        if not (x.parent / "coordinator.json").exists()
    ),
)
p.write_text(json.dumps(j, indent=2) + "\n")
p = D / "learned.json"
j = json.loads(p.read_text())
dest = j["task_times"]["swe_bench_verified"]
seen = {(x.get("run"), x["task"]) for x in dest}
for row in rows:
    if ("swe-extended85-head-bfp8", row["task"]) in seen:
        continue
    cp = L / (row["task"] + "-first") / "calls.json"
    a = json.loads(cp.read_text())["tasks"][0]
    calls = a["calls"]
    dest.append(
        {
            "task": row["task"],
            "run": "swe-extended85-head-bfp8",
            "precision": "stage12_head_bfp8_lofi",
            "calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "cached_tokens": sum(c["cached_tokens"] for c in calls),
            "uncached_input_tokens": sum(c["uncached_input"] for c in calls),
            "output_tokens": sum(c["output"] for c in calls),
            "max_final_context_tokens": max(c["context"] for c in calls),
            "hours": (a["end"] - a["start"]) / 3600,
            "outside_model_seconds": a["outside_s"],
            "solved": None if row["counted_reward"] is None else bool(row["counted_reward"]),
            "source_calls": str(cp),
            "score_evidence": row["evidence"],
            "note": "Completed coordinator-audited task in live85 list; not overall list score/timing.",
        }
    )
p.write_text(json.dumps(j, indent=2) + "\n")
counts = collections.Counter("pass" if x["counted_reward"] == 1 else x["cause"] for x in rows)
anomalies = [c["task"] for c in cs if c.get("anomaly")]
p = D / "REPORT.md"
txt = p.read_text()
paragraph = f"SWE85 live analysis ({len(rows)} reviewed tasks; not a final list score): {counts['pass']} accepted passes, {counts['wrong']} incorrect patches, {counts['fetched_solution']} fetched-solution failures, {counts['anomaly']} primary anomaly result, and {counts['infrastructure']} infrastructure results pending eligible reruns (see `rerun_queue.json`). Anomaly investigations pending: {', '.join(anomalies)}; these can overlap fetched-solution failures. Django16950 has native reward null after the agent deleted its source tree; coordinator adjudicates0 separately. Reviewed response mappings are complete; cap and mechanical-flag counts remain in each task digest, with semantic anomaly adjudication and controls recorded separately. Matched served/fullTT controls are prepared or being prepared while the evaluation owns the device. See `current/swe-extended85-head-bfp8/live_review_summary.json`, `rerun_queue.json`, `anomalies/swe85-precision/pending.json`, and per-task records."
txt = re.sub(r"SWE85 live analysis \(.*?(?=\n\n|\Z)", lambda m: paragraph, txt, flags=re.S)
p.write_text(txt)
print(
    json.dumps(
        {
            "reviewed": len(rows),
            "counts": dict(counts),
            "anomaly_tasks": anomalies,
            "completed": len(list(J.glob("*/result.json"))),
        }
    )
)
