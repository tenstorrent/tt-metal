"""Summarize completed SWE85 traffic and isolate sampling steps without new inference."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
C = D / "current/swe-extended85-head-bfp8/collected-final"
accounting = json.loads((C / "calls.json").read_text())
calls = [c for t in accounting["tasks"] for c in t["calls"]]
ids = {c["id"] for c in calls}
assert len(ids) == len(calls) == 4427
last_id = max(calls, key=lambda c: c["end"])["id"]
log = D / "current/post-gpqa-controls-head-bfp8/server/server-b32/server.log"
probe = "chatcmpl-95262fbf9ec06f39"
active = False
seen = set()
counts = collections.Counter()
first_step = last_step = boundary = end_line = None
for lineno, line in enumerate(log.open(), 1):
    if "eval_call=" in line:
        event, _ = json.JSONDecoder().raw_decode(line.split("eval_call=", 1)[1])
        if event["id"] == probe:
            assert not active
            active = True
            boundary = lineno
            continue
        if active:
            assert event["id"] in ids, ("Unexpected traffic", event["id"], lineno)
            seen.add(event["id"])
            if event["id"] == last_id:
                end_line = lineno
                break
    if active and '"event": "sampling_mode"' in line:
        event, _ = json.JSONDecoder().raw_decode(line[line.index("{") :])
        counts[event["phase"] + ":" + event["mode"]] += 1
        if first_step is None:
            first_step = lineno
        last_step = lineno
assert seen == ids
sampling = {
    "run": "swe-extended85-head-bfp8",
    "counts": dict(counts),
    "steps": sum(counts.values()),
    "source": str(log),
    "source_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "preceding_probe_completion_line": boundary,
    "first_step_line": first_step,
    "last_step_line": last_step,
    "last_task_completion_line": end_line,
    "mapped_response_count": len(seen),
    "boundary_method": "After confirmed logging-probe completion through final mapped response; all intervening eval_call IDs exactly equal list accounting. Untimestamped step lines attributed to this isolated traffic window.",
    "interpretation": "Host prefill is accepted stock intermediate-chunk sampling. Counts do not prove every such batch mixes final and intermediate rows. No scheduler restriction; decode remains device sampling.",
}
(C / "sampling_counts.json").write_text(json.dumps(sampling, indent=2) + "\n")
events = sorted([(c["start"], 1) for c in calls] + [(c["end"], -1) for c in calls])
current = peak = 0
for _, delta in events:
    current += delta
    peak = max(peak, current)
assert current == 0
workload = {
    "run": "swe-extended85-head-bfp8",
    "precision": "stage12_head_bfp8_lofi",
    "target_concurrency": 12,
    "observed_peak_inflight_calls": peak,
    "working_set_tokens": max(c["context"] for c in calls),
    "typical_context_tokens": statistics.median(c["prompt_tokens"] for c in calls),
    "prompt_tokens_min": min(c["prompt_tokens"] for c in calls),
    "prompt_tokens_max": max(c["prompt_tokens"] for c in calls),
    "prompt_tokens_total": sum(c["prompt_tokens"] for c in calls),
    "output_tokens_total": sum(c["output"] for c in calls),
    "cached_tokens_total": sum(c["cached_tokens"] for c in calls),
    "final_context_tokens_median": statistics.median(c["context"] for c in calls),
    "native_calls": len(calls),
    "completed_tasks": 85,
    "served_max_context_tokens": 262144,
    "launch_sizing_working_set_tokens": 131072,
    "sampling": sampling["counts"],
    "source": "All completed native SWE85 calls, including discarded capped replies. Observed context lengths are traffic measurements, not imposed limits.",
    "evidence": [str(C / "calls.json"), str(C / "sampling_counts.json")],
}
(C / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
p = D.parent / "agentic_serving.json"
state = json.loads(p.read_text())
previous = state["workload"]
history = previous.get("prior_measurements", {})
history.setdefault("swe-phase2-ready", previous["measured_by_eval"]["swe_bench_verified"])
by_eval = previous["measured_by_eval"]
by_eval["swe_bench_verified"] = workload.copy()
state["workload"] = dict(workload, prior_measurements=history, measured_by_eval=by_eval)
p.write_text(json.dumps(state, indent=2) + "\n")
print(json.dumps({"sampling": sampling["counts"], "peak": peak, "workload": workload}))
