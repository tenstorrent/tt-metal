#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Join official GPQA samples to native response IDs without retaining reply text."""
import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

PARSER = Path(
    "/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/scripts/reply_digest.py"
)
spec = importlib.util.spec_from_file_location("installed_reply_digest", PARSER)
digest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(digest)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def union_seconds(spans):
    total = 0
    end = float("-inf")
    for start, stop in sorted(spans):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def collect(lines, samples, task_list, agents, server_batch, run_start=None, run_end=None):
    bodies = {}
    for body in digest.responses(iter(lines)):
        if body["id"] in bodies and bodies[body["id"]] != body:
            raise ValueError("Conflicting response bodies for one ID")
        bodies[body["id"]] = body
    boundaries = {}
    for line in digest.framed(iter(lines)):
        if "eval_call=" in line:
            row, _ = json.JSONDecoder().raw_decode(line.split("eval_call=", 1)[1])
            if run_start is not None and row["start"] < run_start:
                continue
            if run_end is not None and row["start"] >= run_end:
                continue
            if run_end is not None and row["end"] > run_end:
                raise ValueError("Call extends beyond run end; use the completed run boundary")
            if row["id"] in boundaries and boundaries[row["id"]] != row:
                raise ValueError("Conflicting timing records for one ID")
            boundaries[row["id"]] = row
    expected = {str(t["doc_id"]): t for t in task_list["tasks"]}
    by_task = {key: [] for key in expected}
    questions = {}
    for sample in samples:
        key = str(sample["doc_id"])
        if key not in expected:
            raise ValueError("Sample outside fixed task list")
        question = sample["doc"]["Question"]
        if sha(question) != expected[key]["question_sha256"]:
            raise ValueError("Question hash contradicts fixed task ID")
        questions[key] = question
        metric = sample.get("exact_match")
        if metric not in (0, 1, 0.0, 1.0):
            raise ValueError("Missing binary exact_match metric")
        by_task[key].append(
            {
                "solved": bool(metric),
                "filter": sample.get("filter"),
                "prompt_hash": sample.get("prompt_hash"),
                "doc_hash": sample.get("doc_hash"),
            }
        )
    matched = {key: [] for key in expected}
    issues = []
    for response_id, boundary in boundaries.items():
        request = json.loads(boundary["request"])
        prompt_strings = list(strings(request.get("messages", request.get("prompt", ""))))
        hits = [key for key, question in questions.items() if any(question in s for s in prompt_strings)]
        if not hits:
            continue  # readiness probes or traffic outside this fixed list
        if len(hits) != 1:
            raise ValueError("Request ambiguously contains multiple task questions")
        key = hits[0]
        if response_id not in bodies:
            issues.append({"task": key, "id": response_id, "reason": "missing response body"})
            continue
        body = bodies[response_id]
        usage = body.get("usage") or {}
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if (
            not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (prompt, output, cached))
            or cached > prompt
        ):
            issues.append({"task": key, "id": response_id, "reason": "missing or invalid token/cache accounting"})
            continue
        start, end = boundary["start"], boundary["end"]
        if not isinstance(start, (float, int)) or not isinstance(end, (float, int)) or end < start:
            raise ValueError("Invalid native call interval")
        matched[key].append(
            {
                "id": response_id,
                "start": start,
                "end": end,
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "uncached_input": prompt - cached,
                "output": output,
                "context": prompt + output,
                "finish_reason": body["choices"][0].get("finish_reason"),
                "request_sha256": sha(boundary["request"]),
            }
        )
    tasks, rows = [], []
    solved = 0
    for key in expected:
        calls = sorted(matched[key], key=lambda x: x["start"])
        attempts = by_task[key]
        if not calls or not attempts:
            issues.append({"task": key, "reason": "missing native calls or scored samples"})
        if len(attempts) > 1:
            issues.append(
                {"task": key, "reason": "multiple sample rows: file order retained; separate run attribution needed"}
            )
        first = attempts[0]["solved"] if attempts else None
        solved += first is True
        row = {
            "task": key,
            "doc_id": int(key),
            "record_id": expected[key].get("record_id"),
            "solved": first,
            "sample_attempts": attempts,
            "calls": calls,
        }
        if calls:
            start, end = min(c["start"] for c in calls), max(c["end"] for c in calls)
            row.update(
                start=start,
                end=end,
                outside_s=max(0, end - start - union_seconds([(c["start"], c["end"]) for c in calls])),
            )
        rows.append(row)
        tasks.append({k: v for k, v in row.items() if k in ("task", "calls", "start", "end", "outside_s")})
    n = len(expected)
    z = 1.959963984540054
    p = solved / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    starts = [c["start"] for row in rows for c in row["calls"]]
    ends = [c["end"] for row in rows for c in row["calls"]]
    accounting = {"agents": agents, "server_batch": server_batch, "tasks": tasks, "accounting_complete": not issues}
    result = {
        "suite": "r1_gpqa_diamond",
        "score": {
            "solved": solved,
            "n": n,
            "interval_95": [max(0, center - radius), min(1, center + radius)],
            "units": "fraction",
            "first_sample_attempt_only": True,
        },
        "tasks": rows,
        "accounting_complete": not issues,
        "issues": issues,
        "measured_server_span_hours": (max(ends) - min(starts)) / 3600 if starts else None,
        "timing_basis": "Native receipt-to-response boundaries; outside_s measures gaps between native calls only. Harness setup, grading and client time outside these boundaries are unknown; obtain overall run wall from eval_runs runs.json.",
        "raw_response_text_retained": False,
    }
    if run_start is not None and run_end is not None:
        if run_end < run_start:
            raise ValueError("Run end precedes start")
        result["measured_hours"] = (run_end - run_start) / 3600
        if starts:
            overhead = (
                run_end - run_start - union_seconds([(task["start"], task["end"]) for task in tasks if task["calls"]])
            )
            accounting["global_overhead_s"] = overhead
            result["global_overhead_s"] = overhead
            result["timing_basis"] = (
                "Native call intervals and intercall gaps; run wall outside the union of task call lifecycles "
                "is global_overhead_s (startup, finalization and all-slots-idle gaps). "
                "Add once after explain_hours replay, never to one concurrent task."
            )
    return result, accounting


def explain_with_global_overhead(rates, calls):
    if not calls["accounting_complete"]:
        raise ValueError("Cannot price incomplete accounting")
    overhead = calls.get("global_overhead_s")
    if overhead is None:
        raise ValueError("Missing global lifecycle accounting; supply run boundaries")
    path = PARSER.with_name("explain_hours.py")
    spec = importlib.util.spec_from_file_location("installed_explain_hours", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.explain(rates, calls, None, calls["server_batch"])
    result["task_replay_hours"] = result["explained_hours"]
    result["global_overhead_s"] = overhead
    result["explained_hours"] += overhead / 3600
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("server-log", "samples", "task-list", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--server-batch", type=int, required=True)
    parser.add_argument("--run-start", type=float, help="Unix seconds; inclusive call-start filter")
    parser.add_argument("--run-end", type=float, help="Unix seconds; exclusive call-start filter")
    parser.add_argument("--run-record", type=Path, help="eval_runs record JSON or runs.json")
    parser.add_argument("--rates", type=Path, help="Replay with installed skill, adding global overhead once")
    parser.add_argument("--run-id", help="Required when --run-record contains multiple runs")
    args = parser.parse_args()
    if args.run_record:
        record = json.loads(args.run_record.read_text())
        records = record.get("runs", [record])
        selected = [r for r in records if not args.run_id or r.get("id") == args.run_id]
        if len(selected) != 1:
            parser.error("--run-record must select exactly one record; supply --run-id")
        record = selected[0]
        if args.run_start is None:
            args.run_start = datetime.fromisoformat(record["start_utc"].replace("Z", "+00:00")).timestamp()
        if args.run_end is None:
            args.run_end = datetime.fromisoformat(record["end_utc"].replace("Z", "+00:00")).timestamp()

    samples = [json.loads(line) for line in args.samples.read_text().split("\n") if line.strip()]
    result, calls = collect(
        args.server_log.read_text().splitlines(),
        samples,
        json.loads(args.task_list.read_text()),
        args.agents,
        args.server_batch,
        args.run_start,
        args.run_end,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in [("collected.json", result), ("calls.json", calls)]:
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n")
    if args.rates:
        explained = explain_with_global_overhead(json.loads(args.rates.read_text()), calls)
        (args.out / "explained.json").write_text(json.dumps(explained, indent=2) + "\n")
    print(
        json.dumps(
            {"accounting_complete": result["accounting_complete"], "score": result["score"], "issues": result["issues"]}
        )
    )


if __name__ == "__main__":
    main()
