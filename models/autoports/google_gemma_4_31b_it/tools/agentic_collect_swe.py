#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect Harbor first attempts and native calls, including discarded agent replies."""

import argparse
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

from agentic_collect_gpqa import digest, sha, strings, union_seconds


def timestamp(value):
    if isinstance(value, (float, int)):
        return float(value)
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Lifecycle timestamp lacks timezone")
    return parsed.timestamp()


def response_ids(node):
    """Native mini-swe messages[].extra.response.id; retain only explicit IDs."""
    return {
        m["extra"]["response"]["id"]
        for m in node.get("messages", [])
        if isinstance((m.get("extra") or {}).get("response"), dict)
        and isinstance(m["extra"]["response"].get("id"), str)
    }


def collect(lines, attempts, statements, agents, server_batch, run_start, run_end):
    issues, bodies, boundaries = [], {}, {}
    for body in digest.responses(iter(lines)):
        if body["id"] in bodies and bodies[body["id"]] != body:
            raise ValueError("Conflicting response bodies")
        bodies[body["id"]] = body
    for line in digest.framed(iter(lines)):
        if "eval_call=" not in line:
            continue
        row, _ = json.JSONDecoder().raw_decode(line.split("eval_call=", 1)[1])
        if not run_start <= row["start"] < run_end:
            continue
        if row["end"] < row["start"] or row["end"] > run_end:
            raise ValueError("Invalid or unfinished call interval")
        if row["id"] in boundaries and boundaries[row["id"]] != row:
            raise ValueError("Conflicting native boundaries")
        boundaries[row["id"]] = row
    assigned = {i: [] for i in range(len(attempts))}
    for rid, boundary in boundaries.items():
        request = json.loads(boundary["request"])
        texts = list(strings(request.get("messages", request.get("prompt", ""))))
        id_hits = {i for i, a in enumerate(attempts) if rid in a["response_ids"]}
        statement_tasks = {task for task, text in statements.items() if text and any(text in s for s in texts)}
        if len(statement_tasks) > 1:
            issues.append({"id": rid, "reason": "ambiguous task statements"})
            continue
        span_hits = {
            i
            for i, a in enumerate(attempts)
            if a["task"] in statement_tasks
            and a["start"] is not None
            and a["end"] is not None
            and a["start"] <= boundary["start"] <= boundary["end"] <= a["end"]
        }
        candidates = id_hits if id_hits else span_hits
        if len(candidates) != 1:
            issues.append({"id": rid, "reason": "missing or ambiguous exact task/attempt mapping"})
            continue
        i = next(iter(candidates))
        a = attempts[i]
        if statement_tasks and a["task"] not in statement_tasks:
            issues.append({"id": rid, "reason": "native ID contradicts request statement"})
            continue
        if a["start"] is None or a["end"] is None or not a["start"] <= boundary["start"] <= boundary["end"] <= a["end"]:
            issues.append({"id": rid, "task": a["task"], "reason": "call outside complete trial lifecycle"})
            continue
        call = {
            "id": rid,
            "start": boundary["start"],
            "end": boundary["end"],
            "assignment": "native_response_id" if id_hits else "exact_statement_and_unique_trial_interval",
            "absent_from_native_trajectory": rid not in a["response_ids"],
            "request_sha256": sha(boundary["request"]),
            "request_limits": {k: request[k] for k in ("max_tokens", "max_completion_tokens") if k in request},
        }
        body = bodies.get(rid)
        usage = (body or {}).get("usage") or {}
        prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        valid = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (prompt, output, cached))
        if not valid or cached > prompt:
            issues.append(
                {"id": rid, "task": a["task"], "reason": "missing response or invalid token/cache accounting"}
            )
        else:
            call.update(
                prompt_tokens=prompt,
                cached_tokens=cached,
                uncached_input=prompt - cached,
                output=output,
                context=prompt + output,
                finish_reason=body["choices"][0].get("finish_reason"),
            )
        assigned[i].append(call)
    grouped = {task: [] for task in statements}
    for i, attempt in enumerate(attempts):
        row = {k: v for k, v in attempt.items() if k != "response_ids"}
        row["calls"] = sorted(assigned[i], key=lambda c: c["start"])
        for rid in attempt["response_ids"] - boundaries.keys():
            issues.append({"task": attempt["task"], "id": rid, "reason": "trajectory response lacks native interval"})
        if attempt["start"] is not None and attempt["end"] is not None:
            row["outside_s"] = (
                attempt["end"] - attempt["start"] - union_seconds([(c["start"], c["end"]) for c in row["calls"]])
            )
        else:
            issues.append({"task": attempt["task"], "reason": "missing trial lifecycle boundary"})
        grouped[attempt["task"]].append(row)
    selected, all_attempts = [], []
    for task, rows in grouped.items():
        rows.sort(key=lambda r: (float("inf") if r["start"] is None else r["start"], r["trial_name"]))
        if not rows:
            issues.append({"task": task, "reason": "no trial result"})
            selected.append({"task": task, "calls": [], "solved": None})
            continue
        if len(rows) > 1 and (rows[0]["start"] is None or rows[0]["start"] == rows[1]["start"]):
            raise ValueError("First attempt order ambiguous")
        for j, row in enumerate(rows):
            row["counted"] = j == 0
        selected.append(rows[0])
        all_attempts.extend(rows)
    calls = {
        "agents": agents,
        "server_batch": server_batch,
        "tasks": [{k: r[k] for k in ("task", "start", "end", "outside_s", "calls") if k in r} for r in selected],
        "accounting_complete": not issues,
    }
    # Missing usage stays visible, but makes this calls file unpriceable: never invent zero cache.
    result = {
        "suite": "swe_bench_verified",
        "score": {
            "solved": sum(r.get("solved") is True for r in selected),
            "n": len(statements),
            "first_attempt_only": True,
            "complete": all(r.get("solved") is not None for r in selected),
        },
        "tasks": selected,
        "attempts": all_attempts,
        "issues": issues,
        "accounting_complete": not issues,
        "measured_hours": (run_end - run_start) / 3600,
        "solution_fetch_audit": "pending trajectory analysis",
        "timing_basis": "Trial lifecycle minus union of exact native HTTP receipt-to-response intervals; includes setup, agent tools, verifier and inter-call time. Overall supervisor wall separately recorded.",
    }
    complete_spans = [(a["start"], a["end"]) for a in attempts if a["start"] is not None and a["end"] is not None]
    if any(start < run_start or stop > run_end or stop < start for start, stop in complete_spans):
        raise ValueError("Trial lifecycle outside run boundaries")
    if complete_spans and len(complete_spans) == len(attempts):
        global_outside = run_end - run_start - union_seconds(complete_spans)
        result["global_outside_trials_s"] = global_outside
        calls["global_outside_trials_s"] = global_outside
        result[
            "global_overhead_basis"
        ] = "Run wall minus union of all trial lifecycles; includes startup, finalization and any all-slots-idle gaps. Add once after replay, never to one concurrent task."
    n = result["score"]["n"]
    successes = result["score"]["solved"]
    z = 1.959963984540054
    center = (successes / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt((successes / n) * (1 - successes / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result["score"].update(interval_95=[max(0, center - radius), min(1, center + radius)], units="fraction")
    mapping = {"tasks": [{"task": r["task"], "calls": [{"id": c["id"]} for c in r["calls"]]} for r in all_attempts]}
    return result, calls, mapping


def explain_with_global_overhead(rates, calls):
    if not calls["accounting_complete"]:
        raise ValueError("Cannot price incomplete accounting")
    path = digest.__file__.replace("reply_digest.py", "explain_hours.py")
    spec = importlib.util.spec_from_file_location("installed_explain_hours", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.explain(rates, calls, None, calls["server_batch"])
    result["task_replay_hours"] = result["explained_hours"]
    overhead = calls.get("global_outside_trials_s")
    if overhead is None:
        raise ValueError("Missing global lifecycle accounting")
    result["global_outside_trials_s"] = overhead
    result["explained_hours"] += overhead / 3600
    return result


def load_attempts(job_dir, task_ids):
    attempts = []
    for path in sorted(job_dir.rglob("result.json")):
        raw = json.loads(path.read_text())
        if "trial_name" not in raw or "task_name" not in raw:
            continue
        names = {raw["task_name"], Path((raw.get("config", {}).get("task") or {}).get("path", "")).name}
        matches = [t for t in task_ids if t in names or "swe-bench/swebench-verified__" + t in names]
        if len(matches) != 1:
            raise ValueError(f"Unknown/ambiguous task identity: {path}")
        native = path.parent / "agent/mini-swe-agent.trajectory.json"
        trajectory = json.loads(native.read_text()) if native.exists() else {}
        rewards = (raw.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        solved = bool(reward) if reward in (0, 1) and raw.get("reward_valid") is not False else None
        attempts.append(
            {
                "task": matches[0],
                "trial_name": raw["trial_name"],
                "source_result": str(path),
                "start": timestamp(raw.get("started_at")),
                "end": timestamp(raw.get("finished_at")),
                "solved": solved,
                "rewards": rewards,
                "reward_valid": raw.get("reward_valid"),
                "exceptions": {"trial": raw.get("exception_info"), "steps": raw.get("step_results")},
                "phases": {k: raw.get(k) for k in ("environment_setup", "agent_setup", "agent_execution", "verifier")},
                "limits_config": raw.get("config", {}).get("agent"),
                "trajectory_info": trajectory.get("info"),
                "native_trajectory_present": native.exists(),
                "response_ids": response_ids(trajectory),
            }
        )
    return attempts


def main():
    ap = argparse.ArgumentParser()
    for key in ("server-log", "job-dir", "task-list", "task-dir", "out"):
        ap.add_argument("--" + key, required=True, type=Path)
    for key in ("agents", "server-batch"):
        ap.add_argument("--" + key, required=True, type=int)
    for key in ("run-start", "run-end"):
        ap.add_argument("--" + key, required=True, type=float)
    ap.add_argument("--rates", type=Path, help="Replay using installed skill and add measured global overhead once")
    args = ap.parse_args()
    ids = [r["task"] for r in json.loads(args.task_list.read_text())["tasks"]]
    statements = {t: (args.task_dir / t / "instruction.md").read_text().strip() for t in ids}
    result, calls, mapping = collect(
        args.server_log.read_text().splitlines(),
        load_attempts(args.job_dir, ids),
        statements,
        args.agents,
        args.server_batch,
        args.run_start,
        args.run_end,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in [("collected.json", result), ("calls.json", calls), ("digest_accounting.json", mapping)]:
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n")
    if args.rates:
        explained = explain_with_global_overhead(json.loads(args.rates.read_text()), calls)
        (args.out / "explained.json").write_text(json.dumps(explained, indent=2) + "\n")
    print(
        json.dumps(
            {"score": result["score"], "accounting_complete": result["accounting_complete"], "issues": result["issues"]}
        )
    )


if __name__ == "__main__":
    main()
