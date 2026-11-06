#!/usr/bin/env python3
"""Read completed Harbor SWE artifacts; emit compact candidates, never raw content."""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
import re


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - half), min(1, center + half)]


def seconds(section):
    try:
        return (
            datetime.datetime.fromisoformat(section["finished_at"].replace("Z", "+00:00"))
            - datetime.datetime.fromisoformat(section["started_at"].replace("Z", "+00:00"))
        ).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def network_tools(command):
    # Command-position matches only; quoted source/URL mentions are not retrieval.
    # This is candidate detection, not a shell interpreter or a validity decision.
    patterns = {
        "curl": r"(?:^|[;|&\n])\s*(?:env\s+)?curl\s",
        "wget": r"(?:^|[;|&\n])\s*wget\s",
        "git_remote": r"(?:^|[;|&\n])\s*git\s+(?:clone|fetch|pull|ls-remote)\b",
        "package_install": r"(?:^|[;|&\n])\s*(?:(?:python[\d.]*\s+-m\s+)?pip[\d.]*|uv\s+pip|npm|apt-get|apt)\s+(?:install|download|update)\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, command)]


def audit_trial(path, cap):
    trial = json.loads(path.read_text())
    source = path.parent / "agent/mini-swe-agent.trajectory.json"
    if not source.exists():
        return {"task": trial.get("task_name"), "missing_trajectory": True, "review_required": True}
    t = json.loads(source.read_text())
    messages = t.get("messages", [])
    records, formats, candidates = [], [], []
    for i, m in enumerate(messages):
        extra = m.get("extra") or {}
        if extra.get("interrupt_type") == "FormatError":
            formats.append(
                {"message_index": i, "failed_response_retained": isinstance(extra.get("failed_response"), dict)}
            )
        for key in ["response", "failed_response"]:
            response = extra.get(key)
            if not isinstance(response, dict):
                continue
            choices = response.get("choices") or []
            choice = choices[0] if choices else {}
            ids = choice.get("token_ids")
            if ids is None:
                ids = (choice.get("provider_specific_fields") or {}).get("token_ids")
            usage = response.get("usage") or {}
            count = usage.get("completion_tokens")
            records.append(
                {
                    "message_index": i,
                    "storage": key,
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": count,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "token_ids_count": len(ids) if isinstance(ids, list) else None,
                    "token_ids_match_usage": len(ids) == count if isinstance(ids, list) and count is not None else None,
                    "cap_candidate": count >= cap if isinstance(count, (int, float)) else None,
                }
            )
        # Prefer parsed actions: these are executable command text, not assistant prose.
        actions = extra.get("actions")
        if not isinstance(actions, list):
            actions = []
            for call in m.get("tool_calls") or []:
                try:
                    args = json.loads(call.get("function", {}).get("arguments", "{}"))
                    if isinstance(args, dict):
                        actions.append(args)
                except (TypeError, ValueError):
                    pass
        for j, action in enumerate(actions):
            command = action.get("command") if isinstance(action, dict) else None
            if not isinstance(command, str):
                continue
            tools = network_tools(command)
            if tools:
                observation = str(messages[i + 1].get("content", "")) if i + 1 < len(messages) else ""
                candidates.append(
                    {
                        "message_index": i,
                        "action_index": j,
                        "tools": tools,
                        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                        "following_message_remote_http_indicator": bool(
                            re.search(r"HTTP/[123](?:\.\d)?\s+[1-5]\d\d", observation)
                        ),
                        "following_message_download_indicator": bool(
                            re.search(r"Downloading |Downloaded |Receiving objects:", observation)
                        ),
                        "classification": "human_review_required",
                    }
                )
    api = t.get("info", {}).get("model_stats", {}).get("api_calls")
    native = sum(x["storage"] == "response" for x in records)
    failed = len(records) - native
    known_outputs = [x["completion_tokens"] for x in records if isinstance(x["completion_tokens"], (int, float))]
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    return {
        "task": trial.get("task_name"),
        "api_calls": api,
        "saved_native_responses": native,
        "saved_failed_responses": failed,
        "response_coverage_count": len(records),
        "missing_response_count": api - len(records) if isinstance(api, int) else None,
        "format_errors": formats,
        "responses": records,
        "network_candidates": candidates,
        "token_id_covered_responses": sum(x["token_ids_count"] is not None for x in records),
        "token_id_usage_mismatches": [x["message_index"] for x in records if x["token_ids_match_usage"] is False],
        "max_saved_output_tokens": max(known_outputs, default=None),
        "cap_candidates": [x["message_index"] for x in records if x["cap_candidate"] or x["finish_reason"] == "length"],
        "agent_seconds": seconds(trial.get("agent_execution") or {}),
        "trial_seconds": seconds(trial),
        "exit_status": t.get("info", {}).get("exit_status"),
        "reward": rewards.get("reward"),
        "exception_type": (trial.get("exception_info") or {}).get("exception_type"),
        "trajectory_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source": str(source),
        "validity": "not_automatically_classified",
    }


def audit(root, cap=32768):
    run = json.loads((root / "result.json").read_text())
    if not run.get("finished_at"):
        raise ValueError("Run is not recorded complete; refusing partial live audit")
    tasks = [audit_trial(p, cap) for p in sorted(root.glob("*/result.json"))]
    rewards = [t.get("reward") for t in tasks]
    binary = len(rewards) > 0 and all(x in [0, 1] for x in rewards)
    return {
        "scope": "completed_frozen_cohort_all_available_trials",
        "tasks": tasks,
        "task_count": len(tasks),
        "evaluation_seconds": seconds(run),
        "output_cap": cap,
        "raw_score": {
            "value": sum(rewards) / len(rewards) if binary else None,
            "wilson95": wilson(sum(rewards), len(rewards)) if binary else None,
            "eligible_for_stage_score": False,
            "reason": "Coordinator must audit validity and limits for the full frozen cohort; no subset selection.",
        },
        "limitations": [
            "Candidates are not network-retrieval or loop verdicts. Shell aliases, embedded scripts and commands inside wrappers can evade lexical detection; review complete native trajectories privately.",
            "Missing response bodies cannot be reconstructed. Token-ID counts alone do not prove semantic output or absence of loops.",
            "Per-task timing is host wall; agent sums overlap and are not device time.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-run-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--output-cap", type=int, default=32768)
    args = parser.parse_args()
    result = audit(args.completed_run_root, args.output_cap)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
