# SPDX-License-Identifier: Apache-2.0
"""Summarize collected Harbor evidence without treating errors as low accuracy."""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def seconds(section):
    if not section or not section.get("started_at") or not section.get("finished_at"):
        return None
    return (stamp(section["finished_at"]) - stamp(section["started_at"])).total_seconds()


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def summarize(root):
    rows = []
    identities = []
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text())
        if "task_name" not in result:
            continue
        trajectory = path.parent / "agent/mini-swe-agent.trajectory.json"
        data = json.loads(trajectory.read_text()) if trajectory.exists() else {}
        messages = [m for m in data.get("messages", []) if m.get("role") == "assistant"]
        responses = [m.get("extra", {}).get("response") for m in messages]
        responses = [r for r in responses if r]
        usages = [r.get("usage") or {} for r in responses]
        endings = Counter(c.get("finish_reason") for r in responses for c in r.get("choices", []))
        info = data.get("info", {})
        exception = result.get("exception_info") or {}
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        calls = info.get("model_stats", {}).get("api_calls")
        rows.append(
            dict(
                task=result["task_name"],
                checksum=result.get("task_checksum"),
                reward=reward,
                exception=exception.get("exception_type"),
                exit_status=info.get("exit_status"),
                started_at=result.get("started_at"),
                finished_at=result.get("finished_at"),
                agent_seconds=seconds(result.get("agent_execution")),
                verifier_seconds=seconds(result.get("verifier")),
                saved_assistants=len(messages),
                saved_responses=len(responses),
                api_calls=calls,
                missing_responses=None if calls is None else calls - len(responses),
                saved_prompt_tokens=sum(u.get("prompt_tokens", 0) for u in usages),
                saved_completion_tokens=sum(u.get("completion_tokens", 0) for u in usages),
                saved_cached_tokens=sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in usages),
                finish_reasons=dict(endings),
            )
        )
    for path in root.rglob("runtime_model_spec_*.json"):
        spec = json.loads(path.read_text())["runtime_model_spec"]
        dev = spec["device_model_spec"]
        identities.append(
            dict(
                implementation=spec["impl"]["code_path"],
                model=spec["hf_model_repo"],
                image=spec.get("docker_image"),
                vllm_args=dev.get("vllm_args"),
                env=dev.get("env_vars"),
            )
        )
    if len({r["task"] for r in rows}) != len(rows):
        raise ValueError("Multiple attempts per task: summarize each attempt root separately")
    scores = [r["reward"] for r in rows if r["reward"] is not None]
    starts = [stamp(r["started_at"]) for r in rows if r["started_at"]]
    ends = [stamp(r["finished_at"]) for r in rows if r["finished_at"]]
    return dict(
        scope="collected Harbor attempts; saved-response usage excludes missing responses",
        task_count=len(rows),
        reward_bearing=len(scores),
        solved=sum(v == 1 for v in scores),
        score_percent=None if not rows else 100 * sum(v == 1 for v in scores) / len(rows),
        no_exception=sum(r["exception"] is None for r in rows),
        exceptions=dict(Counter(r["exception"] for r in rows if r["exception"])),
        trial_envelope_seconds=(max(ends) - min(starts)).total_seconds() if starts and ends else None,
        saved_completion_tokens=sum(r["saved_completion_tokens"] for r in rows),
        saved_prompt_tokens=sum(r["saved_prompt_tokens"] for r in rows),
        saved_cached_tokens=sum(r["saved_cached_tokens"] for r in rows),
        missing_responses=sum(r["missing_responses"] or 0 for r in rows),
        implementation_identities=identities,
        tasks=rows,
        limitations=[
            "No matched GPU reference or parity verdict.",
            "Verifier reward does not imply normal agent completion.",
            "Prior reference-exposure findings require trajectory review.",
            "Trial envelope excludes server startup and artifact collection.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["tasks", "implementation_identities"]}))
