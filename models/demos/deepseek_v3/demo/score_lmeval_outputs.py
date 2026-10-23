#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Collection

TASK_ALIASES = {
    "r1_aime24": "aime24",
}

GPQA_CHOICE_RE = re.compile(r"\(([A-D])\)")
GPQA_BOXED_RE = re.compile(r"\\boxed(?:\s*\{)?\s*\(?([A-D])\)?\s*(?:\})?", re.IGNORECASE)
AIME_BOXED_RE = re.compile(r"\\boxed\s*(?:\{([^{}]*)\}|([^\s$]+))")
AIME_INT_RE = re.compile(r"-?\d+")


def resolve_task_name(task_name: str, task_names: Collection[str]) -> str | None:
    if task_name in task_names:
        return task_name
    alias = TASK_ALIASES.get(task_name)
    if alias in task_names:
        return alias
    if task_name.startswith("r1_"):
        stripped = task_name[len("r1_") :]
        if stripped in task_names:
            return stripped
    return None


def normalize_aime_integer(value: object) -> int | None:
    match = AIME_INT_RE.search(str(value).strip().replace(",", ""))
    if match is None:
        return None
    return int(match.group(0))


def extract_aime_boxed_integer(pred: str) -> int | None:
    matches = list(AIME_BOXED_RE.finditer(pred or ""))
    for match in reversed(matches):
        content = (match.group(1) or match.group(2) or "").strip().replace(",", "")
        value = normalize_aime_integer(content)
        if value is not None:
            return value
    # Some solutions use nested formatting such as \boxed{\textbf{321}},
    # which the simple non-recursive boxed regex above intentionally avoids.
    # For scoring AIME answers, the first integer shortly after the last
    # \boxed marker is the signal we need.
    boxed_markers = list(re.finditer(r"\\boxed", pred or ""))
    for marker in reversed(boxed_markers):
        value = normalize_aime_integer((pred or "")[marker.end() : marker.end() + 128])
        if value is not None:
            return value
    return None


def score_aime24(doc: dict, pred: str) -> dict[str, float]:
    target = normalize_aime_integer(doc["answer"])
    extracted = extract_aime_boxed_integer(pred)
    correct = extracted is not None and target is not None and extracted == target
    return {
        "exact_match": 1.0 if correct else 0.0,
        "no_boxed": 1.0 if extracted is None else 0.0,
        "boxed_wrong": 1.0 if extracted is not None and not correct else 0.0,
    }


def extract_gpqa_choice(pred: str) -> str | None:
    boxed_matches = GPQA_BOXED_RE.findall(pred)
    if boxed_matches:
        return f"({boxed_matches[-1].upper()})"
    choice_matches = GPQA_CHOICE_RE.findall(pred.upper())
    if choice_matches:
        return f"({choice_matches[-1]})"
    answer_match = re.search(r"(?i)(?:the answer is|final answer is|answer:)\s*\(?([A-D])\)?", pred)
    if answer_match:
        return f"({answer_match.group(1).upper()})"
    return None


def score_gpqa(doc: dict, pred: str) -> dict[str, float]:
    target = str(doc["answer"]).strip().upper()
    extracted = extract_gpqa_choice(pred)
    return {"exact_match": 1.0 if extracted == target else 0.0}


LOCAL_TASK_SCORERS = {
    "aime24": score_aime24,
    "r1_aime24": score_aime24,
    "gpqa_diamond": score_gpqa,
    "r1_gpqa_diamond": score_gpqa,
}


def load_generations_by_index(output_path: Path) -> dict[int, str]:
    if output_path.suffix == ".jsonl":
        generations: dict[int, str] = {}
        with open(output_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                index = int(item["index"])
                if index in generations:
                    raise SystemExit(f"Duplicate generation index {index} in {output_path}:{line_number}")
                generations[index] = item.get("text", "") or ""
        return generations

    with open(output_path, "r", encoding="utf-8") as handle:
        output_data = json.load(handle)

    if not isinstance(output_data, dict) or "generations" not in output_data:
        raise SystemExit(f"Expected demo output JSON with a top-level 'generations' array: {output_path}")

    generations = {}
    for item in output_data["generations"]:
        index = int(item["index"])
        if index in generations:
            raise SystemExit(f"Duplicate generation index {index} in {output_path}")
        generations[index] = item.get("text", "") or ""
    return generations


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DeepSeek demo outputs with lm-eval task logic")
    parser.add_argument("--prompts-file", required=True, help="Prompt JSON used for demo.py")
    parser.add_argument("--output-file", required=True, help="demo.py output JSON or checkpoint JSONL")
    parser.add_argument("--summary-json", help="Optional path for writing machine-readable score summary")
    args = parser.parse_args()

    prompts_path = Path(args.prompts_file)
    output_path = Path(args.output_file)

    with open(prompts_path, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list):
        raise SystemExit(f"Expected prompt JSON array in {prompts_path}")

    generations = load_generations_by_index(output_path)
    if not generations:
        raise SystemExit(f"No generations found in {output_path}")

    task_manager = None
    task_index = None
    loaded_tasks = {}
    metrics = defaultdict(lambda: defaultdict(list))
    scored = 0

    for index, prompt_item in enumerate(prompts, start=1):
        pred = generations.get(index)
        if pred is None:
            continue

        task_name = prompt_item["task"]
        doc = prompt_item["doc"]
        scorer = LOCAL_TASK_SCORERS.get(task_name)
        if scorer is not None:
            result = scorer(doc, pred)
        else:
            if task_manager is None:
                from lm_eval.tasks import TaskManager

                task_manager = TaskManager()
                task_index = task_manager.task_index.keys()
            resolved = resolve_task_name(task_name, task_index)
            if resolved is None:
                suggestions = difflib.get_close_matches(task_name, list(task_index), n=5)
                hint = f" Closest: {', '.join(suggestions)}" if suggestions else ""
                raise SystemExit(f"Unknown lm-eval task '{task_name}'.{hint}")
            if resolved not in loaded_tasks:
                loaded_tasks[resolved] = task_manager.load_task_or_group(resolved)[resolved]
            result = loaded_tasks[resolved].process_results(doc, [pred])

        for metric_name, value in result.items():
            metrics[task_name][metric_name].append(value)
        scored += 1

    if scored == 0:
        raise SystemExit("No prompt/generation pairs were scored")

    print(f"Scored {scored}/{len(prompts)} prompts from {output_path}")
    if scored < len(prompts):
        print(f"Skipped {len(prompts) - scored} prompts with no generation record")

    summary = {
        "output_file": str(output_path),
        "prompts_file": str(prompts_path),
        "scored": scored,
        "total_prompts": len(prompts),
        "tasks": {},
    }
    for task_name, task_metrics in metrics.items():
        print(f"\nTask: {task_name}")
        task_summary = {}
        for metric_name, values in task_metrics.items():
            mean = sum(values) / len(values) if values else 0.0
            print(f"  {metric_name}: {mean:.4f} ({len(values)} samples)")
            task_summary[metric_name] = {
                "mean": mean,
                "sum": sum(values),
                "samples": len(values),
            }
        if "exact_match" in task_metrics:
            correct = int(sum(task_metrics["exact_match"]))
            total = len(task_metrics["exact_match"])
            print(f"  correct: {correct}/{total}")
            task_summary["correct"] = {"count": correct, "total": total}
        if "no_boxed" in task_metrics:
            no_boxed = int(sum(task_metrics["no_boxed"]))
            total = len(task_metrics["no_boxed"])
            print(f"  no_boxed: {no_boxed}/{total}")
            task_summary["no_boxed_count"] = {"count": no_boxed, "total": total}
        if "boxed_wrong" in task_metrics:
            boxed_wrong = int(sum(task_metrics["boxed_wrong"]))
            total = len(task_metrics["boxed_wrong"])
            print(f"  boxed_wrong: {boxed_wrong}/{total}")
            task_summary["boxed_wrong_count"] = {"count": boxed_wrong, "total": total}
        summary["tasks"][task_name] = task_summary

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
