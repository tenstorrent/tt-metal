#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
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


def score_aime24(doc: dict, pred: str) -> dict[str, float]:
    from lm_eval.tasks.aime import utils as aime_utils

    return aime_utils.process_results(doc, [pred])


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
    from lm_eval.tasks import TaskManager

    parser = argparse.ArgumentParser(description="Score DeepSeek demo outputs with lm-eval task logic")
    parser.add_argument("--prompts-file", required=True, help="Prompt JSON used for demo.py")
    parser.add_argument("--output-file", required=True, help="demo.py output JSON or checkpoint JSONL")
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

    task_manager = TaskManager()
    task_index = task_manager.task_index.keys()
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

    for task_name, task_metrics in metrics.items():
        print(f"\nTask: {task_name}")
        for metric_name, values in task_metrics.items():
            mean = sum(values) / len(values) if values else 0.0
            print(f"  {metric_name}: {mean:.4f} ({len(values)} samples)")


if __name__ == "__main__":
    main()
