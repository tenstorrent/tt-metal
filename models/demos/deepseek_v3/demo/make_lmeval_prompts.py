#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Generate demo.py prompt JSON files from lm-eval tasks to match tti-eval.

Run with the evals venv Python, e.g.:
  ~/.workflow_venvs/.venv_evals_common/bin/python make_lmeval_prompts.py \
    --task aime24 --out /data/deepseek/prompts/demo_aime24_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lm_eval.tasks import TaskManager

TASK_ALIASES = {
    # Historical/tti-eval naming aliases.
    "r1_aime24": "aime24",
}


def resolve_task_name(task_name: str, manager: TaskManager) -> str | None:
    if task_name in manager.task_index:
        return task_name
    alias = TASK_ALIASES.get(task_name)
    if alias and alias in manager.task_index:
        return alias
    if task_name.startswith("r1_"):
        stripped = task_name[len("r1_") :]
        if stripped in manager.task_index:
            return stripped
    return None


def _jsonify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    # datasets can use numpy types; fall back to string
    try:
        return value.item()
    except Exception:
        return str(value)


def build_items(task_name: str, manager: TaskManager) -> list[dict[str, Any]]:
    resolved = resolve_task_name(task_name, manager)
    if resolved is None:
        raise SystemExit(f"Unknown lm-eval task '{task_name}'")
    task = manager.load_task_or_group(resolved)[resolved]

    items: list[dict[str, Any]] = []
    for idx, doc in task.doc_iterator():
        prompt = task.doc_to_text(doc)
        item = {
            "task": resolved,
            "index": int(idx),
            "prompt": prompt,
            "doc": _jsonify(doc),
        }
        if resolved != task_name:
            item["task_alias"] = task_name
        items.append(item)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo.py prompt files from lm-eval tasks")
    parser.add_argument("--task", action="append", required=True, help="lm-eval task name (repeatable)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional max prompts per output file. If set and total exceeds this, write multiple *_partNNN.json files.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all tasks into a single file (default: yes if multiple tasks)",
    )

    args = parser.parse_args()

    tasks = args.task
    manager = TaskManager()
    combine = args.combine or len(tasks) > 1

    all_items: list[dict[str, Any]] = []
    for task_name in tasks:
        items = build_items(task_name, manager)
        if combine:
            all_items.extend(items)
        else:
            all_items = items

    out_path = Path(args.out)
    batch_size = args.batch_size
    if batch_size is None or batch_size <= 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_items, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(all_items)} prompts to {out_path}")
        return

    chunks = [all_items[i : i + batch_size] for i in range(0, len(all_items), batch_size)]
    if len(chunks) == 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks[0], f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(chunks[0])} prompts to {out_path}")
        return

    out_dir = out_path.parent if out_path.suffix else out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_path.stem if out_path.suffix else "prompts"
    for idx, chunk in enumerate(chunks, start=1):
        part_path = out_dir / f"{base}_part{idx:03d}.json"
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(chunk)} prompts to {part_path}")


if __name__ == "__main__":
    main()
