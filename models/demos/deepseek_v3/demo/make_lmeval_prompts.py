#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Collection

from .score_lmeval_outputs import TASK_ALIASES, resolve_task_name


def _jsonify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    try:
        return value.item()
    except Exception:
        return str(value)


def build_items(task_name: str, task_manager) -> list[dict[str, Any]]:
    resolved = resolve_task_name(task_name, task_manager.task_index.keys())
    if resolved is None:
        raise SystemExit(f"Unknown lm-eval task '{task_name}'")
    task = task_manager.load_task_or_group(resolved)[resolved]

    items: list[dict[str, Any]] = []
    for idx, doc in task.doc_iterator():
        item = {
            "task": resolved,
            "index": int(idx),
            "prompt": task.doc_to_text(doc),
            "doc": _jsonify(doc),
        }
        if resolved != task_name:
            item["task_alias"] = task_name
        items.append(item)
    return items


def main() -> None:
    from lm_eval.tasks import TaskManager

    parser = argparse.ArgumentParser(description="Generate DeepSeek demo prompts from lm-eval tasks")
    parser.add_argument("--task", action="append", required=True, help="lm-eval task name")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional prompt count per output file. Writes *_partNNN.json when splitting.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all requested tasks into one output file.",
    )
    args = parser.parse_args()

    task_manager = TaskManager()
    combine = args.combine or len(args.task) > 1

    items: list[dict[str, Any]] = []
    for task_name in args.task:
        task_items = build_items(task_name, task_manager)
        if combine:
            items.extend(task_items)
        else:
            items = task_items

    out_path = Path(args.out)
    batch_size = args.batch_size
    if batch_size is None or batch_size <= 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {len(items)} prompts to {out_path}")
        return

    chunks = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    if len(chunks) == 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(chunks[0], handle, indent=2, ensure_ascii=False)
        print(f"Wrote {len(chunks[0])} prompts to {out_path}")
        return

    out_dir = out_path.parent if out_path.suffix else out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_path.stem if out_path.suffix else "prompts"
    for idx, chunk in enumerate(chunks, start=1):
        part_path = out_dir / f"{base}_part{idx:03d}.json"
        with open(part_path, "w", encoding="utf-8") as handle:
            json.dump(chunk, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {len(chunk)} prompts to {part_path}")


if __name__ == "__main__":
    main()
