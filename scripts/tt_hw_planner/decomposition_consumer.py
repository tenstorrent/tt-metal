"""Auto-consumer for ``decomposition_plan.json``.

When a user has run ``decompose <model> <component> --write-plan``,
the decomposer persists a JSON plan describing parent → children
relationships. This consumer reads that plan on the next ``up``
invocation and:

  1. Mutates ``bringup_status.json`` to ADD each child as a NEW
     component (with submodule_path, class_name).
  2. Marks the PARENT component as ``no_emit`` (its standalone PCC
     would now duplicate work the children cover).
  3. Archives the consumed plan to ``decomposition_plan.applied.json``
     so we don't re-apply it on every run.

Idempotent: if the plan has already been applied, this is a no-op.
Failure to apply a plan is non-fatal — it logs and proceeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def consume_decomposition_plan(
    *,
    model_id: str,
    demo_dir: Path,
) -> Tuple[int, List[str]]:
    """Apply ``<demo_dir>/decomposition_plan.json`` to ``bringup_status.json``.

    Returns ``(children_added, notes)``. ``children_added`` is the count
    of new NEW-status components added to bringup_status. ``notes`` is
    a list of human-readable lines describing what happened (for the
    auto-iterate banner / RUN_REPORT.md).
    """
    plan_path = demo_dir / "decomposition_plan.json"
    if not plan_path.is_file():
        return 0, []

    try:
        plan = json.loads(plan_path.read_text())
    except Exception as exc:
        return 0, [f"[decomposition-consume] failed to read plan: {type(exc).__name__}: {exc}"]
    if not isinstance(plan, list) or not plan:
        return 0, []

    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return 0, ["[decomposition-consume] no bringup_status.json; cannot apply plan"]

    try:
        status = json.loads(status_path.read_text())
    except Exception as exc:
        return 0, [f"[decomposition-consume] failed to read bringup_status: {exc}"]

    components: List[Dict[str, Any]] = list(status.get("components", []) or [])
    existing_names = {c.get("name") for c in components if c.get("name")}

    children_added = 0
    parents_marked_no_emit: List[str] = []
    notes: List[str] = []

    for entry in plan:
        if not isinstance(entry, dict):
            continue
        parent_name = entry.get("parent_name") or ""
        children = entry.get("children") or []
        if not parent_name or not isinstance(children, list):
            continue
        added_this_entry = 0
        for child in children:
            if not isinstance(child, dict):
                continue
            name = child.get("name")
            sub_path = child.get("submodule_path")
            if not name or not sub_path:
                continue
            if name in existing_names:
                continue  # already in bringup_status (likely from prior apply)
            components.append(
                {
                    "name": name,
                    "status": "NEW",
                    "submodule_path": sub_path,
                    "class_name": child.get("class_name") or "",
                    "_added_by_decomposition_of": parent_name,
                }
            )
            existing_names.add(name)
            children_added += 1
            added_this_entry += 1
        if added_this_entry:
            parents_marked_no_emit.append(parent_name)
            notes.append(f"[decompose] {parent_name} → +{added_this_entry} child component(s)")

    if children_added == 0:
        # All children were already in the status — plan is already applied.
        # Archive it so we don't keep re-reading it.
        _archive_plan(plan_path, demo_dir)
        return 0, ["[decomposition-consume] plan already applied; archiving"]

    # Persist updated bringup_status.json
    status["components"] = components
    status_path.write_text(json.dumps(status, indent=2))

    # Mark parents as no_emit so their standalone PCC doesn't conflict.
    try:
        from .overlay_manager import persist_no_emit_test

        for parent in parents_marked_no_emit:
            persist_no_emit_test(
                model_id,
                parent,
                reason=f"decomposition consumer split parent into children at {time.strftime('%Y-%m-%d')}",
            )
    except Exception as exc:
        notes.append(f"[decompose] no_emit persist failed: {exc}")

    _archive_plan(plan_path, demo_dir)
    notes.append(
        f"[decompose] applied plan: {children_added} child component(s) added, "
        f"{len(parents_marked_no_emit)} parent(s) marked no_emit."
    )
    return children_added, notes


def _archive_plan(plan_path: Path, demo_dir: Path) -> None:
    """Move the applied plan to ``decomposition_plan.applied.<ts>.json``
    so subsequent runs don't re-apply it."""
    try:
        archive_dir = demo_dir / "decomposition_plan.applied"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"plan_{ts}.json"
        archive_path.write_text(plan_path.read_text())
        plan_path.unlink()
    except Exception:
        # Archive failure shouldn't block the consumer.
        pass
