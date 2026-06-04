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
import re
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
    # 2026-06-04 Fix 7: track children that need a PCC test file emitted
    # after the manifest is persisted. The decomposer only WRITES the
    # bringup_status.json entry; without follow-through, _emit_pcc_template
    # never runs for the child and the iter loop's pool gets ghost
    # components with no tests (seamless-m4t's 5 decomposed children
    # vanished from the pool for exactly this reason).
    new_children_to_emit: List[Dict[str, Any]] = []

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
            raw_name = child.get("name")
            sub_path = child.get("submodule_path")
            if not raw_name or not sub_path:
                continue
            # 2026-06-04 Fix 8: namespace the child's name by parent
            # path so collisions don't happen across the model. A bare
            # "ffn" collides across ~32 ffn instances in seamless-m4t;
            # "text_decoder_layers_0_ffn" is unique. Keep the original
            # `name` in `_child_short_name` for human readability.
            _sub_slug = re.sub(r"[^A-Za-z0-9]+", "_", sub_path).strip("_").lower()
            name = _sub_slug if _sub_slug else str(raw_name)
            if name in existing_names:
                continue  # already in bringup_status (likely from prior apply)
            new_entry = {
                "name": name,
                "status": "NEW",
                "submodule_path": sub_path,
                "class_name": child.get("class_name") or "",
                "_added_by_decomposition_of": parent_name,
                "_child_short_name": str(raw_name),
            }
            components.append(new_entry)
            existing_names.add(name)
            new_children_to_emit.append(new_entry)
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

    # 2026-06-04 Fix 7: emit PCC test files for the newly-added children.
    # Without this, children appear in bringup_status.json but have no
    # tests/pcc/test_<child>.py file — the iter loop sees them in the
    # candidate pool, tries to pick them as targets, and they silently
    # drop out because there's nothing to test. Verified failure mode
    # on seamless-m4t: 5 decomposed children vanished from the pool
    # between iter 1 and iter 2 with zero LLM attempts on them.
    if new_children_to_emit:
        try:
            from .bringup_loop import _emit_pcc_template

            _hf_ref = (
                f"transformers/src/transformers/models/{(status.get('new_model_type') or 'unknown')}"
                f"/modeling_{(status.get('new_model_type') or 'unknown')}.py"
            )
            _emitted: List[str] = []
            for _child in new_children_to_emit:
                try:
                    _emit_pcc_template(
                        demo_dir=demo_dir,
                        component_name=_child["name"],
                        model_id=model_id,
                        hf_reference=_hf_ref,
                        new_shape={},
                        repo_root=demo_dir.parent.parent.parent,
                        overwrite=False,
                        discovered_submodule_path=_child.get("submodule_path"),
                    )
                    _emitted.append(_child["name"])
                except Exception as _emit_exc:
                    notes.append(
                        f"[decompose] failed to emit PCC test for child "
                        f"`{_child['name']}`: {type(_emit_exc).__name__}: {_emit_exc}"
                    )
            if _emitted:
                notes.append(
                    f"[decompose] emitted PCC test(s) for " f"{len(_emitted)} child(ren): {', '.join(_emitted)}"
                )
        except Exception as _import_exc:
            notes.append(
                f"[decompose] could not import _emit_pcc_template " f"(skipping child test emission): {_import_exc}"
            )

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

    # Delete each parent's stale standalone PCC test file. After
    # decomposition the parent stub becomes a CPU-fallback shim that
    # delegates to children; the OLD test was written against the
    # parent's original on-device forward signature and now drives
    # mismatched dtypes through the CPU path, producing phantom
    # failures. The children have their own tests which cover the
    # actual work. Leaving the stale test causes the final pytest
    # sweep to report a "failed" component that's actually fine on
    # device — exactly what surfaced for video_memory_encoder on
    # 2026-05-30. We move the file to .stale instead of deleting so
    # users can recover if needed.
    try:
        from .bringup_loop import _safe_id

        tests_dir = demo_dir / "tests" / "pcc"
        for parent in parents_marked_no_emit:
            safe = _safe_id(parent)
            test_file = tests_dir / f"test_{safe}.py"
            if test_file.is_file():
                stale_path = test_file.with_suffix(".py.stale_after_decomposition")
                test_file.rename(stale_path)
                notes.append(
                    f"[decompose] archived stale standalone test for `{parent}` "
                    f"→ {stale_path.name} (children cover the actual work)"
                )
    except Exception as exc:
        notes.append(f"[decompose] stale-test cleanup failed: {exc}")

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
