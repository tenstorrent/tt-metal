from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def assemble_iter_prompt(
    *,
    hw_header: str,
    task_block: str,
    systemic_block: str,
    shape_probe_block: str,
    agentic_block: str,
    budget_clause: str,
    failure_context: str,
    strategy_directive: str,
    escalated_scope_block: str,
    native_directive: str,
    cross_component_block: str,
    components_block: str,
    target_header: str = "",
    constraint_block: str = "",
) -> str:
    return (
        target_header
        + f"{hw_header}"
        + f"{task_block}"
        + f"{systemic_block}"
        + f"{shape_probe_block}"
        + f"{agentic_block}"
        + f"{budget_clause}"
        + f"{constraint_block}"
        + f"{failure_context}"
        + f"STRATEGY DIRECTIVE FOR THIS ITERATION:\n{strategy_directive}\n"
        + f"{escalated_scope_block}"
        + f"{native_directive}\n"
        + f"{cross_component_block}"
        + f"COMPONENTS:\n{components_block}\n"
    )


def build_constraint_block(
    *,
    demo_dir: Path,
    target_component: str,
) -> str:
    """Compute the catalog constraint block for a given target component.
    Used by BOTH the primary-target prompt path and the parallel-extra
    path so all agents get the same hints. Never raises — degrades to
    "" on any failure."""
    from ..constraints import check_component, format_constraint_hints

    safe = _safe_id_local(target_component)
    manifest_path = demo_dir / "_captured" / safe / "manifest.json"
    opplan_path = demo_dir / "_stubs" / f"{safe}.opplan.json"

    hf_class_name = ""
    try:
        import json as _json

        manifest = _json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        hf_class_name = manifest.get("hf_class") or ""
    except Exception:
        manifest = {}
    if not hf_class_name:
        try:
            import json as _json

            status = _json.loads((demo_dir / "bringup_status.json").read_text())
            for c in status.get("components", []) or []:
                if c.get("name") == target_component:
                    hf_class_name = c.get("hf_class_name") or c.get("hf_reference") or ""
                    break
        except Exception:
            pass
    try:
        violations = check_component(
            component_name=target_component,
            hf_class_name=hf_class_name or target_component,
            manifest_path=manifest_path,
            opplan_path=opplan_path,
        )
        return format_constraint_hints(violations)
    except Exception:
        return ""


def _safe_id_local(name: str) -> str:
    """Local minimal copy of bringup_loop._safe_id to avoid a heavyweight
    import here. Mirrors the same `re.sub` rule."""
    import re

    return re.sub(r"[^A-Za-z0-9_]+", "_", (name or "").strip()).strip("_")


def build_target_header(
    *,
    target_component: str,
    attempts_so_far: int = 0,
    prior_failure_class: str = "",
) -> str:
    sep = "=" * 78
    lines = [
        sep,
        f"YOUR TARGET COMPONENT FOR THIS RUN: {target_component}",
        sep,
        "",
        f"Edit ONLY `_synth_responses/{target_component}.py` (and read its "
        f"sibling `_stubs/{target_component}.py` for the current stub source).",
        "Do NOT touch other components' files. Other ungraduated components",
        "are being attempted by parallel agents in this same worktree.",
        "",
    ]
    if attempts_so_far > 0:
        lines.append(
            f"THIS IS ATTEMPT #{attempts_so_far + 1} ON `{target_component}`. "
            f"Previous attempt failure class: {prior_failure_class or 'NONE'}."
        )
        lines.append("")
    lines.append(sep)
    lines.append("")
    return "\n".join(lines)


def build_per_target_blocks(
    *,
    demo_dir: Path,
    target_component: str,
    per_comp_failure: Dict[str, str],
    last_failure_class_per_component: Dict[str, str],
    attempts_per_component: Dict[str, int],
    focused_stub_excerpts: List[str],
    strict_native: bool = False,
) -> Dict[str, str]:
    from ..cli import (
        _build_cross_component_context_block,
        _format_escalated_edit_scope_block,
        _native_directive,
        _strategy_directive_for_failure,
    )

    failure_class = last_failure_class_per_component.get(target_component, "")
    failure_block = per_comp_failure.get(
        target_component,
        f"(no prior failure recorded for `{target_component}` in the latest pytest report)",
    )
    return {
        "failure_class": failure_class,
        "failure_context": failure_block,
        "strategy_directive": _strategy_directive_for_failure(failure_class, strict_native=strict_native),
        "escalated_scope_block": _format_escalated_edit_scope_block(demo_dir, failure_class),
        "native_directive": _native_directive(
            forbidden_excerpt="\n\n".join(focused_stub_excerpts) if focused_stub_excerpts else "",
            strict_native=strict_native,
        ),
        "cross_component_block": _build_cross_component_context_block(
            demo_dir,
            current_target=target_component,
            attempts_per_component=attempts_per_component,
            last_failure_class_per_component=last_failure_class_per_component,
        ),
    }
