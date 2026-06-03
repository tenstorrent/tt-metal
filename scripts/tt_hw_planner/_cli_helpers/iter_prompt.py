from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional


def extract_canonical_api_cheat_sheet(
    tt_reuse_target: str,
    target_component: str,
    *,
    repo_root: Optional[Path] = None,
    max_chars: int = 4000,
) -> str:
    """Build a STRUCTURED 'CANONICAL CLASS API CHEAT SHEET' prompt block by
    AST-parsing the canonical TT class file and surfacing exactly the
    information an ADAPT-iter LLM agent needs to refine the wrapper stub:

      - All `class` definitions in the file (so the agent knows what's
        importable)
      - For each class, its `__init__` signature (arg names + defaults)
      - Each `state_dict[...]` key access pattern (so the agent sees the
        EXACT key format the canonical expects, e.g.
        ``state_dict[f"{wq_str}.weight"]`` reveals the Llama-meta naming
        convention before the agent has to guess)
      - Module-level imports of helper functions (so the agent learns what
        utilities are available, like ``convert_hf_to_meta``)

    Generic across canonical files — works for any path passed as
    ``tt_reuse_target``. NEW components have no ``tt_reuse_target`` and
    bypass this block entirely (caller-gated).

    Returns the empty string if the file can't be read or parsed; the
    caller appends this block to the prompt unconditionally and silently
    drops it when empty.
    """
    if not tt_reuse_target:
        return ""
    try:
        from ..discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        root = repo_root or _BRINGUP_ROOT()
    except Exception:
        root = repo_root or Path.cwd()
    path = (root / tt_reuse_target) if not Path(tt_reuse_target).is_absolute() else Path(tt_reuse_target)
    if not path.is_file():
        return ""
    try:
        src = path.read_text(errors="ignore")
    except Exception:
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""

    lines: List[str] = []
    lines.append(f"CANONICAL CLASS API CHEAT SHEET ({tt_reuse_target})")
    lines.append(
        "(auto-extracted from the canonical file — use this in place of "
        "reading the whole file, which is large and wastes agent budget)"
    )
    lines.append("")

    helper_imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(part in mod for part in ("load_checkpoints", "convert", "tt_ccl", "rope", "model_config", "common")):
                for alias in node.names:
                    helper_imports.append(f"  from {mod} import {alias.name}")
    if helper_imports:
        lines.append("Helper imports visible inside the canonical (you can reuse these):")
        for imp in helper_imports[:12]:
            lines.append(imp)
        lines.append("")

    classes_emitted = 0
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                bases.append("?")
        lines.append(f"class {node.name}({', '.join(bases) or 'object'}):")
        init_emitted = False
        for sub in node.body:
            if not isinstance(sub, ast.FunctionDef):
                continue
            if sub.name != "__init__":
                continue
            try:
                sig = ast.unparse(sub.args)
            except Exception:
                continue
            lines.append(f"    def __init__({sig}):")
            body_keys: List[str] = []
            for inner in ast.walk(sub):
                if isinstance(inner, ast.Subscript):
                    try:
                        sub_src = ast.unparse(inner)
                    except Exception:
                        continue
                    if "state_dict[" in sub_src and sub_src not in body_keys:
                        body_keys.append(sub_src)
            if body_keys:
                lines.append("        # state_dict keys this __init__ reads (literal patterns):")
                for k in body_keys[:8]:
                    lines.append(f"        # - {k}")
            init_emitted = True
            break
        if not init_emitted:
            lines.append("    # (no __init__ defined; uses base class default)")
        lines.append("")
        classes_emitted += 1
        if classes_emitted >= 5:
            break

    layer_name_hits = re.findall(r"layer_name\s*=\s*[^\n]+", src)
    if layer_name_hits:
        lines.append("Per-layer state_dict prefix construction inside the canonical:")
        for hit in layer_name_hits[:3]:
            lines.append(f"  {hit.strip()}")
        lines.append(
            "  -> this typically yields a Meta-style prefix like "
            "`layers.{layer_num}.<module>.` (e.g. `layers.0.attention.`)."
        )
        lines.append("")

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n... (truncated to keep prompt budget)"
    return "\n" + out + "\n"


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
    pcc_trend_block: str = "",
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
        + f"{pcc_trend_block}"
        + f"STRATEGY DIRECTIVE FOR THIS ITERATION:\n{strategy_directive}\n"
        + f"{escalated_scope_block}"
        + f"{native_directive}\n"
        + f"{cross_component_block}"
        + f"COMPONENTS:\n{components_block}\n"
    )


def format_pcc_trend_block(
    *,
    target_component: str,
    pcc_history: Optional[List[float]] = None,
) -> str:
    """Render a per-component PCC-history block for the iter prompt.

    Reads the same ``pcc_history_per_component`` the convergence brain uses
    — entries are **mismatch ratios** (``1.0 - pcc``), smaller is better.
    Surfaces them to the LLM so it can see whether prior attempts moved
    the needle (PCC trending up = make smaller changes; PCC stuck = try
    a different approach).

    Returns empty string when there's no history to show — keeps the
    prompt tight when the component is fresh.

    Notes:
      * Last 5 iters at most. Older iters aren't actionable.
      * Renders raw PCC (1 - mismatch), one decimal place.
      * Tags trend: improving / stagnant / regressing.
    """
    if not pcc_history:
        return ""
    tail = list(pcc_history[-5:])
    pcc_values = [1.0 - float(m) for m in tail]
    if not pcc_values:
        return ""
    # Trend tag
    trend = "first attempt"
    if len(pcc_values) >= 2:
        delta = pcc_values[-1] - pcc_values[0]
        if delta > 0.01:
            trend = f"improving (Δ +{delta:.3f} across last {len(pcc_values)} iters)"
        elif delta < -0.01:
            trend = f"REGRESSING (Δ {delta:.3f} across last {len(pcc_values)} iters — last patch made it worse)"
        else:
            trend = f"STAGNANT (Δ {delta:+.3f} across last {len(pcc_values)} iters — same approach not converging)"
    series = " → ".join(f"{v:.4f}" for v in pcc_values)
    sep = "─" * 78
    return (
        f"\nPCC TREND FOR `{target_component}` (last {len(pcc_values)} iters):\n"
        f"{sep}\n"
        f"  history: {series}\n"
        f"  trend:   {trend}\n"
        f"  target:  ≥ 0.99 to graduate.\n"
        f"  If STAGNANT, your last few patches aren't moving PCC — try a different\n"
        f"  approach (different config, different decomposition, escalated scope).\n"
        f"  If REGRESSING, revert the last change and try a different direction.\n"
        f"{sep}\n\n"
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
        component_block = format_constraint_hints(violations)
    except Exception:
        component_block = ""

    # Bridge: kernel_constraints.py findings (e.g. rotary_embedding_hf
    # head_dim%64 with the explicit "unset use_hf_rope=True" fix) live
    # in a separate static-analysis layer that is computed once at
    # scaffold time. Surface them in every iter prompt so the LLM
    # doesn't reinvent forbidden patches (Phi-3.5 case: iter-5 opus
    # wrote use_hf_rope=True because this block was never appended).
    kernel_block = ""
    try:
        from .kernel_findings import format_kernel_findings_for_prompt, load_kernel_findings

        kernel_findings = load_kernel_findings(Path(demo_dir))
        kernel_block = format_kernel_findings_for_prompt(kernel_findings)
    except Exception:
        kernel_block = ""

    if component_block and kernel_block:
        return component_block + "\n" + kernel_block
    return component_block or kernel_block


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
    component_status: str = "NEW",
    tt_reuse_target: Optional[str] = None,
    last_pcc_value: Optional[float] = None,
    pcc_history: Optional[List[float]] = None,
) -> Dict[str, str]:
    from ..cli import (
        _build_cross_component_context_block,
        _format_escalated_edit_scope_block,
        _native_directive,
        _refinement_directive,
        _strategy_directive_for_failure,
    )

    failure_class = last_failure_class_per_component.get(target_component, "")
    failure_block = per_comp_failure.get(
        target_component,
        f"(no prior failure recorded for `{target_component}` in the latest pytest report)",
    )

    # Prompt routing: ADAPT components get the refinement directive
    # (wrap canonical, edit config only — never rewrite class).
    # NEW components get the from-scratch directive (write ttnn ops).
    if component_status == "ADAPT" and tt_reuse_target:
        directive = _refinement_directive(
            tt_reuse_target=tt_reuse_target,
            pcc_value=last_pcc_value,
        )
        cheat_sheet = extract_canonical_api_cheat_sheet(
            tt_reuse_target=tt_reuse_target,
            target_component=target_component,
        )
        if cheat_sheet:
            directive = directive + cheat_sheet
    else:
        directive = _native_directive(
            forbidden_excerpt="\n\n".join(focused_stub_excerpts) if focused_stub_excerpts else "",
            strict_native=strict_native,
        )

    return {
        "failure_class": failure_class,
        "failure_context": failure_block,
        "strategy_directive": _strategy_directive_for_failure(failure_class, strict_native=strict_native),
        "escalated_scope_block": _format_escalated_edit_scope_block(demo_dir, failure_class),
        "native_directive": directive,
        "cross_component_block": _build_cross_component_context_block(
            demo_dir,
            current_target=target_component,
            attempts_per_component=attempts_per_component,
            last_failure_class_per_component=last_failure_class_per_component,
        ),
        "pcc_trend_block": format_pcc_trend_block(
            target_component=target_component,
            pcc_history=pcc_history,
        ),
    }
