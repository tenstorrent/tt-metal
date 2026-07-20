"""Generate ``RUN_REPORT.md`` — a markdown summary of an auto-iterate run.

Written to ``<demo_dir>/RUN_REPORT.md`` after the auto-iterate loop
finishes (converged or not). Captures everything the operator needs
to understand WHAT the tool decided and WHY:

  * 3-category placement summary (HOT / COLD / KERNEL_MISSING)
  * Cold-evidence per component (kind + signals + reasons)
  * Skip-list contents (grouped by category)
  * Graduated stubs list
  * Demo emission status

Pure read-side: this module only READS state files; the auto-iterate
loop calls it once at the end. Failure to write the report never
blocks the run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPORT_EMITTED = False

_REPORT_NAME = "RUN_REPORT.md"
_SECTION_KEYS = ("bringup", "emit-e2e", "optimize")


def upsert_report_section(demo_dir, key: str, block_md: str):
    try:
        demo_dir = Path(demo_dir)
        path = demo_dir / _REPORT_NAME
        begin, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
        block = f"{begin}\n{block_md.strip()}\n{end}"
        existing = path.read_text() if path.exists() else ""
        if begin in existing and end in existing:
            pre = existing.split(begin, 1)[0].rstrip()
            post = existing.split(end, 1)[1].lstrip()
            parts = [p for p in (pre, block, post) if p]
        else:
            parts = [existing.strip(), block] if existing.strip() else [block]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n".join(parts) + "\n")
        return path
    except Exception as exc:
        import sys

        print(f"  [run-report] upsert '{key}' failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def reset_run_report() -> None:
    """Clear the emit-once guard at the start of a bring-up run."""
    global _REPORT_EMITTED
    _REPORT_EMITTED = False


def emit_run_report(
    model_id: str,
    demo_dir: Path,
    *,
    converged: Optional[bool] = None,
    iterations_run: Optional[int] = None,
    demo_emit_status: Optional[str] = None,
    demo_pytest_status: Optional[str] = None,
    stop_reason: str = "",
    echo_terminal: bool = True,
) -> Optional[Path]:
    """Write ``<demo_dir>/RUN_REPORT.md`` summarizing the run, and (when
    ``echo_terminal``) print a concise version to the terminal.

    This is the SINGLE end-of-run report for auto-up/up/promote — module
    placement + per-module pytest + why-not-graduated + reproduce commands
    (the markdown file) plus, on the terminal, the stop reason, any BLOCKER
    (what to install), and the NEXT STEPS (promote / emit-e2e).

    Returns the path written, or ``None`` if write failed (failure is
    non-fatal — never propagates). Emit-once per run: a second call in the
    same run is a no-op (so run_bringup_cc, the up/promote wrappers, and the
    fsm loop can all call it without producing duplicate reports).
    """
    global _REPORT_EMITTED
    if _REPORT_EMITTED:
        return None
    try:
        _p = _emit_run_report_impl(
            model_id,
            demo_dir,
            converged=converged,
            iterations_run=iterations_run,
            demo_emit_status=demo_emit_status,
            demo_pytest_status=demo_pytest_status,
            stop_reason=stop_reason,
            echo_terminal=echo_terminal,
        )
        _REPORT_EMITTED = True
        return _p
    except Exception as exc:
        import sys

        print(f"  [run-report] failed to emit RUN_REPORT.md: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def refresh_bringup_section(demo_dir, model_id: Optional[str] = None) -> None:
    """Re-render ONLY the bring-up section of ``RUN_REPORT.md`` in the current
    format, from on-disk state.

    A later phase (emit-e2e, optimize) calls this before writing its own
    section so a report assembled across tool versions can never show a stale
    old-format bring-up block next to a fresh section. It bypasses the
    emit-once guard, derives the outcome from the persisted categorization, and
    recovers ``model_id`` from the existing report title when not supplied.
    Best-effort: never raises.
    """
    demo_dir = Path(demo_dir)
    if model_id is None:
        try:
            for _ln in (demo_dir / "RUN_REPORT.md").read_text().splitlines():
                if _ln.lstrip().startswith("# Bring-up run report") and "`" in _ln:
                    model_id = _ln.split("`")[1]
                    break
        except Exception:
            model_id = None
        if not model_id:
            model_id = demo_dir.name
    try:
        _emit_run_report_impl(
            model_id,
            demo_dir,
            converged=None,
            iterations_run=None,
            demo_emit_status=None,
            demo_pytest_status=None,
            echo_terminal=False,
        )
    except Exception:
        pass


def _emit_run_report_impl(
    model_id: str,
    demo_dir: Path,
    *,
    converged: Optional[bool],
    iterations_run: Optional[int],
    demo_emit_status: Optional[str],
    demo_pytest_status: Optional[str],
    stop_reason: str = "",
    echo_terminal: bool = True,
) -> Path:
    from .final_categorization import build_final_categorization
    from .overlay_manager import load_persistent_skips

    report_path = demo_dir / "RUN_REPORT.md"

    cat_report = build_final_categorization(model_id=model_id, demo_dir=demo_dir)
    skips = load_persistent_skips(model_id)

    lines: List[str] = []
    lines.append(f"# Bring-up run report — `{model_id}`")
    lines.append("")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_")
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    if converged is None:
        converged = not cat_report.pending and not cat_report.kernel_missing
    _iters = f"{iterations_run} iteration(s)" if iterations_run else "bring-up"
    if converged is True:
        lines.append(f"**Converged** after {_iters}.")
    else:
        lines.append(f"**Did not converge** after {_iters}.")
    if stop_reason:
        lines.append(f"- Run ended: {stop_reason}")
    if demo_emit_status:
        lines.append(f"- Demo emission: `{demo_emit_status}`")
    if demo_pytest_status:
        lines.append(f"- Demo pytest:   `{demo_pytest_status}`")
    blocker_text = ""
    try:
        blocker_text = (demo_dir / ".loader_blocker.txt").read_text().strip()
    except Exception:
        blocker_text = ""
    if blocker_text:
        lines.append("")
        lines.append("### Blocker — what to install / do")
        lines.append("")
        for bl in blocker_text.splitlines():
            lines.append(f"  {bl}")
    lines.append("")

    # --- Backend & template match --------------------------------------------
    # Same four items the terminal prints in Step 1 (`cli.py::pick_backend_with_quality`
    # + `_emit_backend_match_line`) — hoisted here so RUN_REPORT.md doesn't need the
    # (ephemeral) full log to answer "what backend did the tool pick, and how good was
    # the match". Pure read-side: reads bringup_status.json (already on disk from
    # scaffold time). Best-effort: any parse failure just skips the section, never
    # blocks the rest of the report.
    _backend_lines: List[str] = []
    try:
        _bs_path = demo_dir / "bringup_status.json"
        if _bs_path.is_file():
            _bs = json.loads(_bs_path.read_text())
            _be = _bs.get("backend") or {}
            _new_mt = _bs.get("new_model_type")
            _sib_hf = _bs.get("sibling_hf_id")
            _sib_mt = _bs.get("sibling_model_type")
            _be_name = _be.get("name")
            _be_path = _be.get("demo_path")
            # Derive match quality from what we know (model_type equality is the
            # single strongest signal `pick_backend_with_quality` uses; distinct
            # model_types => sibling-template fallback, i.e. CATEGORY-DEFAULT).
            _quality: Optional[str] = None
            if _new_mt and _sib_mt:
                if _new_mt == _sib_mt:
                    _quality = "EXACT (model_type match)"
                else:
                    _quality = "TEMPLATE-FALLBACK (model_type mismatch — closest sibling by category)"
            if _be_name or _new_mt or _sib_hf:
                _backend_lines.append("## Backend & template match")
                _backend_lines.append("")
                if _be_name:
                    _tail = f"  ({_quality})" if _quality else ""
                    _backend_lines.append(f"- **Backend picked:** `{_be_name}`{_tail}")
                if _be_path:
                    _backend_lines.append(f"- **Closest template:** `{_be_path}`")
                if _new_mt:
                    _backend_lines.append(f"- **Target model_type:** `{_new_mt}`")
                if _sib_hf or _sib_mt:
                    _sib_tail = f" (model_type=`{_sib_mt}`)" if _sib_mt else ""
                    _backend_lines.append(f"- **Sibling / template base:** `{_sib_hf or '?'}`{_sib_tail}")
    except Exception:
        _backend_lines = []
    # Optional: cheaper alternatives from a persisted meta-plan verdict, if the
    # meta-planner ran and wrote its JSON (`.meta_plan_verdict.json`). Silent if
    # not present — meta-plan is opt-in and doesn't run in every bring-up.
    try:
        _mp_path = demo_dir / ".meta_plan_verdict.json"
        if _mp_path.is_file():
            _mp = json.loads(_mp_path.read_text())
            _alts = [str(a).strip() for a in (_mp.get("cheaper_alternatives") or []) if str(a).strip()]
            if _alts:
                if not _backend_lines:
                    _backend_lines.append("## Backend & template match")
                    _backend_lines.append("")
                _backend_lines.append(
                    f"- **Cheaper alternatives (meta-plan):** " + ", ".join(f"`{a}`" for a in _alts[:4])
                )
    except Exception:
        pass
    if _backend_lines:
        lines.extend(_backend_lines)
        lines.append("")

    try:
        from .family_backends import all_backends
        from .sibling_ranker import rank_siblings

        _bs3 = {}
        _bs3_path = demo_dir / "bringup_status.json"
        if _bs3_path.is_file():
            _bs3 = json.loads(_bs3_path.read_text())
        _picked = (_bs3.get("backend") or {}).get("name")
        _cat = _bs3.get("category") or next((b.category for b in all_backends() if b.name == _picked), None)
        if _cat:
            _use_llm = os.environ.get("TT_HW_PLANNER_LLM_SIBLINGS", "1") != "0"
            _ranked = rank_siblings(
                model_id=_bs3.get("model_id") or model_id or "",
                category=_cat,
                model_type=_bs3.get("new_model_type"),
                pipeline_tag=_bs3.get("pipeline_tag"),
                architectures=_bs3.get("architectures"),
                notes=_bs3.get("notes") or "",
                top_n=3,
                use_llm=_use_llm,
            )
            if _ranked:
                lines.append("## Sibling candidates (ranked)")
                lines.append("")
                lines.append(
                    "Top backends by match score — the demo can compose per-component reuse across these, not only rank 1."
                )
                lines.append("")
                lines.append("| Rank | Backend | Score | Match reason |")
                lines.append("|---|---|---|---|")
                for _i, (_b, _score, _reason) in enumerate(_ranked, 1):
                    _sel = " (selected)" if _b.name == _picked else ""
                    lines.append(f"| {_i} | `{_b.name}`{_sel} | {_score} | {_reason} |")
                lines.append("")
    except Exception:
        pass

    lines.append("## Placement summary")
    lines.append("")
    lines.append(f"- **ON_DEVICE** ({len(cat_report.on_device)}): graduated, native ttnn, PCC verified")
    if cat_report.on_device:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.on_device))}")
    lines.append(f"- **KERNEL_MISSING** ({len(cat_report.kernel_missing)}): on CPU temporarily — TTNN op gap")
    if cat_report.kernel_missing:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.kernel_missing))}")
    lines.append(f"- **PENDING** ({len(cat_report.pending)}): retry next run")
    if cat_report.pending:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.pending))}")
    lines.append(
        f"- **CPU_REUSE** ({len(cat_report.cpu_reuse)}): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device"
    )
    if cat_report.cpu_reuse:
        lines.append(f"  - {', '.join(f'`{c}`' for c in sorted(cat_report.cpu_reuse))}")
    lines.append("")

    # --- Module-by-module table (every component: placement + why + its pytest) ---
    from .module_tree import safe_identifier

    _dd = str(demo_dir)
    _mi = _dd.rfind("/models/")
    demo_rel = _dd[_mi + 1 :] if _mi >= 0 else demo_dir.name  # e.g. models/demos/xtts_v2

    def _reason_for(comp: str, placement: str) -> str:
        if placement == "ON_DEVICE":
            return "graduated — native ttnn, PCC-verified"
        if placement == "CPU_REUSE":
            return "REUSE/ADAPT tag not wired to a ttnn module — runs on CPU (eager runner)"
        entry = skips.get(comp) or {}
        r = (entry.get("reason") or "").replace("\n", " ").replace("|", "\\|").strip()
        if r:
            return r[:160]
        return "TTNN op gap" if placement == "KERNEL_MISSING" else "retry next run"

    _rows = (
        [(c, "ON_DEVICE") for c in sorted(cat_report.on_device)]
        + [(c, "KERNEL_MISSING") for c in sorted(cat_report.kernel_missing)]
        + [(c, "PENDING") for c in sorted(cat_report.pending)]
        + [(c, "CPU_REUSE") for c in sorted(cat_report.cpu_reuse)]
    )
    lines.append("## Module placement (all components)")
    lines.append("")
    lines.append("| Module | Status | Placement | Detail | Per-module PCC test |")
    lines.append("|---|---|---|---|---|")
    _status_glyph = {"ON_DEVICE": "[ ok ]", "KERNEL_MISSING": "[ cpu ]", "PENDING": "[wait]", "CPU_REUSE": "[ cpu ]"}
    for comp, placement in _rows:
        safe = safe_identifier(comp)
        glyph = _status_glyph.get(placement, "[ -- ]")
        lines.append(
            f"| `{comp}` | {glyph} | {placement} | {_reason_for(comp, placement)} | "
            f"`{demo_rel}/tests/pcc/test_{safe}.py::test_{safe}` |"
        )
    lines.append("")

    # --- Reproduce: exact commands to re-run every reported result ---
    _demo_files = (
        [p for p in sorted((demo_dir / "demo").glob("*.py")) if p.name != "__init__.py"]
        if (demo_dir / "demo").is_dir()
        else []
    )
    _e2e_files = sorted((demo_dir / "tests" / "e2e").glob("test_*.py")) if (demo_dir / "tests" / "e2e").is_dir() else []
    lines.append("## Reproduce")
    lines.append("")
    lines.append("Run from the repo root. Per-component PCC (on device):")
    lines.append("```bash")
    for comp, _pl in _rows:
        safe = safe_identifier(comp)
        lines.append(f"python -m pytest {demo_rel}/tests/pcc/test_{safe}.py::test_{safe} -svv")
    lines.append("```")
    if _e2e_files or _demo_files:
        lines.append("")
        lines.append("End-to-end / demo:")
        lines.append("```bash")
        for p in _e2e_files:
            lines.append(f"python -m pytest {demo_rel}/tests/e2e/{p.name} -svv")
        for p in _demo_files:
            lines.append(f"python -m pytest {demo_rel}/demo/{p.name}::test_demo -svv")
        lines.append("```")
    lines.append("")

    if skips:
        # Group by category for clearer reading
        by_cat: Dict[str, List[tuple]] = {}
        for name, entry in skips.items():
            cat = (entry.get("category") or "UNKNOWN").upper()
            by_cat.setdefault(cat, []).append((name, entry))

        lines.append("## Skip-list")
        lines.append("")
        for cat in sorted(by_cat.keys()):
            entries = by_cat[cat]
            lines.append(f"### `{cat}` — {len(entries)} entry")
            lines.append("")
            for name, entry in sorted(entries, key=lambda x: x[0]):
                reason = (entry.get("reason") or "").replace("\n", " ")[:200]
                retries = entry.get("retry_count")
                retry_s = f" (retries: {retries})" if retries else ""
                lines.append(f"- `{name}`{retry_s}: {reason}")
            lines.append("")

    lines.append("## Next steps")
    lines.append("")
    if cat_report.kernel_missing:
        lines.append(
            f"- **{len(cat_report.kernel_missing)} component(s) need TTNN work** — see KERNEL_MISSING section above"
        )
    retryable_count = sum(
        1 for entry in skips.values() if (entry.get("category") or "").upper() in {"ITERATION_BUDGET", "AGENT_STUCK"}
    )
    if retryable_count:
        lines.append(f"- **{retryable_count} retryable skip(s)** — next `up --auto` re-attempts them automatically")
    tool_bugs = sum(1 for entry in skips.values() if (entry.get("category") or "").upper() == "TOOL_BUG")
    if tool_bugs:
        lines.append(
            f"- **{tool_bugs} TOOL_BUG entry** — fix the scaffolder, then run "
            f"`overlay-clear-skips --category TOOL_BUG {model_id}`"
        )
    _ungraduated = len(cat_report.kernel_missing) + len(cat_report.pending)
    if _ungraduated:
        lines.append(
            f"- **{_ungraduated} component(s) not graduated** — resume where it left off (already-graduated "
            f"components are kept):"
        )
        lines.append(f"  - `python -m scripts.tt_hw_planner promote {model_id} --box <BOX> --mesh <MESH>`")
    elif cat_report.on_device:
        if cat_report.cpu_reuse:
            lines.append(
                f"- **All NEW components graduated** — {len(cat_report.cpu_reuse)} REUSE/ADAPT component(s) "
                f"run on CPU (not wired to a ttnn module). Wire the end-to-end pipeline:"
            )
        else:
            lines.append("- **All components graduated** — wire the end-to-end pipeline:")
        lines.append(f"  - `python -m scripts.tt_hw_planner emit-e2e {model_id}`")
    lines.append("")

    upsert_report_section(demo_dir, "bringup", "\n".join(lines))

    if echo_terminal:
        bar = "=" * 78
        print("\n" + bar)
        print(f"  BRING-UP REPORT — {model_id}")
        if stop_reason:
            print(f"  RUN ENDED: {stop_reason}")
        _cpu_reuse_n = len(cat_report.cpu_reuse)
        _line = (
            f"  on device: {len(cat_report.on_device)}   kernel-missing: {len(cat_report.kernel_missing)}   "
            f"pending: {len(cat_report.pending)}"
        )
        if _cpu_reuse_n:
            _line += f"   cpu-reuse(not-wired): {_cpu_reuse_n}"
        print(_line)
        if blocker_text:
            print("  " + "-" * 74)
            for bl in blocker_text.splitlines():
                print(f"  {bl}")
            print("  " + "-" * 74)
        if _ungraduated:
            print(f"  NEXT STEP: resume the {_ungraduated} not-graduated component(s):")
            print(f"    python -m scripts.tt_hw_planner promote {model_id} --box <BOX> --mesh <MESH>")
        elif cat_report.on_device:
            print("  NEXT STEP: wire the pipeline:")
            print(f"    python -m scripts.tt_hw_planner emit-e2e {model_id}")
        print(f"  full report + per-module pytest → {report_path}")
        print(bar)
    return report_path
