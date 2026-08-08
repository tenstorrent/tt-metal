"""G5: plan executor.

Orchestrates the generic primitives into a single ``run_iteration``
entry point that ``_run_auto_iterate_loop`` calls per iteration. The
legacy ``_pcc_repair_loop`` was deleted 2026-05-31; Path 2 now
escalates directly to Path 1 so this is the only iterate-loop driver.

Phase order (deterministic, no LLM-driven branching):

  1.  Pre-flight  — capture env snapshot, find HF prompt.
  2.  Dual-probe  — install TT probe (env var, takes effect on
                    next demo run); run HF probe in-process; compare
                    on first iteration after a demo run that includes
                    the probe output.
  3.  Learnings   — lookup (arch_signature, first_diverging_qn) in
                    learned_fixes.json; if hit, apply diff, re-run
                    gate, return early if it passes.
  4.  Mechanical  — try each Action in :func:`default_mechanical_actions`
                    that hasn't been tried this run. After each:
                    re-run gate; if verdict flips, accept; else roll
                    back env override (leave file-system changes
                    in place -- they're idempotent).
  5.  LLM         — invoke agent with the maximal context: divergence
                    table + source files identified via :mod:`.resolve`
                    + history. SINGLE prompt template, no
                    category-specific branching.
  6.  Bisect      — if LLM edits N files, revert subsets to find the
                    minimal-helpful-subset (only if the verdict moved
                    in the right direction).
  7.  Persist     — on gate pass, register the learned fix.

The executor MUST NOT contain category-specific code. Any branching
is on observed signals (verdict, divergence table, probe output) not
on category strings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .actions import (
    Action,
    ActionResult,
    InvalidateCache,
    RevertLastEdits,
    SetEnvVar,
    default_mechanical_actions,
    filter_actions_by_symptoms,
    infer_symptoms,
)
from .convergence import ConvergenceVerdict, predict_convergence
from .diverge import (
    DivergenceReport,
    compute_divergence,
    format_divergence_block,
    load_tt_probe_records,
)
from .learnings import (
    LearnedFix,
    apply_fix,
    compute_arch_signature,
    lookup_fix,
    register_fix,
)
from .probe import HFProbeResult, probe_hf_modules
from .resolve import SuspectSourceFiles, read_file_excerpt, resolve_suspect_files


@dataclass
class AgenticIterationResult:
    """What the executor returns to the calling repair loop."""

    next_action: str
    notes: List[str] = field(default_factory=list)
    diverge_report: Optional[DivergenceReport] = None
    suspect_files: Optional[SuspectSourceFiles] = None
    llm_prompt: Optional[str] = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
    convergence: Optional[ConvergenceVerdict] = None
    applied_actions: List[ActionResult] = field(default_factory=list)


@dataclass
class AgenticContext:
    """State carried across iterations within a single run."""

    model_id: str
    workspace_root: Path
    probe_output_path: Path
    iter_idx: int = 0
    max_iters: int = 12
    history_mismatch: List[float] = field(default_factory=list)
    tried_actions: List[str] = field(default_factory=list)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    hf_probe_result: Optional[HFProbeResult] = None
    last_divergence: Optional[DivergenceReport] = None
    arch_signature: str = ""
    last_diff_files: List[str] = field(default_factory=list)

    consecutive_zero_record_iters: int = 0


def _extract_prompt_from_evidence(evidence_obj: Any) -> str:
    """Best-effort extraction of the input prompt from an evidence
    object. Tries common attribute names. Falls back to a stable
    default so the probe always has something to run on."""
    if evidence_obj is None:
        return "Hello, how are you today?"
    for attr in ("input_hint", "prompt", "prompt_text", "input"):
        v = getattr(evidence_obj, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "Hello, how are you today?"


def _load_hf_config(model_id: str) -> Dict[str, Any]:
    """Pull HF config.json as a dict. Returns {} on failure."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    except Exception:
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(model_id, "config.json")
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def build_llm_prompt(
    *,
    model_id: str,
    iter_idx: int,
    max_iters: int,
    diverge_block: str,
    hf_source_excerpt: str,
    tt_source_excerpt: str,
    suspect_files: Optional[SuspectSourceFiles],
    previous_attempts: Sequence[str],
    convergence: Optional[ConvergenceVerdict],
) -> str:
    """Compose the single, category-agnostic LLM repair prompt.

    The prompt structure is:

      1. Mission (one paragraph, identical for every category).
      2. Empirical divergence table (G1 output).
      3. The two source files (G2 output).
      4. Previous attempts.
      5. Convergence trajectory.
      6. Action required (one Edit, no Read-only iterations).
    """
    hf_path = str(suspect_files.hf_file) if suspect_files and suspect_files.hf_file else "(unresolved)"
    tt_path = str(suspect_files.tt_file) if suspect_files and suspect_files.tt_file else "(unresolved)"

    conv_lines = []
    if convergence is not None:
        conv_lines.append(
            f"  progress_score   : {convergence.progress_score:+.2f}  "
            f"(-1 = regressing, 0 = stagnant, +1 = converging fast)"
        )
        if convergence.stagnant:
            conv_lines.append("  STAGNANT WARNING : last 3 iters made no movement")
        if convergence.predicted_iters_to_zero is not None:
            conv_lines.append(f"  iters_to_zero    : ~{convergence.predicted_iters_to_zero} " f"(at current slope)")

    prev = "\n".join(f"  - {a}" for a in previous_attempts) or "  (none)"

    return f"""\
==============================================================
  AGENTIC PCC REPAIR  iter {iter_idx}/{max_iters}  model={model_id}
==============================================================

MISSION
-------
The TT-side bring-up of `{model_id}` produces output that diverges
from the HF reference. The dual-module-tree probe below has
empirically identified the FIRST submodule whose activation
statistics deviate from HF beyond tolerance. Your job is to make
ONE focused edit to the TT-side source file that fixes that
specific submodule.

The probe ran HF on CPU and TT on the device with the same prompt,
captured per-module mean/std/l2/abs_max, and aligned by qualified
name. The "first diverging" row identifies the bug location.

PROBE OUTPUT (empirical, not guessed)
-------------------------------------
{diverge_block}

SOURCE FILES TO INSPECT (resolved from the diverging classes)
-------------------------------------------------------------
HF reference (read-only):  {hf_path}
TT implementation (edit):  {tt_path}

----- HF source (excerpt) -----
{hf_source_excerpt}

----- TT source (excerpt) -----
{tt_source_excerpt}

PREVIOUS ATTEMPTS
-----------------
{prev}

CONVERGENCE TRAJECTORY
----------------------
{chr(10).join(conv_lines) if conv_lines else "  (first iteration; no history)"}

ACTION REQUIRED
---------------
1. Compare the HF and TT source files. Spot the structural
   difference that produces the divergent statistics.
2. Make EXACTLY ONE Edit to the TT file. Do NOT modify the HF
   file (read-only). Do NOT modify any other TT file unless
   strictly required by your edit.
3. Common root causes when a single submodule diverges:
     - dtype (bfloat8_b vs bfloat16) -- check accumulation
     - missing scaling (1/sqrt(head_dim) on attention)
     - wrong layer-norm eps
     - wrong activation variant (gelu_new vs gelu_pytorch_tanh)
     - residual-add order
     - q/k norm applied along wrong dim
     - weight conversion applied to a 1-D norm vector
     - sliding-window mask collapsing to identity
4. After your Edit, the loop will:
     - re-run the demo
     - re-run the probe
     - re-compute the divergence table
   If your edit was correct, the FIRST DIVERGING row will move
   to a later (deeper) submodule (or disappear).

You MUST end this iteration with at least one Edit. A no-Edit
exit terminates the run as FAIL.
"""


def run_iteration(
    *,
    ctx: AgenticContext,
    captured_demo_output: str,
    evidence_obj: Any,
    current_mismatch_ratio: float,
    hf_probe_max_steps: int = 4,
    probe_verbose: bool = False,
) -> AgenticIterationResult:
    """Run ONE agentic iteration.

    The caller (the repair loop) is responsible for:
      * having already run the demo once with the env var
        ``TT_PLANNER_PROBE_OUTPUT`` pointing at ``ctx.probe_output_path``
      * having already extracted the captured demo output and the
        current evidence object
      * passing the current mismatch_ratio so we can track convergence

    The executor returns a structured :class:`AgenticIterationResult`
    that tells the loop what to do next:

      * ``next_action="graduated"`` -- gate passed, register learning,
        return success.
      * ``next_action="invoke_llm"`` -- the loop should invoke the LLM
        with ``result.llm_prompt`` (and ``result.env_overrides``
        applied to the next demo run).
      * ``next_action="retry"`` -- mechanical action took effect; the
        loop should re-run the demo and call ``run_iteration`` again
        without invoking the LLM.
      * ``next_action="bail"`` -- stagnant / budget exhausted /
        nothing more to try.
    """
    result = AgenticIterationResult(next_action="invoke_llm")
    ctx.iter_idx += 1
    ctx.history_mismatch.append(current_mismatch_ratio)

    if ctx.hf_probe_result is None:
        prompt = _extract_prompt_from_evidence(evidence_obj)
        result.notes.append(f"running HF module probe (model={ctx.model_id}, prompt len={len(prompt)})")
        ctx.hf_probe_result = probe_hf_modules(
            model_id=ctx.model_id,
            prompt_text=prompt,
            max_total_steps=hf_probe_max_steps,
            verbose=probe_verbose,
        )
        if ctx.hf_probe_result is None or not ctx.hf_probe_result.records:
            result.notes.append(
                f"HF probe failed or returned no records "
                f"(note={getattr(ctx.hf_probe_result, 'note', 'None')}); "
                f"falling back to LLM-only path"
            )

    tt_records: List[Dict[str, Any]] = []
    if ctx.probe_output_path.is_file():
        tt_records = load_tt_probe_records(ctx.probe_output_path)
        result.notes.append(f"TT probe records: {len(tt_records)}")
        if not tt_records:
            result.notes.append(
                "TT probe attached but wrapped 0 modules. "
                "Possible causes: (a) the demo uses a TT class outside "
                "tt_transformers.tt.model.Transformer (no probe hook there), "
                "(b) the layer-name heuristic doesn't match the TT class "
                "names for this architecture, (c) the env var didn't reach "
                "the demo subprocess. Grep the demo log for '[tt_probe' "
                "(verbose mode is auto-enabled when the agentic engine is active)."
            )
            ctx.consecutive_zero_record_iters += 1

            if ctx.consecutive_zero_record_iters >= 2:
                result.next_action = "bail"
                result.notes.append(
                    "AGENTIC SMOKE TEST FAILED: the TT probe captured 0 "
                    "records across 2 consecutive iterations. The agentic "
                    "loop has no per-layer divergence signal -- bailing "
                    "now rather than spending more LLM budget flying blind. "
                    "Inspect the demo log for [tt_probe:hook] and [tt_probe] "
                    "lines to diagnose the root cause; common fixes: "
                    "(1) verify trace mode is disabled (TT_PLANNER_NO_TRACE=1 "
                    "should be in the env), (2) verify the demo loads "
                    "tt_transformers.tt.model.Transformer (check stderr for "
                    "the [tt_probe:hook] attach line), (3) extend the probe "
                    "hook to the actual TT root-model class for this category."
                )
        else:
            ctx.consecutive_zero_record_iters = 0

        ctx.env_overrides["TT_PLANNER_PROBE_OUTPUT"] = str(ctx.probe_output_path)
        ctx.env_overrides["TT_PLANNER_PROBE_DEPTH"] = "4"
        ctx.env_overrides["TT_PLANNER_PROBE_VERBOSE"] = "1"
        ctx.env_overrides["TT_PLANNER_NO_TRACE"] = "1"
        result.env_overrides = dict(ctx.env_overrides)
    else:
        result.notes.append(
            f"TT probe sidecar not found at {ctx.probe_output_path} "
            f"(will be generated on next demo run after setting TT_PLANNER_PROBE_OUTPUT)"
        )

        ctx.env_overrides["TT_PLANNER_PROBE_OUTPUT"] = str(ctx.probe_output_path)
        ctx.env_overrides["TT_PLANNER_PROBE_DEPTH"] = "4"
        ctx.env_overrides["TT_PLANNER_PROBE_VERBOSE"] = "1"
        ctx.env_overrides["TT_PLANNER_NO_TRACE"] = "1"
        result.env_overrides = dict(ctx.env_overrides)

    diverge_report: Optional[DivergenceReport] = None
    if ctx.hf_probe_result and ctx.hf_probe_result.records and tt_records:
        diverge_report = compute_divergence(ctx.hf_probe_result, tt_records)
        ctx.last_divergence = diverge_report
        result.diverge_report = diverge_report
        if diverge_report.first_diverging is not None:
            d = diverge_report.first_diverging
            result.notes.append(
                f"first diverging: {d.qualified_name} " f"(rel_err mean={d.rel_err_mean:.1%} l2={d.rel_err_l2:.1%})"
            )
        else:
            result.notes.append("no module diverged above threshold")

    if not ctx.arch_signature:
        cfg = _load_hf_config(ctx.model_id)
        ctx.arch_signature = compute_arch_signature(cfg)
        if ctx.arch_signature:
            result.notes.append(f"arch_signature={ctx.arch_signature}")

    first_qn = ""
    if diverge_report and diverge_report.first_diverging is not None:
        first_qn = diverge_report.first_diverging.qualified_name
    if ctx.iter_idx == 1 and ctx.arch_signature and first_qn:
        learned = lookup_fix(
            arch_signature=ctx.arch_signature,
            first_diverging_qn=first_qn,
        )
        if learned is not None:
            ok, msg = apply_fix(fix=learned, workspace_root=ctx.workspace_root)
            result.notes.append(f"learned fix lookup HIT (source={learned.source_model_id}); " f"apply: {msg}")
            if ok:
                result.next_action = "retry"
                return result

    convergence = predict_convergence(
        ctx.history_mismatch,
        iters_remaining=max(0, ctx.max_iters - ctx.iter_idx),
    )
    result.convergence = convergence

    if convergence.stagnant and ctx.iter_idx >= 2:
        result.notes.append("convergence: STAGNANT -- considering mechanical actions")
    if convergence.predicted_iters_to_zero is not None and convergence.will_hit_zero_by:
        result.notes.append(
            f"convergence: will hit zero by iter {convergence.will_hit_zero_by + ctx.iter_idx} " f"at current slope"
        )

    mechanical_already_tried = bool(ctx.tried_actions)
    should_try_mechanical = (ctx.iter_idx == 1) or (convergence.stagnant and not mechanical_already_tried)
    if should_try_mechanical:
        actions = default_mechanical_actions(
            model_id=ctx.model_id,
            workspace_root=ctx.workspace_root,
            recent_edits=ctx.last_diff_files,  # AUDIT FIX #6: enable RevertLastEdits
        )

        class _VerdictShape:
            mismatch_ratio = float(current_mismatch_ratio)
            max_repeat_ratio = 0.0
            try:
                max_repeat_ratio = float(getattr(evidence_obj, "max_repeat_ratio", 0.0) or 0.0)
            except Exception:
                max_repeat_ratio = 0.0

        symptoms = infer_symptoms(
            verdict_result=_VerdictShape,
            text_evidence=evidence_obj,
            convergence=convergence,
            recent_edits=ctx.last_diff_files,
        )
        if symptoms:
            result.notes.append(f"inferred symptom classes: {sorted(set(symptoms))}")
        filtered = filter_actions_by_symptoms(actions, symptoms)
        next_untried = next((a for a in filtered if a.name not in ctx.tried_actions), None)

        if next_untried is None:
            next_untried = next((a for a in actions if a.name not in ctx.tried_actions), None)
            if next_untried is not None:
                result.notes.append(
                    f"no symptom-matched mechanical action left; "
                    f"falling back to first untried action "
                    f"({next_untried.name})"
                )
        if next_untried is not None:
            ar = next_untried.apply({"env_overrides": ctx.env_overrides})
            ctx.tried_actions.append(next_untried.name)
            result.applied_actions.append(ar)
            for n in ar.notes:
                result.notes.append(f"[{next_untried.name}] {n}")
            if ar.state_changed:
                result.env_overrides = dict(ctx.env_overrides)
                result.next_action = "retry"
                return result

    suspect_files: Optional[SuspectSourceFiles] = None
    hf_excerpt = "(no probe data)"
    tt_excerpt = "(no probe data)"
    diverge_block = "(no probe data available; running on iter 1 bootstrap)"
    if diverge_report and diverge_report.first_diverging is not None:
        suspect_files = resolve_suspect_files(
            diverge_report.first_diverging,
            workspace_root=ctx.workspace_root,
            model_id=ctx.model_id,
        )
        result.suspect_files = suspect_files
        hf_excerpt = read_file_excerpt(suspect_files.hf_file)
        tt_excerpt = read_file_excerpt(suspect_files.tt_file)
        diverge_block = format_divergence_block(diverge_report)

    previous_attempts = [f"iter {i}: mismatch_ratio={r:.0%}" for i, r in enumerate(ctx.history_mismatch, start=1)]
    prompt = build_llm_prompt(
        model_id=ctx.model_id,
        iter_idx=ctx.iter_idx,
        max_iters=ctx.max_iters,
        diverge_block=diverge_block,
        hf_source_excerpt=hf_excerpt,
        tt_source_excerpt=tt_excerpt,
        suspect_files=suspect_files,
        previous_attempts=previous_attempts,
        convergence=convergence,
    )
    result.llm_prompt = prompt
    result.env_overrides = dict(ctx.env_overrides)
    result.next_action = "invoke_llm"
    return result


def register_graduation(
    *,
    ctx: AgenticContext,
    diff_text: str,
    diff_files: List[str],
    notes: str = "",
) -> bool:
    """Persist a learned fix when the gate passed. Returns True on
    successful registration."""
    if not ctx.arch_signature:
        return False
    first_qn = ""
    if ctx.last_divergence and ctx.last_divergence.first_diverging:
        first_qn = ctx.last_divergence.first_diverging.qualified_name
    return register_fix(
        arch_signature=ctx.arch_signature,
        first_diverging_qn=first_qn,
        diff=diff_text,
        diff_files=diff_files,
        source_model_id=ctx.model_id,
        notes=notes,
    )


def compute_diff(workspace_root: Path, files: Sequence[str]) -> str:
    """Get the unified diff for a set of files (used by the executor
    to capture an LLM iteration's edits for learning)."""
    if not files:
        return ""
    try:
        out = subprocess.run(
            ["git", "diff", "HEAD", "--", *files],
            cwd=str(workspace_root),
            capture_output=True,
            check=False,
            timeout=30,
        )
        return out.stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""


__all__ = [
    "AgenticContext",
    "AgenticIterationResult",
    "build_llm_prompt",
    "compute_diff",
    "register_graduation",
    "run_iteration",
]
