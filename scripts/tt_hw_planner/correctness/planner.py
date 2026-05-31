"""Hypothesis-aware prompt builder for the PCC-repair loop.

This is the evidence-engine replacement for the legacy
:func:`output_validation.build_pcc_repair_prompt`. The legacy
prompt was static (same checklist every iteration, no evidence
trail beyond a side-by-side preview). This builder consumes a
running :class:`hypothesis.HypothesisState` plus the latest
:class:`evidence.TextEvidence` and produces a prompt that:

1. Tells the LLM the WHOLE story so far ("3 iterations in,
   collapse moved from token 36 to token 38, suspect X has been
   tested and ruled out, suspect Y is now the top candidate").
2. Surfaces the TOP-RANKED active suspects with concrete file
   hints, instead of a static checklist.
3. Includes the wider-window collapse view (full output, not
   just first 200 chars).
4. Reuses the legacy budget/commit rules (the LLM MUST make at
   least one Edit per iteration; the budget is ~25 min).

The output is a plain string consumed by ``_invoke_agent``; the
prompt format is intentionally a superset of the legacy prompt
so an LLM that handled the legacy prompt also handles this one.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence

from .evidence import TextEvidence, first_mismatch_segment
from .hypothesis import HypothesisState, Suspect


def _format_collapse_banner(evidence: TextEvidence) -> str:
    if evidence.collapse_position is None:
        return (
            "  GATE VIEW: The output is divergent from the HF reference, "
            "but no clear collapse position was detected. The divergence "
            "is spread across the output rather than a sharp regime "
            "shift."
        )
    rs = evidence.first_regime_shift
    rsd = f" ({rs.kind}: {rs.detail})" if rs else ""
    return (
        f"  GATE VIEW: The TT output COLLAPSED at token "
        f"{evidence.collapse_position} (after a {evidence.prefix_match_count}-"
        f"token coherent prefix that matched HF){rsd}. This is the "
        f"'first N tokens fine, then garbage' pattern -- almost always "
        f"a sliding-window-KV-cache or RoPE position-indexing bug, "
        f"not a tokenizer or weight-conversion bug."
    )


def _format_suspect_section(suspects: Sequence[Suspect]) -> str:
    if not suspects:
        return (
            "  No active suspects remain at confidence > floor. "
            "Consider expanding the search to less-likely areas "
            "(activations, KV-cache layout, dequant path)."
        )
    lines = ["  RANKED SUSPECTS (top-confidence first):"]
    for i, s in enumerate(suspects, 1):
        lines.append(f"    {i}. {s.name}  [confidence {s.confidence:.2f}, " f"status {s.status}]")

        for w in textwrap.wrap(
            s.description,
            width=72,
            initial_indent="       ",
            subsequent_indent="       ",
        ):
            lines.append(w)
        if s.files:
            lines.append("       Suggested files to inspect:")
            for f in s.files:
                lines.append(f"         - {f}")
        if s.history:
            lines.append("       Recent history:")
            for h in s.history[-3:]:
                lines.append(f"         - {h}")
    return "\n".join(lines)


def _format_evidence_trail(state: HypothesisState) -> str:
    if not state.iteration_log:
        return "  (no prior iterations -- this is the first repair attempt)"
    lines = ["  ITERATION HISTORY:"]
    for i, entry in enumerate(state.iteration_log, 1):
        verdict = "IMPROVED" if entry["verdict_improved"] else "WORSENED" if entry["verdict_worsened"] else "unchanged"
        files = entry["edited_files"]
        files_short = ", ".join(files[:3])
        if len(files) > 3:
            files_short += f", … (+{len(files) - 3} more)"
        elif not files:
            files_short = "(no files edited)"
        lines.append(f"    iter {i}: edited {files_short} -> verdict {verdict}")
        lines.append(f"      before: {entry['before_summary']}")
        lines.append(f"      after:  {entry['after_summary']}")
    return "\n".join(lines)


def _format_side_by_side(evidence: TextEvidence) -> str:
    """Show the FULL TT output, not just the first 200 chars.
    The medgemma garbage starts AFTER char ~150; truncating
    earlier hides exactly the symptom the LLM needs to see."""
    seg = first_mismatch_segment(evidence)
    seg_block = ""
    if seg:
        seg_block = (
            f"\n  FIRST MISMATCH (token range {seg.tt_start}-{seg.tt_end}, "
            f"{seg.kind}):\n"
            f"    TT: {seg.tt_text[:120]!r}\n"
            f"    HF: {seg.hf_text[:120]!r}\n"
        )
    tt_view = (evidence.tt_text or "")[:600]
    hf_view = (evidence.hf_text or "")[:600]
    return (
        "  ----- TT demo OUTPUT (first 600 chars) -----\n"
        f"  {tt_view}\n"
        "  ----- HF reference OUTPUT (first 600 chars) -----\n"
        f"  {hf_view}\n"
        "  --------------------------------------------" + seg_block
    )


def _format_budget_rules(iter_idx: int, max_iters: int) -> str:
    return (
        f"  BUDGET + COMMIT RULES:\n"
        f"    * This is iteration {iter_idx} of {max_iters}.\n"
        f"    * You have ~25 minutes of wall-clock per iteration. Spend\n"
        f"      time on reading first (HF config, the suggested files), \n"
        f"      then commit ONE focused change. Do NOT exit without an Edit.\n"
        f"    * If the top-1 suspect was tested in a prior iteration\n"
        f"      and produced no improvement, MOVE TO suspect #2 -- do not\n"
        f"      re-test the same hypothesis with a slightly different patch.\n"
        f"    * The next iteration will see your gate verdict and update\n"
        f"      suspect confidences accordingly. Make your edit COUNT."
    )


def build_repair_prompt(
    *,
    model_id: str,
    evidence: TextEvidence,
    state: HypothesisState,
    iter_idx: int,
    max_iters: int,
    prompt_text: str,
    model_config_block: str = "",
    backend_files_block: str = "",
    weight_cache_block: str = "",
    top_n_suspects: int = 3,
    extra_blocks: Optional[Sequence[str]] = None,
    forced_edit_mode: bool = False,
) -> str:
    """Render the iteration ``iter_idx`` repair prompt.

    Parameters mirror the legacy
    :func:`output_validation.build_pcc_repair_prompt` so the
    call-site change is a name swap. Differences vs legacy:

    * ``evidence`` replaces the bare ``ValidationResult`` (richer).
    * ``state`` is new — the running hypothesis tracker. The
      prompt's suspect section is generated from it.
    * The static suspect checklist is gone; suspects are ranked.

    Order of sections (top-down, biggest signal first):

    1. Header (model id, iter / max_iters).
    2. Collapse banner (one-line "the output collapsed at token N").
    3. Iteration history (deltas across iters).
    4. Ranked suspects with history.
    5. Side-by-side preview (TT vs HF).
    6. Model config / backend files / weight cache state.
    7. Budget + commit rules.
    8. Any extra_blocks the caller wants appended (escape hatch).

    Audit 2026-05-24 (P8/L2): when ``forced_edit_mode`` is set, the
    prompt is aggressively trimmed: ONE top suspect with ONE primary
    file (not 3 × 3), no iter history, no backend file surface, no
    side-by-side preview. The agent's cognitive load drops from
    ~6kB of context to ~2kB so the act-vs-analyze tradeoff tips
    back to act. The caller (the per-component iterate loop in
    ``auto_iterate._run_auto_iterate_loop``) sets this when the
    previous iter exited without any edits. (Historically set by
    ``_pcc_repair_loop`` in the deleted ``_cli_helpers/pcc_repair.py``;
    same condition now fires from Path 1.)
    """
    if forced_edit_mode:
        suspects = state.top_active(n=1)
        if suspects:
            top = suspects[0]
            from dataclasses import replace as _dc_replace

            suspects = [_dc_replace(top, files=top.files[:1] if top.files else ())]
        parts: List[str] = []
        parts.append(f"You are debugging {model_id!r}. Pytest passes but the " f"decoded output diverges from HF.")
        parts.append(_format_collapse_banner(evidence))
        parts.append(_format_suspect_section(suspects))

        parts.append(_format_budget_rules(iter_idx, max_iters))
        parts.append("  ORIGINAL PROMPT (what the demo gave the model):\n" f"    {prompt_text[:200]!r}")
        if extra_blocks:
            for block in extra_blocks:
                parts.append(block)
        return "\n\n".join(parts)

    suspects = state.top_active(n=top_n_suspects)
    parts = []
    parts.append(
        f"You are debugging a TT-hardware bring-up. The model "
        f"{model_id!r} runs end-to-end (pytest passes), but the "
        f"decoded output diverges from the HF CPU reference."
    )
    parts.append(_format_collapse_banner(evidence))
    parts.append(_format_evidence_trail(state))
    parts.append(_format_suspect_section(suspects))
    parts.append(_format_side_by_side(evidence))
    if model_config_block:
        parts.append("  MODEL CONFIG (HF AutoConfig snapshot):")
        parts.append(textwrap.indent(model_config_block, "    "))
    if backend_files_block:
        parts.append("  BACKEND FILE SURFACE (models/tt_transformers/tt/):")
        parts.append(textwrap.indent(backend_files_block, "    "))
    if weight_cache_block:
        parts.append("  TT-NATIVE WEIGHT CACHE STATE:")
        parts.append(textwrap.indent(weight_cache_block, "    "))
    parts.append(_format_budget_rules(iter_idx, max_iters))
    parts.append("  ORIGINAL PROMPT (what the demo input the model):\n" f"    {prompt_text[:400]!r}")
    if extra_blocks:
        for block in extra_blocks:
            parts.append(block)
    return "\n\n".join(parts)


__all__ = [
    "build_repair_prompt",
]
