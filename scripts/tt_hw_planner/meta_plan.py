"""
Meta-reasoning planner (2026-05-23, second of the two pure-automation
improvements requested after the audit).

Runs ONE LLM call BEFORE the auto-iterate loop kicks in and asks the
agent to evaluate the bring-up plan as a whole:

  - Is the model's architecture compatible with the chosen TTNN
    primitives? (E.g. exotic norms, dynamic shapes, custom kernels
    that aren't in `op_emitter.py`.)
  - Are there cheaper alternatives? (E.g. "this is a fine-tune of
    Llama-3.1-8B; the existing Llama backend already covers it.")
  - Are there obvious blockers we should warn about before burning
    iteration budget?

The output is ADVISORY ONLY. The planner prints its recommendations
and proceeds with the normal bring-up loop. The user can disable the
meta-plan entirely via `--no-meta-plan` if it ever produces noise.

Why advisory-only:
  - A pre-loop LLM that BLOCKS bring-ups based on its own judgement
    would be too easy to fool by an overcautious model (false
    rejections would be more painful than letting the iterate loop
    figure it out).
  - The valuable signal from meta-reasoning is "you might want to
    reconsider", not "I'm refusing".
  - If a future iteration wants to add hard-gating, the right move
    is a SEPARATE flag (e.g. `--strict-meta-plan`) rather than
    flipping the default.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetaPlanVerdict:
    """Structured output from the meta-reasoning planner.

    `feasibility` is the agent's claimed confidence that the bring-up
    will converge under the normal auto-iterate loop. The CLI uses it
    only for the log-banner; no action is gated on it."""

    feasibility: str = "UNKNOWN"
    summary: str = ""
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    cheaper_alternatives: List[str] = field(default_factory=list)
    raw_llm_response: str = ""
    parse_error: Optional[str] = None


_PROMPT_TEMPLATE = """You are a Tenstorrent hardware bring-up planner. A user is about to
start the `up --auto` iterate loop for the following HF model on TT
hardware. Your job is to evaluate the plan AS A WHOLE and report risks
+ alternatives BEFORE the iteration burns LLM tokens.

Your output MUST be a single JSON object (no surrounding prose, no
markdown fences) with these fields:

{{
  "feasibility": "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN",
  "summary": "<one-paragraph plain-English assessment>",
  "risks": ["<concrete risk>", ...],
  "recommendations": ["<concrete suggestion for the user OR the loop>", ...],
  "cheaper_alternatives": ["<HF model id that might be a better starting point>", ...]
}}

Hard rules:
  - "feasibility" reflects YOUR confidence the auto-iterate loop will
    converge. Be honest -- HIGH for vanilla transformer families
    against existing backends; MEDIUM for arches with one unusual
    block; LOW for custom-kernel-heavy designs (SSMs, exotic
    quantization, dynamic shapes); UNKNOWN if you don't know enough.
  - "risks" are factual hazards. NOT a wishlist; concrete things
    the loop will hit (e.g. "RoPE variant not in op_emitter").
  - "recommendations" are actionable. Prefer ones the LOOP can act
    on (e.g. "graduate vision_encoder before mask_decoder"); user-
    facing recommendations are also fine.
  - "cheaper_alternatives" -- only fill if there's an ACTUALLY
    cheaper path (e.g. a smaller variant of the same family that
    ships ready). Empty list is fine.
  - Output JSON ONLY. No prose, no markdown.

Bring-up context:

Model id:             {model_id}
Inferred category:    {category}
Inferred model_type:  {model_type!r}
Inferred backend:     {backend_name}  (match quality: {match_quality})
Target box / mesh:    {box} / {mesh}

Discovered components ({num_components}):
{components_summary}

Existing TTNN op kinds known to op_emitter:
  REUSE (already implemented natively, no work needed):
    Linear, Conv2d, LayerNorm, Embedding, GELU, SiLU, ReLU,
    Softmax, Dropout, MultiHeadAttention (standard), MatMul
  ADAPT (covered but may need shape/dtype tuning):
    GroupNorm, BatchNorm, ConvTranspose2d, RMSNorm, RoPE (standard)
  NEW (no native impl yet):
    StateSpaceModel (Mamba/RWKV/S5), SparseAttention (windowed/sliding),
    Custom quantization (bnb 4-bit, gptq), DynamicShape, FlashAttention
    variants beyond standard

If the model's discovered components include classes whose
implementation likely requires NEW ops (above), reflect that in
"feasibility" + "risks" + "recommendations".

Respond with the JSON object ONLY."""


def _summarize_components_for_prompt(components: List[Dict[str, Any]]) -> str:
    if not components:
        return "  (no components discovered)"
    lines = []
    for c in components[:20]:
        name = c.get("name", "?")
        cn = c.get("class_name") or c.get("kind") or "?"
        occ = c.get("occurrences", 1)
        leaves = c.get("leaf_op_count", 0)
        path = c.get("submodule_path") or ""
        lines.append(f"  - {name:<32s} class={cn:<35s} path={path:<35s} " f"occ={occ} leaves={leaves}")
    if len(components) > 20:
        lines.append(f"  ... ({len(components) - 20} more truncated)")
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Backwards-compat shim. See
    :func:`llm_synth.extract_json_from_llm_output`."""
    from .llm_synth import extract_json_from_llm_output

    return extract_json_from_llm_output(text)


def _invoke_llm_one_shot(
    prompt: str,
    *,
    agent_bin: str = "claude",
    model: str = "sonnet",
    timeout_s: int = 240,
) -> str:
    """Backwards-compat shim. See
    :func:`llm_synth.invoke_llm_cli_one_shot`."""
    from .llm_synth import invoke_llm_cli_one_shot

    return invoke_llm_cli_one_shot(
        prompt,
        agent_bin=agent_bin,
        model=model,
        timeout_s=timeout_s,
    )


def run_meta_plan(
    *,
    model_id: str,
    category: str,
    model_type: Optional[str],
    backend_name: str,
    match_quality: str,
    box: str,
    mesh: Optional[str],
    components: List[Dict[str, Any]],
    agent_bin: str = "claude",
    agent_model: str = "sonnet",
    timeout_s: int = 240,
    skip_llm: bool = False,
) -> MetaPlanVerdict:
    """Single entrypoint for the meta-reasoning planner.

    Returns a `MetaPlanVerdict` regardless of LLM availability (so
    cmd_up never crashes on a missing claude binary). On any failure
    path, `feasibility` stays "UNKNOWN" and `parse_error` is filled,
    and the caller proceeds with the normal bring-up loop (advisory-
    only contract)."""
    verdict = MetaPlanVerdict()
    if skip_llm:
        verdict.parse_error = "skip_llm=True (no LLM consulted)"
        verdict.feasibility = "UNKNOWN"
        verdict.summary = (
            "meta-plan skipped (skip_llm=True); proceeding with " "auto-iterate loop without advisory context"
        )
        return verdict
    prompt = _PROMPT_TEMPLATE.format(
        model_id=model_id,
        category=category,
        model_type=model_type or "",
        backend_name=backend_name,
        match_quality=match_quality,
        box=box,
        mesh=mesh or "(default)",
        num_components=len(components),
        components_summary=_summarize_components_for_prompt(components),
    )
    try:
        raw = _invoke_llm_one_shot(prompt, agent_bin=agent_bin, model=agent_model, timeout_s=timeout_s)
    except Exception as exc:
        verdict.parse_error = f"{type(exc).__name__}: {exc}"
        verdict.feasibility = "UNKNOWN"
        verdict.summary = (
            f"meta-plan LLM call failed; proceeding with auto-iterate "
            f"loop without advisory context. ({verdict.parse_error})"
        )
        return verdict
    verdict.raw_llm_response = raw
    obj = _extract_json(raw)
    if obj is None:
        verdict.parse_error = "LLM response was not parseable JSON"
        verdict.summary = (
            "meta-plan response could not be parsed; proceeding with " "auto-iterate loop without advisory context."
        )
        return verdict
    verdict.feasibility = str(obj.get("feasibility", "UNKNOWN")).upper()
    if verdict.feasibility not in {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}:
        verdict.feasibility = "UNKNOWN"
    verdict.summary = str(obj.get("summary", "") or "")
    verdict.risks = [str(r) for r in (obj.get("risks") or [])]
    verdict.recommendations = [str(r) for r in (obj.get("recommendations") or [])]
    verdict.cheaper_alternatives = [str(r) for r in (obj.get("cheaper_alternatives") or [])]
    return verdict


def format_verdict_banner(verdict: MetaPlanVerdict, *, sep: str = "=" * 72) -> str:
    """Pretty-print a verdict for the cmd_up banner. Always returns a
    string; never throws."""
    out = [
        sep,
        f"  META-PLAN  feasibility = {verdict.feasibility}",
        sep,
    ]
    if verdict.summary:
        out.append(f"  {verdict.summary}")
    if verdict.risks:
        out.append(f"\n  risks ({len(verdict.risks)}):")
        for r in verdict.risks[:6]:
            out.append(f"    - {r}")
    if verdict.recommendations:
        out.append(f"\n  recommendations ({len(verdict.recommendations)}):")
        for r in verdict.recommendations[:6]:
            out.append(f"    - {r}")
    if verdict.cheaper_alternatives:
        out.append(f"\n  cheaper alternatives:")
        for r in verdict.cheaper_alternatives[:4]:
            out.append(f"    - {r}")
    if verdict.parse_error:
        out.append(f"\n  (note: {verdict.parse_error})")
    out.append("")
    out.append("  (advisory only; proceeding with auto-iterate loop. Disable " "via --no-meta-plan.)")
    return "\n".join(out)
