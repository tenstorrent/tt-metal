# SPDX-License-Identifier: Apache-2.0
"""LLM-ranked sibling backends, constrained to the registry.

Deterministic tag/model_type matching (:func:`family_backends.rank_backends`)
finds the exact and same-tag siblings, but for a NOVEL or COMPOSITE model (e.g. a
text-to-audio model that internally contains an LLM text encoder, a VAE, and a
diffusion transformer) relevance is architectural, not a single-string tag match,
so the deterministic ranker returns too few. This module keeps the deterministic
exact/pipeline matches (free, provably correct) and asks Claude Code to fill the
remaining sibling slots by judging semantic/architectural relevance -- but
CONSTRAINED to the fixed set of registered backends, so every pick maps to a real
template and nothing is invented. When the LLM/SDK is unavailable it degrades to
the deterministic list, so behaviour never regresses.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .family_backends import FamilyBackend, all_backends, rank_backends

_STRONG_SCORE = 60

_LLM_PROMPT = """You are selecting the most architecturally-relevant SIBLING backends for a model
being brought up on Tenstorrent hardware. The planner composes per-component reuse
across these siblings, so relevance is about SHARED ARCHITECTURE / SUB-MODELS, not
just a matching pipeline tag. A composite model (e.g. a text-to-audio model that
internally contains an LLM text encoder, a VAE, and a diffusion transformer) should
surface siblings for EACH of those sub-architectures.

Target model:
  model_id:      {model_id}
  model_type:    {model_type!r}
  pipeline_tag:  {pipeline_tag!r}
  category:      {category}
  architectures: {architectures}
  card notes:    {notes}

You MUST choose ONLY from these registered backends (use the exact `name`):
{backends}

Return a single JSON object, no prose, no markdown fences:
{{"siblings": [{{"name": "<exact backend name from the list>", "score": <0-100 relevance>, "reason": "<why relevant / which sub-component it maps to>"}}]}}

Rules:
  - Every "name" MUST be copied verbatim from the list above; never invent a name.
  - Rank most-relevant first; return up to {top_n} entries.
  - For composite models prefer siblings that match the target's SUB-architectures.
Respond with the JSON object ONLY."""


def _backend_by_name() -> dict:
    return {b.name: b for b in all_backends()}


def rank_backends_llm(
    *,
    model_id: str,
    category: str,
    model_type: Optional[str],
    pipeline_tag: Optional[str],
    architectures: Optional[List[str]] = None,
    notes: str = "",
    top_n: int = 3,
    model: str = "sonnet",
    agent_bin: str = "claude",
    timeout_s: int = 120,
) -> List[Tuple[FamilyBackend, int, str]]:
    """LLM-ranked siblings constrained to the registry. Returns
    ``[(backend, score, reason)]`` best-first with names validated against the
    registry (unknowns dropped), or ``[]`` when the LLM/SDK is unavailable."""
    from .auto_onboard import _summarize_existing_backends
    from .llm_synth import extract_json_from_llm_output, invoke_llm_cli_one_shot

    prompt = _LLM_PROMPT.format(
        model_id=model_id,
        model_type=model_type or "",
        pipeline_tag=pipeline_tag or "",
        category=category,
        architectures=architectures or [],
        notes=(notes or "")[:800],
        backends=_summarize_existing_backends(),
        top_n=top_n,
    )
    try:
        raw = invoke_llm_cli_one_shot(prompt, agent_bin=agent_bin, model=model, timeout_s=timeout_s)
    except Exception:
        return []
    obj = extract_json_from_llm_output(raw)
    if not isinstance(obj, dict):
        return []

    by_name = _backend_by_name()
    out: List[Tuple[FamilyBackend, int, str]] = []
    seen = set()
    for entry in obj.get("siblings") or []:
        if not isinstance(entry, dict):
            continue
        nm = entry.get("name")
        backend = by_name.get(nm)
        if backend is None or nm in seen:
            continue
        seen.add(nm)
        try:
            score = int(entry.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        reason = str(entry.get("reason") or "").strip()[:200] or "LLM-selected sibling"
        out.append((backend, score, reason))
    out.sort(key=lambda x: (-x[1], x[0].name))
    return out[:top_n]


def rank_siblings(
    *,
    model_id: str,
    category: str,
    model_type: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
    architectures: Optional[List[str]] = None,
    notes: str = "",
    top_n: int = 3,
    use_llm: bool = True,
    model: str = "sonnet",
    agent_bin: str = "claude",
    timeout_s: int = 120,
) -> List[Tuple[FamilyBackend, int, str]]:
    """Deterministic exact/pipeline matches first (free, provably correct), then
    an LLM loose-fill constrained to the registry to reach ``top_n`` when the
    deterministic ranker comes up short (novel / composite models). Falls back to
    the deterministic list when the LLM is unavailable, so behaviour never
    regresses. Every returned backend is a real registered backend."""
    det = rank_backends(category=category, model_type=model_type, pipeline_tag=pipeline_tag, top_n=None)
    result: List[Tuple[FamilyBackend, int, str]] = [t for t in det if t[1] >= _STRONG_SCORE]
    seen = {t[0].name for t in result}

    if use_llm and len(result) < top_n:
        for backend, score, reason in rank_backends_llm(
            model_id=model_id,
            category=category,
            model_type=model_type,
            pipeline_tag=pipeline_tag,
            architectures=architectures,
            notes=notes,
            top_n=top_n,
            model=model,
            agent_bin=agent_bin,
            timeout_s=timeout_s,
        ):
            if backend.name in seen:
                continue
            result.append((backend, score, "LLM: " + reason))
            seen.add(backend.name)
            if len(result) >= top_n:
                break

    if len(result) < top_n:
        for t in det:
            if t[0].name in seen:
                continue
            result.append(t)
            seen.add(t[0].name)
            if len(result) >= top_n:
                break

    return result[:top_n]
