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

import os
from typing import Dict, List, Optional, Tuple

from .family_backends import FamilyBackend, all_backends, rank_backends

_STRONG_SCORE = 60

_RESOLVE_CACHE: Dict[Tuple[str, str, str, str], Tuple[Optional[FamilyBackend], str]] = {}

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
  ARCH FINGERPRINT (structural): {target_arch}

Observed module tree (the model's ACTUAL instantiated sub-modules -- use this as the
architecture fingerprint, ESPECIALLY when model_type/config is absent, e.g. a
config-less repo. For a composite model, match siblings to these sub-architectures
one by one: a DiT/transformer block -> a diffusion/DiT backend; RMSNorm/MLP/RoPE
blocks -> the matching LLM family; a VAE/autoencoder/quantizer -> a diffusion/audio
backend; a text encoder -> a text-LLM backend):
{components_hint}

Deterministic pre-ranking (a HINT only -- string/tag matching that can MISS when a
backend's registry key is a path-slug rather than the HF model_type, e.g. key
'qwen25_vl' vs HF model_type 'qwen2_5_vl'. Trust the architecture, not the string;
override this hint whenever a different backend is a closer architectural sibling):
{det_hint}

You MUST choose ONLY from these registered backends (use the exact `name`):
{backends}

Backend ARCH FINGERPRINTS (structural — computed the same way as the target's; match
the target's backbone to these, not the task/pipeline label):
{backend_archs}

Return a single JSON object, no prose, no markdown fences:
{{"siblings": [{{"name": "<exact backend name from the list>", "score": <0-100 relevance>, "reason": "<why relevant / which sub-component it maps to>"}}]}}

Rules:
  - STRUCTURAL MATCH FIRST: compare the target's ARCH FINGERPRINT to each backend's
    ARCH FINGERPRINT and rank by BACKBONE match (decoder-only<->decoder-only,
    DiT<->DiT, encoder-decoder<->encoder-decoder, ViT<->ViT, CNN<->CNN). The rank-1
    backend's backbone MUST match the target's backbone. Task / pipeline_tag
    similarity NEVER overrides a backbone match (an autoregressive causal-LM that
    emits images matches a decoder-only LM backend, NOT a diffusion backend, even
    though its tag says text-to-image).
  - HIGHEST PRIORITY: if any backend's `model_type_keys` denote the SAME model as the
    target's model_type -- allowing for separator / underscore / case / version-digit
    grouping differences (e.g. key 'qwen25_vl' IS 'qwen2_5_vl'; 'qwen2-vl' IS
    'qwen2_vl') -- that backend is the SAME architecture, not just a sibling, and MUST
    be ranked FIRST with the highest score (>=90), ABOVE any backend that is merely
    architecturally similar or a better-documented template of a DIFFERENT model
    family. A well-described template for a different model never outranks the exact
    same model, even if the exact match is in a different category or has sparse tags.
  - NEXT PRIORITY: if there is no same-model match, a backend from the SAME model
    FAMILY / lineage as the target -- sharing the model_type stem or name root (e.g.
    'qwen2_vl', 'qwen2_5_vl', 'qwen3_vl' are one Qwen-VL lineage; 'llama2'/'llama3'
    one Llama lineage) -- outranks an unrelated family's template, even a better-
    documented one, because same-lineage models share the most architecture.
  - IDENTIFY THE BACKBONE: from the module tree, determine the model's GENERATIVE
    BACKBONE -- the dominant, most-repeated core block (e.g. a timestep-conditioned
    DiT/UNet trunk for a diffusion model, or the repeated decoder layer for a causal
    LM) -- versus AUXILIARY parts (text/condition/lyric encoders, tokenizers, VAEs).
    The rank-1 sibling MUST match the BACKBONE; auxiliary-part siblings rank BELOW it.
    (A model whose trunk is a DiT ranks a diffusion/DiT backend first even if it also
    contains a text-encoder sub-block; the text-LLM sibling is then rank 2+.)
  - Every "name" MUST be copied verbatim from the list above; never invent a name.
  - Rank most-relevant first; return up to {top_n} entries.
  - For composite models prefer siblings that match the target's SUB-architectures.
Respond with the JSON object ONLY."""


def _backend_by_name() -> dict:
    return {b.name: b for b in all_backends()}


def _format_det_hint(det: List[Tuple[FamilyBackend, int, str]]) -> str:
    if not det:
        return "  (no exact deterministic match -- decide purely on architecture)"
    return "\n".join(f"  - {b.name}  [score={s}; {r}]" for b, s, r in det[:6])


def _format_components_hint(components: Optional[List[dict]]) -> str:
    """Render the discovered module tree as a compact architecture fingerprint for
    the LLM prompt. This is the substitute for config.json on config-less models:
    the instantiated sub-module classes (+ repeat count / leaf-op size) tell the
    LLM what the model actually is when there is no model_type to match on. When
    size data (occurrences / leaf_op_count) is present the list is sorted biggest-
    first and the dominant block is flagged as the likely backbone, so the ranker
    weights the generative trunk over one-off auxiliary parts."""
    if not components:
        return "  (no module tree available)"
    items = []
    for c in components[:40]:
        nm = (c.get("class_name") or c.get("name") or "").strip()
        if not nm:
            continue
        occ = c.get("occurrences") or 0
        leaf = c.get("leaf_op_count") or 0
        items.append(((occ or 1) * (leaf or 1), nm, occ, leaf))
    if not items:
        return "  (no module tree available)"
    have_sizes = any(w > 1 for w, _, _, _ in items)
    if have_sizes:
        items.sort(key=lambda t: -t[0])
    lines = []
    for i, (_w, nm, occ, leaf) in enumerate(items[:24]):
        extra = []
        if occ:
            extra.append(f"x{occ}")
        if leaf:
            extra.append(f"{leaf} ops")
        tag = "  [largest — likely backbone]" if have_sizes and i == 0 else ""
        lines.append(f"  - {nm}" + (f"  ({', '.join(extra)})" if extra else "") + tag)
    return "\n".join(lines)


def rank_backends_llm(
    *,
    model_id: str,
    category: str,
    model_type: Optional[str],
    pipeline_tag: Optional[str],
    architectures: Optional[List[str]] = None,
    notes: str = "",
    top_n: int = 3,
    det_hint: Optional[List[Tuple[FamilyBackend, int, str]]] = None,
    components: Optional[List[dict]] = None,
    model: str = "sonnet",
    agent_bin: str = "claude",
    timeout_s: int = 120,
) -> List[Tuple[FamilyBackend, int, str]]:
    """LLM-ranked siblings constrained to the registry. Returns
    ``[(backend, score, reason)]`` best-first with names validated against the
    registry (unknowns dropped), or ``[]`` when the LLM/SDK is unavailable.
    ``det_hint`` is the deterministic ranking shown to the LLM as guidance (it
    may override it -- string keys miss the path-slug vs model_type gap).
    ``components`` is the discovered module tree, rendered as the architecture
    fingerprint -- the substitute for config on config-less models."""
    from .auto_onboard import _summarize_existing_backends
    from .fingerprint import arch_descriptor
    from .llm_synth import extract_json_from_llm_output, invoke_llm_cli_one_shot

    target_arch = arch_descriptor(
        model_type=model_type,
        architectures=architectures,
        notes=notes,
        components=components,
        pipeline_tag=pipeline_tag,
    )
    backend_archs = "\n".join(
        f"  - {b.name}: {arch_descriptor(model_type=(b.model_type_keys or [None])[0], notes=b.notes, pipeline_tag=(b.pipeline_tags or [None])[0])}"
        for b in all_backends()
    )

    prompt = _LLM_PROMPT.format(
        model_id=model_id,
        model_type=model_type or "",
        pipeline_tag=pipeline_tag or "",
        category=category,
        architectures=architectures or [],
        notes=(notes or "")[:800],
        target_arch=target_arch,
        components_hint=_format_components_hint(components),
        det_hint=_format_det_hint(det_hint or []),
        backends=_summarize_existing_backends(),
        backend_archs=backend_archs,
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


_SIBLING_CACHE: Dict[Tuple, List[Tuple[FamilyBackend, int, str]]] = {}


def _sibling_votes() -> int:
    try:
        return max(1, int(os.environ.get("TT_HW_PLANNER_SIBLING_VOTES", "5")))
    except (TypeError, ValueError):
        return 5


def _vote_rank_llm(votes: int, **kw) -> List[Tuple[FamilyBackend, int, str]]:
    """Run :func:`rank_backends_llm` ``votes`` times concurrently and aggregate by
    AVERAGE score per backend (a backend absent from a run scores 0 for that run),
    returning the top ``top_n`` by average. Self-consistency: a single flaky ask
    can't decide the rank-1 that seeds the scaffold. votes<=1 -> a single ask."""
    top_n = kw.get("top_n", 3)
    if votes <= 1:
        return rank_backends_llm(**kw)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    runs: List[List[Tuple[FamilyBackend, int, str]]] = []
    with ThreadPoolExecutor(max_workers=min(votes, 5)) as ex:
        futs = [ex.submit(rank_backends_llm, **kw) for _ in range(votes)]
        for f in as_completed(futs):
            try:
                runs.append(f.result() or [])
            except Exception:
                runs.append([])

    totals: Dict[str, float] = {}
    backend_of: Dict[str, FamilyBackend] = {}
    best_reason: Dict[str, Tuple[int, str]] = {}
    for lst in runs:
        for b, s, r in lst:
            totals[b.name] = totals.get(b.name, 0.0) + s
            backend_of[b.name] = b
            if b.name not in best_reason or s > best_reason[b.name][0]:
                best_reason[b.name] = (s, r)
    out = [(backend_of[nm], int(round(totals[nm] / votes)), best_reason[nm][1]) for nm in totals]
    out.sort(key=lambda t: -t[1])
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
    components: Optional[List[dict]] = None,
    model: str = "sonnet",
    agent_bin: str = "claude",
    timeout_s: int = 120,
) -> List[Tuple[FamilyBackend, int, str]]:
    """LLM-primary sibling ranking, constrained to the registry. The LLM ranks
    the full registry (seeded with the deterministic scores as a HINT, so a real
    exact match is never lost) and its order is authoritative -- this is what
    generalizes across the path-slug vs HF-model_type key gap that pure string
    matching misses. The LLM ranking is a self-consistency VOTE (N asks averaged,
    ``TT_HW_PLANNER_SIBLING_VOTES`` default 5) so one flaky ask can't decide rank-1.
    The deterministic ranking is appended for any backend the LLM omitted, and is
    used verbatim as the fallback when the LLM/SDK is unavailable, so behaviour
    never hard-regresses. Cached per model so the vote fires at most once per run."""
    cache_key = (model_id or "", category or "", model_type or "", pipeline_tag or "", top_n, bool(use_llm))
    if use_llm and cache_key in _SIBLING_CACHE:
        return _SIBLING_CACHE[cache_key]

    det = rank_backends(category=category, model_type=model_type, pipeline_tag=pipeline_tag, top_n=None)

    llm: List[Tuple[FamilyBackend, int, str]] = []
    if use_llm:
        llm = _vote_rank_llm(
            _sibling_votes(),
            model_id=model_id,
            category=category,
            model_type=model_type,
            pipeline_tag=pipeline_tag,
            architectures=architectures,
            notes=notes,
            top_n=max(top_n, len(det)),
            det_hint=[t for t in det if t[1] >= 90],
            components=components,
            model=model,
            agent_bin=agent_bin,
            timeout_s=timeout_s,
        )

    if not llm:
        return det[:top_n]

    result: List[Tuple[FamilyBackend, int, str]] = [(b, s, "LLM: " + r) for b, s, r in llm]
    seen = {t[0].name for t in result}
    for t in det:
        if t[0].name in seen:
            continue
        result.append(t)
        seen.add(t[0].name)
    result = result[:top_n]
    if use_llm:
        _SIBLING_CACHE[cache_key] = result
    return result


def resolve_backend_with_quality(
    *,
    model_id: str,
    category: str,
    model_type: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
    architectures: Optional[List[str]] = None,
    notes: str = "",
    use_llm: Optional[bool] = None,
    components: Optional[List[dict]] = None,
    model: str = "sonnet",
    agent_bin: str = "claude",
    timeout_s: int = 120,
) -> Tuple[Optional[FamilyBackend], str]:
    """LLM-primary backend decision for the routing sites (gate / scaffold /
    bringup). Asks the LLM-first sibling ranker (deterministic scores fed in as a
    hint) and takes its top confident pick; ``quality`` is ``"llm"`` when the LLM
    chose. Degrades to the deterministic ``pick_backend_with_quality`` when the
    LLM is unavailable / unconfident, so behaviour never hard-regresses. Gated by
    ``TT_HW_PLANNER_LLM_ROUTE`` (default on; set ``=0`` for pure-deterministic).
    Result is cached per model so the LLM fires at most once per run."""
    from .family_backends import pick_backend_with_quality

    if use_llm is None:
        use_llm = os.environ.get("TT_HW_PLANNER_LLM_ROUTE", "1") != "0"

    det_backend, det_quality = pick_backend_with_quality(
        category=category, model_type=model_type, pipeline_tag=pipeline_tag
    )
    if not use_llm or det_quality == "exact":
        return det_backend, det_quality

    key = (model_id or "", category or "", model_type or "", pipeline_tag or "")
    if key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[key]

    resolved: Tuple[Optional[FamilyBackend], str] = (det_backend, det_quality)
    try:
        ranked = rank_siblings(
            model_id=model_id,
            category=category,
            model_type=model_type,
            pipeline_tag=pipeline_tag,
            architectures=architectures,
            notes=notes,
            top_n=3,
            use_llm=True,
            components=components,
            model=model,
            agent_bin=agent_bin,
            timeout_s=timeout_s,
        )
        top = next((t for t in ranked if str(t[2]).startswith("LLM:")), None)
        if top is not None and top[1] >= _STRONG_SCORE:
            resolved = (top[0], "llm")
    except Exception:
        resolved = (det_backend, det_quality)

    _RESOLVE_CACHE[key] = resolved
    return resolved
