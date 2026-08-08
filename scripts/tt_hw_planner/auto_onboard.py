"""
Auto-onboard: LLM-drafted FamilyBackend for brand-new architectures.

The `up --auto` loop is generic over **components** but not over
**architecture families** -- a new HF `model_type` that's not in any
backend's `model_type_keys` hits the loud-fallback gate (added in
2026-05-23 audit defect 1 fix). Historically the only way past that
was a human writing a new `FamilyBackend(...)` entry plus possibly a
new `_extract_components_*` decomposition.

`auto-onboard` automates that bootstrap step:

  1. Probe the new model (HF config + AutoModel module tree walk).
  2. Identify a structurally-closest existing template (or "none").
  3. Build a structured prompt: HF config + clustered module tree +
     existing backend list + canonical FamilyBackend grammar.
  4. Ask the LLM to draft a `FamilyBackend(...)` entry.
  5. Validate the draft (syntax, required fields, no duplicate
     model_type_keys).
  6. Print the proposal to the user.
  7. If `--accept` is set, write it into `family_backends.py` directly
     (after the matching category's last existing entry).

The drafted backend always sets `use_module_tree=True`, so the
downstream `up --auto` pipeline uses module-tree decomposition for it
(not the legacy filename-grep that requires a sibling template demo).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .family_backends import (
    DEFAULT_TEMPLATE_PYTEST_EXCLUDE_K,
    FamilyBackend,
    all_backends,
)
from .module_tree import (
    DiscoveredComponent,
    _camel_segments,
    _common_prefix_segments,
    discover_components_from_hf_id,
)
from .probe import probe_model


@dataclass
class AutoOnboardProposal:
    """The LLM-drafted FamilyBackend, plus its provenance + validation.

    Stored as a struct so the CLI can pretty-print, optionally write
    to disk, and unit-test against."""

    model_id: str
    new_model_type: Optional[str]
    new_pipeline_tag: Optional[str]
    inferred_category: str
    discovered_components: List[Dict[str, Any]] = field(default_factory=list)
    closest_existing_backend: Optional[str] = None
    backend_dataclass_source: str = ""
    backend_python_repr: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    llm_raw_response: str = ""
    notes: List[str] = field(default_factory=list)


def _summarize_module_tree(components: List[DiscoveredComponent]) -> str:
    """Compact text summary the LLM can read in <= 1k tokens."""
    if not components:
        return "(empty module tree -- model has no decomposable structure)"
    lines = []
    for c in components[:20]:
        lines.append(
            f"  - name={c.name!r:<32s} class={c.class_name:<35s} "
            f"path={c.submodule_path:<40s} occ={c.occurrences:<3d} "
            f"leaves={c.leaf_op_count:<4d} hint={c.status_hint}"
        )
    if len(components) > 20:
        lines.append(f"  ... ({len(components) - 20} more components truncated)")
    return "\n".join(lines)


def _score_existing_backend(
    backend: FamilyBackend,
    *,
    new_category: str,
    new_class_names: List[str],
) -> float:
    """Rough structural similarity score. Used to suggest the closest
    sibling template in the LLM prompt, NOT to auto-pick. We never
    silently fall back -- that's the audit defect 1 fix that started
    this whole effort."""
    if backend.category != new_category:
        return 0.0
    score = 0.5

    new_segs: set = set()
    for cn in new_class_names:
        new_segs.update(seg.lower() for seg in _camel_segments(cn))
    for key in backend.model_type_keys:
        if key.lower() in new_segs:
            score += 0.2
    return min(1.0, score)


def _pick_closest_existing_backend(
    *,
    new_category: str,
    new_class_names: List[str],
) -> Tuple[Optional[FamilyBackend], float]:
    best: Optional[FamilyBackend] = None
    best_score = 0.0
    for b in all_backends():
        s = _score_existing_backend(b, new_category=new_category, new_class_names=new_class_names)
        if s > best_score:
            best = b
            best_score = s
    return best, best_score


def _summarize_existing_backends() -> str:
    """Compact 1-line-per-backend list for the LLM prompt context."""
    rows = []
    for b in all_backends():
        rows.append(
            f"  - name={b.name!r:<60s} category={b.category:<5s} "
            f"routing={b.routing_mode:<10s} "
            f"model_type_keys={b.model_type_keys} "
            f"pipeline_tags={b.pipeline_tags}"
        )
    return "\n".join(rows)


_PROMPT_TEMPLATE = """You are drafting a FamilyBackend entry for the tt-metal `tt_hw_planner`
auto-onboarding pipeline. A new HuggingFace model has shown up whose
`model_type` doesn't match any existing FamilyBackend, and the planner
needs a registered backend to safely scaffold + bring it up on
Tenstorrent hardware.

Your output MUST be a single JSON object (no surrounding prose, no
markdown fences) with these fields. The CLI will parse it directly:

{{
  "category": "<one of LLM | VLM | CNN | Image | STT | Embed | NLP>",
  "name": "<human-readable display name, e.g. 'Whatever (new arch)'>",
  "demo_path": "<repo-relative path to the demo dir for this model; if a structurally-similar existing demo lives in models/demos/, prefer that path so scaffold can clone it; otherwise propose a NEW path under models/demos/<slug>/ and the scaffold step will emit a category-aware skeleton via use_module_tree=True>",
  "routing_mode": "template",
  "canonical_hf_id": "{model_id}",
  "notes": "<short, factual description -- what's this model's role / architecture / family>",
  "model_type_keys": ["<HF config.json model_type values that should match this backend, e.g. 'sam_hiera'>"],
  "pipeline_tags": ["<HF pipeline tags that should match this backend>"],
  "smoke_test_entry": null,
  "use_module_tree": true
}}

Hard rules:
  - `category` MUST be one of the allowed enum values above.
  - `use_module_tree` MUST be `true` -- this is what makes the backend
    work without a hand-written sibling template demo.
  - `routing_mode` MUST be `"template"` (LLM/VLM `"generic"` backends
    exist already; new architectures should never be `"generic"`).
  - `model_type_keys` MUST include `{new_model_type!r}` (lower-cased)
    -- this is what makes the loud-fallback gate consider it a real
    match for this exact HF arch.
  - `name` MUST be unique among existing backend names (listed below).
  - DO NOT invent a `smoke_test_entry`; leave it null. The auto-iterate
    loop will create PCC tests after scaffold.

Context:

New model id:         {model_id}
New model_type:       {new_model_type!r}
New pipeline_tag:     {new_pipeline_tag!r}
Inferred category:    {inferred_category}
Closest existing:     {closest_existing}  (score {closest_score:.2f})

Discovered module tree (top components):
{module_tree_summary}

Existing FamilyBackend entries (for naming context; do NOT duplicate):
{existing_backends}

Respond with the JSON object ONLY."""


def _build_prompt(
    *,
    model_id: str,
    new_model_type: Optional[str],
    new_pipeline_tag: Optional[str],
    inferred_category: str,
    components: List[DiscoveredComponent],
    closest_existing: Optional[FamilyBackend],
    closest_score: float,
) -> str:
    return _PROMPT_TEMPLATE.format(
        model_id=model_id,
        new_model_type=new_model_type or "",
        new_pipeline_tag=new_pipeline_tag or "",
        inferred_category=inferred_category,
        closest_existing=(closest_existing.name if closest_existing else "(none)"),
        closest_score=closest_score,
        module_tree_summary=_summarize_module_tree(components),
        existing_backends=_summarize_existing_backends(),
    )


def _invoke_llm_one_shot(
    prompt: str,
    *,
    agent_bin: str = "claude",
    model: str = "sonnet",
    timeout_s: int = 180,
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


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Backwards-compat shim. See
    :func:`llm_synth.extract_json_from_llm_output`."""
    from .llm_synth import extract_json_from_llm_output

    return extract_json_from_llm_output(text)


_ALLOWED_CATEGORIES = {"LLM", "VLM", "CNN", "Image", "STT", "Embed", "NLP"}


def _validate_proposal(
    obj: Dict[str, Any],
    *,
    new_model_type: Optional[str],
) -> List[str]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return ["LLM response was not a JSON object"]
    for required in (
        "category",
        "name",
        "demo_path",
        "routing_mode",
        "canonical_hf_id",
        "model_type_keys",
        "use_module_tree",
    ):
        if required not in obj:
            errors.append(f"missing required field: {required}")
    if obj.get("category") not in _ALLOWED_CATEGORIES:
        errors.append(f"category must be one of {sorted(_ALLOWED_CATEGORIES)}; " f"got {obj.get('category')!r}")
    if obj.get("routing_mode") != "template":
        errors.append(f"routing_mode must be 'template' for auto-onboard backends; " f"got {obj.get('routing_mode')!r}")
    if obj.get("use_module_tree") is not True:
        errors.append(
            "use_module_tree must be true (this is what lets the backend "
            "work without a hand-written sibling template demo)"
        )
    mtks = obj.get("model_type_keys") or []
    if not isinstance(mtks, list) or not mtks:
        errors.append("model_type_keys must be a non-empty list")
    elif new_model_type and new_model_type.lower() not in [str(k).lower() for k in mtks]:
        errors.append(f"model_type_keys must include {new_model_type!r}; got {mtks}")
    name = obj.get("name")
    if name and name in {b.name for b in all_backends()}:
        errors.append(f"name {name!r} collides with an existing backend")
    return errors


def _render_backend_python_source(obj: Dict[str, Any]) -> str:
    """Render the JSON dict as a Python dataclass instantiation that
    can be spliced into `family_backends.py:_BACKENDS = [...]`."""

    def _py(value: Any) -> str:
        return repr(value)

    return (
        "    FamilyBackend(\n"
        f"        category={_py(obj['category'])},\n"
        f"        name={_py(obj['name'])},\n"
        f"        demo_path={_py(obj['demo_path'])},\n"
        f"        routing_mode={_py(obj['routing_mode'])},\n"
        f"        canonical_hf_id={_py(obj['canonical_hf_id'])},\n"
        f"        notes={_py(obj.get('notes', ''))},\n"
        f"        model_type_keys={_py(list(obj.get('model_type_keys', [])))},\n"
        f"        pipeline_tags={_py(list(obj.get('pipeline_tags', [])))},\n"
        f"        smoke_test_entry={_py(obj.get('smoke_test_entry'))},\n"
        f"        use_module_tree=True,\n"
        "    ),\n"
    )


def auto_onboard(
    model_id: str,
    *,
    agent_bin: str = "claude",
    model: str = "sonnet",
    timeout_s: int = 180,
    skip_llm: bool = False,
) -> AutoOnboardProposal:
    """Top-level entry point. Does probe + module-tree walk +
    LLM-draft + validation. The CLI command `auto-onboard` invokes
    this and then decides whether to print or write.

    `skip_llm=True` is used by tests / dry-runs: it walks the module
    tree and produces a hand-deterministic AutoOnboardProposal whose
    `backend_python_repr` is built from local heuristics rather than
    an LLM call. The resulting proposal won't be production-quality
    (the LLM is much better at naming + pipeline_tags) but it
    exercises the full code path without external dependencies."""
    probe = probe_model(model_id)
    new_model_type = str(probe.raw_config.get("model_type") or "") if probe.raw_config else ""
    new_pipeline_tag = getattr(probe, "pipeline_tag", None)
    category = probe.category or "Unknown"

    try:
        components = discover_components_from_hf_id(model_id)
    except Exception as exc:
        components = []
        notes_init = [
            f"module-tree walk failed: {type(exc).__name__}: {exc}; " f"proposal will have empty discovered_components"
        ]
    else:
        notes_init = []

    class_names = [c.class_name for c in components]
    closest, closest_score = _pick_closest_existing_backend(new_category=category, new_class_names=class_names)

    proposal = AutoOnboardProposal(
        model_id=model_id,
        new_model_type=new_model_type or None,
        new_pipeline_tag=new_pipeline_tag,
        inferred_category=category,
        discovered_components=[asdict(c) for c in components],
        closest_existing_backend=(closest.name if closest else None),
        notes=notes_init,
    )

    if skip_llm:
        prefix_segs = _common_prefix_segments([c.class_name for c in components])
        family_slug = (
            "_".join(s.lower() for s in prefix_segs)
            if prefix_segs
            else (new_model_type or "unknown").lower().replace("-", "_")
        )
        stub = {
            "category": category if category in _ALLOWED_CATEGORIES else "CNN",
            "name": f"{model_id} (auto-onboard stub, not LLM-drafted)",
            "demo_path": f"models/demos/auto_onboard/{family_slug}",
            "routing_mode": "template",
            "canonical_hf_id": model_id,
            "notes": "auto-onboard stub (skip_llm=True); LLM not consulted.",
            "model_type_keys": ([new_model_type.lower()] if new_model_type else [family_slug]),
            "pipeline_tags": [],
            "smoke_test_entry": None,
            "use_module_tree": True,
        }
        proposal.validation_errors = _validate_proposal(stub, new_model_type=new_model_type)
        proposal.backend_python_repr = stub
        if not proposal.validation_errors:
            proposal.backend_dataclass_source = _render_backend_python_source(stub)
        return proposal

    prompt = _build_prompt(
        model_id=model_id,
        new_model_type=new_model_type or None,
        new_pipeline_tag=new_pipeline_tag,
        inferred_category=category,
        components=components,
        closest_existing=closest,
        closest_score=closest_score,
    )
    raw = _invoke_llm_one_shot(prompt, agent_bin=agent_bin, model=model, timeout_s=timeout_s)
    proposal.llm_raw_response = raw
    obj = _extract_json(raw)
    if obj is None:
        proposal.validation_errors = ["LLM response did not contain a parseable JSON object"]
        return proposal
    proposal.backend_python_repr = obj
    proposal.validation_errors = _validate_proposal(obj, new_model_type=new_model_type)
    if not proposal.validation_errors:
        proposal.backend_dataclass_source = _render_backend_python_source(obj)
    return proposal


_BACKENDS_FILE = Path(__file__).resolve().parent / "family_backends.py"


def write_backend_into_registry(
    proposal: AutoOnboardProposal,
    *,
    backends_file: Path = _BACKENDS_FILE,
) -> Tuple[bool, str]:
    """Splice the validated proposal into `family_backends.py`'s
    `_BACKENDS = [...]` list, right before the closing `]`.

    Returns ``(ok, message)``. Refuses on validation errors or if
    the proposal's name already exists in the file (paranoid race-
    check)."""
    if proposal.validation_errors:
        return (
            False,
            "proposal has validation errors; not writing:\n  - " + "\n  - ".join(proposal.validation_errors),
        )
    if not proposal.backend_dataclass_source:
        return (False, "proposal has no rendered Python source; not writing")
    text = backends_file.read_text()
    if f"name={proposal.backend_python_repr['name']!r}" in text:
        return (
            False,
            f"backend name {proposal.backend_python_repr['name']!r} already "
            f"present in {backends_file}; refusing to add duplicate",
        )

    decl = "_BACKENDS: List[FamilyBackend] = ["
    decl_idx = text.find(decl)
    if decl_idx < 0:
        return (False, "could not find `_BACKENDS = [` in target file")

    list_start = decl_idx + len(decl) - 1
    open_count = 0
    list_end = -1
    for i in range(list_start, len(text)):
        ch = text[i]
        if ch == "[":
            open_count += 1
        elif ch == "]":
            open_count -= 1
            if open_count == 0:
                list_end = i
                break
    if list_end < 0:
        return (False, "could not find matching `]` for `_BACKENDS`")
    new_text = text[:list_end] + proposal.backend_dataclass_source + text[list_end:]
    backends_file.write_text(new_text)
    return (True, f"wrote backend {proposal.backend_python_repr['name']!r} " f"into {backends_file}")
