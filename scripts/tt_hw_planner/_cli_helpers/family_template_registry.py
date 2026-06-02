"""Chained-template family registry (Item 6).

When the e2e synthesizer (Item 3) converges on a chained ``demo.py``
that passes end-to-end PCC for some model, this module persists the
result as a family template the next sibling model can reuse via
Step-1 template-reuse.

Distinct from the existing ``learned_bringups.json``:
  * ``learned_bringups.json`` — model-level: "this specific model
    succeeded via this Path, here's the diff trail." One entry per
    model, no template artifact.
  * ``learned_chained_templates.json`` (this module) — family-level:
    "this synthesized chained demo works for any model whose model_type
    is X." One entry per family, points at the source demo.py path.

The family key is the HF ``model_type`` (e.g. "phi3", "qwen2",
"sam2") — same key the existing ``family_backends.py`` registry uses
in ``model_type_keys``. So a template registered for ``model_type=
"sam2"`` is reusable by any future model whose config has that
model_type.

Promotion is gated: Item 7 (multi-model confirmation) decides WHEN a
single successful synthesis turns into a promoted family template
(typically after ≥2 sibling models pass with it). This module just
provides the storage primitives.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REGISTRY_FILENAME = "learned_chained_templates.json"


@dataclass
class ChainedTemplateEntry:
    """One persisted chained-template entry.

    Fields:

      * ``family_key`` — HF model_type (e.g. "sam2"). Lookup index.
      * ``template_demo_source`` — path to the demo.py that defines
        the chained forward. Stored relative to the repo root for
        portability.
      * ``source_model_id`` — the model whose successful synthesis
        produced this template. Listed first in ``confirmed_models``.
      * ``confirmed_models`` — every model id known to pass end-to-end
        PCC with this template. ``len(confirmed_models)`` is what Item 7
        gates promotion on.
      * ``promoted`` — True iff Item 7's multi-model gate has elevated
        the template to "trusted." Set by
        :func:`template_promotion.mark_promoted`. Until promoted, the
        Step-1 template-reuse layer should still run LLM verify
        (Item 2) before trusting the template.
      * ``promoted_at`` — UNIX timestamp of promotion (0.0 if never).
      * ``created_at`` / ``updated_at`` — UNIX timestamps.
      * ``final_pcc`` — last-known end-to-end PCC from the source model
        run. Informational.
      * ``notes`` — free-form provenance.
    """

    family_key: str
    template_demo_source: str
    source_model_id: str
    confirmed_models: List[str] = field(default_factory=list)
    promoted: bool = False
    promoted_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    final_pcc: Optional[float] = None
    notes: str = ""


# ─── Registry I/O ────────────────────────────────────────────────────


def _registry_path(repo_root: Optional[Path] = None) -> Path:
    """Resolve the on-disk registry path.

    ``repo_root`` is injectable so tests don't touch the real
    ``scripts/tt_hw_planner/learned_chained_templates.json``. Real
    callers pass ``None`` to use the canonical location.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    return repo_root / REGISTRY_FILENAME


def load_registry(repo_root: Optional[Path] = None) -> Dict[str, ChainedTemplateEntry]:
    """Read the registry from disk. Returns ``{family_key: entry}``.

    Missing file → empty dict (first-ever call). Malformed JSON →
    empty dict + stderr warning. Never raises.
    """
    path = _registry_path(repo_root)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, ChainedTemplateEntry] = {}
    for key, val in raw.items():
        if not isinstance(val, dict) or not isinstance(key, str):
            continue
        try:
            out[key] = ChainedTemplateEntry(
                family_key=str(val.get("family_key") or key),
                template_demo_source=str(val.get("template_demo_source") or ""),
                source_model_id=str(val.get("source_model_id") or ""),
                confirmed_models=list(val.get("confirmed_models") or []),
                promoted=bool(val.get("promoted", False)),
                promoted_at=float(val.get("promoted_at") or 0.0),
                created_at=float(val.get("created_at") or 0.0),
                updated_at=float(val.get("updated_at") or 0.0),
                final_pcc=val.get("final_pcc"),
                notes=str(val.get("notes") or ""),
            )
        except Exception:
            continue
    return out


def save_registry(registry: Dict[str, ChainedTemplateEntry], repo_root: Optional[Path] = None) -> bool:
    """Write the registry to disk. Returns True on success."""
    path = _registry_path(repo_root)
    blob = {k: asdict(v) for k, v in registry.items()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
        return True
    except Exception:
        return False


# ─── High-level operations ──────────────────────────────────────────


def register_template(
    *,
    family_key: str,
    template_demo_source: str,
    source_model_id: str,
    final_pcc: Optional[float] = None,
    notes: str = "",
    repo_root: Optional[Path] = None,
    clock: Optional[Any] = None,
) -> Optional[ChainedTemplateEntry]:
    """Add or update an entry. Idempotent.

    First call for a ``family_key`` creates a fresh entry with
    ``source_model_id`` as both the source and the first confirmed
    model.

    Subsequent calls with the SAME ``family_key`` update
    ``updated_at`` and add ``source_model_id`` to
    ``confirmed_models`` if it's not already there. Doesn't change
    the original ``source_model_id`` or ``template_demo_source`` —
    re-registration confirms an existing template, doesn't replace
    it. (Item 7's multi-model gate uses this idempotent confirmation
    to count distinct successes.)

    ``clock`` is injectable so tests get deterministic timestamps.
    Returns the entry on success, ``None`` on persistence failure.
    """
    if not family_key or not source_model_id:
        return None
    now = (clock or time.time)()
    registry = load_registry(repo_root)
    existing = registry.get(family_key)
    if existing is None:
        entry = ChainedTemplateEntry(
            family_key=family_key,
            template_demo_source=template_demo_source,
            source_model_id=source_model_id,
            confirmed_models=[source_model_id],
            created_at=now,
            updated_at=now,
            final_pcc=final_pcc,
            notes=notes,
        )
        registry[family_key] = entry
    else:
        entry = existing
        entry.updated_at = now
        if source_model_id not in entry.confirmed_models:
            entry.confirmed_models.append(source_model_id)
        # Update final_pcc to most-recent (informational only).
        if final_pcc is not None:
            entry.final_pcc = final_pcc
        if notes:
            entry.notes = (entry.notes + " | " + notes).strip(" |") if entry.notes else notes
    if not save_registry(registry, repo_root):
        return None
    return entry


def find_template_for_model(
    *,
    model_type: str,
    repo_root: Optional[Path] = None,
) -> Optional[ChainedTemplateEntry]:
    """Look up a template by HF ``model_type``. Returns the entry or
    None. This is the Step-1 template-reuse hook: the e2e flow asks
    "do we have a chained template for this family?" before going
    through Step-2/3 synthesis."""
    if not model_type:
        return None
    return load_registry(repo_root).get(model_type)


def list_all_templates(repo_root: Optional[Path] = None) -> List[ChainedTemplateEntry]:
    """All persisted templates, sorted by family_key. For status /
    debug commands."""
    return sorted(load_registry(repo_root).values(), key=lambda e: e.family_key)


def confirmation_count(entry: ChainedTemplateEntry) -> int:
    """Number of distinct models known to pass with this template.
    Item 7's multi-model gate compares this to a threshold (default 2)."""
    return len(set(entry.confirmed_models))


__all__ = [
    "REGISTRY_FILENAME",
    "ChainedTemplateEntry",
    "confirmation_count",
    "find_template_for_model",
    "list_all_templates",
    "load_registry",
    "register_template",
    "save_registry",
]
