from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from .compatibility import BUILDING_BLOCKS, BuildingBlock, Effort, Status


@dataclass(frozen=True)
class ReuseEntry:
    model_types: Tuple[str, ...]
    hf_class_pattern: re.Pattern
    concept: str
    tt_path: str
    tt_class: str
    status: str
    notes: str = ""

    def matches(self, model_type: Optional[str], hf_class_name: Optional[str]) -> bool:
        if not hf_class_name:
            return False
        if self.model_types and (not model_type or model_type.lower() not in {m.lower() for m in self.model_types}):
            return False
        return bool(self.hf_class_pattern.match(hf_class_name))


def _effort_to_status(effort: Effort, status: Status) -> str:
    """Map (effort, support) to a component status. Every category emits a stub +
    PCC test and goes through the gates. MISSING -> NEW (write from scratch);
    SUPPORTED+DROP_IN -> REUSE (iter-0 tries the existing module as-is; demoted to
    ADAPT after the first attempt if the PCC/native gate fails); SUPPORTED+other
    -> ADAPT (wrap + refine)."""
    if status == Status.MISSING:
        return "NEW"
    if effort == Effort.DROP_IN and status == Status.SUPPORTED:
        return "REUSE"
    return "ADAPT"


def _concept_from_block_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


def _derive_entry(block: BuildingBlock) -> Optional[ReuseEntry]:
    if not block.class_name_pattern or not block.tt_class:
        return None
    tt_path = block.registry_tt_path or block.tt_path
    if not tt_path:
        return None
    if any(token in tt_path for token in (" (wraps", " + ", "{", "}", "ttnn.")):
        return None
    status = _effort_to_status(block.effort_when_needed, block.status_when_needed)
    if status == "NEW":
        return None
    return ReuseEntry(
        model_types=tuple(block.model_type_keys or ()),
        hf_class_pattern=re.compile(block.class_name_pattern),
        concept=_concept_from_block_name(block.name),
        tt_path=tt_path,
        tt_class=block.tt_class,
        status=status,
        notes=(f"derived from compatibility.py BUILDING_BLOCKS '{block.name}'. " f"{block.notes}").strip(),
    )


@lru_cache(maxsize=1)
def _derived_entries() -> Tuple[ReuseEntry, ...]:
    out: List[ReuseEntry] = []
    for block in BUILDING_BLOCKS:
        entry = _derive_entry(block)
        if entry is not None:
            out.append(entry)
    return tuple(out)


_HANDWRITTEN_COMPOSITES: List[ReuseEntry] = [
    ReuseEntry(
        model_types=("qwen3", "qwen3_embedding"),
        hf_class_pattern=re.compile(r"^Qwen3DecoderLayer$"),
        concept="decoder_layer",
        tt_path="models/tt_transformers/tt/decoder.py",
        tt_class="TransformerBlock",
        status="ADAPT",
        notes="Composite Attention+MLP+RMSNorm block; tt_transformers TransformerBlock is the template (presumed adaptable — runs canonical, LLM refines via iterate loop if per-component PCC < 0.99).",
    ),
    ReuseEntry(
        model_types=("qwen3", "qwen3_embedding"),
        hf_class_pattern=re.compile(r"^Qwen3Model$"),
        concept="model_root",
        tt_path="models/tt_transformers/tt/model.py",
        tt_class="Transformer",
        status="ADAPT",
        notes="Top-level decoder stack; tt_transformers/tt/model.py is the template (presumed adaptable — runs canonical, LLM refines via iterate loop if per-component PCC < 0.99).",
    ),
]


@lru_cache(maxsize=1)
def _overlay_entries() -> Tuple[ReuseEntry, ...]:
    """Reuse targets auto-derived from the synced upstream tree (fixes-plan
    Point 2a), loaded as the LOWEST-priority supplement so curated entries always
    win. These let a component wrap an already-implemented upstream module rather
    than write it from scratch; all are status ADAPT (wrapped + PCC-gated, never
    trusted) and participate only in concept lookup (a never-matching hf_class
    pattern keeps them out of the hf-class ``lookup``). Empty if no overlay."""
    out: List[ReuseEntry] = []
    try:
        from .registry_sync import load_generated_overlay

        never = re.compile(r"(?!x)x")
        for c in load_generated_overlay().get("concepts", []):
            concept = c.get("concept")
            tt_path = c.get("tt_path")
            tt_class = c.get("tt_class")
            if not (concept and tt_path and tt_class):
                continue
            out.append(
                ReuseEntry(
                    model_types=tuple(c.get("model_types") or ()),
                    hf_class_pattern=never,
                    concept=_concept_from_block_name(concept),
                    tt_path=tt_path,
                    tt_class=tt_class,
                    status=c.get("status", "ADAPT"),
                    notes="auto-derived from upstream tree (fixes-plan Point 2a); ADAPT => wrapped + PCC-gated, not trusted.",
                )
            )
    except Exception:
        return tuple()
    return tuple(out)


def lookup(model_type: Optional[str], hf_class_name: Optional[str]) -> Optional[ReuseEntry]:
    if not hf_class_name:
        return None
    for entry in _derived_entries():
        if entry.matches(model_type, hf_class_name):
            return entry
    for entry in _HANDWRITTEN_COMPOSITES:
        if entry.matches(model_type, hf_class_name):
            return entry
    return None


_CONCEPT_STOPWORDS = frozenset({"self", "cross", "std", "standard", "text", "type"})


def _normalize_concept_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _concept_tokens(s: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and t not in _CONCEPT_STOPWORDS}


def _concept_overlap(target: str, candidate: str) -> bool:
    t = _normalize_concept_token(target)
    c = _normalize_concept_token(candidate)
    if t and c and (t == c or t in c or c in t):
        return True
    return bool(_concept_tokens(target) & _concept_tokens(candidate))


def _entry_allows_model_type(entry: ReuseEntry, model_type: Optional[str]) -> bool:
    if not entry.model_types:
        return True
    if not model_type:
        return False
    return model_type.lower() in {m.lower() for m in entry.model_types}


def lookup_by_concept(model_type: Optional[str], concept: str) -> Optional[ReuseEntry]:
    if not concept:
        return None
    for entry in _derived_entries():
        if _concept_overlap(concept, entry.concept) and _entry_allows_model_type(entry, model_type):
            return entry
    for entry in _HANDWRITTEN_COMPOSITES:
        if _concept_overlap(concept, entry.concept) and _entry_allows_model_type(entry, model_type):
            return entry
    for entry in _overlay_entries():
        if _concept_overlap(concept, entry.concept) and _entry_allows_model_type(entry, model_type):
            return entry
    return None


def _synthetic_class_for_concept(concept: str) -> str:
    parts = [p for p in concept.split("_") if p]
    return "".join(p.capitalize() for p in parts)


def all_entries() -> List[ReuseEntry]:
    return list(_derived_entries()) + list(_HANDWRITTEN_COMPOSITES) + list(_overlay_entries())


def entries_for_model_type(model_type: Optional[str]) -> List[ReuseEntry]:
    if not model_type:
        return []
    needle = model_type.lower()
    out: List[ReuseEntry] = []
    for entry in all_entries():
        if not entry.model_types:
            out.append(entry)
        elif needle in {m.lower() for m in entry.model_types}:
            out.append(entry)
    return out
