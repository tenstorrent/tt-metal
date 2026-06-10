"""Bridge from the orchestrator to the opt_transfer knowledge base.

Selects mined KBEntry records relevant to a bring-up block so tick.md can
inject them into ttnn/optimization worker specs. Deterministic keyword
ranking — no LLM, no device. Tolerates a missing or empty store (returns
[]), so single-device and pre-mine bring-ups are unaffected.

Import of the opt_transfer schema/store is lazy and guarded: this module
stays importable (and returns []) even if opt_transfer is absent.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_KB_DIR = Path(__file__).resolve().parents[3] / "models/experimental/opt_transfer/kb/records"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(*texts) -> set[str]:
    out: set[str] = set()
    for t in texts:
        # Mined records aren't schema-perfect: fused_op/category may be lists or None.
        if isinstance(t, (list, tuple)):
            t = " ".join(str(x) for x in t)
        out.update(_TOKEN_RE.findall(str(t or "").lower()))
    return out


# Generic tokens that would match almost every entry and carry no signal.
_STOPWORDS = {"ttnn", "op", "ops", "layer", "block", "model", "the", "a", "of", "and", "to"}


def kb_entries_for_block(block_name: str, kind: str, max_entries: int = 8, kb_dir=None) -> list[dict]:
    """Return up to `max_entries` KB entry dicts ranked by keyword overlap with the block.

    Ranking: overlap between the block's name/kind tokens and each entry's
    id/category/fused_op/torch_pattern/applicability_notes tokens. Entries
    with zero overlap are dropped. High-confidence entries win ties.
    """
    try:
        from models.experimental.opt_transfer.kb.store import KBStore
    except Exception:
        return []

    root = Path(kb_dir) if kb_dir else DEFAULT_KB_DIR
    if not root.is_dir():
        return []
    try:
        entries = KBStore(root).load()
    except Exception:
        return []

    query = _tokens(block_name, kind) - _STOPWORDS
    if not query:
        return []

    scored = []
    for e in entries:
        hay = _tokens(
            e.id,
            e.category,
            e.fused_op,
            " ".join(e.torch_pattern or []),
            e.applicability_notes or "",
        )
        score = len(query & hay)
        if score > 0:
            scored.append((score, 0 if e.confidence == "high" else 1, e.id, e))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [e.to_dict() for _, _, _, e in scored[:max_entries]]
