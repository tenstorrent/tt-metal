"""Multi-model template promotion gate (Item 7).

A chained template is registered after the FIRST model's e2e synthesis
converges (Item 6). But one model's success doesn't prove the template
generalizes — it might be overfit to that specific model's quirks.

This module gates "trusted promotion" on a second sibling model also
passing with the same template. Until promotion, the Step-1
template-reuse layer should run LLM verify (Item 2) before each use;
after promotion, it can reuse with high confidence.

Promotion is a one-way flag — once promoted, the template stays
promoted unless a future bring-up explicitly demotes it (e.g. via a
new ``--demote-template`` flag we don't build here). This module
provides the read-side gate (is_template_promoted) plus the
write-side action (mark_promoted) used after the threshold is reached.

Reuses the existing registry; no new file or schema.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

from .family_template_registry import (
    ChainedTemplateEntry,
    confirmation_count,
    load_registry,
    save_registry,
)


# Default threshold: 2 distinct sibling models must confirm before
# promotion. Picked because one success could be overfit; two from
# different siblings is real evidence the template generalizes inside
# the family. Operators can override per-call.
DEFAULT_PROMOTION_THRESHOLD = 2


def is_template_promoted(entry: ChainedTemplateEntry) -> bool:
    """Read-side gate: has Item 7's threshold already promoted this
    template?

    Step-1 template-reuse can short-circuit verify when this returns
    True. Stays False until ``mark_promoted`` flips it (which only
    happens once the confirmation_count crosses the threshold).
    """
    return bool(entry.promoted)


def is_eligible_for_promotion(
    entry: ChainedTemplateEntry,
    threshold: int = DEFAULT_PROMOTION_THRESHOLD,
) -> bool:
    """Pure check: does this entry's confirmation_count meet the
    threshold, AND is it not already promoted?

    Returns False if already promoted (the caller would no-op
    mark_promoted anyway, but this lets callers skip the I/O).
    """
    if entry.promoted:
        return False
    return confirmation_count(entry) >= threshold


def templates_eligible_for_promotion(
    *,
    threshold: int = DEFAULT_PROMOTION_THRESHOLD,
    repo_root: Optional[Path] = None,
) -> List[ChainedTemplateEntry]:
    """All registered templates that pass ``is_eligible_for_promotion``.
    Used by batch-promotion tooling / CI runs to advance multiple
    templates at once."""
    registry = load_registry(repo_root)
    return [e for e in registry.values() if is_eligible_for_promotion(e, threshold)]


def mark_promoted(
    *,
    family_key: str,
    repo_root: Optional[Path] = None,
    clock: Optional[Any] = None,
    threshold: int = DEFAULT_PROMOTION_THRESHOLD,
) -> Optional[ChainedTemplateEntry]:
    """Write-side action: flip the ``promoted`` flag on a template
    entry, set ``promoted_at`` to now, persist.

    Idempotent: re-calling on an already-promoted entry just updates
    ``promoted_at`` (informational). Refuses to promote entries that
    don't yet meet the threshold — returns None.

    Returns the updated entry on success; ``None`` on:
      * family_key not in registry
      * confirmation_count < threshold (not yet eligible)
      * persistence failure
    """
    if not family_key:
        return None
    registry = load_registry(repo_root)
    entry = registry.get(family_key)
    if entry is None:
        return None
    if not entry.promoted and confirmation_count(entry) < threshold:
        return None
    entry.promoted = True
    entry.promoted_at = (clock or time.time)()
    if not save_registry(registry, repo_root):
        return None
    return entry


def auto_promote_after_register(
    *,
    family_key: str,
    repo_root: Optional[Path] = None,
    clock: Optional[Any] = None,
    threshold: int = DEFAULT_PROMOTION_THRESHOLD,
) -> Optional[ChainedTemplateEntry]:
    """Hook to call right after :func:`register_template` succeeds.

    Promotes the template if (and only if) the latest registration
    pushed ``confirmation_count`` over the threshold. Idempotent and
    silent when the entry isn't eligible — callers can wire this into
    every register_template call site without guarding.

    Returns the promoted entry if promotion happened, ``None``
    otherwise. The ``None`` return is informational only — it doesn't
    mean failure, just "not eligible yet."
    """
    registry = load_registry(repo_root)
    entry = registry.get(family_key)
    if entry is None:
        return None
    if not is_eligible_for_promotion(entry, threshold):
        return None
    return mark_promoted(
        family_key=family_key,
        repo_root=repo_root,
        clock=clock,
        threshold=threshold,
    )


__all__ = [
    "DEFAULT_PROMOTION_THRESHOLD",
    "auto_promote_after_register",
    "is_eligible_for_promotion",
    "is_template_promoted",
    "mark_promoted",
    "templates_eligible_for_promotion",
]
