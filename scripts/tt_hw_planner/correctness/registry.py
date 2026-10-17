"""Category → :class:`Comparator` registry.

A single process-global registry maps category strings to the
comparator instance that handles them. Population is done by the
side-effect of importing the per-category module (e.g. importing
``correctness.segmentation`` will register the segmentation
comparator), so the dispatcher only sees comparators for the
categories whose modules have been loaded.

Why a registry instead of an ``if/elif`` ladder
-----------------------------------------------
The audit identified ten distinct categories (LLM, VLM, STT, TTS,
CNN, Image, Embed, NLP, Video, Unknown) plus future ones
(detection, depth-estimation, …). Hard-coding the dispatch in
``cli.py`` would (a) make every new category a cli.py edit
(invariably touching 14k lines), and (b) couple the engine to
LLM-specific assumptions.

The registry pattern lets:

* :func:`register_comparator` be called from the per-category
  module at import time.
* New categories ship as one new file each (the per-category
  module) plus one line in :mod:`__init__` to import it.
* Tests register temporary comparators for fixture scenarios
  without monkey-patching globals.

Thread-safety
-------------
The registry is mutable but the planner is single-threaded.
Register-time happens once at process start (via module imports);
lookup-time is read-only. If the planner ever goes
multi-threaded, the lock at the bottom of this module covers
mutation; reads remain lock-free.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .base import Comparator


_LOCK = threading.RLock()


_REGISTRY: Dict[str, List[Comparator]] = {}


def register_comparator(comparator: Comparator) -> None:
    """Add ``comparator`` to the global registry under
    ``comparator.category``.

    Idempotent: registering the same instance twice is a no-op.
    Registering two different instances under the same category
    appends both (resolution falls through ``supports()``).
    """
    if not isinstance(comparator, Comparator):
        raise TypeError(f"register_comparator expected a Comparator, got " f"{type(comparator).__name__}")
    if not comparator.category:
        raise ValueError(f"comparator {type(comparator).__name__} has no " f"non-empty .category; refusing to register")
    with _LOCK:
        bucket = _REGISTRY.setdefault(comparator.category, [])
        if comparator in bucket:
            return
        bucket.append(comparator)


def get_comparator(
    category: str,
    model_id: str = "",
) -> Optional[Comparator]:
    """Return the first registered comparator whose ``supports``
    method accepts ``(category, model_id)``, or ``None`` if no
    comparator claims this category.

    ``model_id`` defaults to ``""`` so callers that don't have one
    yet still get coarse category-level dispatch.
    """
    with _LOCK:
        bucket = list(_REGISTRY.get(category, ()))
    for c in bucket:
        try:
            if c.supports(category, model_id):
                return c
        except Exception:
            continue
    return None


def list_categories() -> List[str]:
    """Return the sorted list of categories with at least one
    registered comparator. Used by ``--pcc-categories`` for
    validation and by tests."""
    with _LOCK:
        return sorted(_REGISTRY.keys())


def _reset_registry_for_testing() -> None:
    """Clear the registry. **Test-only.** Production code should
    never call this; we expose it so tests can build fixtures
    without polluting the global state of sibling tests."""
    with _LOCK:
        _REGISTRY.clear()


__all__ = [
    "register_comparator",
    "get_comparator",
    "list_categories",
]
