# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Opt-in call tracing for the Gemma text demo / CI token-matching path.

Enable with: export SAKTHI_DEBUG_TRACE=1

Uses the standard library ``logging`` logger named ``sakthi_debug`` (not print).
Each logical method id is logged at most once per process; at the end of a test
call ``sakthi_debug_summary`` for total unique methods and the ordered list.
"""

from __future__ import annotations

import logging
import os
from typing import List, Set

_LOGGER = logging.getLogger("sakthi_debug")

_ENABLED = os.environ.get("SAKTHI_DEBUG_TRACE", "").strip().lower() in ("1", "true", "yes", "y")

_seen_ids: Set[str] = set()
_first_seen_order: List[str] = []


def sakthi_debug_is_enabled() -> bool:
    return _ENABLED


def sakthi_debug_reset() -> None:
    """Clear dedupe state (call once at the start of a test)."""
    global _seen_ids, _first_seen_order
    _seen_ids = set()
    _first_seen_order = []


def sakthi_debug_log_once(method_id: str) -> None:
    """
    Log a single line the first time ``method_id`` is seen in this process.

    ``method_id`` should be stable (e.g. ``Transformer.ttnn_decode_forward``).
    """
    if not _ENABLED:
        return
    if method_id in _seen_ids:
        return
    _seen_ids.add(method_id)
    _first_seen_order.append(method_id)
    _LOGGER.warning("sakthi-debug method=%s unique_index=%d", method_id, len(_first_seen_order))


def sakthi_debug_summary(label: str = "") -> None:
    """Emit total unique methods and full ordered list (each method appeared once in the list)."""
    if not _ENABLED:
        return
    suffix = f" {label}" if label else ""
    _LOGGER.warning(
        "sakthi-debug TOTAL_UNIQUE_METHODS=%d%s",
        len(_seen_ids),
        suffix,
    )
    _LOGGER.warning("sakthi-debug METHOD_ORDER=%s", " | ".join(_first_seen_order))
