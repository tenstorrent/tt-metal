# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LEAN_PY (P.66, tt/settings.py): the per-layer Python glue trimmed (host only; the same ops, the same order, the
same bytes). ``ON[0]`` is read on the hot path (one list index, no settings lookup); ``configure`` sets it at model
load from the setting, ``set_mode`` switches it in a loaded model (in-process A/B tests)."""

from __future__ import annotations

from .settings import settings

ON = [True]


def configure() -> None:
    """Model load: the setting's value."""
    ON[0] = settings.get("LEAN_PY") == "on"


def set_mode(mode: str) -> None:
    """Tests: switch the loaded model's glue (``legacy`` | ``on``)."""
    assert mode in ("legacy", "on"), mode
    ON[0] = mode == "on"
