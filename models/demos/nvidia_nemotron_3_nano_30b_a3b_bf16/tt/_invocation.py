# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tiny process-global registry recording which graduated stubs actually ran.

Each graduated stub calls `record("<name>")` at the top of its forward. The e2e
test resets the registry, runs the pipeline, and asserts every graduated module
appears (Gate 2 — proven by real execution, not by the caller's optimism)."""
from __future__ import annotations

INVOKED: set[str] = set()


def record(name: str) -> None:
    INVOKED.add(name)


def reset() -> None:
    INVOKED.clear()


def snapshot() -> set[str]:
    return set(INVOKED)
