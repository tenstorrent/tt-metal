#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Name the `ttnn` values that appear in a ledger case, for source and for prose.

A sweep grid holds live ttnn objects, and two parts of the harness need to say what one *is* rather
than hand it around: the routing-test emitter, which has to write it as Python source, and the
coverage report, which puts it in a PR body. Neither can use the repr. `ttnn.DRAM_MEMORY_CONFIG`
prints as

    MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,
    shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0,
    per_core_allocation=0)

which is not source, and is not something anyone wants to read twice in a table of strata.

Both needs are met by the same observation: these values are module-level constants of `ttnn`, so
they have names. Finding the name is a search of `ttnn`'s namespace rather than a table here, so an
op whose kwargs carry some other ttnn singleton is served without this file learning about it.
"""

from __future__ import annotations

import re

_CONSTANTS: list[tuple[str, object]] | None = None


def _constants() -> list[tuple[str, object]]:
    """Module-level `ttnn` constants, by convention the SCREAMING_CASE names.

    Restricted to that convention deliberately. Comparing a value against every attribute of `ttnn`
    would walk hundreds of operation callables and invoke `__eq__` on objects with no business being
    compared. Sorted, so a value reachable under two aliases always resolves to the same one: the
    routing test is pinned by re-rendering and comparing, and a label that moved between runs would
    detach a stratum from its own measurements.
    """
    global _CONSTANTS
    if _CONSTANTS is None:
        try:
            import ttnn
        except Exception:  # noqa: BLE001 - without ttnn there are no names to find, and callers cope
            _CONSTANTS = []
            return _CONSTANTS
        found = []
        for name in sorted(dir(ttnn)):
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                continue
            try:
                candidate = getattr(ttnn, name)
            except Exception:  # noqa: BLE001 - an attribute that raises on access is not a constant
                continue
            if not callable(candidate) and not isinstance(candidate, type):
                found.append((name, candidate))
        _CONSTANTS = found
    return _CONSTANTS


def constant_name(value) -> str | None:
    """The name of the `ttnn` constant this value is, or None if it is not one.

    Identity across all candidates first, then equality, so an exact alias always beats a merely
    equal one.
    """
    for name, candidate in _constants():
        if candidate is value:
            return name
    for name, candidate in _constants():
        try:
            if type(candidate) is type(value) and candidate == value:
                return name
        except Exception:  # noqa: BLE001 - an object that refuses comparison simply does not match
            continue
    return None


def readable(value):
    """A short, JSON-safe rendering of a ledger value.

    JSON-safe matters as much as short. This runs in `ledger.py`, where ttnn is importable, and the
    result travels in the ledger JSON to `gate.py`, where the live object does not. Both sides then
    label a stratum from the same token, which is the only reason the coverage table and the graded
    results can be joined at all.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [readable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): readable(v) for k, v in value.items()}
    return constant_name(value) or str(value)
