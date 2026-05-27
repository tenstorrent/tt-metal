# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""State load/save with schema validation for the bringup orchestrator.

The state file is the source of truth for orchestrator progress; see
`skills/orchestrator/SPEC.md` §State schema. This module provides:

- `SchemaError`: raised on missing or invalid required keys.
- `load_state(path)`: read + validate JSON.
- `save_state(path, state)`: validate + atomic write.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# Top-level required keys per SPEC.md §State schema. Deeper validation of nested
# structures (components, locks contents, etc.) is intentionally deferred to
# future tasks.
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_slug",
        "device",
        "arch_name",
        "started_at",
        "updated_at",
        "components",
        "locks",
        "tick_log",
        "config",
    }
)

_SCHEMA_VERSION = 1


class SchemaError(Exception):
    """Raised when a state dict fails schema validation."""


def _validate(state: dict) -> None:
    """Validate top-level required keys and schema_version.

    - Raises ``SchemaError`` for any missing required key.
    - Raises ``SchemaError`` if ``schema_version`` is not 1.
    - Emits a ``UserWarning`` (does not raise) for unknown top-level keys.
    """
    for key in _REQUIRED_KEYS:
        if key not in state:
            raise SchemaError(f"missing key: {key}")

    version = state["schema_version"]
    if version != _SCHEMA_VERSION:
        raise SchemaError(f"schema_version must be 1, got {version!r}")

    extras = set(state.keys()) - _REQUIRED_KEYS
    if extras:
        warnings.warn(
            f"unknown top-level state fields: {sorted(extras)}",
            UserWarning,
            stacklevel=2,
        )


def load_state(path: PathLike) -> dict:
    """Load a state file from JSON and validate it.

    Raises ``FileNotFoundError`` if the file does not exist (default open
    behavior) and ``SchemaError`` if the loaded dict fails validation.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        state = json.load(f)
    _validate(state)
    return state


def save_state(path: PathLike, state: dict) -> None:
    """Validate ``state`` and atomically write it as JSON to ``path``.

    Atomicity is achieved by writing to ``<path>.tmp`` first, then
    ``os.replace`` onto the final path. Parent directories are created
    if they do not exist. If validation fails the target file is not
    created or modified.
    """
    _validate(state)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, p)
