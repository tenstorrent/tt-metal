# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""State load/save with schema validation for the bringup orchestrator.

The state file is the source of truth for orchestrator progress; see
`skills/orchestrator/SPEC.md` §State schema. This module provides:

- `SchemaError`: raised on missing or invalid required keys.
- `load_state(path)`: read + validate JSON.
- `save_state(path, state)`: validate + atomic write.
- `bootstrap(model_id, device, arch_name)`: build a fresh skeleton state.
- `resume_normalize(state)`: make a state safe to resume after a crash.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from datetime import datetime, timezone
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

# The four phase keys present on each component (per SPEC.md). Used by
# resume_normalize to demote in_progress workers whose owning session is gone.
_PHASE_NAMES = ("reference", "ttnn", "debug", "optimization")

# Default tunables baked into a fresh state by bootstrap(). Deep-copied on
# every bootstrap call so the returned state owns its own dict.
_DEFAULT_CONFIG = {
    "max_parallel_reference": 4,
    "max_attempts_per_phase": 10,
    "tick_interval_sec": 60,
}


class SchemaError(Exception):
    """Raised when a state dict fails schema validation."""


def _validate(state: dict) -> None:
    """Validate top-level required keys and schema_version.

    - Raises ``SchemaError`` listing all missing required keys (sorted, so the
      error message is deterministic).
    - Raises ``SchemaError`` if ``schema_version`` is not ``_SCHEMA_VERSION``.
    - Emits a ``UserWarning`` (does not raise) for unknown top-level keys.
    """
    missing = sorted(_REQUIRED_KEYS - state.keys())
    if missing:
        raise SchemaError(f"missing required keys: {missing}")

    version = state["schema_version"]
    if version != _SCHEMA_VERSION:
        raise SchemaError(f"schema_version must be {_SCHEMA_VERSION}, got {version!r}")

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

    # Construct tmp via parent + name (not Path.with_suffix, which raises
    # ValueError on paths with no suffix or trailing dots).
    tmp = p.parent / (p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except BaseException:
        # On any failure during the JSON write, remove the partial tmp file
        # so it cannot be mistaken for a valid state later. The final path
        # is untouched because os.replace has not run yet.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    os.replace(tmp, p)


def _utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 with a trailing 'Z' (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def bootstrap(model_id: str, device: str, arch_name: str) -> dict:
    """Create a fresh state dict for a new model bring-up.

    The returned dict passes ``_validate()``. The caller is responsible for
    persisting it (e.g. via ``save_state``); ``bootstrap`` itself does no I/O.

    ``model_slug`` is derived from ``model_id`` by lowercasing and replacing
    ``/`` and ``-`` with ``_``. Other characters (including ``.``) are kept
    so versioned model identifiers like ``...-1.7B-Base`` remain readable.
    """
    slug = model_id.lower().replace("/", "_").replace("-", "_")
    now = _utc_now_iso()
    return {
        "schema_version": _SCHEMA_VERSION,
        "model_id": model_id,
        "model_slug": slug,
        "device": device,
        "arch_name": arch_name,
        "started_at": now,
        "updated_at": now,
        "components": [],
        "locks": {"device": {"held_by": None, "held_since": None}},
        "tick_log": [],
        "config": copy.deepcopy(_DEFAULT_CONFIG),
    }


def resume_normalize(state: dict) -> dict:
    """Make ``state`` safe to resume after a session crash or ``/clear``.

    The worker that owned any ``in_progress`` phase is, by definition, gone
    by the time we're resuming — its work cannot be assumed complete. Demote
    every such phase to ``pending`` so the next tick re-dispatches it. Also
    drop any held device lock for the same reason.

    Mutates ``state`` in place and returns it (so callers can chain
    ``state = resume_normalize(load_state(path))``).
    """
    for component in state.get("components", []):
        for phase in _PHASE_NAMES:
            phase_dict = component.get(phase)
            if isinstance(phase_dict, dict) and phase_dict.get("status") == "in_progress":
                phase_dict["status"] = "pending"

    device_lock = state.get("locks", {}).get("device")
    if isinstance(device_lock, dict) and device_lock.get("held_by") is not None:
        device_lock["held_by"] = None
        device_lock["held_since"] = None

    return state
