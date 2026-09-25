"""I-2 · Checkpoint with WAL semantics (PLAN section 5).

The checkpoint is the single live control file (`state.json`). Writes are
atomic (via atomic_write). WAL ordering for side-effecting states:

    mark_intent(...)  ->  do the side effect  ->  mark_done()

If a crash happens between intent and done, `is_in_flight()` reports True on
resume so the engine knows to roll back (e.g. `git reset --hard git_sha_clean`)
and re-apply.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .atomic import atomic_write

_IN_FLIGHT = "_in_flight"


class Checkpoint:
    """Durable, atomically-written state checkpoint."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def save(self, state: dict[str, Any]) -> None:
        """Atomically persist the full state dict."""
        atomic_write(self.path, json.dumps(state, indent=2, sort_keys=True))

    def load(self) -> dict[str, Any]:
        """Load the recorded state dict (raises if absent)."""
        return json.loads(self.path.read_text())

    def mark_intent(self, **fields: Any) -> dict[str, Any]:
        """Record intent before a side effect: merge `fields`, set in-flight.

        Typical fields for APPLY: current_lever, git_sha_clean, state.
        """
        state = self.load() if self.exists() else {}
        state.update(fields)
        state[_IN_FLIGHT] = True
        self.save(state)
        return state

    def mark_done(self) -> dict[str, Any]:
        """Clear the in-flight marker after the side effect completed."""
        state = self.load()
        state[_IN_FLIGHT] = False
        self.save(state)
        return state

    def is_in_flight(self) -> bool:
        """True if intent was recorded but the matching done was not."""
        if not self.exists():
            return False
        return bool(self.load().get(_IN_FLIGHT, False))
