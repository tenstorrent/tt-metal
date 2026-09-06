"""I-3 · Append-only ledger (PLAN section 5).

One JSONL row per experiment. Never rewritten — a crash truncates at most the
last line. Appends are idempotent by `experiment_id` (resume-safe). The
"current hypothesis" is the last non-null `hypothesis` across all rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class Ledger:
    """Append-only JSONL experiment ledger."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def rows(self) -> list[dict[str, Any]]:
        """Return all rows (empty list if the file does not exist yet).

        Crash-tolerant: a truncated FINAL line (the exact artifact a crash
        mid-append leaves) is skipped. A malformed NON-final line is real
        corruption and still raises.
        """
        if not self.path.is_file():
            return []
        lines = [ln.strip() for ln in self.path.read_text().splitlines()]
        lines = [ln for ln in lines if ln]
        out: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Only forgive the last line (interrupted write); else re-raise.
                if i == len(lines) - 1:
                    break
                raise
        return out

    def append(self, row: dict[str, Any]) -> bool:
        """Append one row. No-op (returns False) if `experiment_id` exists.

        Returns True when a new line was written, False when skipped as a
        duplicate (idempotent replay).
        """
        exp_id = row.get("experiment_id")
        if exp_id is not None:
            for existing in self.rows():
                if existing.get("experiment_id") == exp_id:
                    return False
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return True

    def current_hypothesis(self) -> Any | None:
        """Last non-null `hypothesis` value across rows, else None."""
        result: Any | None = None
        for row in self.rows():
            hyp = row.get("hypothesis")
            if hyp is not None:
                result = hyp
        return result
