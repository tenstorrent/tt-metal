"""I-1 · atomic_write — all-or-nothing file replace (PLAN section 5).

Write to a sibling `<path>.tmp`, fsync, then os.replace() onto the target so a
crash can never leave a half-written file at `path`. On any failure the tmp
file is removed and the original target is left untouched.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write(path: str | os.PathLike[str], data: str | bytes) -> None:
    """Atomically write `data` to `path` (write tmp, fsync, os.replace).

    Guarantees: the target is either the old content or the new content, never
    a partial write; no `.tmp` artifact is left behind on failure.
    """
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")

    tmp = Path(str(path) + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Failure (including os.replace raising): never leave a tmp behind.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
