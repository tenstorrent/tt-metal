"""Evidence records a serving run leaves behind: phase markers (append-only JSONL) and the one result document."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

Marker = Callable[[str], None]
PhaseRecord = Callable[[Mapping[str, Any]], None]

_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("write made no progress")
        view = view[written:]


def _write_synced(path: Path, payload: bytes, flags: int) -> None:
    descriptor = os.open(path, flags, 0o600)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_phase_record(path: Path, record: Mapping[str, Any]) -> None:
    if type(record) is not dict or "utc" in record:
        raise ValueError("phase record must be an exact JSON object without a caller UTC field")
    phase = record.get("phase")
    if not isinstance(phase, str) or not phase or len(phase) > 256:
        raise ValueError(f"invalid phase marker: {phase!r}")
    payload = (json.dumps({"utc": utc_now(), **record}, sort_keys=True) + "\n").encode()
    _write_synced(path, payload, _OPEN_FLAGS | os.O_APPEND)


def append_marker(path: Path, phase: str) -> None:
    append_phase_record(path, {"phase": phase})


def write_result(path: Path, document: Mapping[str, Any]) -> None:
    """The result document is written once (``O_EXCL``): a second write to the same path is a bug."""

    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    _write_synced(path, payload, _OPEN_FLAGS | os.O_EXCL)
