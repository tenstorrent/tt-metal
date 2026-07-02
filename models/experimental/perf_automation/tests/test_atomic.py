"""I-1 tests: atomic_write is all-or-nothing (PLAN section 5)."""

import os

import pytest

from agent.atomic import atomic_write


def test_atomic_write_roundtrip(tmp_path):
    target = tmp_path / "state.json"
    atomic_write(target, '{"k": 1}')
    assert target.read_text() == '{"k": 1}'
    # Accepts bytes too.
    atomic_write(target, b"bytes")
    assert target.read_bytes() == b"bytes"


def test_atomic_write_all_or_nothing(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text("ORIGINAL")

    def boom(src, dst):
        raise OSError("replace failed after tmp written")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write(target, "NEW CONTENT")

    # Target unchanged ...
    assert target.read_text() == "ORIGINAL"
    # ... and no tmp artifact left behind.
    assert not (tmp_path / "state.json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []
