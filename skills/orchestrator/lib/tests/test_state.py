# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills.orchestrator.lib.state — load/save with schema validation."""

import json

import pytest

from skills.orchestrator.lib.state import SchemaError, load_state, save_state


def _valid_state() -> dict:
    """Return a minimal valid state dict that satisfies the schema."""
    return {
        "schema_version": 1,
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "model_slug": "qwen3_tts",
        "device": "n150",
        "arch_name": "wormhole_b0",
        "started_at": "2026-05-27T14:00:00Z",
        "updated_at": "2026-05-27T15:42:11Z",
        "components": [],
        "locks": {"device": {"held_by": None, "held_since": None}},
        "tick_log": [],
        "config": {
            "max_parallel_reference": 4,
            "max_attempts_per_phase": 10,
            "tick_interval_sec": 60,
        },
    }


def test_load_validates_schema_version(tmp_path):
    """A state file missing schema_version must raise SchemaError on load."""
    state = _valid_state()
    del state["schema_version"]
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))

    with pytest.raises(SchemaError):
        load_state(path)


def test_load_rejects_wrong_schema_version(tmp_path):
    """schema_version != 1 must raise SchemaError on load."""
    state = _valid_state()
    state["schema_version"] = 2
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))

    with pytest.raises(SchemaError):
        load_state(path)


def test_round_trip(tmp_path):
    """save_state then load_state yields an equal dict."""
    state = _valid_state()
    path = tmp_path / "state.json"

    save_state(path, state)
    loaded = load_state(path)

    assert loaded == state


def test_extra_field_warns_but_loads(tmp_path):
    """Unknown top-level fields emit UserWarning but the state still loads."""
    state = _valid_state()
    state["mystery_field"] = "hello"
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))

    with pytest.warns(UserWarning):
        loaded = load_state(path)

    assert loaded["mystery_field"] == "hello"


def test_atomic_write_no_partial_file(tmp_path):
    """save_state leaves no <path>.tmp behind and writes valid JSON."""
    state = _valid_state()
    path = tmp_path / "state.json"

    save_state(path, state)

    # No leftover temp file.
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"unexpected leftover tmp files: {tmp_files}"

    # Final file is valid JSON matching the saved state.
    with open(path, "r", encoding="utf-8") as f:
        parsed = json.load(f)
    assert parsed == state


def test_save_validates_before_writing(tmp_path):
    """An invalid state must raise SchemaError and not create the target file."""
    state = _valid_state()
    del state["model_id"]
    path = tmp_path / "state.json"

    with pytest.raises(SchemaError):
        save_state(path, state)

    assert not path.exists(), "save_state must not create the file when validation fails"
    # And no leftover tmp either.
    assert list(tmp_path.glob("*.tmp")) == []
