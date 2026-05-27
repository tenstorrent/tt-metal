# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills.orchestrator.lib.state — load/save with schema validation."""

import json
import re

import pytest

from skills.orchestrator.lib.state import (
    SchemaError,
    bootstrap,
    load_state,
    resume_normalize,
    save_state,
)


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


def test_load_rejects_missing_schema_version(tmp_path):
    """A state file missing schema_version must raise SchemaError on load."""
    state = _valid_state()
    del state["schema_version"]
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))

    with pytest.raises(SchemaError):
        load_state(path)


def test_validate_reports_all_missing_keys(tmp_path):
    """SchemaError message lists every missing key, not just the first."""
    state = _valid_state()
    del state["model_id"]
    del state["device"]
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))

    with pytest.raises(SchemaError) as excinfo:
        load_state(path)

    msg = str(excinfo.value)
    assert "model_id" in msg
    assert "device" in msg


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


def test_save_torn_write_preserves_previous_file(tmp_path, monkeypatch):
    """If json.dump fails mid-write, the previously committed file is intact."""
    p = tmp_path / "state.json"
    # Establish a known-good baseline file.
    baseline = _valid_state()
    save_state(p, baseline)

    # Make the next dump raise.
    import json as _json

    def boom(*a, **kw):
        raise IOError("disk full")

    monkeypatch.setattr(_json, "dump", boom)

    new = _valid_state()
    new["model_id"] = "different/model"
    with pytest.raises(IOError):
        save_state(p, new)

    # Final path still contains the baseline — the failed write did not corrupt it.
    assert load_state(p) == baseline
    # No leftover tmp.
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_creates_parent_dirs(tmp_path):
    """save_state creates intermediate parent directories as needed."""
    p = tmp_path / "deep" / "nested" / "state.json"
    save_state(p, _valid_state())
    assert p.exists()


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_creates_skeleton():
    """bootstrap returns a fully-formed state dict with the right defaults."""
    state = bootstrap("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "n150", "wormhole_b0")

    assert state["schema_version"] == 1
    assert state["model_id"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    # Only '/' and '-' are replaced with '_'. '.' is preserved.
    assert state["model_slug"] == "qwen_qwen3_tts_12hz_1.7b_base"
    assert state["device"] == "n150"
    assert state["arch_name"] == "wormhole_b0"

    assert state["components"] == []
    assert state["locks"]["device"]["held_by"] is None
    assert state["locks"]["device"]["held_since"] is None
    assert state["tick_log"] == []

    assert state["config"]["max_parallel_reference"] == 4
    assert state["config"]["max_attempts_per_phase"] == 10
    assert state["config"]["tick_interval_sec"] == 60

    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    assert iso_re.match(state["started_at"]), state["started_at"]
    assert state["started_at"] == state["updated_at"]


def test_bootstrap_passes_validate(tmp_path):
    """A freshly bootstrapped state must satisfy the schema (save_state validates)."""
    state = bootstrap("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "n150", "wormhole_b0")
    # save_state runs _validate; if bootstrap is wrong this raises SchemaError.
    save_state(tmp_path / "s.json", state)


# ---------------------------------------------------------------------------
# resume_normalize
# ---------------------------------------------------------------------------


def test_resume_normalize_demotes_in_progress():
    """in_progress phases become pending; other statuses are untouched."""
    state = bootstrap("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "n150", "wormhole_b0")
    state["components"].append(
        {
            "name": "Attention",
            "reference": {"status": "done"},
            "ttnn": {"status": "in_progress"},
            "debug": {"status": "in_progress"},
            "optimization": {"status": "pending"},
        }
    )

    resume_normalize(state)

    comp = state["components"][0]
    assert comp["reference"]["status"] == "done"
    assert comp["ttnn"]["status"] == "pending"
    assert comp["debug"]["status"] == "pending"
    assert comp["optimization"]["status"] == "pending"


def test_resume_normalize_clears_device_lock():
    """A held device lock is released by resume_normalize."""
    state = bootstrap("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "n150", "wormhole_b0")
    state["locks"]["device"] = {
        "held_by": "tick-42",
        "held_since": "2026-05-27T15:00:00Z",
    }

    resume_normalize(state)

    assert state["locks"]["device"]["held_by"] is None
    assert state["locks"]["device"]["held_since"] is None


def test_resume_normalize_handles_missing_phases():
    """Components missing some phase keys are left alone (no KeyError)."""
    state = bootstrap("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "n150", "wormhole_b0")
    state["components"].append(
        {
            "name": "Partial",
            "reference": {"status": "in_progress"},
            # ttnn / debug / optimization intentionally missing
        }
    )

    resume_normalize(state)

    comp = state["components"][0]
    # The one present phase was in_progress, so it should now be pending.
    assert comp["reference"]["status"] == "pending"
    # Missing keys are not synthesized.
    assert "ttnn" not in comp
    assert "debug" not in comp
    assert "optimization" not in comp


def test_resume_normalize_returns_same_object():
    """resume_normalize mutates in place and returns the same dict (for chaining)."""
    state = bootstrap("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "n150", "wormhole_b0")
    out = resume_normalize(state)
    assert out is state
