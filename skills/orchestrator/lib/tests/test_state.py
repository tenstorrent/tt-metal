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
    redo,
    render_log,
    resume_normalize,
    save_state,
    skip,
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
    """in_progress phases become pending; other statuses are preserved.

    The non-in_progress phases here intentionally span done / failing so the
    "unchanged" assertions actually prove preservation rather than re-asserting
    the input (which would be tautological if a phase started as ``pending``).
    """
    state = bootstrap("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "n150", "wormhole_b0")
    state["components"].append(
        {
            "name": "Attention",
            "reference": {"status": "done"},
            "ttnn": {"status": "in_progress"},
            "debug": {"status": "failing"},
            "optimization": {"status": "done"},
        }
    )

    resume_normalize(state)

    comp = state["components"][0]
    # Demoted.
    assert comp["ttnn"]["status"] == "pending"
    # Preserved — not in_progress, not pending.
    assert comp["reference"]["status"] == "done"
    assert comp["debug"]["status"] == "failing"
    assert comp["optimization"]["status"] == "done"


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


# ---------------------------------------------------------------------------
# render_log
# ---------------------------------------------------------------------------


def test_render_log_matches_golden():
    """A representative state renders to a known markdown document, byte-for-byte.

    This is the load-bearing renderer test: it pins down the exact column layout,
    cell-formatting rules (em dash for missing, 6-decimal PCC, last_error vs
    notes selection), and section ordering. The other render_log tests cover
    individual edge cases; this one fixes the overall shape.
    """
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    # Pin timestamps so the golden is deterministic across machines.
    state["started_at"] = "2026-05-27T14:00:00Z"
    state["updated_at"] = "2026-05-27T14:00:00Z"
    state["components"] = [
        {
            "name": "RMSNorm",
            "kind": "norm",
            "reference_impl": "models/common/rmsnorm.py",
            "depends_on": [],
            "reference": {"status": "done", "pcc": 0.999998, "attempts": 1, "notes": "matches HF"},
            "ttnn": {"status": "done", "pcc": 0.999985, "attempts": 1},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
        {
            "name": "Attention",
            "kind": "attention",
            "reference_impl": "models/demos/llama3_70b_galaxy/tt/llama_attention.py",
            "depends_on": ["RMSNorm"],
            "host_resident": {"allowed": False, "justification": None, "reference_link": None},
            "reference": {"status": "done", "pcc": 0.9999, "attempts": 1},
            "ttnn": {"status": "failing", "pcc": 0.81, "attempts": 3, "last_error": "QK-norm mismatch"},
            "debug": {"status": "in_progress", "attempts": 1, "notes": "K matrix amplified by k_norm"},
            "optimization": {"status": "blocked", "blocked_on": "ttnn"},
        },
    ]
    state["tick_log"] = [
        {"tick": 1, "ts": "2026-05-27T14:00:10Z", "action": "architecture", "result": "ok"},
        {"tick": 2, "ts": "2026-05-27T14:01:00Z", "action": "reference[RMSNorm,Attention]", "result": "ok"},
    ]

    golden = (
        "# BRINGUP LOG: Acme/Foo-1B\n"
        "\n"
        "**Model:** `Acme/Foo-1B`\n"
        "**Slug:** `acme_foo_1b`\n"
        "**Target Device:** n150 (wormhole_b0)\n"
        "**Started:** 2026-05-27T14:00:00Z\n"
        "**Updated:** 2026-05-27T14:00:00Z\n"
        "\n"
        "## Block Status\n"
        "\n"
        "| Block | Phase | Status | PCC | Attempts | Notes |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        "| RMSNorm | reference | done | 0.999998 | 1 | matches HF |\n"
        "| RMSNorm | ttnn | done | 0.999985 | 1 |  |\n"
        "| RMSNorm | debug | n/a | — | 0 |  |\n"
        "| RMSNorm | optimization | pending | — | 0 |  |\n"
        "| Attention | reference | done | 0.999900 | 1 |  |\n"
        "| Attention | ttnn | failing | 0.810000 | 3 | QK-norm mismatch |\n"
        "| Attention | debug | in_progress | — | 1 | K matrix amplified by k_norm |\n"
        "| Attention | optimization | blocked | — | 0 |  |\n"
        "\n"
        "## Recent Ticks\n"
        "\n"
        "- tick 1 (2026-05-27T14:00:10Z): architecture — ok\n"
        "- tick 2 (2026-05-27T14:01:00Z): reference[RMSNorm,Attention] — ok\n"
        "\n"
        "## Host-Resident Exceptions\n"
        "\n"
        "_None._\n"
    )

    actual = render_log(state)
    if actual != golden:
        # Helpful diff output when iterating on the renderer.
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                golden.splitlines(),
                actual.splitlines(),
                fromfile="golden",
                tofile="actual",
                lineterm="",
            )
        )
        raise AssertionError(f"render_log output diverged from golden:\n{diff}")


def test_render_log_handles_missing_phases():
    """A component with only `reference` set still produces one row per phase.

    Missing phase keys render as em dash for status/pcc and 0 for attempts.
    Every PHASE_NAMES entry must appear exactly once in the table for the
    component.
    """
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["components"] = [
        {
            "name": "PartialBlock",
            "reference": {"status": "done", "pcc": 0.999, "attempts": 1},
            # ttnn / debug / optimization intentionally absent
        }
    ]

    output = render_log(state)

    # Each phase name appears exactly once (in its own row).
    for phase in ("reference", "ttnn", "debug", "optimization"):
        rows_with_phase = [
            ln for ln in output.splitlines() if ln.startswith("| PartialBlock |") and f"| {phase} |" in ln
        ]
        assert len(rows_with_phase) == 1, f"phase {phase!r} should appear in exactly one row; got {rows_with_phase!r}"

    # Missing phases render em dash / 0.
    assert "| PartialBlock | ttnn | — | — | 0 |  |" in output
    assert "| PartialBlock | debug | — | — | 0 |  |" in output
    assert "| PartialBlock | optimization | — | — | 0 |  |" in output


def test_render_log_empty_components():
    """A bootstrapped state with no components produces an empty table body.

    The section headings and the table header row are still present; the
    Recent Ticks placeholder reads ``_No ticks yet._`` and the Host-Resident
    placeholder reads ``_None._``.
    """
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    output = render_log(state)

    assert "## Block Status" in output
    assert "| Block | Phase | Status | PCC | Attempts | Notes |" in output
    # No data rows: no line starts with "| " followed by anything other than
    # the header / alignment row.
    data_rows = [
        ln
        for ln in output.splitlines()
        if ln.startswith("| ") and not ln.startswith("| Block ") and not ln.startswith("| :--- ")
    ]
    assert data_rows == [], f"expected no data rows, got {data_rows!r}"

    assert "## Recent Ticks" in output
    assert "_No ticks yet._" in output

    assert "## Host-Resident Exceptions" in output
    assert "_None._" in output


def test_render_log_pipe_in_notes_escaped():
    """A pipe in notes must be backslash-escaped so it doesn't break the table."""
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["components"] = [
        {
            "name": "WeirdBlock",
            "reference": {"status": "done", "pcc": 0.9999, "attempts": 1, "notes": "a|b"},
            "ttnn": {"status": "pending"},
            "debug": {"status": "pending"},
            "optimization": {"status": "pending"},
        }
    ]

    output = render_log(state)

    assert "a\\|b" in output, "pipe in notes should be escaped as a\\|b"
    # And the raw `a|b` must NOT appear unescaped in the row — check by looking
    # for it without the preceding backslash.
    for ln in output.splitlines():
        if "WeirdBlock" in ln and "reference" in ln:
            # The escaped form contains "a\|b"; we want to confirm there is no
            # unescaped "a|b" substring (which would mean we forgot the backslash).
            assert "a\\|b" in ln
            assert "a|b" not in ln.replace("a\\|b", "")


def test_render_log_tick_log_truncated_to_10():
    """With 15 tick entries, only the most recent 10 (ticks 6..15) are shown."""
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["tick_log"] = [
        {"tick": i, "ts": f"2026-05-27T14:{i:02d}:00Z", "action": "x", "result": "ok"} for i in range(1, 16)
    ]

    output = render_log(state)

    # Window boundaries are present.
    assert "tick 6" in output
    assert "tick 15" in output
    # Entries outside the window are not.
    assert "tick 1 " not in output  # trailing space avoids matching "tick 10"/"tick 11"
    assert "tick 1:" not in output
    assert "tick 5 " not in output
    assert "tick 5:" not in output


def test_render_log_host_resident_listed():
    """Components with host_resident.allowed=True are listed; otherwise _None._.

    Two states: one with an exception component, one without. We check both
    branches of the renderer in a single test so the contract is fully pinned.
    """
    # Case 1: a qualifying component.
    state_a = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state_a["components"] = [
        {
            "name": "ConvDecoder",
            "host_resident": {
                "allowed": True,
                "justification": "Conv too large for L1",
                "reference_link": "models/foo/decoder.py",
            },
            "reference": {"status": "done"},
            "ttnn": {"status": "n/a"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "n/a"},
        }
    ]
    output_a = render_log(state_a)
    # Section appears with a bullet for the exception.
    assert "## Host-Resident Exceptions" in output_a
    assert "ConvDecoder" in output_a
    assert "Conv too large for L1" in output_a
    assert "models/foo/decoder.py" in output_a
    # Sanity: the _None._ placeholder must NOT appear when there IS an exception.
    # (The Host-Resident section is the only one that can produce it.)
    assert "_None._" not in output_a

    # Case 2: a component with host_resident.allowed = False does NOT qualify.
    state_b = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state_b["components"] = [
        {
            "name": "RegularBlock",
            "host_resident": {"allowed": False, "justification": None, "reference_link": None},
            "reference": {"status": "done"},
            "ttnn": {"status": "done"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        }
    ]
    output_b = render_log(state_b)
    assert "## Host-Resident Exceptions" in output_b
    assert "_None._" in output_b
    # The block name itself can still appear in the table; that's fine. We're
    # only checking the Host-Resident section reads _None._.
    hr_section = output_b.split("## Host-Resident Exceptions", 1)[1]
    assert "RegularBlock" not in hr_section


# ---------------------------------------------------------------------------
# redo / skip — manual nudges
# ---------------------------------------------------------------------------


def _state_with_component(name="Attention", phase_data=None):
    """Build a bootstrapped state with one component pre-populated.

    Default ttnn phase is "failing" with a PCC, attempts, and last_error so
    redo tests can assert these fields are cleared on reset.
    """
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    phase = phase_data or {
        "status": "failing",
        "pcc": 0.81,
        "attempts": 3,
        "last_error": "QK-norm mismatch",
    }
    state["components"].append(
        {
            "name": name,
            "kind": "attention",
            "reference": {"status": "done", "pcc": 0.9999, "attempts": 1},
            "ttnn": phase,
            "debug": {"status": "n/a"},
            "optimization": {"status": "blocked", "blocked_on": "ttnn"},
        }
    )
    return state


def test_redo_resets_phase():
    """redo flips the named phase to pending/attempts=0 with a clean slate."""
    state = _state_with_component()

    redo(state, "Attention", "ttnn")

    comp = state["components"][0]
    assert comp["ttnn"]["status"] == "pending"
    assert comp["ttnn"]["attempts"] == 0
    # Clean slate — previous attempt's fields are dropped.
    assert "pcc" not in comp["ttnn"]
    assert "last_error" not in comp["ttnn"]


def test_redo_unknown_block_raises_KeyError():
    """redo on a non-existent component name raises KeyError."""
    state = _state_with_component()
    with pytest.raises(KeyError):
        redo(state, "DoesNotExist", "ttnn")


def test_redo_unknown_phase_raises_ValueError():
    """redo on an unknown phase name raises ValueError."""
    state = _state_with_component()
    with pytest.raises(ValueError):
        redo(state, "Attention", "nonsense")


def test_redo_appends_tick_log():
    """redo appends a tick_log entry with action='redo[block:phase]', result='ok'.

    Tick number is monotonic from any prior log entries (1 when empty).
    Timestamp matches the ISO-8601-with-Z shape used elsewhere.
    """
    state = _state_with_component()
    assert state["tick_log"] == []  # baseline

    redo(state, "Attention", "ttnn")

    assert len(state["tick_log"]) == 1
    entry = state["tick_log"][-1]
    assert entry["action"].startswith("redo[")
    assert entry["action"] == "redo[Attention:ttnn]"
    assert entry["result"] == "ok"
    assert entry["tick"] == 1
    # Timestamp shape: ISO-8601 with trailing Z, seconds precision.
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", entry["ts"]), entry["ts"]


def test_redo_tick_number_is_max_plus_one():
    """A redo nudge on a state with prior ticks gets tick = max + 1."""
    state = _state_with_component()
    state["tick_log"] = [
        {"tick": 5, "ts": "2026-05-27T14:00:00Z", "action": "x", "result": "ok"},
        {"tick": 7, "ts": "2026-05-27T14:01:00Z", "action": "y", "result": "ok"},
    ]

    redo(state, "Attention", "ttnn")

    assert state["tick_log"][-1]["tick"] == 8  # max(5,7) + 1


def test_redo_updates_updated_at():
    """redo advances updated_at to the current time."""
    state = _state_with_component()
    state["updated_at"] = "1999-01-01T00:00:00Z"

    redo(state, "Attention", "ttnn")

    assert state["updated_at"] != "1999-01-01T00:00:00Z"


def test_redo_returns_same_object():
    """redo returns the same state dict it was given, for chaining."""
    state = _state_with_component()
    out = redo(state, "Attention", "ttnn")
    assert out is state


def test_skip_sets_host_resident_allowed():
    """skip flips host_resident.allowed=True with the justification + ref.

    The named phase becomes ``{"status": "skipped"}`` with all prior fields
    (pcc, attempts, last_error) dropped — the phase is an explicit skip.
    """
    state = _state_with_component()

    skip(state, "Attention", "ttnn", "Conv too large for L1", "models/foo/decoder.py")

    comp = state["components"][0]
    assert comp["host_resident"]["allowed"] is True
    assert comp["host_resident"]["justification"] == "Conv too large for L1"
    assert comp["host_resident"]["reference_link"] == "models/foo/decoder.py"

    assert comp["ttnn"]["status"] == "skipped"
    # Previous phase fields are gone — the phase is now an explicit skip.
    assert "pcc" not in comp["ttnn"]
    assert "attempts" not in comp["ttnn"]
    assert "last_error" not in comp["ttnn"]


def test_skip_appends_tick_log_with_justify():
    """skip's tick_log entry has action='skip[...]' and result includes justification."""
    state = _state_with_component()

    skip(state, "Attention", "ttnn", "Conv too large for L1", "models/foo/decoder.py")

    entry = state["tick_log"][-1]
    assert entry["action"].startswith("skip[")
    assert entry["action"] == "skip[Attention:ttnn]"
    assert "Conv too large for L1" in entry["result"]


def test_skip_rejects_empty_justify():
    """An empty justification is not auditable, so skip raises ValueError."""
    state = _state_with_component()
    with pytest.raises(ValueError):
        skip(state, "Attention", "ttnn", "", "models/foo/decoder.py")


def test_skip_rejects_empty_reference_link():
    """An empty reference_link is not auditable, so skip raises ValueError."""
    state = _state_with_component()
    with pytest.raises(ValueError):
        skip(state, "Attention", "ttnn", "Conv too large for L1", "")


def test_skip_unknown_block_raises_KeyError():
    """skip on a non-existent component name raises KeyError."""
    state = _state_with_component()
    with pytest.raises(KeyError):
        skip(state, "DoesNotExist", "ttnn", "j", "r")


def test_skip_unknown_phase_raises_ValueError():
    """skip on an unknown phase name raises ValueError."""
    state = _state_with_component()
    with pytest.raises(ValueError):
        skip(state, "Attention", "nonsense", "j", "r")


def test_skip_then_render_log_shows_host_resident():
    """After skip(), render_log lists the block in the Host-Resident section.

    Integration sanity check: skip's host_resident dict feeds directly into
    render_log's exception section.
    """
    state = _state_with_component()

    skip(state, "Attention", "ttnn", "Conv too large for L1", "models/foo/decoder.py")
    output = render_log(state)

    # Locate the Host-Resident section and assert the block + justification appear.
    hr_section = output.split("## Host-Resident Exceptions", 1)[1]
    assert "Attention" in hr_section
    assert "Conv too large for L1" in hr_section
    assert "models/foo/decoder.py" in hr_section
    # And the _None._ placeholder must NOT appear in that section.
    assert "_None._" not in hr_section


def test_redo_trims_tick_log_to_100():
    """tick_log is bounded to last 100 entries per SPEC; redo trims overflow."""
    state = _state_with_component()
    # Seed with 105 dummy entries
    state["tick_log"] = [
        {"tick": i, "ts": "2026-05-27T14:00:00Z", "action": f"dummy[{i}]", "result": "ok"} for i in range(1, 106)
    ]
    redo(state, "Attention", "ttnn")
    # Original 105 + 1 new = 106, trimmed to 100
    assert len(state["tick_log"]) == 100
    # The newest entry must be present (it's the redo we just did)
    assert state["tick_log"][-1]["action"] == "redo[Attention:ttnn]"
    # Entries 1..6 should have been dropped; entries 7..105 + new redo retained
    first_ticks = [e["tick"] for e in state["tick_log"][:5]]
    assert 1 not in first_ticks
    assert 7 in first_ticks


def test_skip_accepts_whitespace_only_justify():
    """Whitespace-only justify is currently accepted; tightening this would
    be a contract change and should require an explicit decision."""
    state = _state_with_component()
    out = skip(state, "Attention", "ttnn", "   ", "models/foo/decoder.py")
    comp = out["components"][0]
    assert comp["host_resident"]["allowed"] is True
    assert comp["host_resident"]["justification"] == "   "


def test_skip_overwrites_existing_host_resident():
    """skip overwrites a pre-existing host_resident block with allowed=False.

    Pins behavior in case a future worker (e.g. architecture) stamps a default
    ``host_resident`` block onto components at bootstrap time.
    """
    state = _state_with_component()
    # Simulate a previously-stamped host_resident block with allowed=False
    state["components"][0]["host_resident"] = {
        "allowed": False,
        "justification": None,
        "reference_link": None,
    }
    skip(state, "Attention", "ttnn", "Conv too large", "models/foo/decoder.py")
    hr = state["components"][0]["host_resident"]
    assert hr["allowed"] is True
    assert hr["justification"] == "Conv too large"
    assert hr["reference_link"] == "models/foo/decoder.py"


def test_redo_then_render_log_shows_pending_row():
    """After redo(), the rendered Block Status row shows status=pending, attempts=0.

    Integration counterpart to test_skip_then_render_log_shows_host_resident:
    proves redo's state mutation flows through to the renderer.
    """
    state = _state_with_component()  # Attention.ttnn was failing
    redo(state, "Attention", "ttnn")
    output = render_log(state)
    # The table row for Attention/ttnn should now report status 'pending' and 0 attempts.
    # Find a line that contains both "Attention" and "ttnn" cells.
    lines = [ln for ln in output.splitlines() if "Attention" in ln and "ttnn" in ln]
    assert lines, "no Attention/ttnn row in rendered output"
    row = lines[0]
    assert "pending" in row
    assert " 0 " in row  # attempts column
