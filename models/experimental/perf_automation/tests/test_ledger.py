"""I-3 tests: append-only ledger (PLAN section 5)."""

import json

import pytest

from agent.ledger import Ledger


def test_ledger_append_only(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append({"experiment_id": "e1", "iter": 1})
    first_line = (tmp_path / "ledger.jsonl").read_text()
    led.append({"experiment_id": "e2", "iter": 2})

    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 2
    # First row unchanged after the second append.
    assert lines[0] == first_line.splitlines()[0]
    assert led.rows()[0]["experiment_id"] == "e1"
    assert led.rows()[1]["experiment_id"] == "e2"


def test_ledger_idempotent_by_experiment_id(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    assert led.append({"experiment_id": "e1", "iter": 1}) is True
    # Same experiment_id -> no-op.
    assert led.append({"experiment_id": "e1", "iter": 999}) is False
    rows = led.rows()
    assert len(rows) == 1
    assert rows[0]["iter"] == 1


def test_current_hypothesis_is_last_nonnull(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append({"experiment_id": "e1", "hypothesis": "A"})
    led.append({"experiment_id": "e2", "hypothesis": None})
    led.append({"experiment_id": "e3", "hypothesis": "B"})
    assert led.current_hypothesis() == "B"


def test_current_hypothesis_none_when_empty(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    assert led.current_hypothesis() is None
    assert led.rows() == []


def test_ledger_tolerates_truncated_last_line(tmp_path):
    p = tmp_path / "ledger.jsonl"
    led = Ledger(p)
    led.append({"experiment_id": "e1", "iter": 1})
    led.append({"experiment_id": "e2", "iter": 2})
    # Simulate a crash mid-append: a partial final line, no trailing newline.
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"experiment_id": "e3", "iter":')

    # rows() returns only the intact rows ...
    rows = led.rows()
    assert [r["experiment_id"] for r in rows] == ["e1", "e2"]
    # ... and append still works (it appends a fresh line after the partial).
    assert led.append({"experiment_id": "e4", "iter": 4}) is True


def test_ledger_raises_on_malformed_nonfinal_line(tmp_path):
    p = tmp_path / "ledger.jsonl"
    # A corrupt line that is NOT the last is real corruption, not a crash.
    p.write_text('{"experiment_id": "bad"\n{"experiment_id": "e2"}\n')
    led = Ledger(p)
    with pytest.raises(json.JSONDecodeError):
        led.rows()
