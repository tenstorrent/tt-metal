# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the migration driver's half of PREFILL_FIRST_TOKEN_DIR (row lookup, wait, handoff)."""

import json
import sys
import threading
import time
import types

import pytest
import torch

from models.demos.common.prefill.runners import migration_driver as md

SLOT, REAL_LEN, TOKEN = 2, 96, 4242


class _Adapter:
    """Records the rows it is handed and returns a fixed token, so the test sees which row was used."""

    def __init__(self):
        self.rows = []

    def first_token_from_hidden(self, hidden):
        self.rows.append(hidden.clone())
        return TOKEN


@pytest.fixture
def adapter(monkeypatch):
    # The real producer module configures a prefill from the environment on import; stand in for it.
    adapter = _Adapter()
    fake = types.ModuleType("models.demos.common.prefill.runners.prefill_producer")
    fake.ADAPTER = adapter
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(sys.modules["models.demos.common.prefill.runners"], "prefill_producer", fake, raising=False)
    return adapter


def _driver(monkeypatch, tmp_path, *, timeout_s="5", handoff=""):
    monkeypatch.setenv("PREFILL_FIRST_TOKEN_DIR", str(tmp_path / "rows"))
    monkeypatch.setenv("PREFILL_FIRST_TOKEN_TIMEOUT_S", timeout_s)
    monkeypatch.setenv("PREFILL_MIGRATION_HANDOFF_PATH", handoff)
    (tmp_path / "rows").mkdir(exist_ok=True)
    return md.MigrationDriver(chunk_size=5120, num_layers=4, default_dst_slot_offset=2)


def _row(tmp_path, slot, end, value):
    path = tmp_path / "rows" / f"slot{slot}_end{end}.pt"
    torch.save({"slot_id": slot, "actual_start": 0, "actual_end": end, "hidden": torch.full((8,), float(value))}, path)
    return path


def test_the_row_of_the_requests_slot_and_length_is_used_and_consumed(monkeypatch, tmp_path, adapter):
    driver = _driver(monkeypatch, tmp_path)
    _row(tmp_path, SLOT, REAL_LEN - 32, 1.0)  # an earlier chunk of the same request
    _row(tmp_path, SLOT + 1, REAL_LEN, 2.0)  # another slot's request of the same length
    used = _row(tmp_path, SLOT, REAL_LEN, 3.0)

    assert driver._first_token(SLOT, REAL_LEN) == TOKEN
    assert len(adapter.rows) == 1 and float(adapter.rows[0][0]) == 3.0
    assert not used.exists()  # a later request of the same slot and length must not read it again
    assert (tmp_path / "rows" / f"slot{SLOT + 1}_end{REAL_LEN}.pt").exists()


def test_a_row_that_lands_after_the_acks_is_waited_for(monkeypatch, tmp_path, adapter):
    driver = _driver(monkeypatch, tmp_path)
    late = threading.Timer(0.5, _row, args=(tmp_path, SLOT, REAL_LEN, 5.0))
    late.start()
    t0 = time.monotonic()
    try:
        assert driver._first_token(SLOT, REAL_LEN) == TOKEN
    finally:
        late.join()
    assert time.monotonic() - t0 >= 0.4


def test_a_row_that_never_lands_times_out_naming_the_file(monkeypatch, tmp_path, expect_error, adapter):
    driver = _driver(monkeypatch, tmp_path, timeout_s="0.3")
    with expect_error(TimeoutError, f"slot{SLOT}_end{REAL_LEN}.pt"):
        driver._first_token(SLOT, REAL_LEN)
    assert adapter.rows == []


def test_the_handoff_carries_first_token_per_entry(monkeypatch, tmp_path, adapter):
    handoff = tmp_path / "handoff.json"
    driver = _driver(monkeypatch, tmp_path, handoff=str(handoff))
    _row(tmp_path, SLOT, REAL_LEN, 7.0)
    pool = list(range(1000, 1000 + REAL_LEN))
    driver._write_handoff([(SLOT, SLOT + 2, REAL_LEN)], {SLOT: "trace-a"}, {"trace-a": pool})

    (entry,) = json.loads(handoff.read_text())["slots"]
    assert entry == {"dst_slot": SLOT + 2, "prompt_len": REAL_LEN, "last_prompt_token": pool[-1], "first_token": TOKEN}


def test_without_the_env_var_the_handoff_is_unchanged(monkeypatch, tmp_path, adapter):
    handoff = tmp_path / "handoff.json"
    driver = _driver(monkeypatch, tmp_path, handoff=str(handoff))
    driver.first_token_dir = ""  # PREFILL_FIRST_TOKEN_DIR unset
    driver._write_handoff([(SLOT, SLOT + 2, REAL_LEN)], {SLOT: "trace-a"}, {"trace-a": list(range(REAL_LEN))})

    (entry,) = json.loads(handoff.read_text())["slots"]
    assert "first_token" not in entry and adapter.rows == []
