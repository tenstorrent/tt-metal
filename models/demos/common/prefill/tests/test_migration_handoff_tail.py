# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only: the migration handoff's tail resume, which a recurrent-state model (Kimi-K3's KDA) needs.

Prefill samples no first token, and decode cannot re-run a token the migrated state already applied, so the handoff
leaves the next prompt token(s) for decode: ``prefill_len`` < ``prompt_len`` with ``tail_tokens`` (tt-blaze
``kv_chunk_migration_check._resolve_resume``).
"""

import json

from models.demos.common.prefill.runners.migration_driver import MigrationDriver

POOL = list(range(1000, 1100))


def _driver(tmp_path, monkeypatch, tail):
    monkeypatch.setenv("PREFILL_MIGRATION_HANDOFF_PATH", str(tmp_path / "handoff.json"))
    monkeypatch.setenv("PREFILL_MIGRATION_HANDOFF_TAIL", str(tail))
    return MigrationDriver(chunk_size=32, num_layers=1, default_dst_slot_offset=0)


def _write(driver, triples):
    driver._write_handoff(driver._handoff_slots(triples, {0: "t"}, {"t": POOL}))


def _slots(tmp_path):
    return json.loads((tmp_path / "handoff.json").read_text())["slots"]


def test_no_tail_keeps_the_whole_prompt_applied(tmp_path, monkeypatch):
    _write(_driver(tmp_path, monkeypatch, 0), [(0, 1, 64)])
    assert _slots(tmp_path) == [{"dst_slot": 1, "prompt_len": 64, "last_prompt_token": POOL[63]}]


def test_one_token_tail_leaves_the_next_prompt_token_for_decode(tmp_path, monkeypatch):
    _write(_driver(tmp_path, monkeypatch, 1), [(0, 1, 64)])
    assert _slots(tmp_path) == [
        {"dst_slot": 1, "prompt_len": 65, "prefill_len": 64, "tail_tokens": [POOL[64]], "last_prompt_token": POOL[64]}
    ]


def test_a_tail_past_the_pool_is_refused(tmp_path, monkeypatch, expect_error):
    with expect_error(ValueError, "handoff tail needs prompt tokens up to 101"):
        _driver(tmp_path, monkeypatch, 1)._handoff_slots([(0, 1, 100)], {0: "t"}, {"t": POOL})


def test_a_tail_past_the_pool_stops_the_run_before_any_migration(tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv("PREFILL_MIGRATION_KVM_URL", "http://kvm.invalid:8080")
    driver = _driver(tmp_path, monkeypatch, 1)
    issued = []
    monkeypatch.setattr(driver, "_resolve_pairs", lambda stats, num_slots=None: [(0, 1, 100)])
    monkeypatch.setattr(driver, "_issue", lambda triples, kv_table=None: issued.append(triples))
    with expect_error(ValueError, "handoff tail needs prompt tokens up to 101"):
        driver.run(None, slot_traces={0: "t"}, pools_by_trace={"t": POOL})
    assert issued == []
    assert not (tmp_path / "handoff.json").exists()


def test_the_tail_defaults_to_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("PREFILL_MIGRATION_HANDOFF_TAIL", raising=False)
    assert MigrationDriver(chunk_size=32, num_layers=1, default_dst_slot_offset=0).handoff_tail == 0
