# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Host-only: the migration handoff's tail resume, which a recurrent-state model (Kimi-K3's KDA) needs.

Prefill samples no first token, and decode cannot re-run a token the migrated state already applied, so the handoff
leaves the next prompt token(s) for decode: ``prefill_len`` < ``prompt_len`` with ``tail_tokens`` (tt-blaze
``kv_chunk_migration_check._resolve_resume``).
"""

import json

import pytest

from models.demos.common.prefill.runners.migration_driver import MigrationDriver

POOL = list(range(1000, 1100))


def _driver(tmp_path, monkeypatch, tail):
    monkeypatch.setenv("PREFILL_MIGRATION_HANDOFF_PATH", str(tmp_path / "handoff.json"))
    monkeypatch.setenv("PREFILL_MIGRATION_HANDOFF_TAIL", str(tail))
    return MigrationDriver(chunk_size=32, num_layers=1, default_dst_slot_offset=0)


def _slots(tmp_path):
    return json.loads((tmp_path / "handoff.json").read_text())["slots"]


def test_no_tail_keeps_the_whole_prompt_applied(tmp_path, monkeypatch):
    _driver(tmp_path, monkeypatch, 0)._write_handoff([(0, 1, 64)], {0: "t"}, {"t": POOL})
    assert _slots(tmp_path) == [{"dst_slot": 1, "prompt_len": 64, "last_prompt_token": POOL[63]}]


def test_one_token_tail_leaves_the_next_prompt_token_for_decode(tmp_path, monkeypatch):
    _driver(tmp_path, monkeypatch, 1)._write_handoff([(0, 1, 64)], {0: "t"}, {"t": POOL})
    assert _slots(tmp_path) == [
        {"dst_slot": 1, "prompt_len": 65, "prefill_len": 64, "tail_tokens": [POOL[64]], "last_prompt_token": POOL[64]}
    ]


def test_a_tail_past_the_pool_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="handoff tail needs prompt tokens up to 101"):
        _driver(tmp_path, monkeypatch, 1)._write_handoff([(0, 1, 100)], {0: "t"}, {"t": POOL})


def test_kimi_k3_adapters_default_to_a_one_token_tail():
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter

    assert KimiK3Adapter.migration_handoff_tail_tokens == 1
    assert KimiK31Adapter.migration_handoff_tail_tokens == 1
