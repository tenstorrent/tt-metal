# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks for QWEN36_MTP_SDPA_WINDOW (r9 drafter track): env parsing, the per-instance SDPA kwarg, and the
KV-split opt-out. Needs ttnn importable, no device.

Run: pytest models/demos/blackhole/qwen36/tests/test_mtp_sdpa_window_host.py -q
"""
import pytest

from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention


@pytest.mark.parametrize("raw,want", [(None, 0), ("", 0), ("0", 0), ("-5", 0), ("2048", 2048)])
def test_window_parse(monkeypatch, raw, want):
    monkeypatch.delenv(tpc.MTP_SDPA_WINDOW_ENV, raising=False)
    if raw is not None:
        monkeypatch.setenv(tpc.MTP_SDPA_WINDOW_ENV, raw)
    assert tpc.mtp_sdpa_window() == want


def test_window_bad_value(monkeypatch):
    monkeypatch.setenv(tpc.MTP_SDPA_WINDOW_ENV, "2k")
    with pytest.raises(ValueError):
        tpc.mtp_sdpa_window()


def test_window_kwarg():
    a = TPAttention.__new__(TPAttention)  # bare: no window attribute -> today's call
    assert a._decode_window_kw() == {}
    a.decode_sdpa_window = None
    assert a._decode_window_kw() == {}
    a.decode_sdpa_window = 2048
    assert a._decode_window_kw() == {"sliding_window_size": 2048}


def test_window_disables_draft_split():
    a = TPAttention.__new__(TPAttention)
    a._kv_split_ctx = {"n_users": 8, "T": 1}
    a._kv_split_ready = True
    a.use_paged = True
    a.decode_sdpa_window = 2048
    assert a._kv_split_draft_ok(8, page_table=object()) is False
