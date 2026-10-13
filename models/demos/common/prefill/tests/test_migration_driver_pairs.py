# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Device-free checks of MigrationDriver's src->dst slot resolution when the destination is another
endpoint's table (a decode peer with its own slot count) vs this prefill table (loopback)."""

from types import SimpleNamespace

import pytest

from models.demos.common.prefill.runners.migration_driver import MigrationDriver

PREFILL_EP, DECODE_EP = 1, 0


def _driver(monkeypatch, *, dest_ep, pairs=None):
    monkeypatch.setenv("PREFILL_MIGRATION_SRC_ENDPOINT_ID", str(PREFILL_EP))
    monkeypatch.setenv("PREFILL_MIGRATION_DEST_ENDPOINT_ID", str(dest_ep))
    monkeypatch.delenv("PREFILL_MIGRATION_DST_SLOT_OFFSET", raising=False)
    if pairs is None:
        monkeypatch.delenv("PREFILL_MIGRATION_PAIRS", raising=False)
    else:
        monkeypatch.setenv("PREFILL_MIGRATION_PAIRS", pairs)
    return MigrationDriver(chunk_size=4096, num_layers=60, default_dst_slot_offset=1)


def _resident(**real_lens):
    return SimpleNamespace(resident={int(s[1:]): SimpleNamespace(real_len=n) for s, n in real_lens.items()})


@pytest.mark.parametrize("dst_slot", [0, 1])
def test_cross_endpoint_dst_slot_is_the_decode_tables(monkeypatch, dst_slot):
    # One prefill slot feeds either of the decode peer's two slots; only the peer's table bounds dst.
    driver = _driver(monkeypatch, dest_ep=DECODE_EP, pairs=f"0:{dst_slot}")
    assert driver._resolve_pairs(_resident(s0=1 << 20), num_slots=1) == [(0, dst_slot, 1 << 20)]


def test_cross_endpoint_default_offset_lands_past_the_prefill_table(monkeypatch):
    driver = _driver(monkeypatch, dest_ep=DECODE_EP)
    assert driver._resolve_pairs(_resident(s0=8192), num_slots=1) == [(0, 1, 8192)]


def test_loopback_dst_slot_must_fit_the_prefill_table(monkeypatch, expect_error):
    driver = _driver(monkeypatch, dest_ep=PREFILL_EP, pairs="0:1")
    with expect_error(ValueError, "out of range"):
        driver._resolve_pairs(_resident(s0=8192), num_slots=1)
    assert driver._resolve_pairs(_resident(s0=8192), num_slots=2) == [(0, 1, 8192)]
