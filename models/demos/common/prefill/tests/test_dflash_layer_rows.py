# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_kv_validation import _dflash_layer_rows


class _FakeTable:
    """Answers the only two lookups _dflash_layer_rows makes: the config's layer count, and whether a
    given layer row carries an address for the slot."""

    def __init__(self, num_layers: int, populated_from: int):
        self._num_layers = num_layers
        self._populated_from = populated_from

    def config(self, config_id: int):
        return SimpleNamespace(num_layers=self._num_layers)

    def lookup(self, layer: int, head: int, slot_id: int, config_id: int):
        return SimpleNamespace(noc_addr=0x1000 if layer >= self._populated_from else 0)


def test_layer_rows_start_at_the_first_populated_row():
    assert _dflash_layer_rows(_FakeTable(61, 55), config_id=1, slot_id=0) == range(55, 61)


def test_layer_rows_span_the_config_when_every_row_is_populated():
    assert _dflash_layer_rows(_FakeTable(6, 0), config_id=1, slot_id=0) == range(0, 6)


def test_unpopulated_last_row_is_rejected(expect_error):
    with expect_error(RuntimeError, "unpopulated"):
        _dflash_layer_rows(_FakeTable(61, 61), config_id=1, slot_id=0)
