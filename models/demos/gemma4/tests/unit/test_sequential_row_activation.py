# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequential per-user prefill selects each request's per-layer page-table row
by its position in the batch, not by matching the legacy slice's content:
under lane-sharded bounded rings the sliding table repeats every lane_slots
slots, so two requests can carry identical sliding rows while their
full-attention rows differ."""

import torch

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin


class _Model:
    def __init__(self, tables):
        self._active_page_tables_per_layer = tables
        self.staged = []

    def update_persistent_per_layer_page_tables(self, tables):
        self.staged.append([t.clone() for t in tables])


class _Generator(ChunkedPrefillPageTableGuardMixin):
    def __init__(self, model):
        self.model = [model]


def _tables():
    sliding = torch.tensor([[7, 8, 0, 0], [7, 8, 0, 0]], dtype=torch.int32)
    full = torch.tensor([[101, 102, 0, 0], [201, 202, 0, 0]], dtype=torch.int32)
    return [sliding, full, sliding]


def _slice():
    return torch.tensor([[7, 8, 0, 0]], dtype=torch.int32)


def test_second_request_gets_its_own_full_attention_row():
    model = _Model(_tables())
    gen = _Generator(model)
    gen._activate_sequential_per_layer_row(_slice())
    assert model._active_page_tables_per_layer[1].tolist() == [[101, 102, 0, 0]]
    gen._activate_sequential_per_layer_row(_slice())
    assert model._active_page_tables_per_layer[1].tolist() == [[201, 202, 0, 0]]
    assert [s[1].tolist() for s in model.staged] == [[[101, 102, 0, 0]], [[201, 202, 0, 0]]]
    gen._clear_sequential_batch_page_tables()
    assert int(model._active_page_tables_per_layer[1].shape[0]) == 2
    assert model._sequential_row_cursor == 0


def test_nested_activation_within_one_request_is_a_noop():
    model = _Model(_tables())
    gen = _Generator(model)
    gen._activate_sequential_per_layer_row(_slice())
    model._sequential_row_active = True  # the traced entry holds this across its capture/replay
    gen._activate_sequential_per_layer_row(_slice())
    assert model._active_page_tables_per_layer[1].tolist() == [[101, 102, 0, 0]]
    assert model._sequential_row_cursor == 1


def test_more_requests_than_rows_raises(expect_error):
    model = _Model(_tables())
    gen = _Generator(model)
    gen._activate_sequential_per_layer_row(_slice())
    gen._activate_sequential_per_layer_row(_slice())
    with expect_error(ValueError, "row 2"):
        gen._activate_sequential_per_layer_row(_slice())


def test_legacy_slice_must_match_the_positional_row(expect_error):
    model = _Model(_tables())
    gen = _Generator(model)
    with expect_error(ValueError, "does not match"):
        gen._activate_sequential_per_layer_row(torch.tensor([[9, 9, 0, 0]], dtype=torch.int32))
