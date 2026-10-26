# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode must re-upload a per-layer page-table bucket after prefill wrote the same persistent buffers,
or its trace replays against the last prefilled user's block ids."""

import torch

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin


class _Model:
    def __init__(self, tables):
        self._active_page_tables_per_layer = tables
        self.staged = []

    def update_persistent_per_layer_page_tables(self, tables):
        self.staged.append([t.clone() if isinstance(t, torch.Tensor) else t for t in tables])


class _Generator(ChunkedPrefillPageTableGuardMixin):
    def __init__(self, model):
        self.model = [model]


def _tables(rows):
    sliding = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4) + 1
    return [sliding, sliding + 100, sliding]


def test_prefill_write_forces_decode_reupload_of_the_same_bucket():
    model = _Model(_tables(32))
    gen = _Generator(model)
    decode_rows = _tables(1)
    gen._install_per_layer_page_tables(model, decode_rows, writer="decode")
    assert not gen._page_tables_written_by_prefill([model], [decode_rows])
    # A sequential prefill user slices the batch to one row: same bucket key.
    gen._activate_sequential_per_layer_row(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    assert gen._page_tables_written_by_prefill([model], [decode_rows])
    gen._install_per_layer_page_tables(model, decode_rows, writer="decode")
    assert not gen._page_tables_written_by_prefill([model], [decode_rows])


def test_other_buckets_are_not_marked():
    model = _Model(_tables(32))
    gen = _Generator(model)
    gen._install_per_layer_page_tables(model, _tables(4), writer="decode")
    gen._activate_sequential_per_layer_row(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    # 1-row buffers were written by prefill; the 4-row bucket was not.
    assert gen._page_tables_written_by_prefill([model], [_tables(1)])
    assert not gen._page_tables_written_by_prefill([model], [_tables(4)])


def test_clearing_the_sequential_batch_marks_the_full_batch_rows():
    model = _Model(_tables(32))
    gen = _Generator(model)
    gen._install_per_layer_page_tables(model, _tables(32), writer="decode")
    gen._activate_sequential_per_layer_row(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))
    gen._clear_sequential_batch_page_tables()
    assert gen._page_tables_written_by_prefill([model], [_tables(32)])
    assert len(model.staged) == 3


def test_rows_key_matches_the_model_rule():
    rows = ChunkedPrefillPageTableGuardMixin._page_tables_rows
    assert rows([None, torch.zeros(5, 3, dtype=torch.int32)]) == 5
    assert rows([torch.zeros(7, dtype=torch.int32)]) == 1
    assert rows([None, "device-tensor-stand-in"]) is None
    assert rows(None) is None
