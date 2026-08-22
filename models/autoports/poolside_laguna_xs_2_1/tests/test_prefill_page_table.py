# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for single-shot fill page-table selection."""

from __future__ import annotations

import pytest
import torch

from models.autoports.poolside_laguna_xs_2_1.tt import prefill_page_table as page_table_module
from models.autoports.poolside_laguna_xs_2_1.tt.prefill_page_table import single_shot_fill_page_table


def test_cold_single_shot_prefill_reuses_persistent_full_fill_table(monkeypatch):
    fill_page_table = torch.tensor([[10, 11, -1, -1]], dtype=torch.int32)

    def fail_slice(*_args, **_kwargs):
        raise AssertionError("cold prefill must not allocate a fill-page-table slice")

    monkeypatch.setattr(page_table_module.ttnn, "slice", fail_slice)

    selected = single_shot_fill_page_table(
        fill_page_table,
        start_pos=0,
        seq_len=128,
        block_size=64,
    )

    assert selected is fill_page_table


def test_resumed_single_shot_prefill_slices_to_absolute_start_block(monkeypatch):
    fill_page_table = torch.tensor([[10, 11, 12, 13, 14, -1]], dtype=torch.int32)
    sliced = object()
    calls = []

    def record_slice(tensor, starts, ends):
        calls.append((tensor, starts, ends))
        return sliced

    monkeypatch.setattr(page_table_module.ttnn, "slice", record_slice)

    selected = single_shot_fill_page_table(
        fill_page_table,
        start_pos=128,
        seq_len=129,
        block_size=64,
    )

    assert selected is sliced
    assert calls == [(fill_page_table, [0, 2], [1, 5])]


def test_rebased_resumed_single_shot_reuses_persistent_full_fill_table(monkeypatch):
    fill_page_table = torch.tensor([[12, 13, 14, -1, -1, -1]], dtype=torch.int32)

    def fail_slice(*_args, **_kwargs):
        raise AssertionError("adapter-rebased resumed prefill must not allocate a device slice")

    monkeypatch.setattr(page_table_module.ttnn, "slice", fail_slice)

    selected = single_shot_fill_page_table(
        fill_page_table,
        start_pos=128,
        seq_len=129,
        block_size=64,
        fill_page_table_base_pos=128,
    )

    assert selected is fill_page_table


def test_single_shot_fill_table_rejects_invalid_relative_base():
    fill_page_table = torch.zeros((1, 8), dtype=torch.int32)

    with pytest.raises(ValueError, match="precedes"):
        single_shot_fill_page_table(
            fill_page_table,
            start_pos=64,
            seq_len=32,
            block_size=64,
            fill_page_table_base_pos=128,
        )

    with pytest.raises(ValueError, match="not aligned"):
        single_shot_fill_page_table(
            fill_page_table,
            start_pos=96,
            seq_len=32,
            block_size=64,
            fill_page_table_base_pos=0,
        )
