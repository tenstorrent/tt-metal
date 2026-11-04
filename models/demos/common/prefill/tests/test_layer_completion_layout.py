# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-rank completion record layout for both layer-completion transports (host-only, no ttnn)."""

import pytest

from models.demos.common.prefill.runners.layer_completion_layout import layer_completion_layout


def split(counts):
    out, start = [], 0
    for c in counts:
        out.append((start, c))
        start += c
    return out


def test_dense_model_records_are_its_layers():
    layout = layer_completion_layout(split([16, 15, 15, 15]), rank=2)
    assert layout.completions_per_chunk == 61
    assert (layout.rank_first_record, layout.rank_record_count) == (31, 15)
    assert layout.rank_layer_ids == []


def test_hybrid_rank_counts_only_its_kv_writing_layers():
    ids = [3, 7, 11, 15, 19, 23]
    layout = layer_completion_layout(split([12, 12]), rank=1, kv_slot_layer_ids=ids)
    assert layout.completions_per_chunk == 6
    assert (layout.rank_first_record, layout.rank_record_count) == (3, 3)
    assert layout.rank_layer_ids == [15, 19, 23]


def test_extra_ack_rows_widen_the_last_rank_only():
    for rank, local in ((0, 16), (3, 15 + 2)):
        layout = layer_completion_layout(split([16, 15, 15, 15]), rank=rank, extra_layers=2)
        assert layout.completions_per_chunk == 63
        assert layout.rank_record_count == local


def test_extra_ack_rows_extend_the_hybrid_maps_past_the_trunk():
    ids = [3, 7, 11, 15, 19, 23]
    layout = layer_completion_layout(split([12, 12]), rank=1, kv_slot_layer_ids=ids, extra_layers=2)
    assert layout.completions_per_chunk == 8 and layout.rank_record_count == 5
    assert layout.rank_layer_ids == [15, 19, 23, 24, 25]
    first = layer_completion_layout(split([12, 12]), rank=0, kv_slot_layer_ids=ids, extra_layers=2)
    assert first.rank_record_count == 3 and first.rank_layer_ids == [3, 7, 11]


def test_a_rank_without_a_kv_writing_layer_is_rejected():
    with pytest.raises(ValueError, match="writes a KV slab"):  # allow-pytest.raises: host-only
        layer_completion_layout(split([4, 2, 18]), rank=1, kv_slot_layer_ids=[3, 7, 11, 15, 19, 23])
