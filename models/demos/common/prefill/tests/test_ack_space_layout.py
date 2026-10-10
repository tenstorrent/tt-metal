# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ACK-space derivation for both layer-completion transports (host-only, no ttnn)."""

import pytest

from models.demos.common.prefill.runners.ack_space import ack_space_layout


def split(counts):
    out, start = [], 0
    for c in counts:
        out.append((start, c))
        start += c
    return out


def test_dense_model_ack_space_is_layer_space():
    layout = ack_space_layout(split([16, 15, 15, 15]), rank=2)
    assert layout.num_ack_layers == 61
    assert (layout.ack_first_idx, layout.ack_local_count) == (31, 15)
    assert layout.ack_idx_of_layer is None and layout.ack_layer_ids == []


def test_hybrid_rank_counts_only_its_kv_writing_layers():
    ids = [3, 7, 11, 15, 19, 23]
    layout = ack_space_layout(split([12, 12]), rank=1, kv_slot_layer_ids=ids)
    assert layout.num_ack_layers == 6
    assert (layout.ack_first_idx, layout.ack_local_count) == (3, 3)
    assert layout.ack_layer_ids == [15, 19, 23]
    assert layout.ack_idx_of_layer == {3: 0, 7: 1, 11: 2, 15: 3, 19: 4, 23: 5}


def test_extra_ack_rows_widen_the_last_rank_only():
    for rank, local in ((0, 16), (3, 15 + 2)):
        layout = ack_space_layout(split([16, 15, 15, 15]), rank=rank, extra_ack_layers=2)
        assert layout.num_ack_layers == 63
        assert layout.ack_local_count == local


def test_extra_ack_rows_extend_the_hybrid_maps_past_the_trunk():
    ids = [3, 7, 11, 15, 19, 23]
    layout = ack_space_layout(split([12, 12]), rank=1, kv_slot_layer_ids=ids, extra_ack_layers=2)
    assert layout.num_ack_layers == 8 and layout.ack_local_count == 5
    assert layout.ack_layer_ids == [15, 19, 23, 24, 25]
    assert layout.ack_idx_of_layer[24] == 6 and layout.ack_idx_of_layer[25] == 7
    first = ack_space_layout(split([12, 12]), rank=0, kv_slot_layer_ids=ids, extra_ack_layers=2)
    assert first.ack_local_count == 3 and first.ack_layer_ids == [3, 7, 11]


def test_a_rank_without_a_kv_writing_layer_is_rejected():
    with pytest.raises(ValueError, match="writes a KV slab"):  # allow-pytest.raises: host-only
        ack_space_layout(split([4, 2, 18]), rank=1, kv_slot_layer_ids=[3, 7, 11, 15, 19, 23])
