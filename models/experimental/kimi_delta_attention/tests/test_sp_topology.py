# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only layout checks for KDA sequence/tensor parallel topology."""

from models.experimental.kimi_delta_attention.tt.sp_layer import SPTPTopology


def test_loudbox_sp2_tp4_is_two_contiguous_tp_groups() -> None:
    topology = SPTPTopology(sp_size=2, tp_size=4)
    assert topology.validate_mesh_shape((1, 8), name="test")
    assert [topology.span_start(rank, flattened=True) for rank in range(2)] == [(0, 0), (0, 4)]


def test_galaxy_sp8_tp4_is_native_eight_by_four() -> None:
    topology = SPTPTopology(sp_size=8, tp_size=4)
    assert not topology.validate_mesh_shape((8, 4), name="test")
    assert [topology.span_start(rank, flattened=False) for rank in range(8)] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
    ]


def test_topology_rejects_ambiguous_or_incomplete_mesh(expect_error) -> None:
    with expect_error(ValueError, "requires either flattened"):
        SPTPTopology(sp_size=8, tp_size=4).validate_mesh_shape((1, 8), name="test")
