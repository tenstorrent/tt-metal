# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Host-only checks for helpers that must agree on which mesh axis carries SP (the dispatch groups).

Weight placement: `ExpertMapping.gather_weights_for_mesh_distribution` lays weights out (mesh_rows, mesh_cols), while
the (group, chip, local_expert) global expert table is sharded with `ep_mesh_dims(sp_axis)`. Both
have to put the same global expert on every mesh position, for SP on either mesh axis; otherwise
dispatch routes tokens to a chip that holds a different expert's weights.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, ep_mesh_dims
from models.demos.deepseek_v3_d_p.tt.tt_ccl import resolve_per_axis_topology


@pytest.mark.parametrize("mesh_shape", [(2, 4), (4, 2), (4, 4), (1, 8), (8, 1)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp0", "sp1"])
@pytest.mark.parametrize("is_col_major", [True, False], ids=["col_major", "row_major"])
def test_weight_placement_matches_global_expert_table(mesh_shape, sp_axis, is_col_major):
    mesh_rows, mesh_cols = mesh_shape
    dispatch_group_size, num_dispatch_groups = mesh_shape[sp_axis], mesh_shape[1 - sp_axis]
    experts_per_chip = 3
    num_experts = experts_per_chip * mesh_rows * mesh_cols
    # Each "weight" is its own global expert id, so placement can be read back directly.
    torch_weights = [
        {name: torch.tensor([float(e)]) for name in ("gate_proj", "up_proj", "down_proj")} for e in range(num_experts)
    ]

    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip, dispatch_group_size, num_dispatch_groups, is_col_major=is_col_major
    )
    table_dims = ep_mesh_dims(sp_axis)  # table dim placed on mesh axis 0 and on mesh axis 1

    for local_expert in range(experts_per_chip):
        gate, up, down = ExpertMapping.gather_weights_for_mesh_distribution(
            torch_weights, local_expert, mesh_rows, mesh_cols, experts_per_chip, is_col_major, sp_axis=sp_axis
        )
        for weights in (gate, up, down):
            placed = torch.stack(weights).reshape(mesh_rows, mesh_cols).int()
            for row in range(mesh_rows):
                for col in range(mesh_cols):
                    idx = [0, 0]
                    idx[table_dims[0]], idx[table_dims[1]] = row, col
                    expected = int(table[idx[0], idx[1], local_expert])
                    assert int(placed[row, col]) == expected, (
                        f"mesh ({row}, {col}) local_expert={local_expert}: weights hold expert "
                        f"{int(placed[row, col])}, the global expert table says {expected}"
                    )

    # Every expert is placed exactly once across the mesh.
    all_placed = sorted(
        int(w)
        for local_expert in range(experts_per_chip)
        for w in ExpertMapping.gather_weights_for_mesh_distribution(
            torch_weights, local_expert, mesh_rows, mesh_cols, experts_per_chip, is_col_major, sp_axis=sp_axis
        )[0]
    )
    assert all_placed == list(range(num_experts))


@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp0", "sp1"])
def test_resolve_per_axis_topology_indexes_by_axis(sp_axis):
    tp_axis = 1 - sp_axis
    # TorusX-style fabric: only mesh axis 1 wraps.
    per_axis = (ttnn.Topology.Linear, ttnn.Topology.Ring)
    sp_topology, tp_topology = resolve_per_axis_topology(per_axis, sp_axis, tp_axis)
    assert sp_topology == per_axis[sp_axis]
    assert tp_topology == per_axis[tp_axis]
    # A scalar applies to both axes.
    assert resolve_per_axis_topology(ttnn.Topology.Ring, sp_axis, tp_axis) == (ttnn.Topology.Ring, ttnn.Topology.Ring)


def test_resolve_per_axis_topology_rejects_same_axis(expect_error):
    with expect_error(ValueError, "sp_axis and tp_axis must be 0 and 1"):
        resolve_per_axis_topology((ttnn.Topology.Linear, ttnn.Topology.Ring), 1, 1)


def test_mesh_cache_key_separates_sp_axes():
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import mesh_cache_key

    # SP on axis 0 keeps the existing key so caches already on disk stay valid.
    assert mesh_cache_key((8, 4)) == "8x4"
    assert mesh_cache_key((8, 4), 0) == "8x4"
    # A square mesh must not share a cache between the two SP placements.
    assert mesh_cache_key((4, 4), 0) != mesh_cache_key((4, 4), 1)
    assert mesh_cache_key((4, 4), 1) == "4x4_sp1"
