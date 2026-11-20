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


@pytest.mark.parametrize("mesh_shape", [(2, 4), (4, 2), (4, 4), (8, 4)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp0", "sp1"])
def test_metadata_source_is_the_row_major_mesh_index(mesh_shape, sp_axis):
    """The re-encoded source of every (chip, group) is the row-major index of that device's mesh
    coordinate, the index the dispatch op writes and the combine op reads; unfilled slots stay -1."""
    mesh_rows, mesh_cols = mesh_shape
    dispatch_group_size, num_dispatch_groups = mesh_shape[sp_axis], mesh_shape[1 - sp_axis]
    sources, expected = [-1], [-1]
    for group in range(num_dispatch_groups):
        for chip in range(dispatch_group_size):
            sources.append(ExpertMapping.compute_linearized_mesh_coord(chip, group, num_dispatch_groups))
            row, col = (chip, group) if sp_axis == 0 else (group, chip)
            expected.append(row * mesh_cols + col)
    metadata = torch.tensor([[source, 7, 1] for source in sources], dtype=torch.int32)

    out = ExpertMapping.metadata_source_for_sp_axis(metadata, num_dispatch_groups, dispatch_group_size, sp_axis)

    assert out[:, 0].tolist() == expected
    assert torch.equal(out[:, 1:], metadata[:, 1:])


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


def test_stage_sp_axis_is_the_knob_over_the_stage_default(monkeypatch, expect_error):
    from models.demos.common.prefill.runners.runner_utils import SP_AXIS_ENV, default_sp_axis, stage_sp_axis

    monkeypatch.delenv(SP_AXIS_ENV, raising=False)
    # A 4x4 tray half rings only axis 1; every other stage keeps SP on axis 0.
    assert (default_sp_axis((4, 4)), default_sp_axis((8, 4))) == (1, 0)
    assert (stage_sp_axis((4, 4)), stage_sp_axis((8, 4))) == (1, 0)
    monkeypatch.setenv(SP_AXIS_ENV, "0")
    assert (stage_sp_axis((4, 4)), default_sp_axis((4, 4))) == (0, 1)
    monkeypatch.setenv(SP_AXIS_ENV, "")
    assert stage_sp_axis((4, 4)) == 1
    monkeypatch.setenv(SP_AXIS_ENV, "2")
    with expect_error(AssertionError, "must be 0 or 1"):
        stage_sp_axis((4, 4))


def test_kimi_k3_weight_cache_key_follows_the_sp_axis(monkeypatch, tmp_path):
    from models.demos.common.prefill.adapter import get_adapter

    checkpoint = tmp_path / "Kimi-K3.1-Test"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text("{}")
    monkeypatch.setenv("PREFILL_HF_MODEL", str(checkpoint))
    adapter = get_adapter("kimi_k3_1")
    default, flipped = adapter.weight_cache_key((4, 4), 1), adapter.weight_cache_key((4, 4), 0)
    # One checkpoint directory; the TP axis (the other one) picks the placement.
    assert default.parent == flipped.parent and len(default.parent.name) == 16
    assert (default.name, flipped.name) == ("mesh4x4.tpaxis0", "mesh4x4.tpaxis1")
    assert adapter.weight_cache_key((8, 4), 0).name == "mesh8x4.tpaxis1"
    # A different checkpoint never shares a directory.
    other = tmp_path / "Kimi-K3.1-Other"
    other.mkdir()
    (other / "model.safetensors.index.json").write_text("{}")
    monkeypatch.setenv("PREFILL_HF_MODEL", str(other))
    assert adapter.weight_cache_key((4, 4), 1).parent != default.parent


def test_kimi_k3_weight_cache_key_needs_the_checkpoint(monkeypatch, expect_error):
    from models.demos.common.prefill.adapter import get_adapter

    # K3's hf_model_default is a config-only directory: no index to key the cache by, so the runner must be told.
    monkeypatch.delenv("PREFILL_HF_MODEL", raising=False)
    with expect_error(AssertionError, "set PREFILL_HF_MODEL"):
        get_adapter("kimi_k3").weight_cache_key((4, 4), 1)


@pytest.mark.parametrize(
    "fabric_mode, descriptor, sp_axis, expected",
    [
        # An explicit mode wins over the descriptor.
        ("2d_torus_y", "some_mgd.textproto", 0, "FABRIC_2D_TORUS_Y"),
        # A descriptor: the control plane wraps its RING axes, on either SP axis.
        ("", "some_mgd.textproto", 0, "MGD_DEFAULT"),
        ("", "some_mgd.textproto", 1, "MGD_DEFAULT"),
        # No descriptor: torus XY with SP on axis 0, torus X (the tray half's ring) with SP on axis 1.
        ("", "", 0, "FABRIC_2D_TORUS_XY"),
        ("", "", 1, "FABRIC_2D_TORUS_X"),
    ],
)
def test_open_mesh_device_fabric_follows_mode_then_descriptor(monkeypatch, fabric_mode, descriptor, sp_axis, expected):
    from models.demos.common.prefill.runners import runner_utils

    requested = []
    monkeypatch.setenv("PREFILL_FABRIC_MODE", fabric_mode)
    monkeypatch.setenv("TT_MESH_GRAPH_DESC_PATH", descriptor)
    monkeypatch.setattr(runner_utils.ttnn, "set_fabric_config", lambda config, *args: requested.append(config))
    monkeypatch.setattr(runner_utils.ttnn, "open_mesh_device", lambda **kwargs: "mesh")
    monkeypatch.setattr(runner_utils.ttnn, "get_fabric_config", lambda mesh_device: requested[-1])
    assert runner_utils.open_mesh_device((4, 4), 4096, sp_axis=sp_axis) == "mesh"
    assert requested == [getattr(ttnn.FabricConfig, expected)]
