# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.common.tests.conftest import (
    _allowed_req_shapes_for_system,
    _default_fabric_config,
    _is_physical_p150x4_cluster,
    _pick_parent_shape_for_submesh,
)


@pytest.mark.parametrize(
    ("logical_shape", "expected"),
    [
        ((1, 1), None),
        ((1, 2), ttnn.FabricConfig.FABRIC_1D),
        ((1, 8), ttnn.FabricConfig.FABRIC_1D_RING),
        ((1, 16), ttnn.FabricConfig.FABRIC_1D_RING),
        ((2, 4), ttnn.FabricConfig.FABRIC_1D),
        ((4, 8), ttnn.FabricConfig.FABRIC_1D),
        ((8, 4), ttnn.FabricConfig.FABRIC_1D),
    ],
)
def test_default_fabric_config(logical_shape, expected):
    assert _default_fabric_config(logical_shape) == expected


@pytest.mark.parametrize(
    ("system_shape", "expected"),
    [
        ((1, 1), {(1, 1)}),
        ((1, 2), {(1, 2), (1, 1)}),
        ((2, 1), {(1, 2), (2, 1), (1, 1)}),
        ((1, 4), {(1, 4), (1, 2), (1, 1)}),
        ((4, 1), {(1, 4), (4, 1), (1, 2), (1, 1)}),
        ((2, 2), {(2, 2), (1, 4), (1, 2), (1, 1)}),
        ((2, 4), {(2, 4), (1, 8), (1, 4), (1, 2), (1, 1)}),
        ((8, 4), {(8, 4), (4, 8), (1, 8), (1, 4), (1, 2), (1, 1)}),
    ],
)
def test_allowed_req_shapes_for_system(system_shape, expected):
    assert _allowed_req_shapes_for_system(system_shape) == expected


def test_quietbox_square_system_uses_linear_full_device_view():
    system_shape = (2, 2)
    allowed = _allowed_req_shapes_for_system(system_shape, blackhole_selected=True)

    assert (1, 4) in allowed
    assert (2, 2) not in allowed
    assert _pick_parent_shape_for_submesh(system_shape, (1, 4)) == (1, 4)


@pytest.mark.parametrize("cluster_type", [ttnn.cluster.ClusterType.P150_X4, ttnn.cluster.ClusterType.P300_X2])
def test_physical_four_die_bh_systems_admit_p150x4(cluster_type):
    assert _is_physical_p150x4_cluster(cluster_type)


@pytest.mark.parametrize(
    ("cluster_type", "num_devices", "expected"),
    [
        (ttnn.cluster.ClusterType.T3K, 8, ttnn.Topology.Ring),
        (ttnn.cluster.ClusterType.P150_X4, 4, ttnn.Topology.Ring),
        (ttnn.cluster.ClusterType.GALAXY, 8, ttnn.Topology.Linear),
        (ttnn.cluster.ClusterType.GALAXY, 4, ttnn.Topology.Linear),
        (ttnn.cluster.ClusterType.GALAXY, 1, None),
    ],
)
def test_default_ccl_topology(monkeypatch, cluster_type, num_devices, expected):
    class MeshDevice:
        def get_num_devices(self):
            return num_devices

    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster_type)

    assert default_topology(MeshDevice()) == expected
