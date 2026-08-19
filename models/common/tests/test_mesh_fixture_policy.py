# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.common.tests.conftest import _default_fabric_config


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
    ("cluster_type", "num_devices", "expected"),
    [
        (ttnn.cluster.ClusterType.T3K, 8, ttnn.Topology.Ring),
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
