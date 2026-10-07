# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported meshes before loading weights or constructing the decoder."""

from unittest.mock import Mock

import pytest

import ttnn
from models.demos.qwen38_27b_qb2.tt import model
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder, validate_qb2_mesh
from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
from models.demos.qwen38_27b_qb2.tt.topology import resolve_tp4_topology


@pytest.mark.parametrize("entrypoint", ["model", "decoder"])
@pytest.mark.parametrize("unsupported", ["architecture", "product", "shape", "device_count"])
def test_rejects_unsupported_mesh_before_loading_weights(monkeypatch, expect_error, entrypoint, unsupported):
    mesh = Mock(shape=(2, 2) if unsupported == "shape" else (1, 4))
    mesh.arch.return_value = ttnn.Arch.WORMHOLE_B0 if unsupported == "architecture" else ttnn.Arch.BLACKHOLE
    mesh.get_num_devices.return_value = 3 if unsupported == "device_count" else 4
    monkeypatch.setattr(
        ttnn.cluster,
        "get_cluster_type",
        lambda: ttnn.cluster.ClusterType.P150_X4 if unsupported == "product" else ttnn.cluster.ClusterType.P300_X2,
    )
    precision_loader = Mock()
    checkpoint_loader = Mock()
    monkeypatch.setattr(model, "load_precision", precision_loader)
    monkeypatch.setattr(model, "checkpoint_path", checkpoint_loader)
    with expect_error(ValueError, "requires a Blackhole TP4"):
        if entrypoint == "model":
            model.Qwen38Model(mesh)
        else:
            Qwen38TPDecoder.from_state_dict(None, hf_config=None, layer_idx=0, mesh_device=mesh)
    precision_loader.assert_not_called()
    checkpoint_loader.assert_not_called()


@pytest.mark.parametrize("cluster", [ttnn.cluster.ClusterType.P300_X2, ttnn.cluster.ClusterType.BLACKHOLE_GALAXY])
def test_accepts_tp4_replica(monkeypatch, cluster):
    mesh = Mock(shape=(1, 4))
    mesh.arch.return_value = ttnn.Arch.BLACKHOLE
    mesh.get_num_devices.return_value = 4
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster)
    validate_qb2_mesh(mesh)


@pytest.mark.parametrize(
    "cluster,topology,fabric",
    [
        (ttnn.cluster.ClusterType.P300_X2, ttnn.Topology.Ring, ttnn.FabricConfig.FABRIC_1D_RING),
        (ttnn.cluster.ClusterType.BLACKHOLE_GALAXY, ttnn.Topology.Linear, ttnn.FabricConfig.FABRIC_1D),
    ],
)
def test_default_topology_matches_fabric(monkeypatch, cluster, topology, fabric):
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster)
    configure = Mock()
    monkeypatch.setattr(ttnn, "set_fabric_config", configure)
    assert resolve_tp4_topology() == topology
    configure_fabric()
    assert configure.call_args.args == (fabric,)
    assert configure.call_args.kwargs["router_config"].max_packet_payload_size_bytes == 8192


def test_explicit_qualified_ring_and_invalid_topology(monkeypatch, expect_error):
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.BLACKHOLE_GALAXY)
    configure = Mock()
    monkeypatch.setattr(ttnn, "set_fabric_config", configure)
    configure_fabric(topology=ttnn.Topology.Ring)
    assert configure.call_args.args == (ttnn.FabricConfig.FABRIC_1D_RING,)
    with expect_error(ValueError, "must be Linear or Ring"):
        resolve_tp4_topology("ring")
