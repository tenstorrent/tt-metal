# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One TP4 topology contract for fabric, decoder, embedding and sampling."""

import ttnn


def validate_tp4_mesh(mesh_device):
    """Validate the replica, not the full parent Galaxy mesh."""
    arch = mesh_device.arch()
    cluster = ttnn.cluster.get_cluster_type()
    count, shape = mesh_device.get_num_devices(), tuple(mesh_device.shape)
    if (
        arch != ttnn.Arch.BLACKHOLE
        or count != 4
        or shape != (1, 4)
        or cluster not in (ttnn.cluster.ClusterType.P300_X2, ttnn.cluster.ClusterType.BLACKHOLE_GALAXY)
    ):
        raise ValueError(
            "Qwen3.8-27B requires a Blackhole TP4 (1, 4) replica on QB2 or Galaxy; "
            f"got arch={arch}, cluster_type={cluster}, num_devices={count}, mesh_shape={shape}"
        )


def resolve_tp4_topology(topology=None):
    """Galaxy defaults to linear until wraparound links have been qualified.

    Passing Ring explicitly requires matching fabric setup before opening the
    mesh. The existing qualified QB2 ring remains the default on that product.
    """
    if topology is None:
        topology = (
            ttnn.Topology.Linear
            if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY
            else ttnn.Topology.Ring
        )
    if topology not in (ttnn.Topology.Linear, ttnn.Topology.Ring):
        raise ValueError("Qwen TP4 topology must be Linear or Ring")
    return topology
