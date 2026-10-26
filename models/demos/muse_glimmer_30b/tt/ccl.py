# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TP collectives for Muse-Glimmer (1 x TP mesh, cluster_axis=1)."""

from loguru import logger

import ttnn


def is_blackhole_device(mesh_device) -> bool:
    return mesh_device.arch() == ttnn.device.Arch.BLACKHOLE


def default_topology(tp: int):
    """Ring if tp >= 8 else Linear."""
    return ttnn.Topology.Ring if tp >= 8 else ttnn.Topology.Linear


def default_num_links(mesh_device) -> int:
    """2 links on Blackhole, 1 otherwise."""
    return 2 if is_blackhole_device(mesh_device) else 1


def fabric_config_for(tp: int):
    """Fabric config to set BEFORE opening the mesh (None when tp == 1)."""
    if tp <= 1:
        return None
    return ttnn.FabricConfig.FABRIC_1D


class MuseCCL:
    """Synchronous all_reduce / all_gather along mesh axis 1.

    ``topology``/``num_links`` are not passed to the synchronous ops: the fabric config (FABRIC_1D) decides the
    routing. They are kept for logging and the sampling all-gather.
    """

    def __init__(self, mesh_device, topology=None, num_links=None):
        self.mesh_device = mesh_device
        self.tp = int(mesh_device.shape[1])
        self.cluster_axis = 1
        self.topology = topology if topology is not None else default_topology(self.tp)
        self.num_links = num_links if num_links is not None else default_num_links(mesh_device)
        logger.info(f"MuseCCL: tp={self.tp} topology={self.topology} num_links={self.num_links}")

    def all_reduce(self, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        if self.tp == 1:
            return x
        return ttnn.all_reduce(x, cluster_axis=self.cluster_axis, memory_config=memory_config)

    def all_gather(self, x, dim: int, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        if self.tp == 1:
            return x
        return ttnn.all_gather(x, dim=dim, cluster_axis=self.cluster_axis, memory_config=memory_config)
