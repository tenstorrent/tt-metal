# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCLManager:
    """Lightweight CCL manager for Gemma4 tensor parallelism."""

    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
