# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

# =============================================================================
# CCL tuning defaults - shared across all TTTv2 modules
# =============================================================================

# Default number of chunks per synchronization barrier in CCL operations.
# Higher values reduce sync overhead but increase latency per chunk.
CCL_CHUNKS_PER_SYNC = 10

# Default number of worker threads per Ethernet link for CCL operations.
CCL_NUM_WORKERS_PER_LINK = 2

# Default number of double-buffered channels per CCL link.
CCL_NUM_BUFFERS_PER_CHANNEL = 2

# =============================================================================
# TT_CCL cache - one instance per mesh_device (semaphores are hardware resources)
# =============================================================================


_tt_ccl_cache: dict[int, "TT_CCL"] = {}


def get_tt_ccl(mesh_device: ttnn.MeshDevice) -> "TT_CCL":
    """Get or create TT_CCL for mesh_device (cached per device id)."""
    mesh_id = mesh_device.id()
    if mesh_id not in _tt_ccl_cache:
        _tt_ccl_cache[mesh_id] = TT_CCL(mesh_device)
    return _tt_ccl_cache[mesh_id]


def clear_tt_ccl_cache():
    """Clear cache (for testing)."""
    _tt_ccl_cache.clear()


# =============================================================================
# TT_CCL class
# =============================================================================


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self.mesh_device.compute_with_storage_grid_size().x - 1,
                        self.mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )

        self.barrier_semaphore_idx = [0, 0, 0]
        self.barrier_semaphore_handles = [[], [], []]

        self.ag_semaphores_idx = [0, 0, 0]
        self.ag_semaphore_handles = [[], [], []]

        self.rs_semaphores_idx = [0, 0, 0]
        self.rs_semaphore_handles = [[], [], []]

        # cluster-axis-0, cluster-axis-1, no-cluster-axis
        for i in range(3):
            # double buffered semaphores
            for _ in range(2):
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
                )

                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                )

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        current_idx = self.barrier_semaphore_idx[semaphore_index]
        self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
        return self.barrier_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        current_idx = self.ag_semaphores_idx[semaphore_index]
        self.ag_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.ag_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        current_idx = self.rs_semaphores_idx[semaphore_index]
        self.rs_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.rs_semaphore_handles[semaphore_index][current_idx]

    def get_num_links(self, cluster_axis=None):
        """Get the number of available Ethernet links for CCL operations on this mesh device."""
        return get_num_links(self.mesh_device, cluster_axis)


# =============================================================================
# Topology auto-detection
# =============================================================================


# todo)) work with the CCL team to find opportunity to simplify this --> e.g., build into TTNN APIs?
def default_topology(mesh_device: ttnn.MeshDevice) -> Optional[ttnn.Topology]:
    """Auto-detect CCL topology based on cluster type and device count."""
    num_devices = mesh_device.get_num_devices()
    cluster_type = ttnn.cluster.get_cluster_type()
    if (num_devices == 8 and cluster_type == ttnn.cluster.ClusterType.T3K) or (
        num_devices == 4 and cluster_type == ttnn.cluster.ClusterType.P150_X4
    ):
        # NOTE: we always want to do ring if it is available
        return ttnn.Topology.Ring
    elif num_devices > 1:
        # NOTE: this should be a fallback when the ring is not available
        return ttnn.Topology.Linear
    return None


def get_num_links(mesh_device: ttnn.MeshDevice, cluster_axis: int | None = None) -> int:
    """
    Get the number of available Ethernet links for CCL operations.

    Args:
        mesh_device: The mesh device to query.
        cluster_axis: Optional cluster axis to query links for.
            - 0: vertical axis (North-South).
            - 1: horizontal axis (East-West).
            - None: minimum across all axes.

    Returns:
        int: The number of available links. 0 for a single-device mesh.
    """
    if cluster_axis not in (None, 0, 1):
        raise ValueError(f"Unsupported cluster_axis: {cluster_axis}")
    if mesh_device.get_num_devices() == 1:
        return 0
    return ttnn.get_num_links(mesh_device, cluster_axis)
