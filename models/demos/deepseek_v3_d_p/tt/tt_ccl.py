# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

# NOTE: This file is forked from models/common/modules/tt_ccl.py
#       This is done to include logic for divifing the grid of cores for ring attention
#       One row is taken for the CCL communication of the op

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


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
        full_compute_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        full_compute_grid.x - 1,
                        full_compute_grid.y - 1,
                    ),
                )
            }
        )

        self.ring_attention_ccl_core_grid_offset = (0, full_compute_grid.y - 1)
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        self.worker_sub_device = ttnn.SubDevice(
            [
                ccl_sub_device_crs,
            ]
        )
        self.worker_sub_device_id = ttnn.SubDeviceId(0)
        self.sub_device_stall_group = [self.worker_sub_device_id]

        self.sub_device_manager = self.mesh_device.create_sub_device_manager([self.worker_sub_device], 0)
        self.mesh_device.load_sub_device_manager(self.sub_device_manager)
        self.mesh_device.set_sub_device_stall_group(self.sub_device_stall_group)

        # create global semaphore handles
        self.ring_attention_ccl_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0)

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
    if num_devices == 8 and ttnn.cluster.get_cluster_type() in [
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
    ]:
        # NOTE: we always want to do ring if it is available
        return ttnn.Topology.Ring
    elif num_devices > 1:
        # NOTE: this should be a fallback when the ring is not available
        return ttnn.Topology.Linear
    return None


# =============================================================================
# Device name / link count helpers (copied from TTTv1 ccl.py + model_config.py
# to avoid importing from tt_transformers)
# =============================================================================


def _determine_device_name(mesh_device: ttnn.MeshDevice) -> str:
    """Determine device name based on number of devices and architecture."""
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None

    if num_devices == 0:
        return "CPU"

    if "blackhole" in arch_name:
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif "wormhole_b0" in arch_name:
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices in dict_device_names:
        return dict_device_names[num_devices]
    raise ValueError(f"Unsupported number of devices: {num_devices} for {arch_name}")


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
        int: The number of available links.
    """
    device_name = _determine_device_name(mesh_device)
    link_dict = {
        "P100": (0, 0),
        "P150": (0, 0),
        "N150": (0, 0),
        "N300": (1, 1),
        "T3K": (1, 1),
        "P150x4": (2, 2),
        "P150x8": (2, 2),
        "P300": (2, 2),
        "BHGLX": (2, 2),  # NOTE: Possible increase to 4 when it's enabled
        "TG": (4, 4),
        "N150x4": (1, 1),
    }
    device_links = link_dict[device_name]
    if cluster_axis is None:
        return min(device_links)
    if cluster_axis in (0, 1):
        return device_links[cluster_axis]
    return min(device_links)
