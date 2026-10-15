# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the GPT-OSS demo.
"""

import ttnn
from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def get_compute_grid_size(mesh_device, max_x=8, max_y=8):
    """Compute grid to use on the given device: the device grid, limited to max_x by max_y.

    The model's core grids were sized for 8x8. Devices with a larger grid still use 8x8, and devices
    with a smaller grid (such as the emulated Quasar device) use their full grid.
    """
    device_grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(max_x, device_grid.x), min(max_y, device_grid.y))


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole exposes 2 fabric links per device; Wormhole exposes 4. Single-row meshes
    (shape[0] == 1) only need 1 link regardless of arch.
    """
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4


def throughput_experts_supported_on_arch():
    """Whether the throughput-optimized experts path (all_to_all dispatch/combine, moe_gpt,
    selective_reduce_combine) is supported on the current arch.

    Currently disabled on Blackhole; only the batch=1 low-latency expert path runs there.
    """
    return not is_blackhole()
