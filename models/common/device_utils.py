# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device topology naming helpers shared by TTTv2 modules."""

from __future__ import annotations

import ttnn


def is_blackhole() -> bool:
    return "blackhole" in ttnn.get_arch_name()


def get_device_name(mesh_device: ttnn.MeshDevice, num_devices: int | None = None) -> str:
    """Return the product/topology name for a TT mesh device.

    By default, the full mesh device count is used. CCL callers can pass a
    host-local device count when they need link-count tuning for the current
    process rather than for the full mesh.
    """
    num_devices = mesh_device.get_num_devices() if num_devices is None else num_devices
    dram_grid_size = mesh_device.dram_grid_size()

    if ttnn.device.is_blackhole(mesh_device):
        device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif ttnn.device.is_wormhole_b0(mesh_device):
        device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {ttnn.get_arch_name()}")

    if num_devices in device_names:
        return device_names[num_devices]
    raise ValueError(f"Unsupported number of devices: {num_devices} for {ttnn.get_arch_name()}")
