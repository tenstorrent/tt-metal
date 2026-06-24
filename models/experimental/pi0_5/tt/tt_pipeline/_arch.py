# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native arch introspection for the standalone pi0.5 streamed-denoise port.

Vendored verbatim from ``tt_symbiote.core.arch`` (itself vendored from tt-metal
``models/common/utility_functions.py`` + ``models/tt_transformers/tt/model_config.py``).
Pure ttnn -- no torch, no tt_symbiote. The linchpin: a 1x1 Blackhole submesh
(num_devices==1, dram_grid_size.x != 7) introspects as ``"P150"``, so a stage's
``@run_on_devices(DeviceArch.P150, ...)`` guard passes on both a standalone P150 board
AND under a BH-Galaxy parent (same Blackhole silicon, each stage = one P150 chip).
"""
from __future__ import annotations

import ttnn

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"


def is_blackhole() -> bool:
    return "blackhole" in ttnn.get_arch_name()


def is_wormhole_b0() -> bool:
    return "wormhole_b0" in ttnn.get_arch_name()


def determine_device_name(mesh_device) -> str:
    """Map a mesh device to a device-name string (e.g. "P150", "P150x4", "BHGLX")."""
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None

    if num_devices == 0:
        return "CPU"

    if is_blackhole():
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif is_wormhole_b0():
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
