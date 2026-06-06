# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures/helpers for the dots_ocr_tp4 rebuild tests."""

import os

import ttnn

# MESH_DEVICE -> (rows, cols). Default to the 4-chip 1x4 ring on this host.
MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "P100": (1, 1),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "T3K": (1, 8),
}


def resolve_mesh_shape():
    md = os.environ.get("MESH_DEVICE")
    if md in MESH_DEVICE_MAP:
        return MESH_DEVICE_MAP[md]
    try:
        n = len(ttnn.get_device_ids())
    except Exception:
        n = 1
    return (1, max(1, n))


def mesh_num_devices_for_shape(shape) -> int:
    if isinstance(shape, (tuple, list)):
        n = 1
        for d in shape:
            n *= int(d)
        return n
    return int(shape)


def device_params():
    n = mesh_num_devices_for_shape(resolve_mesh_shape())
    dp = {"trace_region_size": 0, "num_command_queues": 1}
    dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING if n > 1 else ttnn.FabricConfig.DISABLED
    return dp
