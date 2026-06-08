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


def _trace_enabled() -> bool:
    return os.environ.get("DOTS_OCR_TP4_TRACE", "").lower() in {"1", "true", "yes", "on"}


def device_params():
    n = mesh_num_devices_for_shape(resolve_mesh_shape())
    # Reserve a trace region only when traced decode is requested (capture/replay
    # needs it); otherwise keep it 0 so prefill-only runs don't reserve DRAM.
    trace_size = 300_000_000 if _trace_enabled() else 0
    dp = {"trace_region_size": trace_size, "num_command_queues": 1}
    dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING if n > 1 else ttnn.FabricConfig.DISABLED
    return dp
