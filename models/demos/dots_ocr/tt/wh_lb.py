# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Wormhole **LB** (single Wormhole card / 1×1 mesh) helpers.

Use with:
- `MESH_DEVICE=N150` or `MESH_DEVICE=N300` (same as other tt-metal demos)
- `ttnn.MeshShape(1, 1)` — one chip, no galaxy / T3K fabric row.

For long-document OCR, reduce peak DRAM via `DOTS_MAX_SEQ_LEN_WH_LB` (see `DotsModelArgs`).
"""

from __future__ import annotations

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn


def default_mesh_shape_wh_lb():
    """Single-chip Wormhole: one device in the mesh."""
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")
    return ttnn.MeshShape(1, 1)


def assert_single_wormhole_device(mesh_device) -> None:
    n = mesh_device.get_num_devices()
    assert n == 1, f"Wormhole LB expects 1 device, got {n}"
