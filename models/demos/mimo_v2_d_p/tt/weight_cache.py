# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN weight cache: ttnn.as_tensor / flatbuffer tensorbins per mesh shape. On by default under the extracted
checkpoint (``<LOCAL>/ttnn_cache``); ``MIMO_TTNN_CACHE=<dir>`` moves it, ``MIMO_TTNN_CACHE=0`` turns it off.

as_tensor encodes dtype + layout in the file name but not the mesh mapping, so the directory is keyed by mesh shape.
"""

import os
from pathlib import Path


def cache_dir(mesh_device) -> Path | None:
    root = os.environ.get("MIMO_TTNN_CACHE")
    if root in ("0", "off", "none", ""):
        return None
    if root is None:
        from models.demos.mimo_v2_d_p.reference.remote_st import LOCAL

        root = LOCAL / "ttnn_cache"
    rows, cols = tuple(mesh_device.shape)
    return Path(root) / f"mesh{rows}x{cols}"


def cache_name(mesh_device, prefix: str | None, name: str) -> str | None:
    d = cache_dir(mesh_device)
    return None if d is None or prefix is None else str(d / f"{prefix}.{name}")
