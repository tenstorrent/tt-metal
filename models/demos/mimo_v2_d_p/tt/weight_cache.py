# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TTNN weight cache (``MIMO_TTNN_CACHE=<dir>``): ttnn.as_tensor tensorbins per mesh shape.

as_tensor encodes dtype + layout in the file name but not the mesh mapping, so the directory is keyed by mesh shape.
"""

import os
from pathlib import Path


def cache_dir(mesh_device) -> Path | None:
    root = os.environ.get("MIMO_TTNN_CACHE")
    if not root:
        return None
    rows, cols = tuple(mesh_device.shape)
    return Path(root) / f"mesh{rows}x{cols}"


def cache_name(mesh_device, prefix: str | None, name: str) -> str | None:
    d = cache_dir(mesh_device)
    return None if d is None or prefix is None else str(d / f"{prefix}.{name}")
