# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN weight cache: ttnn.as_tensor / flatbuffer tensorbins per mesh shape. On by default under the extracted
checkpoint (``<LOCAL>/ttnn_cache``); ``MiMoRuntimeOptions.ttnn_cache_root`` moves it, ``ttnn_cache=False`` turns it off.

as_tensor encodes dtype + layout in the file name but not the mesh mapping, so the directory is keyed by mesh shape.
"""

from pathlib import Path

from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions


def cache_dir(mesh_device, options: MiMoRuntimeOptions | None = None) -> Path | None:
    options = options or MiMoRuntimeOptions()
    if not options.ttnn_cache:
        return None
    root = options.ttnn_cache_root
    if root is None:
        from models.demos.mimo_v2_d_p.reference.remote_st import LOCAL

        root = LOCAL / "ttnn_cache"
    rows, cols = tuple(mesh_device.shape)
    return Path(root) / f"mesh{rows}x{cols}"


def cache_name(mesh_device, prefix: str | None, name: str, options: MiMoRuntimeOptions | None = None) -> str | None:
    d = cache_dir(mesh_device, options)
    return None if d is None or prefix is None else str(d / f"{prefix}.{name}")
