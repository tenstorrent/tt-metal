# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Loader for the Inkling-2-Mini tt weight cache v1 (design.md §4).

``load_layer(root, L, device)`` / ``load_global(root, device)`` return ``{§4 name: ttnn.Tensor}`` in the stored
dtype + layout. ``device=None`` (default) returns host tensors, so a module can place them once itself; with a
``device`` they go to DRAM interleaved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ttnn

DEFAULT_ROOT = Path(os.environ.get("INKLING_CACHE", "/work/inkling-bringup/cache/v1"))


def manifest(root: Path | str = DEFAULT_ROOT) -> dict:
    m = json.loads((Path(root) / "manifest.json").read_text())
    if m.get("schema") != 1:
        raise ValueError(f"{root}: manifest schema {m.get('schema')} != 1")
    return m


def _load(root: Path | str, layer: int | None, device) -> dict[str, ttnn.Tensor]:
    out = {}
    for e in manifest(root)["tensors"]:
        if e["layer"] != layer:
            continue
        t = ttnn.load_tensor(Path(root) / e["file"])
        if device is not None:
            t = ttnn.to_device(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out[e["name"]] = t
    if not out:
        raise KeyError(f"{root}: no tensors for {'global' if layer is None else f'layer {layer}'} in manifest")
    return out


def load_layer(root: Path | str, layer: int, device=None) -> dict[str, ttnn.Tensor]:
    return _load(root, layer, device)


def load_global(root: Path | str, device=None) -> dict[str, ttnn.Tensor]:
    return _load(root, None, device)
