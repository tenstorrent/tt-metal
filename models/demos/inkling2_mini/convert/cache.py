# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Loader for the Inkling-2-Mini tt weight cache v1 (written by convert.py).

``load_layer(root, L, device)`` / ``load_global(root, device)`` return ``{cache name: ttnn.Tensor}`` in the stored
dtype + layout. ``device=None`` (default) returns host tensors, so a module can place them once itself; with a
``device`` they go to DRAM interleaved.
"""

from __future__ import annotations

import json
from pathlib import Path

import ttnn
from models.demos.inkling2_mini import modes

# The weight caches loaded by default: the run mode's (modes.py: own3-def8, or own3-l8 with INKLING_MODE=fast).
# Decode: the PF expert layout (expert prefetch). Prefill: the TILE twin (prefill reads the experts as tiles).
# $INKLING_CACHE / $INKLING_TILE_CACHE override them; the earlier defaults were cache/own2-def-pf and -tile.
CACHE_DIR = modes.CACHE_DIR
DEFAULT_ROOT, DEFAULT_TILE_ROOT = modes.roots()


def _root(root) -> Path:
    if root is None:
        raise ValueError("no weight cache given: set INKLING_CACHE or pass the cache directory")
    return Path(root)


def manifest(root: Path | str | None = DEFAULT_ROOT) -> dict:
    root = _root(root)
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"{root}: no weight cache here. Set INKLING_CACHE_DIR to the directory that holds the run modes' caches "
            f"(now {modes.CACHE_DIR}), or INKLING_CACHE / INKLING_TILE_CACHE to one cache"
        )
    m = json.loads((root / "manifest.json").read_text())
    if m.get("schema") != 1:
        raise ValueError(f"{root}: manifest schema {m.get('schema')} != 1")
    if not m.get("complete"):
        raise ValueError(
            f"{root}: manifest is not complete (conversion unfinished or interrupted); rerun the converter"
        )
    return m


def layouts(root: Path | str, layer: int | None) -> dict[str, str]:
    """{cache name: "RM" | "TILE" | "STICK"} for one layer (None = global). A "STICK" tensor (convert/stick.py) is a
    [1, 1, rows, 32] TILE tensor holding the expert tiles of its ``tile_shape`` reordered for the fused MoE kernel;
    it must sit DRAM interleaved on a device with ``stick_banks`` DRAM banks."""
    return {e["name"]: e["layout"] for e in manifest(root)["tensors"] if e["layer"] == layer}


def _load(root: Path | str, layer: int | None, device) -> dict[str, ttnn.Tensor]:
    root = _root(root)
    out = {}
    for e in manifest(root)["tensors"]:
        if e["layer"] != layer:
            continue
        t = ttnn.load_tensor(root / e["file"])
        if device is not None and e["layout"] == "STICK" and device.dram_grid_size().x != e["stick_banks"]:
            raise ValueError(f"{e['file']}: stick layout for {e['stick_banks']} DRAM banks, device has "
                             f"{device.dram_grid_size().x}")  # fmt: skip
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
