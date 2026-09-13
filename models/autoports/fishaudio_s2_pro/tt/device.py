"""Mesh open/close for the four board profiles, with the fabric config each needs.

Profiles: p150 (1x1), p300 (1x2, FABRIC_1D), p300x2/QB2 (1x4, FABRIC_1D), p300x4 (1x8, FABRIC_1D_RING).
The env the tt-model launcher sets: MESH_DEVICE (label) and FISH_S2_MESH_SHAPE ("RxC"); the profile adds
FISH_S2_FABRIC_CONFIG. Smaller profiles on a bigger box open the full parent and take a submesh (fabric
cannot be initialised on a subset of devices).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import ttnn

PRESETS = {
    "P150": (1, 1),
    "P100": (1, 1),
    "N150": (1, 1),
    "P300": (1, 2),
    "P150x2": (1, 2),
    "N300": (1, 2),
    "P300x2": (1, 4),
    "P150x4": (1, 4),
    "QB2": (1, 4),
    "P300x4": (1, 8),
    "P150x8": (1, 8),
    "T3K": (1, 8),
}
DEFAULT_FABRIC = {1: None, 2: "FABRIC_1D", 4: "FABRIC_1D", 8: "FABRIC_1D_RING"}
# MAX_PREFILL_CHUNK_SIZE (in K tokens) tt_transformers should use per chip count; the table lookup for an
# unknown model name falls back to 4 with a warning, so we set it explicitly (model_config.py:2607-2618).
PREFILL_CHUNK_K = {1: 4, 2: 32, 4: 128, 8: 128}


def parse_shape(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    s = s.strip()
    if "x" in s.lower():
        r, c = s.lower().split("x")
        return int(r), int(c)
    if s in PRESETS:
        return PRESETS[s]
    import ast

    v = ast.literal_eval(s)
    return int(v[0]), int(v[1])


def requested_shape() -> Tuple[int, int]:
    shape = parse_shape(os.environ.get("FISH_S2_MESH_SHAPE")) or parse_shape(os.environ.get("MESH_DEVICE"))
    if shape is None:
        n = ttnn.get_num_devices()
        shape = (1, n)
    return shape


class MeshHandle:
    def __init__(self, parent, mesh):
        self.parent, self.mesh = parent, mesh

    @property
    def shape(self):
        return tuple(self.mesh.shape)

    @property
    def num_devices(self):
        return self.mesh.get_num_devices()

    def close(self):
        ttnn.close_mesh_device(self.parent)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


def open_mesh(
    shape: Optional[Tuple[int, int]] = None,
    fabric: Optional[str] = None,
    l1_small_size: int = 65536,
    trace_region_size: int = 0,
    num_command_queues: int = 1,
) -> MeshHandle:
    shape = shape or requested_shape()
    r, c = shape
    n_req = r * c
    n_sys = ttnn.get_num_devices()
    assert n_req <= n_sys, f"profile needs {n_req} chips, system has {n_sys}"
    fabric = os.environ.get("FISH_S2_FABRIC_CONFIG", fabric) if fabric is None else fabric
    if fabric is None:
        fabric = DEFAULT_FABRIC.get(n_req)
    if fabric and fabric.lower() != "none" and n_req > 1:
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric))
    os.environ.setdefault("MAX_PREFILL_CHUNK_SIZE", str(PREFILL_CHUNK_K.get(n_req, 4)))
    params = dict(
        l1_small_size=l1_small_size, trace_region_size=trace_region_size, num_command_queues=num_command_queues
    )
    if n_req == n_sys:
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(r, c), **params)
        return MeshHandle(parent, parent)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, n_sys), **params)
    sub = parent.create_submesh(ttnn.MeshShape(r, c))
    return MeshHandle(parent, sub)
