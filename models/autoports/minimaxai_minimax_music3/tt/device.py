"""Mesh open/close for the single-chip profile (and, later, multi-chip ones).

The tt-model launcher sets MESH_DEVICE (label) and MUSIC3_MESH_SHAPE ("RxC"). A 1x1 profile on a multi-chip box
must be a TRUE 1x1 mesh: TT_METAL_VISIBLE_DEVICES=0 is set before ttnn touches the cluster (a submesh of the
4-chip parent hangs on the first large host->device write, see the QB2 notes). ttnn is imported lazily for that.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

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
PREFILL_CHUNK_K = {
    1: 4,
    2: 32,
    4: 128,
    8: 128,
}  # MAX_PREFILL_CHUNK_SIZE (K tokens) per chip count (model_config.py table default)


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
    shape = parse_shape(os.environ.get("MUSIC3_MESH_SHAPE")) or parse_shape(os.environ.get("MESH_DEVICE"))
    return shape or (1, 1)


def pin_visible_devices(shape: Tuple[int, int]) -> None:
    """Must run before the first ttnn cluster query. A 1x1 mesh pins device 0 unless the operator chose otherwise."""
    n = shape[0] * shape[1]
    if n == 1:
        os.environ.setdefault("TT_METAL_VISIBLE_DEVICES", "0")
    elif n == 2:
        os.environ.setdefault("TT_METAL_VISIBLE_DEVICES", "0,1")


class MeshHandle:
    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def shape(self):
        return tuple(self.mesh.shape)

    @property
    def num_devices(self):
        return self.mesh.get_num_devices()

    def close(self):
        import ttnn

        ttnn.close_mesh_device(self.mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


def open_mesh(
    shape: Optional[Tuple[int, int]] = None,
    fabric: Optional[str] = None,
    l1_small_size: int = 32768,
    trace_region_size: Optional[int] = None,
    num_command_queues: int = 1,
) -> MeshHandle:
    shape = shape or requested_shape()
    pin_visible_devices(shape)
    import ttnn

    r, c = shape
    n_req, n_sys = r * c, ttnn.get_num_devices()
    # ttnn.get_num_devices() reports the whole cluster even under TT_METAL_VISIBLE_DEVICES; open_mesh_device honours the
    # variable. Refuse to open a sub-mesh of a bigger parent (it hangs on large host->device writes on the QB2).
    vis = os.environ.get("TT_METAL_VISIBLE_DEVICES", "")
    n_vis = len([v for v in vis.split(",") if v.strip()]) if vis else n_sys
    assert n_req <= n_sys, f"profile needs {n_req} chips, system has {n_sys}"
    assert (
        n_req == n_vis
    ), f"profile needs a {r}x{c} mesh but TT_METAL_VISIBLE_DEVICES={vis!r} exposes {n_vis} of {n_sys} devices (true mesh, never a submesh)"
    fabric = os.environ.get("MUSIC3_FABRIC_CONFIG", fabric) if fabric is None else fabric
    if fabric is None:
        fabric = DEFAULT_FABRIC.get(n_req)
    if fabric and str(fabric).lower() != "none" and n_req > 1:
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric))
    os.environ.setdefault(
        "MAX_PREFILL_CHUNK_SIZE", os.environ.get("MUSIC3_PREFILL_CHUNK_K", str(PREFILL_CHUNK_K.get(n_req, 4)))
    )
    if trace_region_size is None:
        trace_region_size = int(float(os.environ.get("MUSIC3_TRACE_REGION_MB", "256")) * 1024 * 1024)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(r, c),
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
    )
    return MeshHandle(mesh)
