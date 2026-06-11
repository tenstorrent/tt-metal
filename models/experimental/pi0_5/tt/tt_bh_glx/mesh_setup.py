# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Open the 8x4 BH Galaxy parent mesh and carve into vision/prefill/denoise submeshes.

Host-bounce only — no fabric, no P2P, no sockets. Just submesh creation and
per-chip materialization for layer-paired execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import ttnn

from . import stages
from .stages import MeshHandles


def _carve_per_chip(parent_mesh, submesh_shape, submesh_offset, num_chips):
    """Create num_chips 1x1 submeshes covering submesh_shape at submesh_offset.

    Returns a list of (1,1) submeshes in row-major order. The parent submesh
    is implied (= shape × offset) but we don't materialize it separately —
    only the per-chip 1x1's, since host-bounce only ever talks to single chips.
    """
    rows, cols = submesh_shape
    base_r, base_c = submesh_offset
    per_chip = []
    for r in range(rows):
        for c in range(cols):
            sm = parent_mesh.create_submesh(
                ttnn.MeshShape(1, 1),
                ttnn.MeshCoordinate(base_r + r, base_c + c),
            )
            per_chip.append(sm)
    if len(per_chip) != num_chips:
        raise RuntimeError(f"expected {num_chips} chips, materialized {len(per_chip)}")
    return per_chip


@contextmanager
def open_galaxy_mesh(l1_small_size: Optional[int] = None, trace_region_size: Optional[int] = None):
    """Open parent (8,4) mesh and carve 4/18/6 submeshes + per-chip 1x1s.

    Yields MeshHandles. On exit, every submesh is closed before the parent
    (parent-last close avoids the wedged-device firmware-init hang we hit
    when closing out of order — see option_c/mesh_setup.py).
    """
    parent_shape = ttnn.MeshShape(*stages.PARENT_MESH_SHAPE)
    open_kwargs = {"mesh_shape": parent_shape}
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    if trace_region_size is not None:
        open_kwargs["trace_region_size"] = trace_region_size

    parent = ttnn.open_mesh_device(**open_kwargs)
    all_submeshes = []
    try:
        if parent.get_num_devices() != stages.PARENT_MESH_SHAPE[0] * stages.PARENT_MESH_SHAPE[1]:
            raise RuntimeError(
                f"Parent mesh has {parent.get_num_devices()} devices, expected "
                f"{stages.PARENT_MESH_SHAPE[0] * stages.PARENT_MESH_SHAPE[1]} for BH Galaxy"
            )

        vision_submesh = parent.create_submesh(
            ttnn.MeshShape(*stages.VISION_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*stages.VISION_SUBMESH_OFFSET),
        )
        all_submeshes.append(vision_submesh)

        prefill_submesh = parent.create_submesh(
            ttnn.MeshShape(*stages.PREFILL_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*stages.PREFILL_SUBMESH_OFFSET),
        )
        all_submeshes.append(prefill_submesh)

        denoise_submesh = parent.create_submesh(
            ttnn.MeshShape(*stages.DENOISE_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*stages.DENOISE_SUBMESH_OFFSET),
        )
        all_submeshes.append(denoise_submesh)

        vision_per_chip = _carve_per_chip(
            parent, stages.VISION_SUBMESH_SHAPE, stages.VISION_SUBMESH_OFFSET, stages.VISION_NUM_CHIPS
        )
        all_submeshes.extend(vision_per_chip)
        prefill_per_chip = _carve_per_chip(
            parent, stages.PREFILL_SUBMESH_SHAPE, stages.PREFILL_SUBMESH_OFFSET, stages.PREFILL_NUM_CHIPS
        )
        all_submeshes.extend(prefill_per_chip)
        denoise_per_chip = _carve_per_chip(
            parent, stages.DENOISE_SUBMESH_SHAPE, stages.DENOISE_SUBMESH_OFFSET, stages.DENOISE_NUM_CHIPS
        )
        all_submeshes.extend(denoise_per_chip)

        yield MeshHandles(
            parent=parent,
            vision_submesh=vision_submesh,
            prefill_submesh=prefill_submesh,
            denoise_submesh=denoise_submesh,
            vision_per_chip=vision_per_chip,
            prefill_per_chip=prefill_per_chip,
            denoise_per_chip=denoise_per_chip,
        )
    finally:
        for sm in reversed(all_submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
