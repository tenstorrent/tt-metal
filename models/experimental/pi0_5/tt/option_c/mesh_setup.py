# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh setup for pi0.5 Option C — heterogeneous submeshes on a Galaxy.

Unlike Option B (4 uniform 4x2 submeshes via `create_submeshes`), Option C
carves the 8x4 parent into 3 different-shape submeshes by calling
`MeshDevice.create_submesh(shape, offset)` once per stage.

Physical placement (see stages.py):

    col→  0 1 2 3
 row↓  0  V V _ _    V = vision  shape (2,2) offset (0,0)  4 chips
       1  V V _ _    _ = spare   shape (2,2) offset (0,2)  4 chips (not opened)
       2  P P P D    P = prefill shape (6,3) offset (2,0) 18 chips
       3  P P P D    D = denoise shape (6,1) offset (2,3)  6 chips
       4  P P P D
       5  P P P D
       6  P P P D
       7  P P P D
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import ttnn

from .stages import StageLayout


@contextmanager
def open_galaxy_mesh(
    layout: StageLayout,
    enable_fabric: bool = False,
    l1_small_size: Optional[int] = None,
    open_spare: bool = False,
):
    """Open the parent 8x4 mesh, carve into 3 (or 4) heterogeneous submeshes
    per Option C layout.

    Args:
        layout: the StageLayout to honour (3 stages).
        enable_fabric: if True, call `ttnn.set_fabric_config(FABRIC_1D)`
            before opening the mesh. Option C has no all_reduce within a
            stage, so fabric is only needed if inter-stage transport uses
            ttnn.all_gather / D2D sockets. Default False.
        l1_small_size: bytes per core reserved for static circular buffers.
            With Option C's L1-resident weights, the matmul kernels still
            need a small static CB region; 1 MB is plenty since we don't
            run all_reduce. Pass None to use the device default.
        open_spare: if True, also open the 2x2 spare submesh and yield it
            as the 4th element. Default False — spares stay closed so they
            don't consume L1.

    Yields:
        (parent_mesh, [vision_submesh, prefill_submesh, denoise_submesh])
        or, if open_spare, (parent_mesh, [vision, prefill, denoise, spare]).

    On exit, every submesh and the parent mesh are closed.
    """
    parent_shape = ttnn.MeshShape(*layout.parent_mesh_shape)

    if enable_fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    open_kwargs = {"mesh_shape": parent_shape}
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    parent = ttnn.open_mesh_device(**open_kwargs)
    try:
        submeshes: List = []
        for spec in layout.stages:
            sm = parent.create_submesh(
                ttnn.MeshShape(*spec.submesh_shape),
                ttnn.MeshCoordinate(*spec.submesh_offset),
            )
            expected = spec.num_chips
            actual = sm.get_num_devices()
            if actual != expected:
                raise RuntimeError(
                    f"Stage {spec.name}: expected {expected} chips from "
                    f"shape={spec.submesh_shape} offset={spec.submesh_offset}, "
                    f"got {actual}"
                )
            submeshes.append(sm)

        if open_spare:
            from .stages import SPARE_SUBMESH_SHAPE, SPARE_SUBMESH_OFFSET

            spare = parent.create_submesh(
                ttnn.MeshShape(*SPARE_SUBMESH_SHAPE),
                ttnn.MeshCoordinate(*SPARE_SUBMESH_OFFSET),
            )
            submeshes.append(spare)

        yield parent, submeshes
    finally:
        # Close submeshes before the parent. Otherwise tt-metal throws
        # "MeshDevice cq ID 0 is in use by parent mesh ID N during close of
        # mesh ID M" and leaves device firmware state torn down out-of-order
        # — Device 10 in particular ends up wedged for the next process and
        # subsequent ttnn.open_mesh_device() hangs 10s in firmware init.
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        submeshes.clear()
        ttnn.close_mesh_device(parent)
        if enable_fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def describe_submesh(submesh, name: str = "") -> str:
    """Human-readable submesh summary for logging."""
    rows, cols = submesh.shape[0], submesh.shape[1]
    n = submesh.get_num_devices()
    prefix = f"{name}: " if name else ""
    return f"{prefix}submesh shape=({rows},{cols}) devices={n}"


def create_per_chip_submeshes(parent_submesh, count: Optional[int] = None) -> List:
    """Carve `parent_submesh` into `count` single-chip (1, 1) sub-submeshes.

    Returns one ``ttnn.MeshDevice`` per chip in row-major order
    (col-fastest), i.e. submesh[0] = parent chip at (0, 0), submesh[1] =
    (0, 1), ..., submesh[cols-1] = (0, cols-1), submesh[cols] = (1, 0).

    This is the primitive Option C uses for layer-paired L1 placement:
    one VLM layer's weights live on exactly one chip, dispatched
    independently of the others, with host-bounce (or fabric, when it
    lands) activation transport between adjacent layers.

    Args:
        parent_submesh: an already-opened MeshDevice (typically the
            prefill or denoise stage submesh from `open_galaxy_mesh`).
        count: if given, must satisfy ``count <= parent.get_num_devices()``
            and only the first `count` chips (row-major) are carved.
            Default = all chips.

    Returns: list of 1-chip MeshDevices. Each remains a child of
    `parent_submesh` and is closed implicitly when the parent closes.
    """
    rows, cols = parent_submesh.shape[0], parent_submesh.shape[1]
    total = rows * cols
    if count is None:
        count = total
    if count <= 0 or count > total:
        raise ValueError(f"create_per_chip_submeshes: count must be in [1, {total}]; got {count}")

    one_shape = ttnn.MeshShape(1, 1)
    out: List = []
    for chip_idx in range(count):
        r, c = divmod(chip_idx, cols)
        sm = parent_submesh.create_submesh(one_shape, ttnn.MeshCoordinate(r, c))
        if sm.get_num_devices() != 1:
            raise RuntimeError(
                f"create_per_chip_submeshes: expected 1 device at ({r},{c}), " f"got {sm.get_num_devices()}"
            )
        out.append(sm)
    return out
