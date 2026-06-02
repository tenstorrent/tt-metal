# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh setup for pi0.5 Option B — opens a Galaxy and slices into 4× 4×2 submeshes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import ttnn

from .stages import StageLayout


@contextmanager
def open_galaxy_mesh(
    layout: StageLayout,
    enable_fabric: bool = False,
    l1_small_size: Optional[int] = None,  # default; bump when L1 weights land
):
    """Open the parent 8×4 mesh, partition into 4× 4×2 submeshes per Option B layout.

    Args:
        layout: the StageLayout to honour (parent + submesh shape).
        enable_fabric: if True, call `ttnn.set_fabric_config(FABRIC_1D)` before
            opening the mesh and `FabricConfig.DISABLED` on exit. Required for
            collective ops (`ttnn.all_reduce`) on the TP path. Adds ~10s to
            mesh open. Default False for non-collective tests.
        l1_small_size: bytes per core reserved for static circular buffers.
            When weights are L1-resident, the all_reduce / SDPA kernels need
            a fixed L1 block for static CBs. Without enough headroom the
            allocator places L1-interleaved buffers inside the CB region and
            the kernel errors with "Statically allocated CBs clash with L1
            buffers". 1 MB is enough for our all_reduce + SDPA pattern;
            bump if a new kernel needs more.

    Yields (parent_mesh, [submesh_0, submesh_1, submesh_2, submesh_3]) in stage order.
    On exit, all submeshes and the parent mesh are closed.
    """
    parent_shape = ttnn.MeshShape(*layout.parent_mesh_shape)
    submesh_shape = ttnn.MeshShape(*layout.submesh_shape)

    if enable_fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    open_kwargs = {"mesh_shape": parent_shape}
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    parent = ttnn.open_mesh_device(**open_kwargs)
    try:
        submeshes = parent.create_submeshes(submesh_shape)
        if len(submeshes) != len(layout.stages):
            raise RuntimeError(
                f"Expected {len(layout.stages)} submeshes from {layout.parent_mesh_shape} parent / "
                f"{layout.submesh_shape} submesh, got {len(submeshes)}"
            )
        yield parent, submeshes
    finally:
        ttnn.close_mesh_device(parent)
        if enable_fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def describe_submesh(submesh) -> str:
    """Human-readable submesh summary for logging."""
    rows, cols = submesh.shape[0], submesh.shape[1]
    n = submesh.get_num_devices()
    return f"submesh shape=({rows},{cols}) devices={n}"
