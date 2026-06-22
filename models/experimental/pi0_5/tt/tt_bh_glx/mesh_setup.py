# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Open the 8x4 BH Galaxy parent mesh and carve into vision/prefill/denoise submeshes.

v2: FABRIC_1D enabled at mesh open so sockets (ttnn.create_socket_pair +
ttnn.experimental.send_direct_async / recv_direct_async) can route between
the 1x1 per-chip submeshes. trace_region_size defaults to 128 MiB so the
parent mesh has room for the Phase B sample_actions trace capture.

Setting fabric requires re-disabling it on context exit; otherwise a
subsequent ttnn.open_mesh_device call in the same process gets a stale
fabric config and the underlying mesh fails to initialize.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import ttnn

from . import stages
from .stages import MeshHandles, TracedMeshHandles


@contextmanager
def open_galaxy_mesh_traced(l1_small_size: Optional[int] = None, trace_region_size: Optional[int] = 134_217_728):
    """Open the (8,4) torus under FABRIC_1D and carve the 3 fully-traced stage
    meshes (collinear layout — see stages.py). Yields TracedMeshHandles.

    The parent is kept whole (8,4) so every torus ethernet link trains; the
    stages occupy a collinear subset (rows 0–6) so cross-stage sockets +
    in-trace point_to_point both work under one FABRIC_1D config.
    """
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    open_kwargs = {"mesh_shape": ttnn.MeshShape(*stages.PARENT_MESH_SHAPE)}
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    if trace_region_size is not None:
        open_kwargs["trace_region_size"] = trace_region_size

    parent = ttnn.open_mesh_device(**open_kwargs)
    submeshes = []
    try:
        if parent.get_num_devices() != stages.PARENT_MESH_SHAPE[0] * stages.PARENT_MESH_SHAPE[1]:
            raise RuntimeError(f"Parent mesh has {parent.get_num_devices()} devices, expected 32 for BH Galaxy")

        def carve(shape, offset):
            sm = parent.create_submesh(ttnn.MeshShape(*shape), ttnn.MeshCoordinate(*offset))
            submeshes.append(sm)
            return sm

        vision_mesh = carve(stages.TRACED_VISION_SHAPE, stages.TRACED_VISION_OFFSET)
        prefill_mesh = carve(stages.TRACED_PREFILL_SHAPE, stages.TRACED_PREFILL_OFFSET)
        denoise_mesh = carve(stages.TRACED_DENOISE_SHAPE, stages.TRACED_DENOISE_OFFSET)

        yield TracedMeshHandles(
            parent=parent, vision_mesh=vision_mesh, prefill_mesh=prefill_mesh, denoise_mesh=denoise_mesh
        )
    finally:
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


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


_DEFAULT_TRACE_REGION_SIZE = 134_217_728  # 128 MiB — matches single-chip trace tests


@contextmanager
def open_prefill_tp4_mesh(
    tp: int = 4,
    l1_small_size: Optional[int] = None,
    trace_region_size: Optional[int] = None,
    enable_fabric: bool = True,
):
    """Open a 1×tp mesh for TP=4 VLM prefill on an 8-chip (or larger) device.

    Yields the mesh device directly.  The caller is responsible for uploading
    inputs as ReplicateTensorToMesh and downloading outputs via
    ttnn.get_device_tensors(out)[0] + ttnn.to_torch().

    Args:
        tp:               tensor-parallel degree (number of chips, default 4).
        l1_small_size:    bytes per core for static CBs.
        trace_region_size: bytes per chip for trace capture.
        enable_fabric:    set FABRIC_1D for AllReduce collectives (default True).
    """
    if enable_fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    open_kwargs = {"mesh_shape": ttnn.MeshShape(1, tp)}
    if l1_small_size is not None:
        open_kwargs["l1_small_size"] = l1_small_size
    if trace_region_size is not None:
        open_kwargs["trace_region_size"] = trace_region_size

    mesh = ttnn.open_mesh_device(**open_kwargs)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        if enable_fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@contextmanager
def open_galaxy_mesh(
    l1_small_size: Optional[int] = None,
    trace_region_size: Optional[int] = _DEFAULT_TRACE_REGION_SIZE,
    enable_fabric: bool = True,
):
    """Open parent (8,4) mesh + FABRIC_1D and carve 4/18/6 submeshes + per-chip 1x1s.

    Yields MeshHandles. On exit, every submesh is closed before the parent
    (parent-last close avoids the wedged-device firmware-init hang we hit
    when closing out of order — see option_c/mesh_setup.py). Fabric is
    disabled on the same exit path so subsequent mesh opens start clean.

    Args:
        l1_small_size: bytes per core reserved for static circular buffers.
        trace_region_size: bytes per chip for trace capture (Phase B). 128 MiB
            is enough for the full sample_actions trace on the parent mesh.
        enable_fabric: set False to revert to v1 host-bounce-only behavior.
            Default True (required for socket-based transport).
    """
    if enable_fabric:
        # FABRIC_2D so sockets can route between chips that don't share a row
        # OR column (the cross-stage hops vision[3]→prefill[0] and
        # prefill[k]→denoise[k//3] cross both axes). FABRIC_1D fails with
        # "sender_global_coord[0] == recv_global_coord[0] || ..." on those.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

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

        # Compute submesh = the 28 used chips (rows 0..6). The physical parent
        # stays (8,4) so fabric trains every torus link; all compute (and the
        # trace root) lives on this (7,4) view. Stage + per-chip submeshes are
        # carved from `compute`, not `parent`, so a trace rooted on `compute`
        # finishes over exactly these 28 chips (no idle row-7 deadlock).
        compute = parent.create_submesh(
            ttnn.MeshShape(*stages.COMPUTE_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*stages.COMPUTE_SUBMESH_OFFSET),
        )
        all_submeshes.append(compute)

        vision_submesh = compute.create_submesh(
            ttnn.MeshShape(*stages.VISION_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*stages.VISION_SUBMESH_OFFSET),
        )
        all_submeshes.append(vision_submesh)

        prefill_submesh = compute.create_submesh(
            ttnn.MeshShape(*stages.PREFILL_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*stages.PREFILL_SUBMESH_OFFSET),
        )
        all_submeshes.append(prefill_submesh)

        denoise_submesh = compute.create_submesh(
            ttnn.MeshShape(*stages.DENOISE_SUBMESH_SHAPE),
            ttnn.MeshCoordinate(*stages.DENOISE_SUBMESH_OFFSET),
        )
        all_submeshes.append(denoise_submesh)

        vision_per_chip = _carve_per_chip(
            compute, stages.VISION_SUBMESH_SHAPE, stages.VISION_SUBMESH_OFFSET, stages.VISION_NUM_CHIPS
        )
        all_submeshes.extend(vision_per_chip)
        prefill_per_chip = _carve_per_chip(
            compute, stages.PREFILL_SUBMESH_SHAPE, stages.PREFILL_SUBMESH_OFFSET, stages.PREFILL_NUM_CHIPS
        )
        all_submeshes.extend(prefill_per_chip)
        denoise_per_chip = _carve_per_chip(
            compute, stages.DENOISE_SUBMESH_SHAPE, stages.DENOISE_SUBMESH_OFFSET, stages.DENOISE_NUM_CHIPS
        )
        all_submeshes.extend(denoise_per_chip)

        yield MeshHandles(
            parent=parent,
            trace_root=compute,
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
        if enable_fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
