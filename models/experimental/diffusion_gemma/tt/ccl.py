# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-owned Gemma-4 CCL helpers with pre-created semaphores.

The shared Gemma-4 helpers route collectives through plain ``ttnn.all_reduce``
and ``ttnn.all_gather``. On a program-cache miss those operations create global
semaphores, whose setup performs a blocking command-queue drain. During trace
capture commands are recorded rather than executed, so that drain can never
finish and the process deadlocks.

DiffusionGemma instead decomposes all-reduce into the experimental minimal
reduce-scatter and all-gather operations. Both receive semaphores created by
``CCLManager`` before trace capture, keeping semaphore creation and blocking
writes off captured and post-capture execution paths.

``all_gather_async`` has one additional trap: ROW_MAJOR inputs and TILE inputs
padded on the gather dimension are automatically routed through
``composite_all_gather``, which is implemented with ``all_broadcast`` and ignores
the supplied semaphores. ``ccl_allgather`` therefore rejects those layouts.
Callers must tilize and tile-align the gather dimension first.
"""

import ttnn

# DG reuses Gemma-4's CCLManager unchanged: it pre-creates the double-buffered
# reduce-scatter / all-gather / barrier semaphores before trace capture, which is
# all the DG collectives below need.
from models.demos.gemma4.tt.ccl import CCLManager, default_num_links  # noqa: F401  (re-exported)


def replicate_mapper(mesh_device):
    """``ReplicateTensorToMesh`` for a multi-device mesh, ``None`` for a single device."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce across TP devices without creating semaphores."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    scattered = ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        dim=3,
        cluster_axis=tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    gathered = ttnn.experimental.all_gather_async(
        scattered,
        dim=3,
        cluster_axis=tp_axis,
        mesh_device=ccl_manager.mesh_device,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    scattered.deallocate(True)
    return gathered


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """DG-local replacement for Gemma-4 attention's ``apply_allreduce``.

    ``hidden_size`` is accepted and ignored so call sites keep the exact Gemma-4
    signature and stay diffable against the shared originals.
    """
    return ccl_allreduce(tensor, mesh_config, ccl_manager)


def _validate_minimal_allgather_input(tensor, dim: int):
    rank = len(tensor.shape)
    gather_dim = dim if dim >= 0 else rank + dim
    if gather_dim < 0 or gather_dim >= rank:
        raise ValueError(f"invalid all-gather dim {dim} for rank-{rank} tensor")
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            "DG ccl_allgather requires TILE_LAYOUT; ROW_MAJOR silently selects "
            "composite all_broadcast and bypasses the supplied semaphores"
        )
    if gather_dim == rank - 1 and int(tensor.shape[gather_dim]) % ttnn.TILE_SIZE != 0:
        raise ValueError(f"DG ccl_allgather dim {gather_dim} must be tile-aligned; got shape {tensor.shape}")
    if gather_dim == rank - 2 and int(tensor.shape[gather_dim]) % ttnn.TILE_SIZE != 0:
        raise ValueError(f"DG ccl_allgather dim {gather_dim} must be tile-aligned; got shape {tensor.shape}")


def ccl_allgather(tensor, mesh_config, ccl_manager, dim=3, memory_config=None):
    """All-gather across TP devices using caller-owned semaphores."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    _validate_minimal_allgather_input(tensor, dim)
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    gathered = ttnn.experimental.all_gather_async(
        tensor,
        dim=dim,
        cluster_axis=tp_axis,
        mesh_device=ccl_manager.mesh_device,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        memory_config=memory_config,
    )
    tensor.deallocate(True)
    return gathered
