# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The D3 mesh prerequisite: the target mesh opens, MeshConfig and CCLManager stand up, and an
all-gather + all-reduce round-trips.

No module work happens before this passes. It is deliberately the cheapest possible device test, so
a broken fabric or a stale runtime root fails here in seconds instead of inside a PCC test.

Run it in whichever topology the pod supports, and in both where both work — this Galaxy is a plain
grid, so only linear maps here.
"""

import pytest
import torch
from loguru import logger

import ttnn

from ..test_factory import TARGET_MESH, comp_pcc, parametrize_target_mesh


@parametrize_target_mesh()
def test_mesh_opens(mesh_device, device_params):
    """The mesh opens at the shape the spec asks for, with the SP/TP axes the right way round."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == TARGET_MESH, f"opened {(rows, cols)}, expected {TARGET_MESH}"
    assert mesh_device.get_num_devices() == rows * cols
    logger.info(f"mesh {rows}x{cols}, {mesh_device.get_num_devices()} devices, grid {mesh_device.compute_with_storage_grid_size()}")


@parametrize_target_mesh()
def test_ccl_manager_stands_up(mesh_device, device_params, ccl_manager, mesh_config):
    """Semaphores and the ring-attention core offset exist, and TP/SP are the spec's."""
    assert ccl_manager.num_links >= 1
    assert len(ccl_manager.ring_attention_ccl_semaphore_handles) == 2
    grid = mesh_device.compute_with_storage_grid_size()
    assert ccl_manager.ring_attention_ccl_core_grid_offset == (grid.x - 1, 0), (
        "ring-attention CCL workers must live in the LAST compute column, or they overlap the "
        "ring_joint SDPA compute cores and the op's grid assert fires"
    )
    assert (mesh_config.sp, mesh_config.tp) == (8, 4)
    assert (mesh_config.sp_axis, mesh_config.tp_axis) == (0, 1)


@parametrize_target_mesh()
@pytest.mark.parametrize("axis_name", ["tp", "sp"])
def test_all_gather(mesh_device, device_params, ccl_manager, mesh_config, axis_name):
    """All-gather along each mesh axis reconstructs the full tensor from its shards."""
    axis = mesh_config.tp_axis if axis_name == "tp" else mesh_config.sp_axis
    extent = tuple(mesh_device.shape)[axis]
    rows, cols = tuple(mesh_device.shape)

    width = 128 * extent
    torch.manual_seed(0)
    full = torch.randn(1, 1, 32, width)

    dims = [None, None]
    dims[axis] = 3
    tt = ttnn.from_torch(
        full,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
    )
    gathered = mesh_config.allgather(tt, ccl_manager, axis=axis, dim=3)
    ttnn.synchronize_device(mesh_device)

    got = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0])
    assert got.shape[-1] == width, f"gathered width {got.shape[-1]}, expected {width}"
    pcc = comp_pcc(full, got)
    logger.info(f"all_gather {axis_name} (extent {extent}): PCC {pcc:.6f}")
    assert pcc > 0.999


@parametrize_target_mesh()
def test_all_reduce_tp(mesh_device, device_params, ccl_manager, mesh_config):
    """All-reduce across TP sums the per-column partials — the MLP/attention collective tail."""
    rows, cols = tuple(mesh_device.shape)
    tp = mesh_config.tp
    torch.manual_seed(0)
    # Each TP column holds a different partial; the reduction must produce their sum on every column.
    per_col = [torch.randn(1, 1, 32, 256) for _ in range(tp)]
    stacked = torch.cat(per_col, dim=-1)  # shard dim 3 across the TP cols gives col c -> per_col[c]

    dims = [None, None]
    dims[mesh_config.tp_axis] = 3
    tt = ttnn.from_torch(
        stacked,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
    )
    reduced = mesh_config.allreduce(tt, ccl_manager, axis=mesh_config.tp_axis)
    ttnn.synchronize_device(mesh_device)

    got = ttnn.to_torch(ttnn.get_device_tensors(reduced)[0])
    expected = sum(per_col)
    pcc = comp_pcc(expected, got[..., : expected.shape[-1]])
    logger.info(f"all_reduce tp (extent {tp}): PCC {pcc:.6f}")
    assert pcc > 0.99
