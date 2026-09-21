# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D3 prerequisite: the target mesh opens, ``MeshConfig`` and ``CCLManager`` stand up, and an
all-gather + all-reduce round-trips on every shape this pod can reach.

Nothing else in the suite runs until this is green — every module PCC test exercises sharding and
collectives from the first one, so a fabric or semaphore problem must surface here rather than as
a mysterious PCC number three stages later.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ..test_factory import mesh_setup, parametrize_mesh


@parametrize_mesh()
def test_mesh_opens(mesh, submesh_shape, device_params):
    mesh_config, _ccl = mesh_setup(mesh)
    rows, cols = tuple(mesh.shape)
    assert mesh_config.sp == rows and mesh_config.tp == cols
    logger.info(f"opened {mesh_config}, grid={mesh.compute_with_storage_grid_size()}")


@parametrize_mesh()
def test_tp_all_gather(mesh, submesh_shape, device_params):
    """Column-sharded -> full width. The collective every norm and projection tail depends on."""
    mesh_config, ccl = mesh_setup(mesh)
    if mesh_config.tp == 1:
        pytest.skip("tp=1: no TP collective to exercise")
    width = 32 * mesh_config.tp
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, width)
    tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.column_parallel(mesh),
    )
    out = mesh_config.allgather(tt, ccl, axis=mesh_config.tp_axis, dim=3)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, 1, 64, width)
    passing, pcc = comp_pcc(x, got, 0.99)
    logger.info(f"tp all_gather {mesh_config}: {pcc}")
    assert passing, pcc


@parametrize_mesh()
def test_tp_all_reduce(mesh, submesh_shape, device_params):
    """Replicated in -> tp x the value out. Exercises reduce_scatter + all_gather together, which
    is the closing collective of every MLP and attention block."""
    mesh_config, ccl = mesh_setup(mesh)
    if mesh_config.tp == 1:
        pytest.skip("tp=1: no TP collective to exercise")
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 256)
    tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.replicate(mesh),
    )
    out = mesh_config.allreduce(tt, ccl, axis=mesh_config.tp_axis)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, 1, 64, 256)
    passing, pcc = comp_pcc(x * mesh_config.tp, got, 0.99)
    logger.info(f"tp all_reduce {mesh_config}: {pcc}")
    assert passing, pcc


@parametrize_mesh()
def test_sp_all_gather_and_partition_round_trip(mesh, submesh_shape, device_params):
    """The SP pair the Gated DeltaNet rests on: all-gather the sequence, then ``mesh_partition``
    it back. ``mesh_partition`` is the inverse of all-gather and is a per-device slice with no
    fabric traffic — it is how each SP row takes its own token block out of a tensor computed
    redundantly over the whole chunk (see ``tt/gdn/prefill.py``)."""
    mesh_config, ccl = mesh_setup(mesh)
    if mesh_config.sp == 1:
        pytest.skip("sp=1: no SP collective to exercise")
    seq_total = 32 * mesh_config.sp
    torch.manual_seed(0)
    x = torch.randn(1, 1, seq_total, 128)
    tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.sequence_parallel(mesh, seq_dim=2),
    )
    gathered = mesh_config.allgather(tt, ccl, axis=mesh_config.sp_axis, dim=2)
    assert tuple(gathered.shape) == (1, 1, seq_total, 128)
    got_full = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).reshape(1, 1, seq_total, 128)
    passing, pcc = comp_pcc(x, got_full, 0.99)
    logger.info(f"sp all_gather {mesh_config}: {pcc}")
    assert passing, pcc

    back = ttnn.mesh_partition(gathered, 2, cluster_axis=mesh_config.sp_axis)
    s_local = seq_total // mesh_config.sp
    assert tuple(back.shape) == (1, 1, s_local, 128)
    shards = ttnn.get_device_tensors(back)
    for r in range(mesh_config.sp):
        # Device index r*cols is row r, column 0.
        got = ttnn.to_torch(shards[r * mesh_config.tp]).reshape(1, 1, s_local, 128)
        expected = x[:, :, r * s_local : (r + 1) * s_local]
        ok, pcc = comp_pcc(expected, got, 0.99)
        assert ok, f"row {r} got the wrong token block back: {pcc}"
