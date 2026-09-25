# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mesh-up prerequisite (recipe D3 step 1): the 8x4 mesh opens, MeshConfig + CCLManager are in place,
and all-gather (both axes) + reduce-scatter + all-reduce (TP axis) match torch exactly.

Topology follows MISTRAL_FABRIC (linear default, ring where the pod has wrap-around links).
"""

import torch
from loguru import logger

import ttnn
from models.demos.mistral_medium_3_5_128b.tt.fabric import fabric_mode


def _per_device(mesh, tensor, shape):
    return [ttnn.to_torch(t).reshape(shape).float() for t in ttnn.get_device_tensors(tensor)]


def test_mesh_smoke(galaxy_mesh, mesh_config, ccl_manager):
    mesh = galaxy_mesh
    sp, tp = mesh_config.sp, mesh_config.tp
    assert tuple(mesh.shape) == (8, 4), f"expected the 8x4 galaxy mesh, got {tuple(mesh.shape)}"
    logger.info(
        f"mesh {tuple(mesh.shape)} fabric={fabric_mode()} topology={ccl_manager.topology} "
        f"links={ccl_manager.num_links} dram_banks={mesh.dram_grid_size().x} "
        f"grid={mesh.compute_with_storage_grid_size()}"
    )
    dram = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    logger.info(
        f"DRAM per chip: {dram.num_banks} banks x {dram.total_bytes_per_bank / 2**30:.3f} GiB, "
        f"free per bank {dram.total_bytes_free_per_bank / 2**30:.3f} GiB"
    )

    rows, cols = 64, 128
    torch.manual_seed(0)
    # One distinct [rows, cols] block per chip, laid out as [sp*rows, tp*cols].
    full = torch.randn(1, 1, sp * rows, tp * cols).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        full,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.mapper(mesh, sp_dim=2, tp_dim=3),
    )
    blocks = full.reshape(sp, rows, tp, cols).permute(0, 2, 1, 3)  # [sp, tp, rows, cols]

    # all-gather over TP (dim 3): each chip holds its whole SP row.
    ag_tp = mesh_config.allgather(tt_x, ccl_manager, axis=mesh_config.tp_axis, dim=3)
    for idx, got in enumerate(_per_device(mesh, ag_tp, (rows, tp * cols))):
        r = idx // tp
        assert torch.equal(got, full[0, 0, r * rows : (r + 1) * rows].float()), f"TP all-gather mismatch on chip {idx}"

    # all-gather over SP (dim 2): each chip holds its whole TP column.
    ag_sp = mesh_config.allgather(tt_x, ccl_manager, axis=mesh_config.sp_axis, dim=2)
    for idx, got in enumerate(_per_device(mesh, ag_sp, (sp * rows, cols))):
        c = idx % tp
        assert torch.equal(got, full[0, 0, :, c * cols : (c + 1) * cols].float()), f"SP all-gather mismatch {idx}"

    # reduce-scatter over TP (dim 3) on a TP-replicated-shape input: sum of the 4 column blocks,
    # scattered back into cols/tp slices.
    rs = mesh_config.reduce_scatter(tt_x, ccl_manager, axis=mesh_config.tp_axis, dim=3)
    for idx, got in enumerate(_per_device(mesh, rs, (rows, cols // tp))):
        r, c = idx // tp, idx % tp
        ref = blocks[r].float().sum(0)[:, c * (cols // tp) : (c + 1) * (cols // tp)]
        assert torch.allclose(got, ref, atol=0.1, rtol=0.02), f"TP reduce-scatter mismatch on chip {idx}"

    # all-reduce over TP = RS + AG: every chip of row r holds the row's column-block sum.
    ar = mesh_config.allreduce(ttnn.clone(tt_x), ccl_manager, axis=mesh_config.tp_axis, dim=3)
    for idx, got in enumerate(_per_device(mesh, ar, (rows, cols))):
        r = idx // tp
        assert torch.allclose(got, blocks[r].float().sum(0), atol=0.1, rtol=0.02), f"TP all-reduce mismatch {idx}"
    logger.info("mesh smoke: TP/SP all-gather exact, TP reduce-scatter + all-reduce match torch")
