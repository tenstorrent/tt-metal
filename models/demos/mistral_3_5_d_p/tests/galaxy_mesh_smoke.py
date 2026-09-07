#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D3 mesh prerequisite: open the spec's target mesh, build MeshConfig + CCLManager, and prove the
collectives every module depends on actually work.

Bring-up goes straight to the target mesh — there is no single-card step — so this must pass before
the first module is written. It also tells the two fabric topologies apart on THIS pod:

  * ``FABRIC_1D_RING`` + a torus descriptor: a Galaxy wired as a 2D torus. Ring collectives map.
  * ``FABRIC_1D`` + the plain mesh descriptor: no wrap-around links, so a ring cannot be mapped at
    all ("Graph specified in MGD could not fit in the discovered physical topology"). Use
    ``ttnn.Topology.Linear`` in the CCLManager there.

Correctness is topology-independent; PERFORMANCE is not (every ring-gather collective changes), so
never compare linear-fabric timings against ring ones.

Run:
    MISTRAL_LINEAR_FABRIC=0 python models/demos/mistral_3_5_d_p/tests/galaxy_mesh_smoke.py
"""

from __future__ import annotations

import os
import sys

import torch

import ttnn


def linear_fabric_requested() -> bool:
    """``MISTRAL_LINEAR_FABRIC=1`` selects the plain mesh (no torus wrap-around links)."""
    return os.getenv("MISTRAL_LINEAR_FABRIC", "0").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    from loguru import logger

    from models.demos.mistral_3_5_d_p.spec import SPEC
    from models.demos.mistral_3_5_d_p.tt.ccl import CCLManager
    from models.demos.mistral_3_5_d_p.tt.config import MeshConfig
    from models.demos.mistral_3_5_d_p.utils.general_utils import get_default_num_links

    rows, cols = SPEC.mesh_shape
    if ttnn.get_num_devices() < rows * cols:
        logger.warning(
            f"SKIP: the spec targets {SPEC.target_hw} = mesh {(rows, cols)} ({rows * cols} devices); "
            f"this host exposes {ttnn.get_num_devices()}"
        )
        return 0

    linear = linear_fabric_requested()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if linear else ttnn.FabricConfig.FABRIC_1D_RING)
    topology = ttnn.Topology.Linear if linear else ttnn.Topology.Ring
    logger.info(f"opening mesh {(rows, cols)} with {'FABRIC_1D (linear)' if linear else 'FABRIC_1D_RING (torus)'}")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        logger.info(f"mesh open: shape={tuple(mesh.shape)} ndev={mesh.get_num_devices()}")
        mesh_config = MeshConfig(tuple(mesh.shape), tp=cols)
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=topology)
        logger.info(f"{mesh_config} num_links={ccl.num_links} grid={ccl.compute_grid_size}")

        # --- all-gather on the TP axis: each column contributes a distinct slice of the last dim ---
        width, tokens = 256, 64
        base = torch.arange(cols, dtype=torch.float32).reshape(1, 1, 1, cols)
        shard = base.expand(1, 1, tokens, cols).reshape(1, 1, tokens, cols)
        full = torch.cat([torch.full((1, 1, tokens, width), float(c)) for c in range(cols)], dim=-1)
        tt_shard = ttnn.from_torch(
            full,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.column_parallel(mesh),
        )
        gathered = mesh_config.allgather(tt_shard, ccl, axis=mesh_config.tp_axis, dim=3)
        got = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
        assert tuple(got.shape) == (1, 1, tokens, width * cols), f"all-gather shape {tuple(got.shape)}"
        assert torch.allclose(got, full, atol=1e-2), "all-gather did not reconstruct the full tensor"
        logger.info(f"all-gather OK: {tuple(tt_shard.shape)} -> {tuple(got.shape)}")
        del shard, base

        # --- all-reduce on the TP axis: every column holds 1.0, so the sum must be `cols` ---
        ones = torch.ones(1, 1, tokens, width * cols)
        tt_ones = ttnn.from_torch(
            ones,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        reduced = mesh_config.allreduce(tt_ones, ccl, axis=mesh_config.tp_axis)
        got = ttnn.to_torch(ttnn.get_device_tensors(reduced)[0]).float()
        assert torch.allclose(got, ones * cols, atol=1e-2), f"all-reduce gave {got.flatten()[:4]}, want {cols}"
        logger.info(f"all-reduce OK: every element == {cols}")

        # --- all-gather on the SP axis: the sequence dim, which every module shards ---
        s_local = 32
        seq = (
            torch.arange(rows * s_local, dtype=torch.float32)
            .reshape(1, 1, rows * s_local, 1)
            .expand(1, 1, rows * s_local, 32)
            .contiguous()
        )
        tt_seq = ttnn.from_torch(
            seq,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(2, None)),
        )
        gathered_seq = mesh_config.allgather(tt_seq, ccl, axis=mesh_config.sp_axis, dim=2)
        got = ttnn.to_torch(ttnn.get_device_tensors(gathered_seq)[0]).float()
        assert torch.allclose(got, seq, atol=1e-1), "SP all-gather did not reconstruct the sequence"
        logger.info(f"SP all-gather OK: {tuple(tt_seq.shape)} -> {tuple(got.shape)}")

        ttnn.synchronize_device(mesh)
        logger.info("MESH PREREQUISITE PASSED")
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
