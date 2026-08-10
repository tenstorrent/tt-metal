# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole-galaxy port test for the fused RMSAllGather (ttnn.fused_rms_minimal).

The fused op's writer used raw 1D MulticastRoutingCommandHeaders for its stats all-gather, which
no-op on the BH 2D-torus fabric (HybridMeshPacketHeader routing). The port emits host-computed
route info via ccl_routing_utils (same mechanism as the BH-proven all_gather_async writers).

This test mirrors the Qwen3-32B decode norm geometry exactly:
- 8x4 mesh, cluster_axis=1 (4 devices per row), Ring topology
- hidden 5120 -> 1280 per device, width-sharded on the 10-core LN grid at (7,0)-(8,4)
- stats persistent buffer (1,1,32,128) on the sender core (7,0)
- eager + trace-replay correctness against a torch RMS reference
"""

import os

import pytest
import torch
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)

HIDDEN = 5120
RING_SIZE = 4  # devices per mesh row (cluster_axis=1)
SHARD_H = 32
LN_GRID_OFFSET = ttnn.CoreCoord(7, 0)
LN_GRID_END = ttnn.CoreCoord(8, 4)  # 2 cols x 5 rows = 10 cores
NUM_CORES_LN = 10
PER_DEV = HIDDEN // RING_SIZE  # 1280
SHARD_W = PER_DEV // NUM_CORES_LN  # 128


def torch_rms(x, gamma, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _FABRIC_CONFIG,
            "trace_region_size": 23887872,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
def test_bh_fused_rms_allgather(mesh_device, reset_seeds):
    torch.manual_seed(1234)
    eps = 1e-6
    num_iters = int(os.environ.get("FUSED_RMS_TEST_ITERS", "8"))
    trace_mode = os.environ.get("FUSED_RMS_TEST_TRACE", "1") == "1"

    ln_grid = ttnn.CoreRangeSet([ttnn.CoreRange(LN_GRID_OFFSET, LN_GRID_END)])
    input_memcfg = ttnn.create_sharded_memory_config(
        shape=(SHARD_H, SHARD_W),
        core_grid=ln_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    output_memcfg = input_memcfg  # skip_write_back path, same as model when out grid == in grid

    prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 5),
        subblock_w=SHARD_W // 32,
        block_h=SHARD_H // 32,
        block_w=SHARD_W // 32,
        inplace=False,
    )

    # Stats gather buffer on the sender core (first core of the LN grid), like the model's
    # LAYERNORM persistent buffer. Width = ring_size tiles.
    stats_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 32 * RING_SIZE),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(LN_GRID_OFFSET, LN_GRID_OFFSET)]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_stats = ttnn.from_torch(
        torch.zeros((1, 1, 32, 32 * RING_SIZE)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=stats_memcfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Semaphores on a grid covering the LN cores (model uses a worker-grid pool).
    sem_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(10, 9))])
    semaphores = [ttnn.create_global_semaphore(mesh_device, sem_crs, 0) for _ in range(2)]

    ref_in, ref_gamma, tt_in, tt_gamma = [], [], [], []
    for i in range(num_iters):
        x = torch.randn(1, 1, SHARD_H, HIDDEN)
        g = torch.randn(1, 1, 1, HIDDEN) * 0.5 + 1.0
        ref_in.append(x)
        ref_gamma.append(g)
        tt_in.append(
            ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_memcfg,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(8, 4)),
            )
        )
        tt_gamma.append(
            ttnn.from_torch(
                g.reshape([1, 1, HIDDEN // 32, 32]),
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(8, 4)),
            )
        )

    def run_op(i):
        return ttnn.fused_rms_minimal(
            tt_in[i],
            prg_cfg,
            1,  # cluster_axis
            mesh_device,
            semaphores[i % 2],
            topology=ttnn.Topology.Ring,
            residual_input_tensor=None,
            num_links=1,
            epsilon=eps,
            weight=tt_gamma[i],
            stats=tt_stats,
            memory_config=output_memcfg,
            use_noc1_only=False,
        )

    def check(i, tt_out, tag):
        out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(8, 4)),
        )  # [8, 1, 32, 5120]
        ref = torch_rms(ref_in[i], ref_gamma[i], eps)[0]  # [1, 32, 5120]
        ok_all = True
        for row in range(8):
            passing, msg = comp_pcc(out[row : row + 1], ref.unsqueeze(0), 0.999)
            if not passing:
                logger.error(f"{tag} iter {i} mesh-row {row}: {msg}")
                ok_all = False
        assert ok_all, f"{tag} iter {i}: PCC failure (see log)"
        logger.info(f"{tag} iter {i}: PCC ok")

    # Eager
    logger.info("eager run")
    eager_outs = [run_op(i) for i in range(num_iters)]
    ttnn.synchronize_device(mesh_device)
    for i in range(num_iters):
        check(i, eager_outs[i], "eager")
        eager_outs[i].deallocate(True)

    if trace_mode:
        logger.info("capturing trace")
        trace_outs = []
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            trace_outs.append(run_op(i))
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for rep in range(3):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
            for i in range(num_iters):
                check(i, trace_outs[i], f"trace-rep{rep}")
        ttnn.release_trace(mesh_device, trace_id)

    logger.info("test_bh_fused_rms_allgather passed")
