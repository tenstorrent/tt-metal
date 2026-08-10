# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole-galaxy repro/port test for ttnn.experimental.llama_rs_matmul (fused gathered ring
matmul + llama_reduce_scatter connected by MatmulFusedOpSignaler).

This is the same fused program the model's matmul_line_reduce_scatter /
double_matmul_line_reduce_scatter paths use for FF1/FF3 (the two-weight variant additionally
requires the prefetcher global CB with DRAM-sharded weights, which cannot be replicated
standalone - matmul validation rejects 3 input tensors without a global CB). The RS-side
fabric routing was ported to 2D-torus routes (test_bh_llama_reduce_scatter passes); this test
adds the matmul + signaler on top, which is where the model-level QWEN_BH_FUSED_RS_MATMUL hang
lives.

Geometry mirrors Qwen3-32B FF1/FF3 decode: 24-core BH ring (PREFETCHER_NOC1_GRID_BH, cols 1-3
rows 0-7), hop core (3,8), scatter over the 4 devices of each mesh row (cluster_axis=1, Ring),
RS input 24 cores x [32,160], RS output 30 cores x [32,32].
"""

import os

import pytest
import torch
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.demos.llama3_70b_galaxy.tt.model_config import PREFETCHER_NOC1_GRID_BH, num_to_coregrid
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)
from tests.ttnn.unit_tests.operations.ccl.test_llama_reduce_scatter_async_TG import gen_tensor
from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import (
    SUB_DEVICE_CRS,
    FF1_CRS_RS_OUT,
)

M = 32
K = 1536  # per-device K (padded), 48 tiles -> in0_block_w=2 on 24 cores
N = 3840  # per-device N (padded), 120 tiles -> per_core_N=5 on 24 cores
RING_SIZE = 24
NUM_SCATTER_DEVICES = 4
NUM_FRACTURE_DEVICES = 8
CLUSTER_SHAPE = (8, 4)
SHARD_HEIGHT = 32
SHARD_WIDTH = N // RING_SIZE  # 160
NUM_PAGES_PER_PACKET = 4


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
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_bh_llama_rs_matmul(mesh_device, reset_seeds):
    torch.manual_seed(1234)
    num_iters = int(os.environ.get("BH_RS_MM_TEST_ITERS", "4"))
    trace_mode = os.environ.get("BH_RS_MM_TEST_TRACE", "1") == "1"
    num_links = int(os.environ.get("BH_RS_MM_TEST_LINKS", "2"))

    ring_crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in PREFETCHER_NOC1_GRID_BH]
    )
    hop_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 8), ttnn.CoreCoord(3, 8))])

    # Matmul tensors (identical on all devices).
    in0_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ring_crs, [M, K // RING_SIZE], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ring_crs, [K, N // RING_SIZE], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mm_out_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ring_crs, [M, N // RING_SIZE], ttnn.ShardOrientation.ROW_MAJOR),
    )
    # RS tensors: same geometry as the passing test_bh_llama_reduce_scatter harness.
    rs_in_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ring_crs, [SHARD_HEIGHT, SHARD_WIDTH], ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Packet-worker (interim) cores must NOT overlap the matmul ring: the matmul factory subtracts
    # the RS cores (restricted_cores) from its worker set, so any ring core inside the RS grid gets
    # no matmul kernel and the ring gather deadlocks (this was the BH QWEN_BH_FUSED_RS_MATMUL hang).
    # BH ring = cols 1-3 rows 0-7 + hop (3,8); RS senders = (5,3),(6,3). Use cols 5-6 off row 3.
    packet_worker_crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(6, 4)),
        ]
    )
    interim_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            packet_worker_crs,
            [SHARD_HEIGHT, NUM_SCATTER_DEVICES * NUM_PAGES_PER_PACKET * 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    rs_out_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(FF1_CRS_RS_OUT, [SHARD_HEIGHT, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )

    grid = num_to_coregrid(RING_SIZE)
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=K // RING_SIZE // 32,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=M // 32,
        per_core_N=N // RING_SIZE // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_crs,
        untilize_out=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    in0 = torch.randn(1, 1, M, K)
    w1 = torch.randn(1, 1, K, N)
    in0_t = ttnn.from_torch(
        in0, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=in0_memcfg
    )
    w1_t = ttnn.from_torch(
        w1, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=in1_memcfg
    )

    rs_input = gen_tensor(
        3, SHARD_HEIGHT, SHARD_WIDTH, NUM_SCATTER_DEVICES, NUM_FRACTURE_DEVICES, RING_SIZE, scheme="random"
    )
    # Golden: reduce the 4 column-device shards, then scatter chunks back across the column.
    intermediate_outputs = torch.chunk(rs_input, chunks=NUM_SCATTER_DEVICES, dim=1)
    reduced = torch.zeros(intermediate_outputs[0].shape)
    for t in intermediate_outputs:
        reduced += t
    rs_golden = torch.cat(torch.chunk(reduced, chunks=NUM_SCATTER_DEVICES, dim=3), dim=1)
    mm_golden = in0.float() @ w1.float()

    rs_in_t = ttnn.from_torch(
        rs_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=rs_in_memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=list(CLUSTER_SHAPE)),
    )
    interim_t = ttnn.from_torch(
        torch.zeros(
            (
                NUM_FRACTURE_DEVICES,
                NUM_SCATTER_DEVICES,
                SHARD_HEIGHT,
                NUM_SCATTER_DEVICES * NUM_PAGES_PER_PACKET * 32 * packet_worker_crs.num_cores(),
            )
        ),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=interim_memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=list(CLUSTER_SHAPE)),
    )

    worker_sub_device = ttnn.SubDevice([SUB_DEVICE_CRS])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    semaphores = [ttnn.create_global_semaphore(mesh_device, SUB_DEVICE_CRS, 0) for _ in range(2)]

    def run_once(i):
        # Same unpacking as the model's matmul_line_reduce_scatter: (matmul_out, rs_out).
        mm_out, rs_out = ttnn.experimental.llama_rs_matmul(
            in0_t,
            w1_t,
            interim_t,
            3,
            semaphores[i % 2],
            1,  # cluster_axis
            mesh_device,
            num_links,
            worker_sub_device_id,
            rs_tensor=rs_in_t,
            memory_config_rs=rs_out_memcfg,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            memory_config_mm=mm_out_memcfg,
            second_weight_tensor=None,
            topology=ttnn.Topology.Ring,
        )
        return mm_out, rs_out

    logger.info(f"eager compile run (links={num_links})")
    mm_out_t, rs_out_t = run_once(0)
    ttnn.synchronize_device(mesh_device)
    logger.info("eager run done")

    # Identify outputs by shape in case the binding orders them differently.
    if mm_out_t.shape[-1] != N:
        mm_out_t, rs_out_t = rs_out_t, mm_out_t

    def check(rs_tensor_dev, mm_tensor_dev, tag):
        rs_torch = ttnn.to_torch(
            rs_tensor_dev,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=list(CLUSTER_SHAPE)),
        )
        ok, pcc = comp_pcc(rs_golden, rs_torch, 0.99)
        logger.info(f"[{tag}] rs pcc: {pcc}")
        assert ok, f"[{tag}] rs output mismatch: {pcc}"
        if mm_tensor_dev is not None:
            mm_torch = ttnn.to_torch(
                mm_tensor_dev,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=list(CLUSTER_SHAPE)),
            )
            ok, pcc = comp_pcc(mm_golden[0], mm_torch[0:1, 0], 0.99)
            logger.info(f"[{tag}] matmul pcc: {pcc}")
            assert ok, f"[{tag}] matmul output mismatch: {pcc}"

    check(rs_out_t, mm_out_t, "eager")

    if trace_mode:
        logger.info("capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            mm_o, rs_o = run_once(i)
            if mm_o.shape[-1] != N:
                mm_o, rs_o = rs_o, mm_o
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info("executing trace")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
        check(rs_o, mm_o, "trace")

    mesh_device.reset_sub_device_stall_group()
    logger.info("PASSED bh llama_rs_matmul")
