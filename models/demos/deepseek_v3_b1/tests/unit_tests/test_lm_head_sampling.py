# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN LM Head Sampling CCL Broadcast + Mcast + Matmul Op Test

In multi-device mode: CCL broadcasts input_a [1, 7168] from sender device to all
devices, then on each device the sender core multicasts to 101 matmul cores.
Each matmul core holds a weight shard [7168, N_per_core] and computes
[1, 7168] x [7168, N_per_core] -> [1, N_per_core].
Output stays width-sharded across matmul cores.

In single-device mode (skip_ccl=True): CCL is skipped and the input is used directly.
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pod_pipeline import (
    PodPipeline,
    create_random_weights_single_iteration,
    create_stage_kind,
    create_synthetic_weights,
    token_page_size_bytes,
)
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.perf.benchmarking_utils import BenchmarkProfiler


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _is_lm_head_sampling_perf_enabled():
    return os.getenv("RUN_LM_HEAD_SAMPLING_PERF", "0") == "1"


def _is_persistent_mode_enabled():
    return os.getenv("RUN_PERSISTENT_MODE", "0") == "1"


@pytest.mark.skipif(not _is_lm_head_sampling_perf_enabled(), reason="Set RUN_LM_HEAD_SAMPLING_PERF=1 to run perf test")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (1, 0), (2, 1), (2, 0)])
@pytest.mark.parametrize("num_iters,num_warmup_iters", [(20, 6)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1163264,
        }
    ],
    indirect=True,
)
def test_perf(bh_2d_mesh_device, use_fp32, final_mesh_coord, num_iters, num_warmup_iters):
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    seed = 7

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    device_inputs = []
    device_intermediate = []
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            if r == sender_coord[0] and c == sender_coord[1]:
                device_inputs.append(torch_a)
            else:
                device_inputs.append(torch.zeros_like(torch_a))
            device_intermediate.append(torch.zeros_like(torch_a))
    mesh_input = torch.cat(device_inputs, dim=0)
    mesh_intermediate = torch.cat(device_intermediate, dim=0)

    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        mesh_intermediate,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    output_buffers = [
        ttnn.from_torch(
            torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=submesh,
            memory_config=output_index_mem_config,
            mesh_mapper=mesh_mapper,
        )
        for _ in range(num_iters)
    ]
    scratch_buffers = [
        ttnn.from_torch(
            torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=submesh,
            memory_config=scratch_mem_config,
            mesh_mapper=mesh_mapper,
        )
        for _ in range(num_iters)
    ]

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    stage1_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    stage2_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    ttnn.synchronize_device(submesh)

    submesh.enable_program_cache()
    profiler = BenchmarkProfiler()

    _ = LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=output_buffers[0],
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
        global_semaphore=stage1_semaphores[0],
        global_stage2_semaphore=stage2_semaphores[0],
        fabric_scratch_tensor=scratch_buffers[0],
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
    )
    ttnn.synchronize_device(submesh)

    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=output_buffers[i % num_iters],
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=scratch_buffers[i % num_iters],
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=output_buffers[i],
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=scratch_buffers[i],
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    profiler.start("lm-head-sampling-mesh-4x2-trace-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end("lm-head-sampling-mesh-4x2-trace-warmup")

    signpost("start")
    profiler.start("lm-head-sampling-mesh-4x2-trace")
    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)
    profiler.end("lm-head-sampling-mesh-4x2-trace")
    signpost("stop")

    trace_duration_ns = profiler.get_duration("lm-head-sampling-mesh-4x2-trace")
    warmup_duration_ns = profiler.get_duration("lm-head-sampling-mesh-4x2-trace-warmup")
    effective_duration_ns = max(0.0, trace_duration_ns - warmup_duration_ns)
    avg_iter_ns = effective_duration_ns / float(max(1, num_iters))
    logger.info(
        f"LMHead+Argmax mesh(4x2) trace perf: final_mesh_coord={final_mesh_coord}, "
        f"iters={num_iters}, total_ns={effective_duration_ns:.2f}, avg_iter_ns={avg_iter_ns:.2f}"
    )

    final_output_shards = ttnn.get_device_tensors(output_buffers[-1])
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_torch = ttnn.to_torch(final_output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        final_output_torch, torch_expected_idx
    ), f"Perf run fused mesh argmax mismatch. expected={torch_expected_idx.item()}, got={int(final_output_torch.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [123, 1337, 52098])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_single_device(
    bh_2d_mesh_device,
    use_fp32,
    seed,
):
    """Single-device fused LM-head + argmax sampling with pre-cached width-sharded indices."""
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        torch.zeros_like(torch_a),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
    )
    ttnn.synchronize_device(submesh)

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Output index: {output_index_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [1337])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_single_device_d2h(
    bh_2d_mesh_device,
    use_fp32,
    seed,
):
    """Single-device fused LM-head + argmax with optional D2H token emission enabled."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 64

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        torch.zeros_like(torch_a),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        socket_output=d2h_socket,
    )

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    d2h_socket.barrier()
    ttnn.synchronize_device(submesh)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [7, 1337, 4242])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_multidevice(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
):
    """4x2 mesh fused LM-head + k=1 sampling (argmax) with CCL enabled."""
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    # Global indices are unique across mesh devices.
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    device_inputs = []
    device_intermediate = []
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            if r == sender_coord[0] and c == sender_coord[1]:
                device_inputs.append(torch_a)
            else:
                device_inputs.append(torch.zeros_like(torch_a))
            device_intermediate.append(torch.zeros_like(torch_a))
    mesh_input = torch.cat(device_inputs, dim=0)
    mesh_intermediate = torch.cat(device_intermediate, dim=0)

    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        mesh_intermediate,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output index: {final_output_index}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_d2h(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
):
    """4x2 mesh fused LM-head + argmax with optional D2H token emission on final mesh device."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 64

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    device_inputs = []
    device_intermediate = []
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            if r == sender_coord[0] and c == sender_coord[1]:
                device_inputs.append(torch_a)
            else:
                device_inputs.append(torch.zeros_like(torch_a))
            device_intermediate.append(torch.zeros_like(torch_a))
    mesh_input = torch.cat(device_inputs, dim=0)
    mesh_intermediate = torch.cat(device_intermediate, dim=0)

    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        mesh_intermediate,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(final_mesh_coord[0], final_mesh_coord[1]), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=d2h_socket,
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_d2d_to_d2h_pipeline(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
):
    """4x2 mesh fused LM-head + argmax with D2D output routed through D2D forwarding to D2H."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    socket_page_size_bytes = 64
    socket_fifo_size = 256

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    lmhead_input_core_x = 10
    lmhead_input_core_y = 9
    lmhead_input_core = ttnn.CoreCoord(lmhead_input_core_x, lmhead_input_core_y)
    lmhead_input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    argmax_final_core = ttnn.CoreCoord(0, 0)
    argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

    mcast_bbox = matmul_core_grid.bounding_box()
    reserved_cores = {(argmax_final_core.x, argmax_final_core.y), (lmhead_input_core.x, lmhead_input_core.y)}
    extra_cores = []
    for y in range(device_grid_size.y):
        for x in range(device_grid_size.x):
            if (x, y) in reserved_cores:
                continue
            if mcast_bbox.contains(ttnn.CoreCoord(x, y)):
                continue
            extra_cores.append(ttnn.CoreCoord(x, y))
    logger.info(f"Extra cores: {extra_cores}")
    if len(extra_cores) < 4:
        pytest.skip("Test requires at least 4 spare cores for D2D/D2H pipeline wiring")
    d2d1_core = ttnn.CoreCoord(11, 0)
    d2d2_core = ttnn.CoreCoord(11, 1)
    d2h_core = ttnn.CoreCoord(11, 2)
    dummy_h2d_core = ttnn.CoreCoord(11, 3)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(lmhead_input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    device_inputs = []
    device_intermediate = []
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            if r == sender_coord[0] and c == sender_coord[1]:
                device_inputs.append(torch_a)
            else:
                device_inputs.append(torch.zeros_like(torch_a))
            device_intermediate.append(torch.zeros_like(torch_a))
    mesh_input = torch.cat(device_inputs, dim=0)
    mesh_intermediate = torch.cat(device_intermediate, dim=0)

    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        mesh_intermediate,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    global_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)

    final_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        argmax_final_core,
    )

    d2d1_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d1_core,
    )
    d2d2_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d2_core,
    )
    d2h_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2h_core,
    )
    dummy_h2d_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        dummy_h2d_core,
    )

    logger.info(f"final_mesh_core: {final_mesh_core}")
    logger.info(f"d2d1_mesh_core: {d2d1_mesh_core}")
    logger.info(f"d2d2_mesh_core: {d2d2_mesh_core}")
    logger.info(f"d2h_mesh_core: {d2h_mesh_core}")
    logger.info(f"dummy_h2d_mesh_core: {dummy_h2d_mesh_core}")

    h2d_socket = ttnn.H2DSocket(
        submesh, dummy_h2d_mesh_core, ttnn.BufferType.L1, socket_fifo_size, ttnn.H2DMode.HOST_PUSH
    )
    d2h_socket = ttnn.D2HSocket(submesh, d2h_mesh_core, socket_fifo_size)
    logger.info("Creating HostInterface")
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        socket_page_size_bytes,
        socket_page_size_bytes,
        core_to_core_socket_buffer_size=socket_fifo_size,
        h2d_downstream_core=dummy_h2d_mesh_core,
        d2h_upstream_core=d2d2_mesh_core,
    )
    logger.info("Creating SocketInterface")
    socket_interface = SocketInterface(
        socket_page_size_bytes,
        socket_fifo_size,
        socket_page_size_bytes,
        d2d1_mesh_core,
        d2d2_mesh_core,
        upstream_core_coord=final_mesh_core,
        downstream_socket=host_io.get_upstream_socket(),
        sender_mesh=MeshWrapper(mesh_device=submesh),
        receiver_mesh=MeshWrapper(mesh_device=submesh),
    )

    logger.info("Running HostInterface")
    host_io.run()
    logger.info("Running SocketInterface")
    socket_interface.run()
    logger.info("Running LMHeadSampling")
    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=argmax_final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=socket_interface.get_upstream_socket(),
    )
    d2h_page_words = socket_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2D->D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2D->D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"

    host_io.terminate(False)
    socket_interface.terminate(True)

    ttnn.synchronize_device(submesh)


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_4stage_galaxy_1_iteration(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
):
    """4x2 mesh lm_head pipeline with H2D ingress + D2D ingress before compute, then D2D->D2H egress."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    activation_page_size_bytes = K * 2  # bf16 [1, 7168]
    activation_fifo_size = activation_page_size_bytes * 2
    socket_page_size_bytes = 64
    socket_fifo_size = 512
    assert activation_page_size_bytes == 14336
    assert socket_fifo_size == 8 * socket_page_size_bytes

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    lmhead_input_core_x = 10
    lmhead_input_core_y = 9
    lmhead_input_core = ttnn.CoreCoord(lmhead_input_core_x, lmhead_input_core_y)
    lmhead_input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    argmax_final_core = ttnn.CoreCoord(0, 0)
    argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

    reserved_cores = {(argmax_final_core.x, argmax_final_core.y), (lmhead_input_core.x, lmhead_input_core.y)}
    extra_cores = []
    for y in range(device_grid_size.y):
        for x in range(device_grid_size.x):
            if (x, y) in reserved_cores:
                continue
            if matmul_core_grid.bounding_box().contains(ttnn.CoreCoord(x, y)):
                continue
            extra_cores.append(ttnn.CoreCoord(x, y))
    if len(extra_cores) < 4:
        pytest.skip("Test requires at least 4 spare cores for H2D/D2D and D2D/D2H pipeline wiring")

    ingress_forward_core = ttnn.CoreCoord(11, 0)
    egress_sink_core = ttnn.CoreCoord(11, 1)
    d2h_endpoint_core = ttnn.CoreCoord(11, 2)
    h2d_endpoint_core = ttnn.CoreCoord(11, 3)
    ingress_relay_core = ttnn.CoreCoord(11, 4)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(lmhead_input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    device_inputs = []
    device_intermediate = []
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            if r == sender_coord[0] and c == sender_coord[1]:
                device_inputs.append(torch_a)
            else:
                device_inputs.append(torch.zeros_like(torch_a))
            device_intermediate.append(torch.zeros_like(torch_a))
    mesh_input = torch.cat(device_inputs, dim=0)
    mesh_intermediate = torch.cat(device_intermediate, dim=0)

    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    intermediate_tensor_mesh = ttnn.from_torch(
        mesh_intermediate,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    global_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    sender_mesh_coord = ttnn.MeshCoordinate(int(sender_coord[0]), int(sender_coord[1]))

    lmhead_input_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, lmhead_input_core)
    ingress_relay_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, ingress_relay_core)
    ingress_forward_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, ingress_forward_core)
    h2d_endpoint_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, h2d_endpoint_core)

    argmax_final_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        argmax_final_core,
    )

    egress_forward_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        ingress_forward_core,
    )
    egress_sink_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        egress_sink_core,
    )
    d2h_endpoint_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2h_endpoint_core,
    )
    h2d_host_socket = ttnn.H2DSocket(
        submesh,
        h2d_endpoint_mesh_core,
        ttnn.BufferType.L1,
        activation_fifo_size,
        ttnn.H2DMode.HOST_PUSH,
    )
    d2h_host_socket = ttnn.D2HSocket(submesh, d2h_endpoint_mesh_core, socket_fifo_size)
    host_io_bridge = HostInterface(
        h2d_host_socket,
        d2h_host_socket,
        activation_page_size_bytes,
        socket_page_size_bytes,
        core_to_core_socket_buffer_size=activation_fifo_size,
        h2d_downstream_core=ingress_relay_mesh_core,
        d2h_upstream_core=egress_sink_mesh_core,
    )
    ingress_d2d_link = SocketInterface(
        activation_page_size_bytes,
        activation_fifo_size,
        activation_page_size_bytes,
        ingress_relay_mesh_core,
        ingress_forward_mesh_core,
        upstream_socket=host_io_bridge.get_downstream_socket(),
        downstream_core_coord=lmhead_input_mesh_core,  # LMHead sender/socket-receiver core
        sender_mesh=MeshWrapper(submesh),
        receiver_mesh=MeshWrapper(submesh),
    )
    egress_d2d_link = SocketInterface(
        socket_page_size_bytes,
        socket_fifo_size,
        socket_page_size_bytes,
        egress_forward_mesh_core,
        egress_sink_mesh_core,
        upstream_core_coord=argmax_final_mesh_core,  # sampling winner core / socket sender core
        downstream_socket=host_io_bridge.get_upstream_socket(),
        sender_mesh=MeshWrapper(submesh),
        receiver_mesh=MeshWrapper(submesh),
    )

    logger.info("Running HostInterface")
    host_io_bridge.run()
    logger.info("Running Input SocketInterface")
    ingress_d2d_link.run()
    logger.info("Running Output SocketInterface")
    egress_d2d_link.run()

    try:
        h2d_activation_tensor = ttnn.from_torch(
            torch_a.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        logger.info("Running H2D socket write")
        h2d_host_socket.write_tensor(h2d_activation_tensor)

        logger.info("Running LMHeadSampling")
        LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=argmax_final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
            global_semaphore=global_semaphore,
            global_stage2_semaphore=global_stage2_semaphore,
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            socket_input=ingress_d2d_link.get_downstream_socket(),
            socket_output=egress_d2d_link.get_upstream_socket(),
        )
        logger.info("Running D2H socket read")
        d2h_page_words = socket_page_size_bytes // 4
        d2h_read_tensor = ttnn.from_torch(
            torch.zeros((1, d2h_page_words), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        d2h_host_socket.read_tensor(d2h_read_tensor)
        d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
        assert torch.equal(
            d2h_token, torch_expected_idx
        ), f"Mesh H2D->D2D->LMHead->D2D->D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"
    finally:
        host_io_bridge.terminate(False)
        ingress_d2d_link.terminate(False)
        egress_d2d_link.terminate(True)
        ttnn.synchronize_device(submesh)


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_lm_head_sampling_pipeline_block_4stage_single_galaxy(mesh_device, use_fp32):
    """
    4-stage 4x2 single-galaxy pipeline:
    P1(H2D) -> P2(LMHead+Sampling) -> P3(forward) -> P4(forward) -> P1(D2H).
    One-shot LMHead (no persistent mode); single token; terminate in finally.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    embedding_tensor, lmhead_weights, torch_expected_idx = create_random_weights_single_iteration(seed=5449)
    stage_kind = create_stage_kind(
        my_mesh_id,
        lmhead_stage=1,
        embedding_tensor=embedding_tensor,
        lmhead_weights=lmhead_weights,
        fp32_dest_acc_en=use_fp32,
        persistent_mode=False,
    )
    pipeline = PodPipeline(mesh_device, stage_kind)
    try:
        pipeline.setup()
        pipeline.run()

        if pipeline.my_mesh_id == 0:
            torch_token = torch.zeros(1, token_page_size_bytes // 4, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, token_page_size_bytes // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            assert torch.equal(
                got, torch_expected_idx
            ), f"PipelineBlock 4-stage token mismatch. expected={int(torch_expected_idx.item())}, got={int(got.item())}"

        pipeline.barrier()
    finally:
        pipeline.terminate()


@pytest.mark.skipif(not _is_persistent_mode_enabled(), reason="Set RUN_PERSISTENT_MODE=1 to run persistent mode test")
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode(mesh_device, use_fp32):
    """
    4-stage 4x2 single-galaxy pipeline:
    P1(H2D) -> P2(LMHead+Sampling) -> P3(forward) -> P4(forward) -> P1(D2H).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 100
    embedding_tensor, lmhead_weights, torch_expected_indices = create_synthetic_weights(iterations)
    stage_kind = create_stage_kind(
        my_mesh_id,
        lmhead_stage=1,
        embedding_tensor=embedding_tensor,
        lmhead_weights=lmhead_weights,
        fp32_dest_acc_en=use_fp32,
    )
    pipeline = PodPipeline(mesh_device, stage_kind)
    pipeline.setup()
    pipeline.run()

    if pipeline.my_mesh_id == 0:
        for iteration in range(iterations):
            torch_token = torch.zeros(1, token_page_size_bytes // 4, dtype=torch.uint32)
            torch_token[0, 0] = iteration
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, token_page_size_bytes // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            expected_idx = torch_expected_indices[iteration]
            logger.info(f"Iteration {iteration} output token: {got}, expected: {expected_idx}")
            assert torch.equal(
                got, expected_idx
            ), f"PipelineBlock 4-stage token mismatch. expected={int(expected_idx.item())}, got={int(got.item())}"

    logger.info(f"Barrier for P{pipeline.my_mesh_id}")
    pipeline.barrier()
    logger.info(f"Barrier completed for P{pipeline.my_mesh_id}")


# @pytest.mark.skipif(not _is_persistent_mode_enabled(), reason="Set RUN_PERSISTENT_MODE=1 to run persistent mode test")
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode_pod(mesh_device, use_fp32):
    """
    16-stage 4x2 pod pipeline (4 galaxies):
    Stage1(H2D+Embed) -> Stage2..14(activation fwd) -> Stage15(LMHead+Sampling) -> Stage16(token fwd) -> Stage1(D2H).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 16:
        pytest.skip("This test requires exactly 16 distributed pipeline processes (pod: 4 galaxies)")

    iterations = 100
    embedding_tensor, lmhead_weights, torch_expected_indices = create_synthetic_weights(iterations)
    stage_kind = create_stage_kind(
        my_mesh_id,
        lmhead_stage=14,
        embedding_tensor=embedding_tensor,
        lmhead_weights=lmhead_weights,
        fp32_dest_acc_en=use_fp32,
    )
    pipeline = PodPipeline(mesh_device, stage_kind)
    pipeline.setup()
    pipeline.run()

    if pipeline.my_mesh_id == 0:
        for iteration in range(iterations):
            torch_token = torch.zeros(1, token_page_size_bytes // 4, dtype=torch.uint32)
            torch_token[0, 0] = iteration
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, token_page_size_bytes // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            expected_idx = torch_expected_indices[iteration]
            logger.info(f"Iteration {iteration} output token: {got}, expected: {expected_idx}")
            assert torch.equal(
                got, expected_idx
            ), f"Pod 16-stage token mismatch at iter {iteration}. expected={int(expected_idx.item())}, got={int(got.item())}"

    logger.info(f"Barrier for stage {pipeline.my_mesh_id + 1}")
    pipeline.barrier()
    logger.info(f"Barrier completed for stage {pipeline.my_mesh_id + 1}")
