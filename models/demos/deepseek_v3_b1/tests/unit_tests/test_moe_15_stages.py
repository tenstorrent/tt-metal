# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-stage pipeline integration test for forward + MoE fused op + reduce-to-all.

Uses parallel sockets across all 8 devices in the mesh. Each device independently
receives its input via a per-device forward operation (socket read → residual tensor)
and produces its output via reduce-to-all (all devices hold the full reduced result).

Stage 0:
  8 parallel H2D/D2H + HostInterface (one per device) with embedding lookup.
  ParallelSocketInterface connects exit (→ stage 1) and entry (← last stage loopback).
Stage 1+:
  PipelineBlock with pipeline_device_coords for per-device parallel forwarding.
  Entry D2D → moe_sender_core socket → forward + fused MoE + reduce-to-all → exit D2D.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import (
    MeshWrapper,
    ParallelSocketInterface,
    _combine_overlapping_programs,
    _group_by_device,
)
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    ROUTED_EXPERT_LAYER_IDX,
    RoutedExpert,
    create_routed_expert_tensors,
    create_shared_expert_tensors,
)
from models.demos.deepseek_v3_b1.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _dispatch_merged_programs(all_entries, mesh_device):
    """Merge (device_coord, program) entries by device and dispatch in a single generic_op."""
    dummy_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
    )
    groups = _group_by_device(all_entries)
    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    for device_coord, progs in groups:
        if len(progs) > 1:
            progs = _combine_overlapping_programs(progs)
            merged = ttnn.merge_program_descriptors(progs) if len(progs) > 1 else progs[0]
        else:
            merged = progs[0]
        mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = merged
    return ttnn.generic_op([dummy_tensor, dummy_tensor], mesh_program_descriptor)


def build_worker_grid_excluding_cores(device_grid_size, excluded_cores):
    """Build a CoreRangeSet covering the full device grid minus a set of excluded cores."""
    full_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
    )
    excluded_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in excluded_cores])
    return full_grid.subtract(excluded_set)


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
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
@pytest.mark.parametrize("token_id", [0])
@pytest.mark.timeout(120000)
def test_moe_15_stages(mesh_device, vocab_size, embedding_dim, token_id, device_params, get_reference_model_state_dict):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert len(pipeline_config) == num_procs + 1

    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0

    M = 1
    K = embedding_dim

    # Core allocation — single pipeline_core for both entry and exit d2d exchange.
    # The entry d2d_exchange runs on BRISC, the multi-upstream exit runs on NCRISC,
    # so they can share one core without RISC conflict.
    pipeline_core = ttnn.CoreCoord(12, 8)
    core_io = ttnn.CoreCoord(0, 2)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_cores(device_grid, [pipeline_core])

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    reduce_root_coord = ttnn.MeshCoordinate(0, 0)

    # ── Core setup for reduce aggregation ──
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    reduce_payload_per_shard = embedding_size_bytes // num_gate_proj_cores
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    # Device coordinates for parallel channels (all devices in mesh)
    mesh_rows, mesh_cols = mesh_device.shape
    device_coords = [ttnn.MeshCoordinate(r, c) for r in range(int(mesh_rows)) for c in range(int(mesh_cols))]

    # ── Pipeline block setup (collective — all hosts must participate simultaneously) ──
    pipeline_block = None
    h2d_sockets = None
    d2h_sockets = None
    host_ios = None
    exit_socket_interface = None
    entry_socket_interface = None
    stage0_program_entries = None

    if is_stage0:
        # PipelineBlock.__init__ calls generate_blitz_decode_pipeline (collective).
        # Stage 0 must match so all processes participate.
        ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)

        embedding_tensor = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in device_coords]

        exit_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=d2d_cores,
            recv_core_coords=d2d_cores,
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
        )

        entry_socket_interface = ParallelSocketInterface(
            embedding_size_bytes,
            embedding_fifo_size,
            send_core_coords=d2d_cores,
            recv_core_coords=d2d_cores,
            sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
            receiver_mesh=MeshWrapper(mesh_device),
            receiver_use_reader_config=True,
        )

        # Local H2D/D2H + HostInterface setup (no cross-mesh handshake)
        h2d_sockets = []
        d2h_sockets = []
        host_ios = []

        for dc in device_coords:
            h2d = ttnn.H2DSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                ttnn.BufferType.L1,
                token_size_bytes * 2,
                ttnn.H2DMode.HOST_PUSH,
            )
            d2h = ttnn.D2HSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                embedding_fifo_size,
            )
            hio = HostInterface(
                h2d,
                d2h,
                token_size_bytes,
                embedding_size_bytes,
                core_to_core_socket_buffer_size=embedding_fifo_size,
                h2d_downstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                d2h_upstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                embedding_tensor=embedding_tensor,
            )
            h2d_sockets.append(h2d)
            d2h_sockets.append(d2h)
            host_ios.append(hio)

        # Wire upstream/downstream sockets now that HostInterface instances exist
        for i, hio in enumerate(host_ios):
            exit_socket_interface._upstream_sockets[i] = hio.get_downstream_socket()
            entry_socket_interface._downstream_sockets[i] = hio.get_upstream_socket()

        all_entries = []
        for hio in host_ios:
            all_entries.extend(hio._build_programs())
        exit_progs = exit_socket_interface.build_programs()
        combined_progs = entry_socket_interface.build_programs(base_programs=exit_progs)
        all_entries.extend(combined_progs)
        stage0_program_entries = all_entries
        logger.info(f"[rank=0] parallel stage 0 programs built ({len(device_coords)} channels), dispatch deferred")
    else:
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            pipeline_device_coords=device_coords,
            pipeline_exit_core_coord=pipeline_core,
            entry_downstream_core=moe_sender_core,
            exit_upstream_cores=shard_cores_list,
            exit_upstream_page_size=reduce_payload_per_shard,
        )
        logger.info(f"[rank={my_mesh_id}] parallel pipeline block created")

    # ── Get per-device sockets for forward (input) and reduce (output) ──
    forward_sockets = None
    downstream_sockets = None
    if my_mesh_id >= 1:
        forward_sockets = pipeline_block.get_downstream_sockets()
        downstream_sockets = pipeline_block.get_upstream_sockets()
        logger.info(
            f"[rank={my_mesh_id}] {len(forward_sockets)} forward sockets, "
            f"{len(downstream_sockets)} downstream socket groups"
        )

    # ── MoE tensor setup ──
    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    r = create_routed_expert_tensors(
        mesh_device,
        use_hardcoded_expert_index=True,
        mesh_mapper=mesh_mapper,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        skip_attention_weights=True,
    )
    mcast_grid = moe_worker_core_grid
    s = create_shared_expert_tensors(
        mesh_device,
        M,
        K,
        mcast_grid,
        mesh_mapper=mesh_mapper,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
    )
    logger.info(f"[rank={my_mesh_id}] MoE tensors created")

    if my_mesh_id >= 1:
        kv_cache_shard_height = 256
        kvpe_dim = 576
        num_mcast_cores = mcast_grid.num_cores()
        kv_cache_shard_spec = ttnn.ShardSpec(
            mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR
        )
        sdpa_kv_cache_buffer = ttnn.from_torch(
            torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
            ),
        )
        sdpa_out_interm_shard_height = 48
        sdpa_out_interm_shard_width = 544
        num_worker_cores = moe_worker_core_grid.num_cores()
        sdpa_out_interm_shard_spec = ttnn.ShardSpec(
            moe_worker_core_grid,
            (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sdpa_out_interm_buffer = ttnn.from_torch(
            torch.zeros(
                (sdpa_out_interm_shard_height * num_worker_cores, sdpa_out_interm_shard_width), dtype=torch.bfloat16
            ),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
            ),
            tile=ttnn.Tile([8, 32]),
        )
        logger.info(f"[rank={my_mesh_id}] SDPA buffers created")

        tile_1x32 = ttnn.Tile([1, 32])
        final_output_total_width = r.final_output_total_width

        # ── Reduce-to-all tensors ──
        reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
        )
        reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

        final_output_mem_config = r.final_output_mem_config
        intermediate_shard_spec = final_output_mem_config.shard_spec
        intermediate_shard_shape = [intermediate_shard_spec.shape[0], intermediate_shard_spec.shape[1] * 3]
        intermediate_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.types.BufferType.L1,
            ttnn.ShardSpec(
                intermediate_shard_spec.grid,
                intermediate_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        reduce_intermediate_tensors = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=intermediate_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        logger.info(f"[rank={my_mesh_id}] reduce intermediate tensors created")

        reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
        reduce_output_shard_spec = ttnn.ShardSpec(
            reduce_output_shard_grid,
            (1, final_output_total_width),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        reduce_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
        )
        reduce_output_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
        reduce_output_tensor = ttnn.from_torch(
            reduce_output_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=reduce_output_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        logger.info(f"[rank={my_mesh_id}] reduce output tensor created")

        compute_grid = mesh_device.compute_with_storage_grid_size()
        num_cores = compute_grid.x * compute_grid.y
        reduce_available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_available_cores, 0) for _ in range(4)]

        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        logger.info(f"[rank={my_mesh_id}] semaphores created")

    # ── Launch pipeline programs ──────────────────────────────────────────────
    if is_stage0:
        _dispatch_merged_programs(stage0_program_entries, mesh_device)
        logger.info(f"[rank=0] stage0 programs dispatched (deferred)")
    else:
        pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    ttnn.distributed_context_barrier()

    if is_stage0:
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for h2d in h2d_sockets:
            h2d.write_tensor(token_tensor)
        logger.info(f"[rank=0] token {token_id} injected to {len(h2d_sockets)} channels")

    ttnn.distributed_context_barrier()

    if my_mesh_id >= 1:
        logger.info(f"[rank={my_mesh_id}] launching MoE forward + reduce-to-all (num_iterations=1)")
        MoeOp.op(
            r.ttnn_residual_mcast_src,
            gate_mm_weights_tensor=r.ttnn_gate_mm_weights,
            gate_bias_tensor=r.ttnn_gate_bias,
            gate_indices_tensor=r.ttnn_gate_indices,
            gate_output_scores_tensor=r.gate_output_scores_tensor,
            gate_output_indices_tensor=r.gate_output_indices_tensor,
            gate_proj_weights_tensor=r.gate_proj_weights,
            up_proj_weights_tensor=r.up_proj_weights,
            down_proj_weights_tensor=r.down_proj_weights,
            final_output_tensor=r.final_output_tensor,
            rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
            shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
            shared_up_weights_overlapped=s.shared_up_weights_overlapped,
            shared_down_weights_tensor=s.ttnn_down_weights,
            shared_k_parallel=s.k_parallel,
            shared_n_parallel=s.n_parallel,
            use_hardcoded_expert_index=True,
            sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer=sdpa_out_interm_buffer,
            num_iterations=1,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=reduce_root_coord,
            forward_sockets=forward_sockets,
            semaphores=moe_semaphores,
            worker_core_grid=moe_worker_core_grid,
            is_torus=is_torus,
            downstream_sockets=downstream_sockets,
        )
        logger.info(f"[rank={my_mesh_id}] MoE + reduce-to-all completed")

    # ── Stage 0: D2H loopback read + golden validation ───────────────────────
    if is_stage0:
        logger.info(f"[rank=0] waiting for D2H result from pipeline loopback")
        num_elements = embedding_size_bytes // 2
        received_tensor_torch = torch.zeros(1, num_elements, dtype=torch.bfloat16)
        d2h_output_tensor = ttnn.from_torch(received_tensor_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        d2h_sockets[0].read_tensor(d2h_output_tensor)
        d2h_result_torch = ttnn.to_torch(d2h_output_tensor)
        logger.info(f"[rank=0] D2H read complete, shape={d2h_result_torch.shape}")
        logger.info(f"[rank=0] D2H first 5 values: {d2h_result_torch[0, :5]}")

        d2h_nonzero = torch.count_nonzero(d2h_result_torch)
        logger.info(f"[rank=0] D2H non-zero elements: {d2h_nonzero}/{d2h_result_torch.numel()}")
        assert d2h_nonzero > 0, "D2H output is all zeros — reduce or pipeline failed"

    ttnn.distributed_context_barrier()

    # ── Pipeline teardown ───────────────────────────────────────────────────
    logger.info(f"[rank={my_mesh_id}] waiting for pipeline termination")
    if is_stage0:
        for hio in host_ios:
            hio.terminate(False)
        entry_socket_interface.terminate(False)
        exit_socket_interface.terminate(True)
    else:
        pipeline_block.terminate()
    logger.info(f"[rank={my_mesh_id}] programs terminated")

    logger.info(f"[rank={my_mesh_id}] test PASSED")


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
@pytest.mark.parametrize("embedding_dim", [7168])
@pytest.mark.parametrize("iterations", [4000])
@pytest.mark.timeout(120000)
def test_persistent_moe_15_stages(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent-mode 15-stage MoE pipeline test with parallel sockets.

    Pipeline topology:
      Stage 0  : 8 parallel H2D/D2H + HostInterface with embedding
      Stage 1-14: forward + fused MoE + reduce-to-all (persistent while-true loop)

    Socket flow control handles inter-iteration synchronization (no persistent semaphore needed).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert len(pipeline_config) == num_procs + 1

    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0
    is_moe_stage = my_mesh_id >= 1

    M = 1
    K = embedding_dim
    pipeline_core = ttnn.CoreCoord(12, 8)
    core_io = ttnn.CoreCoord(0, 2)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_cores(device_grid, [pipeline_core])

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch.manual_seed(42)
    print("Create torch embedding")
    torch_embedding = torch.randn(iterations, 1, 1, K, dtype=torch.bfloat16)
    print("Torch embedding created")

    reduce_root_coord = ttnn.MeshCoordinate(0, 0)

    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    reduce_payload_per_shard = embedding_size_bytes // num_gate_proj_cores
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    mesh_rows, mesh_cols = mesh_device.shape
    device_coords = [ttnn.MeshCoordinate(r, c) for r in range(int(mesh_rows)) for c in range(int(mesh_cols))]

    # ── Pipeline block setup (collective) ──
    pipeline_block = None
    h2d_sockets = None
    d2h_sockets = None
    host_ios = None
    exit_socket_interface = None
    entry_socket_interface = None

    try:
        if is_stage0:
            # PipelineBlock.__init__ calls generate_blitz_decode_pipeline (collective).
            # Stage 0 must match so all processes participate.
            ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)

            print("Write embedding to tensor")
            embedding_tensor = ttnn.from_torch(
                torch_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            print("Embedding tensor written to device")

            d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in device_coords]

            exit_socket_interface = ParallelSocketInterface(
                embedding_size_bytes,
                embedding_fifo_size,
                send_core_coords=d2d_cores,
                recv_core_coords=d2d_cores,
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
            )

            entry_socket_interface = ParallelSocketInterface(
                embedding_size_bytes,
                embedding_fifo_size,
                send_core_coords=d2d_cores,
                recv_core_coords=d2d_cores,
                sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
                receiver_mesh=MeshWrapper(mesh_device),
                receiver_use_reader_config=True,
            )

            h2d_sockets = []
            d2h_sockets = []
            host_ios = []

            for dc in device_coords:
                h2d = ttnn.H2DSocket(
                    mesh_device,
                    ttnn.MeshCoreCoord(dc, core_io),
                    ttnn.BufferType.L1,
                    token_size_bytes * 2,
                    ttnn.H2DMode.HOST_PUSH,
                )
                d2h = ttnn.D2HSocket(
                    mesh_device,
                    ttnn.MeshCoreCoord(dc, core_io),
                    embedding_fifo_size,
                )
                hio = HostInterface(
                    h2d,
                    d2h,
                    token_size_bytes,
                    embedding_size_bytes,
                    core_to_core_socket_buffer_size=embedding_fifo_size,
                    h2d_downstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                    d2h_upstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                    embedding_tensor=embedding_tensor,
                )
                h2d_sockets.append(h2d)
                d2h_sockets.append(d2h)
                host_ios.append(hio)

            for i, hio in enumerate(host_ios):
                exit_socket_interface._upstream_sockets[i] = hio.get_downstream_socket()
                entry_socket_interface._downstream_sockets[i] = hio.get_upstream_socket()

            all_entries = []
            for hio in host_ios:
                all_entries.extend(hio._build_programs())
            exit_progs = exit_socket_interface.build_programs()
            combined_progs = entry_socket_interface.build_programs(base_programs=exit_progs)
            all_entries.extend(combined_progs)
            _dispatch_merged_programs(all_entries, mesh_device)
            logger.info(f"[rank=0] parallel stage 0 programs dispatched ({len(device_coords)} channels)")
        else:
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                pipeline_device_coords=device_coords,
                pipeline_exit_core_coord=pipeline_core,
                entry_downstream_core=moe_sender_core,
                exit_upstream_cores=shard_cores_list,
                exit_upstream_page_size=reduce_payload_per_shard,
            )

        forward_sockets = None
        downstream_sockets = None
        if my_mesh_id >= 1:
            forward_sockets = pipeline_block.get_downstream_sockets()
            downstream_sockets = pipeline_block.get_upstream_sockets()

        # ── MoE tensors (MoE stages only) ──
        state_dict = get_reference_model_state_dict(
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            is_moe=True,
            seed=RoutedExpert.SEED,
            num_routed_experts=256,
            include_global=False,
        )

        r = None
        s = None
        if is_moe_stage:
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            r = create_routed_expert_tensors(
                mesh_device,
                use_hardcoded_expert_index=True,
                mesh_mapper=mesh_mapper,
                state_dict=state_dict,
                is_moe=True,
                layer_idx=ROUTED_EXPERT_LAYER_IDX,
                skip_attention_weights=True,
            )
            mcast_grid = moe_worker_core_grid
            s = create_shared_expert_tensors(
                mesh_device,
                M,
                K,
                mcast_grid,
                mesh_mapper=mesh_mapper,
                state_dict=state_dict,
                is_moe=True,
                layer_idx=ROUTED_EXPERT_LAYER_IDX,
            )
            logger.info(f"[rank={my_mesh_id}] MoE tensors created")

            kv_cache_shard_height = 256
            kvpe_dim = 576
            num_mcast_cores = mcast_grid.num_cores()
            kv_cache_shard_spec = ttnn.ShardSpec(
                mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR
            )
            sdpa_kv_cache_buffer = ttnn.from_torch(
                torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
                ),
            )
            sdpa_out_interm_shard_height = 48
            sdpa_out_interm_shard_width = 544
            num_worker_cores = moe_worker_core_grid.num_cores()
            sdpa_out_interm_shard_spec = ttnn.ShardSpec(
                moe_worker_core_grid,
                (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            sdpa_out_interm_buffer = ttnn.from_torch(
                torch.zeros(
                    (sdpa_out_interm_shard_height * num_worker_cores, sdpa_out_interm_shard_width),
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
                ),
                tile=ttnn.Tile([8, 32]),
            )

            tile_1x32 = ttnn.Tile([1, 32])
            final_output_total_width = r.final_output_total_width

            reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
                [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
            )
            reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

            final_output_mem_config = r.final_output_mem_config
            intermediate_shard_spec = final_output_mem_config.shard_spec
            intermediate_shard_shape = [intermediate_shard_spec.shape[0], intermediate_shard_spec.shape[1] * 3]
            intermediate_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.BufferType.L1,
                ttnn.ShardSpec(
                    intermediate_shard_spec.grid,
                    intermediate_shard_shape,
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            reduce_intermediate_tensors = ttnn.from_torch(
                torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=intermediate_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )

            compute_grid = mesh_device.compute_with_storage_grid_size()
            reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
            reduce_output_shard_spec = ttnn.ShardSpec(
                reduce_output_shard_grid,
                (1, final_output_total_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            reduce_output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
            )
            reduce_output_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
            reduce_output_tensor = ttnn.from_torch(
                reduce_output_data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=reduce_output_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )

            num_cores = compute_grid.x * compute_grid.y
            reduce_available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
            reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_available_cores, 0) for _ in range(4)]

            moe_semaphores = MoeOp.create_semaphores(mesh_device)
            logger.info(f"[rank={my_mesh_id}] MoE setup complete")

        # ── Launch pipeline ──
        if not is_stage0:
            pipeline_block.run()
        logger.info(f"[rank={my_mesh_id}] pipeline launched")

        # ── MoE stages: submit persistent kernel ──
        if my_mesh_id >= 1:
            logger.info(f"[rank={my_mesh_id}] submitting persistent MoE kernel")
            MoeOp.op(
                r.ttnn_residual_mcast_src,
                gate_mm_weights_tensor=r.ttnn_gate_mm_weights,
                gate_bias_tensor=r.ttnn_gate_bias,
                gate_indices_tensor=r.ttnn_gate_indices,
                gate_output_scores_tensor=r.gate_output_scores_tensor,
                gate_output_indices_tensor=r.gate_output_indices_tensor,
                gate_proj_weights_tensor=r.gate_proj_weights,
                up_proj_weights_tensor=r.up_proj_weights,
                down_proj_weights_tensor=r.down_proj_weights,
                final_output_tensor=r.final_output_tensor,
                rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
                shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
                shared_up_weights_overlapped=s.shared_up_weights_overlapped,
                shared_down_weights_tensor=s.ttnn_down_weights,
                shared_k_parallel=s.k_parallel,
                shared_n_parallel=s.n_parallel,
                use_hardcoded_expert_index=True,
                sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
                sdpa_out_interm_buffer=sdpa_out_interm_buffer,
                num_iterations=1,
                persistent_mode=True,
                reduce_intermediate_tensors=reduce_intermediate_tensors,
                reduce_output_tensor=reduce_output_tensor,
                reduce_semaphores=reduce_semaphores,
                reduce_root_coord=reduce_root_coord,
                forward_sockets=forward_sockets,
                semaphores=moe_semaphores,
                worker_core_grid=moe_worker_core_grid,
                is_torus=is_torus,
                downstream_sockets=downstream_sockets,
            )
            logger.info(f"[rank={my_mesh_id}] persistent MoE kernel submitted")
        ttnn.distributed_context_barrier()

        # ── Stage 0: drive pipeline with multiple tokens ──
        if is_stage0:
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            num_elements = embedding_size_bytes // 2
            torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            start_time = time.time()
            for iteration in range(iterations):
                for h2d in h2d_sockets:
                    h2d.write_tensor(token_tensor)
                logger.info(f"[rank=0] token {iteration} injected to {len(h2d_sockets)} channels")

                d2h_output_tensor = ttnn.from_torch(
                    torch.zeros(1, num_elements, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                print(f"[rank={my_mesh_id}] iteration {iteration} waiting for D2H result")
                d2h_sockets[0].read_tensor(d2h_output_tensor)
                print(f"[rank={my_mesh_id}] iteration {iteration} D2H result read")
                d2h_result = ttnn.to_torch(d2h_output_tensor)

                d2h_nonzero = torch.count_nonzero(d2h_result)
                logger.info(
                    f"[rank={my_mesh_id}] iteration {iteration}: non-zero={d2h_nonzero}/{d2h_result.numel()}, "
                    f"first 5={d2h_result[0, :5]}"
                )
                assert (
                    d2h_nonzero > 0
                ), f"D2H output is all zeros at iteration {iteration} — persistent MoE pipeline failed"
            end_time = time.time()
            print(f"[rank=0] time taken to move {iterations} tokens: {end_time - start_time} seconds")

            logger.info(f"[rank={my_mesh_id}] all {iterations} iterations passed")

        logger.info(f"[rank={my_mesh_id}] waiting for barrier")
        ttnn.distributed_context_barrier()
        logger.info(f"[rank={my_mesh_id}] barrier completed")

    finally:
        pass

    logger.info(f"[rank={my_mesh_id}] persistent 15-stage MoE test PASSED")


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
@pytest.mark.parametrize("embedding_dim", [7168])
@pytest.mark.parametrize("iterations", [4000])
@pytest.mark.timeout(120000)
def test_persistent_moe_multi_token(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent-mode multi-token-in-flight MoE pipeline test with parallel sockets.

    Same topology as test_persistent_moe_15_stages but injects multiple tokens
    before reading results, testing pipelined throughput.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for MoE (need >= 13x10)")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert len(pipeline_config) == num_procs + 1

    is_torus = device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_2D_TORUS_Y

    is_stage0 = my_mesh_id == 0
    is_moe_stage = my_mesh_id >= 1

    M = 1
    K = embedding_dim
    pipeline_core = ttnn.CoreCoord(12, 8)
    core_io = ttnn.CoreCoord(0, 2)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_cores(device_grid, [pipeline_core])

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch.manual_seed(42)
    print("Create torch embedding")
    torch_embedding = torch.randn(iterations, 1, 1, K, dtype=torch.bfloat16)
    print("Torch embedding created")

    reduce_root_coord = ttnn.MeshCoordinate(0, 0)

    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    reduce_payload_per_shard = embedding_size_bytes // num_gate_proj_cores
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    mesh_rows, mesh_cols = mesh_device.shape
    device_coords = [ttnn.MeshCoordinate(r, c) for r in range(int(mesh_rows)) for c in range(int(mesh_cols))]

    pipeline_block = None
    h2d_sockets = None
    d2h_sockets = None
    host_ios = None
    exit_socket_interface = None
    entry_socket_interface = None

    try:
        if is_stage0:
            # PipelineBlock.__init__ calls generate_blitz_decode_pipeline (collective).
            # Stage 0 must match so all processes participate.
            ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)

            print("Write embedding to tensor")
            embedding_tensor = ttnn.from_torch(
                torch_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            print("Embedding tensor written to device")

            d2d_cores = [ttnn.MeshCoreCoord(dc, pipeline_core) for dc in device_coords]

            exit_socket_interface = ParallelSocketInterface(
                embedding_size_bytes,
                embedding_fifo_size,
                send_core_coords=d2d_cores,
                recv_core_coords=d2d_cores,
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(mesh_id=my_mesh_id + 1),
            )

            entry_socket_interface = ParallelSocketInterface(
                embedding_size_bytes,
                embedding_fifo_size,
                send_core_coords=d2d_cores,
                recv_core_coords=d2d_cores,
                sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
                receiver_mesh=MeshWrapper(mesh_device),
                receiver_use_reader_config=True,
            )

            h2d_sockets = []
            d2h_sockets = []
            host_ios = []

            for dc in device_coords:
                h2d = ttnn.H2DSocket(
                    mesh_device,
                    ttnn.MeshCoreCoord(dc, core_io),
                    ttnn.BufferType.L1,
                    token_size_bytes * 2,
                    ttnn.H2DMode.HOST_PUSH,
                )
                d2h = ttnn.D2HSocket(
                    mesh_device,
                    ttnn.MeshCoreCoord(dc, core_io),
                    embedding_fifo_size,
                )
                hio = HostInterface(
                    h2d,
                    d2h,
                    token_size_bytes,
                    embedding_size_bytes,
                    core_to_core_socket_buffer_size=embedding_fifo_size,
                    h2d_downstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                    d2h_upstream_core=ttnn.MeshCoreCoord(dc, pipeline_core),
                    embedding_tensor=embedding_tensor,
                )
                h2d_sockets.append(h2d)
                d2h_sockets.append(d2h)
                host_ios.append(hio)

            for i, hio in enumerate(host_ios):
                exit_socket_interface._upstream_sockets[i] = hio.get_downstream_socket()
                entry_socket_interface._downstream_sockets[i] = hio.get_upstream_socket()

            all_entries = []
            for hio in host_ios:
                all_entries.extend(hio._build_programs())
            exit_progs = exit_socket_interface.build_programs()
            combined_progs = entry_socket_interface.build_programs(base_programs=exit_progs)
            all_entries.extend(combined_progs)
            _dispatch_merged_programs(all_entries, mesh_device)
            logger.info(f"[rank=0] parallel stage 0 programs dispatched ({len(device_coords)} channels)")
        else:
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                pipeline_device_coords=device_coords,
                pipeline_exit_core_coord=pipeline_core,
                entry_downstream_core=moe_sender_core,
                exit_upstream_cores=shard_cores_list,
                exit_upstream_page_size=reduce_payload_per_shard,
            )

        forward_sockets = None
        downstream_sockets = None
        if my_mesh_id >= 1:
            forward_sockets = pipeline_block.get_downstream_sockets()
            downstream_sockets = pipeline_block.get_upstream_sockets()

        # ── MoE tensors (MoE stages only) ──
        state_dict = get_reference_model_state_dict(
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            is_moe=True,
            seed=RoutedExpert.SEED,
            num_routed_experts=256,
            include_global=False,
        )

        r = None
        s = None
        if is_moe_stage:
            mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            r = create_routed_expert_tensors(
                mesh_device,
                use_hardcoded_expert_index=True,
                mesh_mapper=mesh_mapper,
                state_dict=state_dict,
                is_moe=True,
                layer_idx=ROUTED_EXPERT_LAYER_IDX,
                skip_attention_weights=True,
            )
            mcast_grid = moe_worker_core_grid
            s = create_shared_expert_tensors(
                mesh_device,
                M,
                K,
                mcast_grid,
                mesh_mapper=mesh_mapper,
                state_dict=state_dict,
                is_moe=True,
                layer_idx=ROUTED_EXPERT_LAYER_IDX,
            )
            logger.info(f"[rank={my_mesh_id}] MoE tensors created")

            kv_cache_shard_height = 256
            kvpe_dim = 576
            num_mcast_cores = mcast_grid.num_cores()
            kv_cache_shard_spec = ttnn.ShardSpec(
                mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR
            )
            sdpa_kv_cache_buffer = ttnn.from_torch(
                torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
                ),
            )
            sdpa_out_interm_shard_height = 48
            sdpa_out_interm_shard_width = 544
            num_worker_cores = moe_worker_core_grid.num_cores()
            sdpa_out_interm_shard_spec = ttnn.ShardSpec(
                moe_worker_core_grid,
                (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            sdpa_out_interm_buffer = ttnn.from_torch(
                torch.zeros(
                    (sdpa_out_interm_shard_height * num_worker_cores, sdpa_out_interm_shard_width),
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
                ),
                tile=ttnn.Tile([8, 32]),
            )

            tile_1x32 = ttnn.Tile([1, 32])
            final_output_total_width = r.final_output_total_width

            reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
                [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
            )
            reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

            final_output_mem_config = r.final_output_mem_config
            intermediate_shard_spec = final_output_mem_config.shard_spec
            intermediate_shard_shape = [intermediate_shard_spec.shape[0], intermediate_shard_spec.shape[1] * 3]
            intermediate_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.BufferType.L1,
                ttnn.ShardSpec(
                    intermediate_shard_spec.grid,
                    intermediate_shard_shape,
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            reduce_intermediate_tensors = ttnn.from_torch(
                torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=intermediate_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )

            compute_grid = mesh_device.compute_with_storage_grid_size()
            reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
            reduce_output_shard_spec = ttnn.ShardSpec(
                reduce_output_shard_grid,
                (1, final_output_total_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            reduce_output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
            )
            reduce_output_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
            reduce_output_tensor = ttnn.from_torch(
                reduce_output_data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=reduce_output_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )

            num_cores = compute_grid.x * compute_grid.y
            reduce_available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
            reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_available_cores, 0) for _ in range(4)]

            moe_semaphores = MoeOp.create_semaphores(mesh_device)
            logger.info(f"[rank={my_mesh_id}] MoE setup complete")

        # ── Launch pipeline ──
        if not is_stage0:
            pipeline_block.run()
        logger.info(f"[rank={my_mesh_id}] pipeline launched")

        # ── MoE stages: submit persistent kernel ──
        if my_mesh_id >= 1:
            logger.info(f"[rank={my_mesh_id}] submitting persistent MoE kernel")
            MoeOp.op(
                r.ttnn_residual_mcast_src,
                gate_mm_weights_tensor=r.ttnn_gate_mm_weights,
                gate_bias_tensor=r.ttnn_gate_bias,
                gate_indices_tensor=r.ttnn_gate_indices,
                gate_output_scores_tensor=r.gate_output_scores_tensor,
                gate_output_indices_tensor=r.gate_output_indices_tensor,
                gate_proj_weights_tensor=r.gate_proj_weights,
                up_proj_weights_tensor=r.up_proj_weights,
                down_proj_weights_tensor=r.down_proj_weights,
                final_output_tensor=r.final_output_tensor,
                rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
                shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
                shared_up_weights_overlapped=s.shared_up_weights_overlapped,
                shared_down_weights_tensor=s.ttnn_down_weights,
                shared_k_parallel=s.k_parallel,
                shared_n_parallel=s.n_parallel,
                use_hardcoded_expert_index=True,
                sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
                sdpa_out_interm_buffer=sdpa_out_interm_buffer,
                num_iterations=1,
                persistent_mode=True,
                reduce_intermediate_tensors=reduce_intermediate_tensors,
                reduce_output_tensor=reduce_output_tensor,
                reduce_semaphores=reduce_semaphores,
                reduce_root_coord=reduce_root_coord,
                forward_sockets=forward_sockets,
                semaphores=moe_semaphores,
                worker_core_grid=moe_worker_core_grid,
                is_torus=is_torus,
                downstream_sockets=downstream_sockets,
            )
            logger.info(f"[rank={my_mesh_id}] persistent MoE kernel submitted")
        ttnn.distributed_context_barrier()

        # ── Stage 0: drive pipeline with multiple tokens in flight ──
        if is_stage0:
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            num_elements = embedding_size_bytes // 2
            torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            start_time = time.time()
            num_tokens_in_flight = 64
            for iteration in range(num_tokens_in_flight):
                for h2d in h2d_sockets:
                    h2d.write_tensor(token_tensor)
                logger.info(f"[rank=0] token {iteration} injected to {len(h2d_sockets)} channels")
            for iter_ in range(num_tokens_in_flight):
                d2h_output_tensor = ttnn.from_torch(
                    torch.zeros(1, num_elements, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                print(f"[rank={my_mesh_id}] iteration {iter_} waiting for D2H result")
                d2h_sockets[0].read_tensor(d2h_output_tensor)
                print(f"[rank={my_mesh_id}] iteration {iter_} D2H result read")
                d2h_result = ttnn.to_torch(d2h_output_tensor)

                d2h_nonzero = torch.count_nonzero(d2h_result)
                logger.info(
                    f"[rank={my_mesh_id}] iteration {iter_}: non-zero={d2h_nonzero}/{d2h_result.numel()}, "
                    f"first 5={d2h_result[0, :5]}"
                )
                assert d2h_nonzero > 0, f"D2H output is all zeros at iteration {iter_} — persistent MoE pipeline failed"
            end_time = time.time()
            print(f"[rank=0] time taken to move {num_tokens_in_flight} tokens: {end_time - start_time} seconds")

            logger.info(f"[rank={my_mesh_id}] all {num_tokens_in_flight} iterations passed")

        logger.info(f"[rank={my_mesh_id}] waiting for barrier")
        ttnn.distributed_context_barrier()
        logger.info(f"[rank={my_mesh_id}] barrier completed")

    finally:
        pass

    logger.info(f"[rank={my_mesh_id}] persistent multi-token MoE test PASSED")
