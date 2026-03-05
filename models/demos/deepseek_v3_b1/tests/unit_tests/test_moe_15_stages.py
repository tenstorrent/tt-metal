# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for socket-fed bcast + MoE fused op + reduce-to-one.

Stage 0:
  host token -> fused embedding (HostInterface) -> cross-stage D2D
  D2H loopback: read aggregated reduce output from pipeline
Stage 1:
  entry D2D receiver -> moe sender core socket input -> bcast + fused MoE
  -> reduce-to-one -> D2D_0 aggregator -> pipeline exit
Stage 2+ (if applicable):
  passive forwarding, no downstream op
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    ROUTED_EXPERT_LAYER_IDX,
    RoutedExpert,
    create_routed_expert_tensors,
    create_shared_expert_tensors,
)


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def build_worker_grid_excluding_core(device_grid_size, excluded_core):
    max_x = device_grid_size.x - 1
    max_y = device_grid_size.y - 1
    ranges = []

    if excluded_core.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_x, excluded_core.y - 1)))
    if excluded_core.y < max_y:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, excluded_core.y + 1), ttnn.CoreCoord(max_x, max_y)))
    if excluded_core.x > 0:
        ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, excluded_core.y), ttnn.CoreCoord(excluded_core.x - 1, excluded_core.y))
        )
    if excluded_core.x < max_x:
        ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(excluded_core.x + 1, excluded_core.y), ttnn.CoreCoord(max_x, excluded_core.y))
        )

    return ttnn.CoreRangeSet(ranges)


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
@pytest.mark.timeout(1200)
def test_bcast_moe_two_stage_pipeline(
    mesh_device, vocab_size, embedding_dim, token_id, device_params, get_reference_model_state_dict
):
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
    pipeline_core = ttnn.CoreCoord(12, 8)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_core(device_grid, pipeline_core)

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    # The root_coord for reduce-to-one; for host 1 this is the exit node
    reduce_root_coord = ttnn.MeshCoordinate(0, 0)

    # ── Core setup for reduce aggregation ──
    stage_entry_device = None
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    reduce_payload_per_shard = embedding_size_bytes // num_gate_proj_cores
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    # Aggregator is the first worker core (shard_idx == 0)
    aggregator_core = shard_cores_list[0]

    if my_mesh_id >= 1:
        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord
        logger.info(f"[rank={my_mesh_id}] stage entry device: {stage_entry_device}")
        logger.info(f"[rank={my_mesh_id}] reduce aggregator core: {aggregator_core}")

    # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
    if is_stage0:
        embedding_tensor = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            h2d_socket_fifo_size=token_size_bytes * 2,
            d2h_socket_fifo_size=embedding_fifo_size,
            d2h_socket_page_size=embedding_size_bytes,
            embedding_tensor=embedding_tensor,
        )
    else:
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, moe_sender_core),
            exit_node_upstream=ttnn.MeshCoreCoord(reduce_root_coord, aggregator_core),
        )

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    # ── Get downstream socket for reduce aggregator to send to pipeline exit ──
    downstream_socket = None
    if my_mesh_id >= 1:
        downstream_socket = pipeline_block.exit_socket_interface.get_upstream_socket()
        logger.info(f"[rank={my_mesh_id}] downstream socket wired to pipeline exit")

    # ── MoE tensor setup (stage 0: golden validation, stage 1: MoE compute + validation) ──
    result_scores = None
    result_indices = None
    result_output = None
    r = None
    s = None

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
        # SDPA buffers for CB memory overlap
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
        sdpa_out_interm_shard_height = 40
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

        # ── Reduce-to-one tensors (follows test_moe_mlp.py pattern) ──────────
        reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
        )
        reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

        final_output_mem_config = r.final_output_mem_config
        reduce_intermediate_tensors = []
        for _ in range(3):
            intermediate_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
            intermediate_tensor = ttnn.from_torch(
                intermediate_data,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=final_output_mem_config,
                tile=tile_1x32,
                mesh_mapper=reduce_mesh_mapper,
            )
            reduce_intermediate_tensors.append(intermediate_tensor)
        logger.info(f"[rank={my_mesh_id}] reduce intermediate tensors created")

        # Reduce output tensor (single-core sharded)
        compute_grid = mesh_device.compute_with_storage_grid_size()
        reduce_output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
        reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(reduce_output_core, reduce_output_core)})
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

        # 4 global semaphores for reduce synchronization
        num_cores = compute_grid.x * compute_grid.y
        reduce_available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, reduce_available_cores, 0) for _ in range(4)]
        logger.info(f"[rank={my_mesh_id}] reduce semaphores created")

        # ── Bcast + MoE semaphores ───────────────────────────────────────────
        available_cores = moe_worker_core_grid
        bcast_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(3)]
        moe_semaphores = MoeOp.create_semaphores(mesh_device)
        logger.info(f"[rank={my_mesh_id}] bcast + moe semaphores created")

        # Bcast tensors
        input_core_grid = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid
        bcast_shard_spec = ttnn.ShardSpec(input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
        bcast_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, bcast_shard_spec
        )
        bcast_input_tensor = ttnn.from_torch(
            torch.zeros((M, K), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=bcast_mem_config,
            tile=tile_1x32,
            mesh_mapper=mesh_mapper,
        )
        bcast_intermediate_tensor = r.ttnn_residual_mcast_src

        # recv_socket: the downstream socket of the entry node
        recv_socket = pipeline_block.get_downstream_socket()
        sender_coord = recv_socket.get_active_cores()[0].device_coord
        bcast_sender_coord = stage_entry_device
        logger.info(
            f"[rank={my_mesh_id}] recv_socket created with sender_coord {sender_coord} "
            f"and bcast_sender_coord {bcast_sender_coord}"
        )

    # ── Launch pipeline programs ──────────────────────────────────────────────
    pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    ttnn.distributed_context_barrier()

    if is_stage0:
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pipeline_block.write_token(token_tensor)
        logger.info(f"[rank=0] token {token_id} injected")

    ttnn.distributed_context_barrier()

    if my_mesh_id >= 1:
        logger.info(f"[rank={my_mesh_id}] launching MoE bcast + reduce (num_iterations=1)")
        stage_downstream_socket = downstream_socket
        result_scores, result_indices, result_output = MoeOp.op(
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
            bcast_input_tensor=bcast_input_tensor,
            bcast_intermediate_tensor=bcast_intermediate_tensor,
            bcast_semaphores=bcast_semaphores,
            bcast_sender_coord=bcast_sender_coord,
            socket=recv_socket,
            semaphores=moe_semaphores,
            worker_core_grid=moe_worker_core_grid,
            is_torus=is_torus,
            downstream_socket=stage_downstream_socket,
        )
        logger.info(f"[rank={my_mesh_id}] MoE + reduce completed")

    # ── Stage 0: D2H loopback read + golden validation ───────────────────────
    if is_stage0:
        logger.info("[rank=0] waiting for D2H result from pipeline loopback")
        num_elements = embedding_size_bytes // 2
        received_tensor_torch = torch.zeros(1, num_elements, dtype=torch.bfloat16)
        d2h_output_tensor = ttnn.from_torch(received_tensor_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        pipeline_block.read_output(d2h_output_tensor)
        d2h_result_torch = ttnn.to_torch(d2h_output_tensor)
        logger.info(f"[rank=0] D2H read complete, shape={d2h_result_torch.shape}")
        logger.info(f"[rank=0] D2H first 5 values: {d2h_result_torch[0, :5]}")

        d2h_nonzero = torch.count_nonzero(d2h_result_torch)
        logger.info(f"[rank=0] D2H non-zero elements: {d2h_nonzero}/{d2h_result_torch.numel()}")
        assert d2h_nonzero > 0, "D2H output is all zeros — reduce or D2D0 pipeline failed"

    ttnn.distributed_context_barrier()

    # ── Pipeline teardown ───────────────────────────────────────────────────
    logger.info(f"[rank={my_mesh_id}] waiting for pipeline block termination")
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
@pytest.mark.parametrize("iterations", [1024])
@pytest.mark.timeout(1200)
def test_persistent_moe_15_stages(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent-mode 15-stage MoE pipeline test.

    Pipeline topology:
      Stage 0  : H2D embedding → downstream D2D, D2H loopback ← pipeline
      Stage 1-14 (15 MoE stages): socket bcast → fused MoE → reduce-to-one → pipeline exit
      Stage 15 : passive forwarding back to stage 0

    The MoE kernel on stages 1-14 runs in a while(true) loop.  Stage 0 drives
    the pipeline by writing *iterations* tokens and reading back each D2H result.
    Validates every result is non-zero.
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
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_core(device_grid, pipeline_core)

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    torch.manual_seed(42)
    torch_embedding = torch.randn(1, 1, 1, K, dtype=torch.bfloat16)

    reduce_root_coord = ttnn.MeshCoordinate(0, 0)

    stage_entry_device = None
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    if is_moe_stage:
        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        reduce_root_coord = pipeline_config[my_mesh_id].exit_node_coord
        logger.info(f"[rank={my_mesh_id}] stage entry device: {stage_entry_device}")
        logger.info(f"[rank={my_mesh_id}] reduce aggregator core: {aggregator_core}")

    # ── Pipeline block setup (collective) ──
    pipeline_block = None
    try:
        # ── Pipeline block setup (collective — all hosts must participate simultaneously) ────
        if is_stage0:
            embedding_tensor = ttnn.from_torch(
                torch_embedding,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                h2d_socket_fifo_size=token_size_bytes * 2,
                d2h_socket_fifo_size=embedding_fifo_size,
                d2h_socket_page_size=embedding_size_bytes,
                embedding_tensor=embedding_tensor,
            )
        else:
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
                entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, moe_sender_core),
                exit_node_upstream=ttnn.MeshCoreCoord(reduce_root_coord, aggregator_core),
            )

        logger.info(f"[rank={my_mesh_id}] pipeline block created")

        downstream_socket = None
        if my_mesh_id >= 1:
            downstream_socket = pipeline_block.exit_socket_interface.get_upstream_socket()

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
            sdpa_out_interm_shard_height = 40
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
            reduce_intermediate_tensors = []
            for _ in range(3):
                intermediate_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
                intermediate_tensor = ttnn.from_torch(
                    intermediate_data,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=final_output_mem_config,
                    tile=tile_1x32,
                    mesh_mapper=reduce_mesh_mapper,
                )
                reduce_intermediate_tensors.append(intermediate_tensor)

            compute_grid = mesh_device.compute_with_storage_grid_size()
            reduce_output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
            reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(reduce_output_core, reduce_output_core)})
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

            available_cores = moe_worker_core_grid
            bcast_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(3)]
            moe_semaphores = MoeOp.create_semaphores(mesh_device)

            device_grid_size = mesh_device.compute_with_storage_grid_size()
            worker_crs = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
            )

            persistent_next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, worker_crs, 1)

            input_core_grid = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid
            bcast_shard_spec = ttnn.ShardSpec(input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
            bcast_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, bcast_shard_spec
            )
            bcast_input_tensor = ttnn.from_torch(
                torch.zeros((M, K), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=bcast_mem_config,
                tile=tile_1x32,
                mesh_mapper=mesh_mapper,
            )
            bcast_intermediate_tensor = r.ttnn_residual_mcast_src

            recv_socket = pipeline_block.get_downstream_socket()
            bcast_sender_coord = stage_entry_device
            logger.info(f"[rank={my_mesh_id}] MoE setup complete")

        # ── Launch pipeline ──
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
                persistent_next_iter_semaphore=persistent_next_iter_semaphore,
                reduce_intermediate_tensors=reduce_intermediate_tensors,
                reduce_output_tensor=reduce_output_tensor,
                reduce_semaphores=reduce_semaphores,
                reduce_root_coord=reduce_root_coord,
                bcast_input_tensor=bcast_input_tensor,
                bcast_intermediate_tensor=bcast_intermediate_tensor,
                bcast_semaphores=bcast_semaphores,
                bcast_sender_coord=bcast_sender_coord,
                socket=recv_socket,
                semaphores=moe_semaphores,
                worker_core_grid=moe_worker_core_grid,
                is_torus=is_torus,
                downstream_socket=downstream_socket,
            )
            logger.info(f"[rank={my_mesh_id}] persistent MoE kernel submitted")
        ttnn.distributed_context_barrier()

        # ── Stage 0: drive pipeline with multiple tokens ──
        if is_stage0:
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            num_elements = embedding_size_bytes // 2
            for iteration in range(iterations):
                print(f"[rank={my_mesh_id}] iteration {iteration} start")
                torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
                torch_token[0, 0] = 0
                token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                pipeline_block.write_token(token_tensor)
                print(f"[rank={my_mesh_id}] iteration {iteration} token written")
                print(f"[rank={my_mesh_id}] iteration {iteration} waiting for D2H result")
                d2h_output_tensor = ttnn.from_torch(
                    torch.zeros(1, num_elements, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                pipeline_block.read_output(d2h_output_tensor)
                d2h_result = ttnn.to_torch(d2h_output_tensor)

                d2h_nonzero = torch.count_nonzero(d2h_result)
                logger.info(
                    f"[rank=0] iteration {iteration}: non-zero={d2h_nonzero}/{d2h_result.numel()}, "
                    f"first 5={d2h_result[0, :5]}"
                )
                assert (
                    d2h_nonzero > 0
                ), f"D2H output is all zeros at iteration {iteration} — persistent MoE 15-stage pipeline failed"
                ttnn.distributed_context_barrier()

            logger.info(f"[rank=0] all {iterations} iterations passed")

        logger.info(f"[rank={my_mesh_id}] waiting for barrier")
        ttnn.distributed_context_barrier()
        logger.info(f"[rank={my_mesh_id}] barrier completed")

    finally:
        pass

    logger.info(f"[rank={my_mesh_id}] persistent 15-stage MoE test PASSED")
