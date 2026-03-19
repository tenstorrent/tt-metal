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
from models.common.utility_functions import comp_pcc, is_slow_dispatch
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe_mlp import (
    ROUTED_EXPERT_LAYER_IDX,
    RoutedExpert,
    create_routed_expert_tensors,
    create_shared_expert_tensors,
    extract_routed_expert_output,
)
from models.demos.deepseek_v3_b1.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment


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
def test_bcast_moe_reduce_pipeline(
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
    is_stage1 = my_mesh_id == 1

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
    reduce_root_coord = pipeline_config[1].exit_node_coord if num_procs >= 2 else ttnn.MeshCoordinate(0, 0)

    # ── Core setup for reduce aggregation ──
    stage_entry_device = None
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    num_gate_proj_cores = len(gate_proj_worker_cores)
    reduce_payload_per_shard = embedding_size_bytes // num_gate_proj_cores
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    # Aggregator is the first worker core (shard_idx == 0)
    aggregator_core = shard_cores_list[0]

    if is_stage1:
        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
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
    elif is_stage1:
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
    else:
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
        )

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    # ── Get downstream socket for reduce aggregator to send to pipeline exit ──
    downstream_socket = None
    if is_stage1:
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

    if is_stage0 or is_stage1:
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

    if is_stage1:
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
        # Single intermediate tensor with 3x shard width for all 3 reduction rounds
        orig_shard_spec = final_output_mem_config.shard_spec
        intermediate_shard_shape = [orig_shard_spec.shape[0], orig_shard_spec.shape[1] * 3]
        intermediate_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.types.BufferType.L1,
            ttnn.ShardSpec(
                orig_shard_spec.grid,
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

        # Reduce output tensor (single-core sharded)
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

    if is_stage1:
        logger.info(f"[rank=1] launching MoE bcast + reduce (num_iterations=1)")
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
            downstream_socket=downstream_socket,
        )
        logger.info("[rank=1] MoE + reduce completed")

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

        # Validate D2H output against golden expected MoE reduce output
        from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore

        mesh_rows, mesh_cols = mesh_device.shape
        K_down = s.K_down
        torch_input_row = torch_embedding[0, 0, token_id : token_id + 1, :]

        x_raw = torch_input_row.float()
        variance = x_raw.pow(2).mean(-1, keepdim=True)
        normalized_input = ((x_raw * torch.rsqrt(variance + 1e-6)) * r.torch_rmsnorm_gamma.float()).bfloat16().float()
        logits = normalized_input.float() @ r.torch_gate_mm_weights.float()
        scores = torch.sigmoid(logits)
        gate_input = scores.reshape(1, 8, 32)
        cpu_gate_scores, _ = DeepseekMoeGateSingleCore.golden(
            gate_input, r.torch_bias.float(), r.gate_eps, r.gate_scaling_factor, enable_sigmoid=False
        )

        expected_final_outputs = []
        for dev_idx in range(mesh_rows * mesh_cols):
            actual_expert_scale = cpu_gate_scores.flatten()[dev_idx].float()

            shared_gate_shard = s.torch_gate_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_up_shard = s.torch_up_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_down_shard = s.torch_down_weights[dev_idx * K_down : (dev_idx + 1) * K_down, :]

            _, _, expected_final = MoeOp.golden_single_device(
                torch_input_row,
                shared_gate_weights=shared_gate_shard,
                shared_up_weights=shared_up_shard,
                shared_down_weights=shared_down_shard,
                gate_proj_weights_dict=r.expert_weights_dict,
                up_proj_weights_dict=r.up_proj_weights_dict,
                down_proj_weights_dict=r.down_proj_weights_dict,
                routing_weights_tensor=r.torch_gate_mm_weights,
                bias_tensor=r.torch_bias,
                eps=r.gate_eps,
                scaling_factor=r.gate_scaling_factor,
                use_hardcoded_expert_index=True,
                hardcoded_expert_index=dev_idx,
                explicit_expert_scale=actual_expert_scale,
                rmsnorm_gamma=r.torch_rmsnorm_gamma,
                rmsnorm_epsilon=1e-6,
            )
            expected_final_outputs.append(expected_final)

        expected_reduce_output = sum(expected_final_outputs)

        reduce_shard_width = reduce_payload_per_shard // dtype_size(ttnn.bfloat16)
        d2h_valid = extract_routed_expert_output(
            d2h_result_torch, r.num_gate_proj_cores, reduce_shard_width, r.per_core_down_proj_N
        )

        passing, pcc_msg = comp_pcc(expected_reduce_output.flatten(), d2h_valid.flatten(), 0.9)
        logger.info(f"Pipeline Stage 0 D2H Reduce PCC: {pcc_msg}")
        assert passing, f"Pipeline Stage 0 D2H PCC check failed: {pcc_msg}"

    ttnn.distributed_context_barrier()

    # ── Pipeline teardown ───────────────────────────────────────────────────
    logger.info(f"[rank={my_mesh_id}] waiting for pipeline block termination")
    pipeline_block.terminate()
    logger.info(f"[rank={my_mesh_id}] programs terminated")

    # ── Stage 1: validate MoE golden ─────────────────────────────────────────
    if is_stage1:
        mesh_rows, mesh_cols = mesh_device.shape
        K_down = s.K_down

        torch_input_row = torch_embedding[0, 0, token_id : token_id + 1, :]

        device_gate_scores = ttnn.to_torch(result_scores, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

        # Validate reduce output (sum of all per-device MoE outputs)
        reduce_output_torch = ttnn.to_torch(
            result_output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        root_device_idx = reduce_root_coord[0] * mesh_cols + reduce_root_coord[1]

        expected_final_outputs = []
        for dev_idx in range(mesh_rows * mesh_cols):
            actual_expert_idx = dev_idx
            actual_expert_scale = device_gate_scores[0].flatten()[dev_idx].float()

            shared_gate_shard = s.torch_gate_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_up_shard = s.torch_up_weights[:, dev_idx * K_down : (dev_idx + 1) * K_down]
            shared_down_shard = s.torch_down_weights[dev_idx * K_down : (dev_idx + 1) * K_down, :]

            _, _, expected_final = MoeOp.golden_single_device(
                torch_input_row,
                shared_gate_weights=shared_gate_shard,
                shared_up_weights=shared_up_shard,
                shared_down_weights=shared_down_shard,
                gate_proj_weights_dict=r.expert_weights_dict,
                up_proj_weights_dict=r.up_proj_weights_dict,
                down_proj_weights_dict=r.down_proj_weights_dict,
                routing_weights_tensor=r.torch_gate_mm_weights,
                bias_tensor=r.torch_bias,
                eps=r.gate_eps,
                scaling_factor=r.gate_scaling_factor,
                use_hardcoded_expert_index=True,
                hardcoded_expert_index=actual_expert_idx,
                explicit_expert_scale=actual_expert_scale,
                rmsnorm_gamma=r.torch_rmsnorm_gamma,
                rmsnorm_epsilon=1e-6,
            )
            expected_final_outputs.append(expected_final)

        expected_reduce_output = sum(expected_final_outputs)

        reduce_output_root = reduce_output_torch[root_device_idx]
        reduce_output_valid = extract_routed_expert_output(
            reduce_output_root.unsqueeze(0),
            r.num_gate_proj_cores,
            r.final_output_width_per_core,
            r.per_core_down_proj_N,
        )

        passing, pcc_msg = comp_pcc(expected_reduce_output.flatten(), reduce_output_valid.flatten(), 0.97)
        logger.info(f"Pipeline Stage 1 Reduce PCC: {pcc_msg}")
        logger.info(f"[rank=1] expected first 5 values: {reduce_output_valid.flatten()[:5]}")
        assert passing, f"Pipeline Stage 1 Reduce PCC check failed: {pcc_msg}"

    logger.info(f"[rank={my_mesh_id}] test PASSED")


# ---------------------------------------------------------------------------
# Persistent-mode pipeline test
# ---------------------------------------------------------------------------


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
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.timeout(1200)
def test_persistent_mode_pipeline(
    mesh_device, embedding_dim, iterations, device_params, get_reference_model_state_dict
):
    """
    Persistent mode: MoE kernel runs in a while(true) loop on Stage 1.
    Stage 0 drives the pipeline by writing *iterations* tokens and reading
    back each D2H result.  Validates every result is non-zero.

    Pipeline topology (same as test_bcast_moe_reduce_pipeline):
      Stage 0  : H2D embedding → downstream D2D
      Stage 1  : socket bcast → fused MoE → reduce-to-one → D2D₀ → pipeline exit
      Stage 2+ : passive forwarding
      Stage 0  : D2H loopback ← pipeline
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
    is_stage1 = my_mesh_id == 1

    M = 1
    K = embedding_dim
    pipeline_core = ttnn.CoreCoord(12, 8)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_core(device_grid, pipeline_core)

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    # Embedding table: one row per iteration (like lm_head persistent test)
    torch.manual_seed(42)
    torch_embedding = torch.randn(iterations, 1, 1, K, dtype=torch.bfloat16)

    reduce_root_coord = pipeline_config[1].exit_node_coord if num_procs >= 2 else ttnn.MeshCoordinate(0, 0)

    # ── Core setup for reduce aggregation ──
    stage_entry_device = None
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(mesh_device, gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in gate_proj_worker_cores])
    shard_cores_list = ttnn.corerange_to_cores(gate_proj_core_ranges, row_wise=True)
    aggregator_core = shard_cores_list[0]

    if is_stage1:
        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        logger.info(f"[rank={my_mesh_id}] stage entry device: {stage_entry_device}")

    # ── Pipeline blocks (collective) ──────────────────────────────────────────
    pipeline_block = None
    try:
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
        elif is_stage1:
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
        else:
            pipeline_block = PipelineBlock(
                mesh_device,
                pipeline_core,
                upstream_d2d_socket_fifo_size=embedding_fifo_size,
                downstream_d2d_socket_fifo_size=embedding_fifo_size,
                upstream_d2d_socket_page_size=embedding_size_bytes,
                downstream_d2d_socket_page_size=embedding_size_bytes,
            )

        logger.info(f"[rank={my_mesh_id}] pipeline block created")

        # ── Downstream socket for reduce aggregator ──
        downstream_socket = None
        if is_stage1:
            downstream_socket = pipeline_block.exit_socket_interface.get_upstream_socket()

        # ── MoE tensors (Stage 1 only — no golden needed) ────────────────────
        state_dict = get_reference_model_state_dict(
            layer_idx=ROUTED_EXPERT_LAYER_IDX,
            is_moe=True,
            seed=RoutedExpert.SEED,
            num_routed_experts=256,
            include_global=False,
        )

        r = None
        s = None
        if is_stage1:
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

            # SDPA buffers
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

            # Reduce-to-one tensors
            reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
                [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
            )
            reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

            final_output_mem_config = r.final_output_mem_config
            # Single intermediate tensor with 3x shard width for all 3 reduction rounds
            orig_shard_spec = final_output_mem_config.shard_spec
            intermediate_shard_shape = [orig_shard_spec.shape[0], orig_shard_spec.shape[1] * 3]
            intermediate_mem_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.types.BufferType.L1,
                ttnn.ShardSpec(
                    orig_shard_spec.grid,
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

            available_cores = moe_worker_core_grid
            bcast_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(3)]
            moe_semaphores = MoeOp.create_semaphores(mesh_device)
            persistent_next_iter_semaphore = ttnn.create_global_semaphore(mesh_device, available_cores, 1)

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

        # ── Launch pipeline ───────────────────────────────────────────────────
        pipeline_block.run()
        logger.info(f"[rank={my_mesh_id}] pipeline launched")

        # ── Stage 1: submit persistent MoE kernel ────────────────────────────
        if is_stage1:
            logger.info(f"[rank={my_mesh_id}] submitting MoE persistent kernel")
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

        # ── Stage 0: drive pipeline with multiple tokens ──────────────────────
        if is_stage0:
            token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
            num_elements = embedding_size_bytes // 2

            for iteration in range(iterations):
                print(f"[rank={my_mesh_id}] iteration {iteration} start")
                torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
                torch_token[0, 0] = iteration
                token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

                d2h_output_tensor = ttnn.from_torch(
                    torch.zeros(1, num_elements, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                pipeline_block.write_token(token_tensor)
                print(f"[rank={my_mesh_id}] iteration {iteration} token written")
                print(f"[rank={my_mesh_id}] iteration {iteration} waiting for D2H result")
                pipeline_block.read_output(d2h_output_tensor)
                d2h_result = ttnn.to_torch(d2h_output_tensor)

                d2h_nonzero = torch.count_nonzero(d2h_result)
                logger.info(
                    f"[rank=0] iteration {iteration}: non-zero={d2h_nonzero}/{d2h_result.numel()}, "
                    f"first 5={d2h_result[0, :5]}"
                )
                assert (
                    d2h_nonzero > 0
                ), f"D2H output is all zeros at iteration {iteration} — persistent MoE pipeline failed"

            logger.info(f"[rank=0] all {iterations} iterations passed")

        logger.info(f"[rank={my_mesh_id}] waiting for barrier")
        ttnn.distributed_context_barrier()
        logger.info(f"[rank={my_mesh_id}] barrier completed")

    finally:
        # Persistent kernel runs forever — terminate is not safe here.
        # Device cleanup happens when mesh_device is destroyed by the fixture.
        pass

    logger.info(f"[rank={my_mesh_id}] persistent mode test PASSED")
