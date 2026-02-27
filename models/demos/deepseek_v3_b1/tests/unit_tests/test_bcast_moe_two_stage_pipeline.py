# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for socket-fed bcast + MoE fused op.

Stage 0:
  host token -> fused embedding (HostInterface) -> cross-stage D2D
Stage 1:
  entry D2D receiver -> moe sender core socket input -> bcast + fused MoE
  (reduce-to-one included)
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
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe import (
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
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
@pytest.mark.parametrize("token_id", [0, 31, 63])
def test_bcast_moe_two_stage_pipeline(mesh_device, vocab_size, embedding_dim, token_id):
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

    pipeline_config = ttnn._ttnn.operations.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert len(pipeline_config) == num_procs + 1

    is_stage0 = my_mesh_id == 0
    is_stage1 = my_mesh_id == 1

    M = 1
    K = embedding_dim
    # Hardcoded test/prod placement:
    # pipeline core is excluded from MoE worker grid; MoE sender/mcaster is at (12, 9).
    pipeline_core = ttnn.CoreCoord(12, 8)
    moe_sender_core = ttnn.CoreCoord(12, 9)
    moe_worker_core_grid = build_worker_grid_excluding_core(device_grid, pipeline_core)

    token_size_bytes = 64
    embedding_size_bytes = K * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    # Deterministic embedding table: row i = arange offset by i*K
    torch_embedding = torch.arange(vocab_size * K, dtype=torch.float32).reshape(1, 1, vocab_size, K).to(torch.bfloat16)

    # ── Pipeline block setup ──────────────────────────────────────────────────
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
        stage_entry_device = pipeline_config[my_mesh_id].entry_node_coord
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, moe_sender_core),
            # Detach stage-1 exit from the MoE socket chain
            exit_node_upstream=ttnn.MeshCoreCoord(stage_entry_device, pipeline_core),
        )
    else:
        # Passive forwarding for rank >= 2
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
        )

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    # ── Stage 1: MoE tensor setup ─────────────────────────────────────────────
    result_scores = None
    result_indices = None
    result_output = None

    if is_stage1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        r = create_routed_expert_tensors(
            mesh_device,
            use_hardcoded_expert_index=True,
            mesh_mapper=mesh_mapper,
            input_core=moe_sender_core,
            mcast_output_core_grid=moe_worker_core_grid,
        )
        mcast_grid = r["ttnn_mcast_output"].memory_config().shard_spec.grid
        gate_proj_grid = r["gate_proj_output"].memory_config().shard_spec.grid
        gate_proj_cores = ttnn.corerange_to_cores(gate_proj_grid, row_wise=True)
        worker_cores = {(c.x, c.y) for c in ttnn.corerange_to_cores(moe_worker_core_grid, row_wise=True)}
        assert all(
            (c.x, c.y) in worker_cores for c in gate_proj_cores
        ), "gate_proj worker cores must stay inside MoE worker_core_grid"
        s = create_shared_expert_tensors(mesh_device, M, K, mcast_grid, mesh_mapper=mesh_mapper)

        # Bcast tensors: bcast_input_tensor backs bcast_pkt_cb (CB 46).
        # Socket writes the received embedding directly here each iteration.
        # bcast_intermediate_tensor is the broadcast destination (CB 25 backing).
        tile_1x32 = ttnn.Tile([1, 32])
        input_core_grid = r["ttnn_residual_mcast_src"].memory_config().shard_spec.grid
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
        bcast_intermediate_tensor = r["ttnn_residual_mcast_src"]

        # Global semaphores for bcast sync (3) and reduce sync (4)
        num_cores = device_grid.x * device_grid.y
        available_cores = ttnn.num_cores_to_corerangeset(num_cores, device_grid, row_wise=True)
        ttnn.synchronize_device(mesh_device)
        bcast_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(3)]
        ttnn.synchronize_device(mesh_device)

        # ReduceToOne tensors — same setup as test_moe_fused_with_bcast
        root_coord = (1, 1)
        final_output_total_width = r["final_output_total_width"]
        final_output_mem_config = r["final_output_mem_config"]
        reduce_mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape
        )
        reduce_mesh_mapper = ttnn.create_mesh_mapper(mesh_device, reduce_mesh_mapper_config)

        reduce_intermediate_tensors = []
        for _ in range(3):
            reduce_intermediate_tensors.append(
                ttnn.from_torch(
                    torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=final_output_mem_config,
                    tile=tile_1x32,
                    mesh_mapper=reduce_mesh_mapper,
                )
            )

        reduce_output_core = moe_sender_core
        reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(reduce_output_core, reduce_output_core)})
        reduce_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(reduce_output_shard_grid, (1, final_output_total_width), ttnn.ShardOrientation.ROW_MAJOR),
        )
        reduce_output_tensor = ttnn.from_torch(
            torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=reduce_output_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )

        ttnn.synchronize_device(mesh_device)
        reduce_semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(4)]
        ttnn.synchronize_device(mesh_device)

        # recv_socket: the downstream socket of the entry node, delivering the embedding
        # to moe_sender_core on this device. bcast_sender_coord must be the sender *device* coord.
        recv_socket = pipeline_block.get_downstream_socket()
        sender_coord = recv_socket.get_active_cores()[0].device_coord
        bcast_sender_coord = stage_entry_device

    # ── Launch pipeline programs ──────────────────────────────────────────────
    pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    # Ensure all stages are launched before injecting the token
    ttnn.distributed_context_barrier()

    if is_stage0:
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pipeline_block.write_token(token_tensor)
        logger.info(f"[rank=0] token {token_id} injected")

    # Ensure token is injected before stage-1 enters potentially blocking op launch
    ttnn.distributed_context_barrier()

    if is_stage1:
        logger.info("[rank=1] launching MoE bcast + socket (num_iterations=1)")
        result_scores, result_indices, result_output = MoeOp.op(
            r["ttnn_rmsnorm_output"],
            r["ttnn_mcast_output"],
            r["ttnn_gate_mm_weights"],
            r["ttnn_gate_mm_output"],
            r["ttnn_gate_input"],
            r["ttnn_gate_bias"],
            r["ttnn_gate_indices"],
            r["gate_output_scores_tensor"],
            r["gate_output_indices_tensor"],
            r["expert_index_tensor"],
            r["expert_scale_tensor"],
            r["gate_proj_weights"],
            r["gate_proj_output"],
            r["up_proj_weights"],
            r["up_proj_mm_out_tensor"],
            r["fused_output_tensor"],
            r["down_proj_gather_output_tensor"],
            r["down_proj_mcast_output_tensor"],
            r["down_proj_weights"],
            r["down_proj_output"],
            s["ttnn_output_mcast_dst"],
            r["final_output_tensor"],
            r["gate_proj_in1_buf_tensor"],
            r["down_proj_in1_buf_tensor"],
            r["mul_scalar_buf_tensor"],
            rmsnorm_gamma_tensor=r["ttnn_rmsnorm_gamma"],
            shared_residual_mcast_src_tensor=r["ttnn_residual_mcast_src"],
            shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
            shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
            shared_residual_mcast_dst_tensor=r["ttnn_residual_mcast_dst"],
            shared_down_mcast_dst_tensor=s["ttnn_down_mcast_dst"],
            shared_down_weights_tensor=s["ttnn_down_weights"],
            shared_output_tensor=s["ttnn_output"],
            shared_ag_gather_dst_tensor=s["ttnn_ag_gather_dst"],
            shared_bg_gather_dst_tensor=s["ttnn_bg_gather_dst"],
            shared_gu_out_tensor=s["ttnn_gu_out"],
            shared_intermed_tensor=s["ttnn_intermed"],
            shared_down_mcast_src_tensor=s["ttnn_down_mcast_src"],
            shared_down_matmul_out_tensor=s["ttnn_down_matmul_out"],
            shared_residual_add_out_tensor=s["ttnn_residual_add_out"],
            shared_k_parallel=s["k_parallel"],
            shared_n_parallel=s["n_parallel"],
            use_hardcoded_expert_index=True,
            num_iterations=1,
            reduce_intermediate_tensors=reduce_intermediate_tensors,
            reduce_output_tensor=None,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=ttnn.MeshCoordinate(root_coord),
            bcast_input_tensor=bcast_input_tensor,
            bcast_intermediate_tensor=bcast_intermediate_tensor,
            bcast_semaphores=bcast_semaphores,
            bcast_sender_coord=bcast_sender_coord,
            socket=recv_socket,
            worker_core_grid=moe_worker_core_grid,
        )
        logger.info("[rank=1] MoE completed")

    ttnn.distributed_context_barrier()

    pipeline_block.terminate()
    logger.info(f"[rank={my_mesh_id}] programs terminated")

    # ── Stage 1: validate ─────────────────────────────────────────────────────
    if is_stage1:
        mesh_rows, mesh_cols = mesh_device.shape
        num_devices = mesh_rows * mesh_cols
        K_down = s["K_down"]

        # 1. Grab the exact row from the embedding table for this token
        torch_input_row = torch_embedding[0, 0, token_id : token_id + 1, :]

        # 2. Reconstruct expected output per device
        device_gate_indices = ttnn.to_torch(result_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        device_gate_scores = ttnn.to_torch(result_scores, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        result_output_torch = ttnn.to_torch(result_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

        # We validate the output of the rank 1 device itself (the stage entry device)
        device_idx = stage_entry_device[0] * mesh_cols + stage_entry_device[1]

        # 1. Grab the exact row from the embedding table for this token
        torch_input_row = torch_embedding[0, 0, token_id : token_id + 1, :]

        # 3. Get local MoE properties
        actual_expert_idx = device_idx
        actual_expert_scale = device_gate_scores[0].flatten()[device_idx].float()

        shared_gate_shard = s["torch_gate_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_up_shard = s["torch_up_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_down_shard = s["torch_down_weights"][device_idx * K_down : (device_idx + 1) * K_down, :]

        # 4. Compute Golden MoE for this device only
        _, _, expected_final = MoeOp.golden(
            torch_input_row,
            r["torch_gate_mm_weights"],
            r["torch_bias"],
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights_dict=r["expert_weights_dict"],
            up_proj_weights_dict=r["up_proj_weights_dict"],
            down_proj_weights_dict=r["down_proj_weights_dict"],
            eps=r["gate_eps"],
            scaling_factor=r["gate_scaling_factor"],
            use_hardcoded_expert_index=True,
            hardcoded_expert_index=actual_expert_idx,
            explicit_expert_scale=actual_expert_scale,
            rmsnorm_gamma=r["torch_rmsnorm_gamma"],
            rmsnorm_epsilon=1e-6,
        )

        # 5. Extract valid data (strip padding) from the raw result tensor
        actual_final = result_output_torch[device_idx]
        from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe import extract_routed_expert_output

        actual_valid = extract_routed_expert_output(
            actual_final.unsqueeze(0),
            r["num_gate_proj_cores"],
            r["final_output_width_per_core"],
            r["per_core_down_proj_N"],
        )

        # 6. PCC Comparison
        from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe import comp_pcc

        passing, pcc_msg = comp_pcc(expected_final.flatten(), actual_valid.flatten(), 0.97)
        logger.info(f"Pipeline Stage 1 Device {device_idx} PCC: {pcc_msg}")
        assert passing, f"Pipeline Stage 1 Device {device_idx} failed PCC: {pcc_msg}"

        logger.info("bcast + MoE two-stage pipeline test passed (non-reduced output)!")
