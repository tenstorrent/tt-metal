# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage pipeline integration test for socket-fed BroadcastRMSNorm.
Stage 0:
  host token -> fused embedding (HostInterface) -> cross-stage D2D
Stage 1:
  entry D2D receiver -> bcast core socket input -> fused BroadcastRMSNorm
Stage 2 onward (if applicable):
  passive for now, but can be used for downstream ops with exit D2D forwarding
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_slow_dispatch
from models.demos.deepseek_v3_b1.fused_ops.broadcast_rms.op import BroadcastRMSNorm
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


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
@pytest.mark.parametrize("epsilon", [1e-6])
def test_broadcast_rms_two_stage_pipeline(mesh_device, vocab_size, embedding_dim, token_id, epsilon, device_params):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs < 2:
        pytest.skip(f"Requires at least 2 distributed processes, got {num_procs}")

    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert len(pipeline_config) == num_procs + 1
    assert 0 <= token_id < vocab_size

    is_stage0 = my_mesh_id == 0
    is_stage1 = my_mesh_id == 1

    pipeline_core = ttnn.CoreCoord(0, 1)
    bcast_core = ttnn.CoreCoord(0, 0)
    token_size_bytes = 64
    output_shape = [1, embedding_dim]
    embedding_size_bytes = embedding_dim * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2

    # Deterministic embedding for token-to-row mapping; random gamma like existing RMS tests.
    torch_embedding = (
        torch.arange(vocab_size * embedding_dim, dtype=torch.float32)
        .reshape(1, 1, vocab_size, embedding_dim)
        .to(torch.bfloat16)
    )
    torch_gamma = torch.randn(tuple(output_shape), dtype=torch.bfloat16)
    expected_input = torch_embedding[0, 0, token_id, :].reshape(tuple(output_shape))
    expected_output = BroadcastRMSNorm.golden(expected_input, torch_gamma, epsilon=epsilon)

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
            entry_node_downstream=ttnn.MeshCoreCoord(stage_entry_device, bcast_core),
            # Detach stage-1 exit path from the bcast input socket chain. This keeps the
            # stage-0->stage-1 input path clean while allowing later stages to be passive.
            exit_node_upstream=ttnn.MeshCoreCoord(stage_entry_device, pipeline_core),
        )
    else:
        # Passive forwarding stages for rank >=2 when running on larger clusters.
        pipeline_block = PipelineBlock(
            mesh_device,
            pipeline_core,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
        )

    logger.info(f"[rank={my_mesh_id}] pipeline block created")

    result = None
    sender_coord = None
    input_tensor_mesh = None
    intermediate_tensor_mesh = None
    gamma_tensor = None
    output_tensor = None
    semaphores = None
    recv_socket = None
    if is_stage1:
        input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(bcast_core, bcast_core)})
        input_shard_spec = ttnn.ShardSpec(input_shard_grid, tuple(output_shape), ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=input_shard_spec,
        )

        input_tensor_mesh = ttnn.from_torch(
            torch.zeros(tuple(output_shape), dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((1, 32)),
            dtype=ttnn.bfloat16,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        intermediate_tensor_mesh = ttnn.from_torch(
            torch.zeros(tuple(output_shape), dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((1, 32)),
            dtype=ttnn.bfloat16,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        gamma_tensor = ttnn.from_torch(
            torch_gamma,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((1, 32)),
            dtype=ttnn.bfloat16,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        output_tensor = ttnn.from_torch(
            torch.zeros(tuple(output_shape), dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((1, 32)),
            dtype=ttnn.bfloat16,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        num_cores = compute_grid_size.x * compute_grid_size.y
        available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
        out_ready_semaphore = ttnn.create_global_semaphore(mesh_device, available_cores, 0)
        barrier_semaphore = ttnn.create_global_semaphore(mesh_device, available_cores, 0)
        secondary_sync_semaphore = ttnn.create_global_semaphore(mesh_device, available_cores, 0)
        semaphores = [out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore]

        recv_socket = pipeline_block.get_downstream_socket()
        sender_coord = pipeline_config[my_mesh_id].entry_node_coord

    pipeline_block.run()
    logger.info(f"[rank={my_mesh_id}] pipeline programs launched")

    # Ensure all stages are launched and stage-1 op inputs are fully prepared.
    # TODO: confirm if this is actually needed
    ttnn.distributed_context_barrier()

    if is_stage0:
        token_size_datums = token_size_bytes // dtype_size(ttnn.uint32)
        torch_token = torch.zeros(1, token_size_datums, dtype=torch.uint32)
        torch_token[0, 0] = token_id
        token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pipeline_block.write_token(token_tensor)

    # Ensure token is injected before rank-1 enters potentially blocking op launch.
    ttnn.distributed_context_barrier()

    if is_stage1:
        logger.info("[rank=1] launching BroadcastRMSNorm")
        result = BroadcastRMSNorm.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            gamma_tensor,
            sender_coord,
            output_tensor,
            semaphores=semaphores,
            cluster_axis=0,
            secondary_cluster_axis=1,
            epsilon=epsilon,
            skip_ccl=False,
            socket=recv_socket,
            is_torus=(device_params["fabric_config"] == ttnn.FabricConfig.FABRIC_2D_TORUS_Y),
        )
        logger.info("[rank=1] BroadcastRMSNorm completed")

    ttnn.distributed_context_barrier()
    pipeline_block.terminate()

    if is_stage1:
        mesh_rows, mesh_cols = mesh_device.shape
        slice_rows = output_shape[0]
        result_torch = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        for device_idx in range(mesh_rows * mesh_cols):
            start = device_idx * slice_rows
            end = start + slice_rows
            received = result_torch[start:end, :]
            max_diff = torch.max(torch.abs(received - expected_output)).item()
            mean_diff = torch.mean(torch.abs(received - expected_output)).item()
            passing, pcc_message = comp_pcc(expected_output, received, 0.999)
            assert passing, pcc_message

        logger.info(f"[rank=1] validation complete (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
