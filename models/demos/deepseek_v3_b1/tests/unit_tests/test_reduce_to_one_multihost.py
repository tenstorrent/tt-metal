# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Host Reduce-to-One B1 Test with PipelineBlock (4-Stage Pipeline).
This test validates reduce-to-one operation across 4 hosts using PipelineBlock:
  Host 0 (Stage 0): PipelineBlock with dummy H2D/D2H → Send dummy → Host 1
  Host 1 (Stage 1): Entry (dummy) + Reduce-to-One → D2D_0 → Exit → Host 2
  Host 2-15 (Stage 2-15): Entry → Forward → Exit → Next Host
  Host 0 (last stage): Entry (from last Host) → D2H → Validation
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size, ttnn_dtype_from_torch_dtype
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock
from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import ReduceToOneB1


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(7168),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [1, 7168],
    ],
)
@pytest.mark.parametrize("vocab_size, embedding_dim", [(64, 7168)])
def test_reduce_to_one_b1_multihost(mesh_device, tensor_shape, vocab_size, embedding_dim, device_params):
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    logger.info(f"mesh_device shape: {mesh_device.shape}")
    logger.info(f"mesh_device num_devices: {mesh_device.get_num_devices()}")

    # Get this host's identity in the pipeline
    my_mesh_id = mesh_device.get_system_mesh_id()
    num_procs = int(ttnn.distributed_context_get_size())

    logger.info(f"Running on Host {my_mesh_id} of {num_procs} total hosts")

    # Enable async dispatch
    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    # Get pipeline configuration
    pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
    assert (
        len(pipeline_config) == num_procs + 1
    ), f"Expected {num_procs + 1} pipeline stages, got {len(pipeline_config)}"

    # Log pipeline configuration for debugging
    for i, stage in enumerate(pipeline_config):
        logger.info(f"Pipeline Stage {i}: entry={stage.entry_node_coord}, exit={stage.exit_node_coord}")

    # Configuration
    element_size = 2  # bfloat16
    payload_size_bytes = 896 * element_size  # 1,792 bytes per worker
    aggregated_size_bytes = 8 * payload_size_bytes  # 14,336 bytes total (from 8 workers)

    # Pipeline parameters
    pipeline_core_coord = ttnn.CoreCoord(11, 9)

    # ============================================================
    # PHASE 1: All hosts instantiate their pipeline infrastructure
    # ============================================================
    logger.info(f"\n=== HOST {my_mesh_id}: Instantiating pipeline infrastructure ===")

    # Variables to hold state for each host
    pipeline_block = None
    data_per_device = None
    d2d0_infra = None
    input_tensor = None
    intermediate_tensors = None
    output_tensor = None
    semaphores = None

    embedding_size_bytes = embedding_dim * dtype_size(ttnn.bfloat16)
    embedding_fifo_size = embedding_size_bytes * 2
    token_size_bytes = 64

    root_coord = pipeline_config[my_mesh_id].exit_node_coord

    data_per_device = []
    torch.manual_seed(42)
    for device_idx in range(8):
        data = torch.randn(tensor_shape, dtype=torch.bfloat16)
        data_per_device.append(data)

    data_all = torch.stack(data_per_device, dim=0).reshape(4, 2, *tensor_shape)

    if my_mesh_id == 0:
        # HOST 0: Pipeline orchestrator with dummy H2D and validation D2H
        logger.info("Stage 0: Dummy H2D → Exit (dummy data) → Host 1")

        # Create dummy embedding tensor for PipelineBlock
        embedding_dtype = torch.bfloat16
        torch_embedding = (
            torch.arange(vocab_size * embedding_dim, dtype=torch.float32)
            .reshape(1, 1, vocab_size, embedding_dim)
            .to(torch.bfloat16)
        )
        embedding_tensor = ttnn.from_torch(
            torch_embedding, dtype=ttnn_dtype_from_torch_dtype(embedding_dtype), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        embedding_tensor = ttnn.to_device(embedding_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        pipeline_block = PipelineBlock(
            mesh_device=mesh_device,
            pipeline_core_coord=pipeline_core_coord,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            h2d_socket_fifo_size=token_size_bytes * 2,
            d2h_socket_fifo_size=embedding_fifo_size,
            d2h_socket_page_size=embedding_size_bytes,
            embedding_tensor=embedding_tensor,
        )
        logger.info("✓ Created PipelineBlock for Host 0 (orchestrator)")

    elif my_mesh_id == 1:
        # HOST 1: Reduce-to-one with D2D_0 aggregator
        logger.info("Compute: Reduce workers → D2D_0 aggregator")
        logger.info("Exit: Aggregated result → Host 2")

        # Validate mesh
        mesh_rows, mesh_cols = mesh_device.shape
        if mesh_rows * mesh_cols < 8:
            pytest.skip(f"Need at least 8 devices, got {mesh_rows * mesh_cols}")

        print("entry node coord:", pipeline_config[my_mesh_id].entry_node_coord)
        print("exit node coord:", pipeline_config[my_mesh_id].exit_node_coord)
        dtype = ttnn.bfloat16
        layout = ttnn.TILE_LAYOUT
        tile = ttnn.Tile((1, 32))

        # Get optimal cores for DRAM access
        compute_cores = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        num_shard_cores = 8
        shard_cores = compute_cores[:num_shard_cores]
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in shard_cores})

        shard_shape = [1, 896]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )

        mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape)
        mesh_mapper = ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config)

        # Create intermediate tensors
        intermediate_tensors = []
        for _ in range(3):
            intermediate_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
            intermediate_tensor = ttnn.from_torch(
                intermediate_data,
                device=mesh_device,
                layout=layout,
                tile=tile,
                dtype=dtype,
                memory_config=mem_config,
                mesh_mapper=mesh_mapper,
            )
            intermediate_tensors.append(intermediate_tensor)

        # Create output tensor
        compute_grid = mesh_device.compute_with_storage_grid_size()
        output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
        output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(output_core, output_core)})
        output_shard_spec = ttnn.ShardSpec(output_shard_grid, tensor_shape, ttnn.ShardOrientation.ROW_MAJOR)
        output_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, output_shard_spec
        )

        output_data = torch.zeros([4, 2] + tensor_shape, dtype=torch.bfloat16)
        output_tensor = ttnn.from_torch(
            output_data,
            device=mesh_device,
            layout=layout,
            tile=tile,
            dtype=dtype,
            memory_config=output_mem_config,
            mesh_mapper=mesh_mapper,
        )

        # Generate test data (same seed as Host 0)

        input_tensor = ttnn.from_torch(
            data_all,
            device=mesh_device,
            layout=layout,
            tile=tile,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

        # Create synchronization semaphores
        all_devices_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))])
        sem_round1 = ttnn.create_global_semaphore(mesh_device, all_devices_grid, 0)
        sem_round2 = ttnn.create_global_semaphore(mesh_device, all_devices_grid, 0)
        sem_round3 = ttnn.create_global_semaphore(mesh_device, all_devices_grid, 0)
        sem_exit = ttnn.create_global_semaphore(mesh_device, all_devices_grid, 0)
        semaphores = [sem_round1, sem_round2, sem_round3, sem_exit]  # Create D2D_0 infrastructure before PipelineBlock
        # IMPORTANT: Use same core ordering as op.py will use (row_wise=True)
        shard_cores_list = ttnn.corerange_to_cores(shard_grid, row_wise=True)
        d2d0_infra_temp = ReduceToOneB1.create_d2d0_infrastructure(
            mesh_device, root_coord, shard_cores_list, payload_size_bytes, None
        )

        d2d0_core = d2d0_infra_temp["d2d0_core"]
        logger.info(f"✓ Created D2D_0 infrastructure at core {d2d0_core}")
        dummy_drain_core = ttnn.CoreCoord(10, 9)  # Core to run dummy drain kernel on Host 1
        # Create PipelineBlock - connect D2D_0 core to exit socket
        pipeline_block = PipelineBlock(
            mesh_device=mesh_device,
            pipeline_core_coord=pipeline_core_coord,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
            entry_node_downstream=ttnn.MeshCoreCoord(pipeline_config[my_mesh_id].entry_node_coord, dummy_drain_core),
            exit_node_upstream=ttnn.MeshCoreCoord(root_coord, d2d0_core),
        )

        logger.info("✓ Created PipelineBlock with D2D_0→Exit connection")

        # Connect D2D_0 downstream socket to exit socket's upstream
        d2d0_downstream_socket = pipeline_block.exit_socket_interface.get_upstream_socket()
        d2d0_infra_temp["d2d0_downstream_socket"] = d2d0_downstream_socket
        d2d0_infra = d2d0_infra_temp

    else:
        logger.info("Entry → Forward → Exit")

        pipeline_block = PipelineBlock(
            mesh_device=mesh_device,
            pipeline_core_coord=pipeline_core_coord,
            upstream_d2d_socket_fifo_size=embedding_fifo_size,
            downstream_d2d_socket_fifo_size=embedding_fifo_size,
            upstream_d2d_socket_page_size=embedding_size_bytes,
            downstream_d2d_socket_page_size=embedding_size_bytes,
        )
        logger.info("✓ Created PipelineBlock for Host 2")

    logger.info(f"\n=== HOST {my_mesh_id}: Running pipeline ===")
    pipeline_block.run()
    logger.info(f"✓ PipelineBlock started on Host {my_mesh_id}")

    if my_mesh_id == 1:
        logger.info("\n=== HOST 1: Running Reduce-to-One with D2D_0 ===")

        result = ReduceToOneB1.op(
            input_tensor,
            intermediate_tensors,
            output_tensor,
            semaphores,
            root_coord,
            enable_d2d0_output=True,
            d2d0_infrastructure=d2d0_infra,
            num_iterations=100,
            is_torus=(device_params["fabric_config"] == ttnn.FabricConfig.FABRIC_2D_TORUS_Y),
        )

        if isinstance(result, tuple):
            output_tensor, _ = result

        logger.info("✓ Reduce-to-one completed")

    if my_mesh_id == 0:
        logger.info("\n=== HOST 0: Waiting for result from D2H ===")

        num_elements = aggregated_size_bytes // 2
        received_tensor_torch = torch.zeros(1, num_elements, dtype=torch.bfloat16)
        d2h_output_tensor = ttnn.from_torch(received_tensor_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        pipeline_block.read_output(d2h_output_tensor)
        logger.info("✓ Successfully read from D2H")

        # Validate result
        d2h_result_torch = ttnn.to_torch(d2h_output_tensor)
        logger.info(f"D2H output (first 5): {d2h_result_torch[0, :5]}")

        expected_value = ReduceToOneB1.golden(data_per_device)
        logger.info(f"Expected value (first 5): {expected_value[0, :5]}")

        rtol = 0.01
        atol = 0.05
        assert torch.allclose(
            d2h_result_torch, expected_value, rtol=rtol, atol=atol
        ), f"D2H output mismatch! Expected {expected_value[0, 0]}, got {d2h_result_torch[0, 0]}"
        logger.info("✓ D2H output matches expected value")

    logger.info(f"\n=== HOST {my_mesh_id}: Terminating PipelineBlock ===")
    if my_mesh_id == 1:
        # Signal termination to D2D_0 aggregator
        termination_semaphore = d2d0_infra["d2d0_termination_semaphore"]
        ttnn.reset_global_semaphore_value(termination_semaphore, 1)
        logger.info("✓ D2D_0 termination signaled")
    pipeline_block.terminate()
    logger.info(f"✓ PipelineBlock terminated on Host {my_mesh_id}")
