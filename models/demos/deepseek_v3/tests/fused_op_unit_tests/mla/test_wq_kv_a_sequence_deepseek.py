# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_wq_kv_a_sequence_with_trace(
    mesh_device,
    input_tensor_mesh,
    weight_tensor,
    ccl_semaphore_handles,
    output_mem_config,
    num_iter=100,
    warmup_iters=10,
    subdevice_id=None,
    profiler=BenchmarkProfiler(),
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    bsz=32,
):
    """Run wq_kv_a sequence with trace mode for performance measurement."""
    # Compile Run
    logger.info("Compiling wq_kv_a sequence")

    # Linear projection
    tt_q_kv = ttnn.linear(
        input_tensor_mesh,
        weight_tensor,
        bias=None,
        memory_config=output_mem_config,
    )

    # All-gather
    tt_q_kv = ttnn.experimental.all_gather_async(
        tt_q_kv,
        dim=1,
        multi_device_global_semaphore=[ccl_semaphore_handles[0], ccl_semaphore_handles[1]],
        num_links=1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        subdevice_id=subdevice_id,
    )
    reduce_config = {
        "dims": [1],
        "output": None,
        "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        "memory_config": ttnn.L1_MEMORY_CONFIG,
    }

    # Fast reduce
    tt_q_kv = ttnn.experimental.fast_reduce_nc(tt_q_kv, **reduce_config)

    # Slice operations
    tt_q = ttnn.slice(
        tt_q_kv,
        [0, 0, 0, 0],
        [1, 1, bsz, q_lora_rank],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_kv_nope = ttnn.slice(
        tt_q_kv,
        [0, 0, 0, q_lora_rank],
        [1, 1, bsz, q_lora_rank + kv_lora_rank],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_kv_rope = ttnn.slice(
        tt_q_kv,
        [0, 0, 0, q_lora_rank + kv_lora_rank],
        [1, 1, bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(warmup_iters):
        # Linear projection
        tt_q_kv = ttnn.linear(
            input_tensor_mesh,
            weight_tensor,
            bias=None,
            memory_config=output_mem_config,
        )

        # All-gather
        tt_q_kv = ttnn.experimental.all_gather_async(
            tt_q_kv,
            dim=1,
            multi_device_global_semaphore=[ccl_semaphore_handles[2 * i], ccl_semaphore_handles[2 * i + 1]],
            num_links=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            subdevice_id=subdevice_id,
        )

        # Fast reduce
        tt_q_kv = ttnn.experimental.fast_reduce_nc(tt_q_kv, **reduce_config)

        # Slice operations
        tt_q = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, 0],
            [1, 1, bsz, q_lora_rank],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_kv_nope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank],
            [1, 1, bsz, q_lora_rank + kv_lora_rank],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_kv_rope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank + kv_lora_rank],
            [1, 1, bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        tt_q_kv.deallocate(True)
        tt_q.deallocate(True)
        tt_kv_nope.deallocate(True)
        tt_kv_rope.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iter} iterations")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        # Linear projection
        tt_q_kv = ttnn.linear(
            input_tensor_mesh,
            weight_tensor,
            bias=None,
            memory_config=output_mem_config,
        )

        # All-gather
        tt_q_kv = ttnn.experimental.all_gather_async(
            tt_q_kv,
            dim=1,
            multi_device_global_semaphore=[ccl_semaphore_handles[2 * i], ccl_semaphore_handles[2 * i + 1]],
            num_links=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            subdevice_id=subdevice_id,
        )

        # Fast reduce
        tt_q_kv = ttnn.experimental.fast_reduce_nc(tt_q_kv, **reduce_config)

        # Slice operations
        tt_q = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, 0],
            [1, 1, bsz, q_lora_rank],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_kv_nope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank],
            [1, 1, bsz, q_lora_rank + kv_lora_rank],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_kv_rope = ttnn.slice(
            tt_q_kv,
            [0, 0, 0, q_lora_rank + kv_lora_rank],
            [1, 1, bsz, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        if i != num_iter - 1:
            tt_q_kv.deallocate(True)
            tt_q.deallocate(True)
            tt_kv_nope.deallocate(True)
            tt_kv_rope.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler.start("wq-kv-a-sequence-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    profiler.end("wq-kv-a-sequence-warmup")

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("wq-kv-a-sequence")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    profiler.end("wq-kv-a-sequence")
    signpost("stop")

    return tt_q, tt_kv_nope, tt_kv_rope


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, weight_shape, q_output_shape, kv_nope_output_shape, kv_rope_output_shape",
    [
        (
            "wq_kv_a_sequence",
            [1, 1, 32, 896],  # Input: [1, 1, bsz, hidden_size]
            [896, 2112],  # Weight: [hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
            [1, 1, 32, 1536],  # tt_q output: [1, 1, bsz, q_lora_rank]
            [1, 1, 32, 512],  # tt_kv_nope output: [1, 1, bsz, kv_lora_rank]
            [1, 1, 32, 64],  # tt_kv_rope output: [1, 1, bsz, qk_rope_head_dim]
        ),
    ],
    ids=["wq_kv_a_sequence"],
)
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 100)])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 2154500,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_wq_kv_a_sequence_trace_mode(
    mesh_device,
    op_name,
    batch_size,
    input_shape,
    weight_shape,
    q_output_shape,
    kv_nope_output_shape,
    kv_rope_output_shape,
    trace_mode,
    warmup_iters,
    num_iters,
    function_level_defaults,
):
    """
    Test the complete _fwd_decode_wq_kv_a sequence from mla1d.py (lines 1092-1134).

    This test captures the entire operation sequence:
    1. Linear projection: [1, 1, 32, 896] x [896, 2112] -> [1, 1, 32, 2112]
    2. All-gather: [1, 1, 32, 2112] -> [1, 8, 32, 2112]
    3. Fast reduce: [1, 8, 32, 2112] -> [1, 1, 32, 2112]
    4. Slice q: [1, 1, 32, 2112] -> [1, 1, 32, 1536]
    5. Slice kv_nope: [1, 1, 32, 2112] -> [1, 1, 32, 512]
    6. Slice kv_rope: [1, 1, 32, 2112] -> [1, 1, 32, 64]

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Input memory: WIDTH_SHARDED 7x4 grid, shard [32, 32]
    - Output memory: WIDTH_SHARDED (linear output)
    - CCL topology: Linear
    """
    torch.manual_seed(0)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    # Set up sub-devices and semaphores for async operation
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create global semaphore handles (need 2 per iteration)
    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters * 2)
    ]

    # Create input and weight tensors
    logger.info(f"Running wq_kv_a sequence test: {op_name}")
    logger.info(f"Running on {mesh_device.get_num_devices()} devices")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Weight shape: {weight_shape}")
    logger.info(f"Q output shape: {q_output_shape}")
    logger.info(f"KV nope output shape: {kv_nope_output_shape}")
    logger.info(f"KV rope output shape: {kv_rope_output_shape}")

    # Create golden reference tensors
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)

    # Golden reference: linear + reduce + slices
    torch_linear_out = torch_input @ torch_weight  # [1, 1, 32, 2112]
    # After all-gather and reduce, we get back [1, 1, 32, 2112]
    torch_q = torch_linear_out[:, :, :, :1536]
    torch_kv_nope = torch_linear_out[:, :, :, 1536:2048]
    torch_kv_rope = torch_linear_out[:, :, :, 2048:2112]

    # Create WIDTH_SHARDED memory config for input (matching model line 1103)
    # 7x4 grid with shard shape [32, 32]
    input_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(6, 3),  # 7x4 grid
            )
        }
    )
    input_shard_shape = [32, 32]
    input_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input_shard_shape,
        core_grid=input_core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Convert input to ttnn with proper mesh mapping for 8 devices in linear topology
    # Use MeshShape(1, 8) to indicate 1 row of 8 devices (linear arrangement)
    input_tensor_mesh = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, 8)),
        ),
    )

    # Apply sharding to input
    input_tensor_mesh = ttnn.to_memory_config(input_tensor_mesh, input_sharded_mem_config)

    # Convert weight to ttnn (replicated across devices with proper mesh shape)
    weight_tensor = ttnn.from_torch(
        torch_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, 8)),
        ),
    )

    # Output memory config - WIDTH_SHARDED (matching linear output)
    output_mem_config = ttnn.L1_MEMORY_CONFIG

    profiler = BenchmarkProfiler()

    try:
        if trace_mode:
            # Run sequence with trace
            tt_q, tt_kv_nope, tt_kv_rope = run_wq_kv_a_sequence_with_trace(
                mesh_device,
                input_tensor_mesh,
                weight_tensor,
                ccl_semaphore_handles=ccl_semaphore_handles,
                output_mem_config=output_mem_config,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                subdevice_id=worker_sub_device_id,
                profiler=profiler,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                bsz=batch_size,
            )
        else:
            pytest.skip("Non-trace mode not implemented for this test")

        # Verify correctness for all three outputs
        logger.info("Verifying correctness")
        passed = True

        # Verify tt_q
        logger.info("Verifying tt_q output")
        for i, t in enumerate(ttnn.get_device_tensors(tt_q)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking tt_q for device {t.device().id()}")

            passed_q, output_q = assert_with_pcc(tt_output_tensor, torch_q, pcc=0.99)
            if not passed_q:
                logger.error(f"tt_q output mismatch for device {i}: {output_q}")
                passed = False

        # Verify tt_kv_nope
        logger.info("Verifying tt_kv_nope output")
        for i, t in enumerate(ttnn.get_device_tensors(tt_kv_nope)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking tt_kv_nope for device {t.device().id()}")

            passed_kv_nope, output_kv_nope = assert_with_pcc(tt_output_tensor, torch_kv_nope, pcc=0.99)
            if not passed_kv_nope:
                logger.error(f"tt_kv_nope output mismatch for device {i}: {output_kv_nope}")
                passed = False

        # Verify tt_kv_rope
        logger.info("Verifying tt_kv_rope output")
        for i, t in enumerate(ttnn.get_device_tensors(tt_kv_rope)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking tt_kv_rope for device {t.device().id()}")

            passed_kv_rope, output_kv_rope = assert_with_pcc(tt_output_tensor, torch_kv_rope, pcc=0.99)
            if not passed_kv_rope:
                logger.error(f"tt_kv_rope output mismatch for device {i}: {output_kv_rope}")
                passed = False

        assert passed, "Output verification failed for one or more tensors"

        logger.info("✓ wq_kv_a sequence trace mode test passed with correct outputs")

    finally:
        # Clean up sub-device configuration
        mesh_device.reset_sub_device_stall_group()
