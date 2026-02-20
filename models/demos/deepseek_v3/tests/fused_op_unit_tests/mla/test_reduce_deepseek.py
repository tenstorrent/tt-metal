# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("hidden_size", [2112])  # q_lora_rank + kv_lora_rank + qk_rope_head_dim = 1536 + 512 + 64
@pytest.mark.parametrize("num_devices", [8])  # Number of devices along the reduction dimension
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 567296,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_wq_kv_a_fast_reduce_nc_trace_mode(
    mesh_device, batch_size, hidden_size, num_devices, warmup_iters, num_iters, function_level_defaults
):
    """
    Test the fast_reduce_nc operation from line 1111 of mla1d.py with trace mode for performance measurement.

    This operation follows the all-gather in the MLA forward pass and reduces the gathered tensor
    along dim=1 (the device dimension).

    Configuration:
    - Input shape: [1, num_devices, batch_size, hidden_size] = [1, 8, 32, 2112]
    - Reduction along dim=1 (sum across devices)
    - Output shape: [1, 1, batch_size, hidden_size] = [1, 1, 32, 2112]
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Memory: L1 interleaved
    - Compute config: HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    - Mesh device: 8x4 grid with FABRIC_1D
    """
    torch.manual_seed(0)

    # Set up sub-devices and semaphores (matching sequence test context)
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

    input_shape = [1, num_devices, batch_size, hidden_size]
    output_shape = [1, 1, batch_size, hidden_size]

    logger.info(f"Running on {mesh_device.get_num_devices()} devices")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output shape: {output_shape}")

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output: sum along dim=1
    torch_output_tensor = torch.sum(torch_input_tensor, dim=1, keepdim=True)

    # Create ttnn mesh tensor - replicated across devices (simulating post-all-gather state)
    # After all-gather, each device has the full [1, 8, 32, 2112] tensor
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, num_devices)
            ),
        ),
    )

    # Configure fast_reduce_nc matching mla1d.py line 1111-1114
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

    try:
        # Compile run
        logger.info("Compiling fast_reduce_nc operation")
        tt_output_tensor = ttnn.experimental.fast_reduce_nc(tt_input_tensor, **reduce_config)
        ttnn.synchronize_device(mesh_device)

        # Capture warmup trace
        logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
        trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(warmup_iters):
            tt_output_tensor = ttnn.experimental.fast_reduce_nc(tt_input_tensor, **reduce_config)
        ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        # Capture main trace
        logger.info(f"Capturing main trace with {num_iters} iterations")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_output_tensor = ttnn.experimental.fast_reduce_nc(tt_input_tensor, **reduce_config)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        # Execute warmup trace
        logger.info("Executing warmup trace")
        profiler = BenchmarkProfiler()
        profiler.start("warmup")
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        profiler.end("warmup")
        ttnn.synchronize_device(mesh_device)

        # Execute main trace with signposts
        logger.info("Executing main trace")
        signpost("start")
        profiler.start("main")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        profiler.end("main")
        signpost("stop")
        ttnn.synchronize_device(mesh_device)

        # Verify correctness
        # Since the output is replicated across all devices, we need to extract from individual devices
        logger.info("Verifying correctness")
        passed = True
        for i, t in enumerate(ttnn.get_device_tensors(tt_output_tensor)):
            tt_output_from_device = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking output for device {t.device().id()}")

            # Verify shape
            assert (
                tt_output_from_device.shape == torch_output_tensor.shape
            ), f"Shape mismatch on device {i}: {tt_output_from_device.shape} != {torch_output_tensor.shape}"

            # Use PCC for comparison since this is a reduction operation
            passed_pcc, output_pcc = assert_with_pcc(torch_output_tensor, tt_output_from_device, pcc=0.99)
            if not passed_pcc:
                logger.error(f"Output mismatch for device {i}: {output_pcc}")
                passed = False

        assert passed, "Output verification failed for one or more devices"

        logger.info("✓ Trace mode fast_reduce_nc test passed with correct output")

    finally:
        # Clean up sub-device configuration
        mesh_device.reset_sub_device_stall_group()
