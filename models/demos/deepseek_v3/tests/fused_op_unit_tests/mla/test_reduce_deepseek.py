# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 567296,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_wq_kv_a_fast_reduce_nc_trace_mode(
    device, batch_size, hidden_size, num_devices, warmup_iters, num_iters
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
    """
    torch.manual_seed(0)

    input_shape = [1, num_devices, batch_size, hidden_size]
    output_shape = [1, 1, batch_size, hidden_size]

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output: sum along dim=1
    torch_output_tensor = torch.sum(torch_input_tensor, dim=1, keepdim=True)

    # Create ttnn tensor
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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

    # Compile run
    logger.info("Compiling fast_reduce_nc operation")
    tt_output_tensor = ttnn.experimental.fast_reduce_nc(tt_input_tensor, **reduce_config)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.experimental.fast_reduce_nc(tt_input_tensor, **reduce_config)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.experimental.fast_reduce_nc(tt_input_tensor, **reduce_config)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(device)

    # Calculate performance metrics
    warmup_time_ms = profiler.get_duration("warmup") * 1000
    main_time_ms = profiler.get_duration("main") * 1000
    avg_time_per_iter_us = (main_time_ms / num_iters) * 1000

    logger.info(f"Warmup time: {warmup_time_ms:.2f} ms ({warmup_iters} iterations)")
    logger.info(f"Main trace time: {main_time_ms:.2f} ms ({num_iters} iterations)")
    logger.info(f"Average time per iteration: {avg_time_per_iter_us:.2f} µs")

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert (
        torch_output_from_tt.shape == torch_output_tensor.shape
    ), f"Shape mismatch: {torch_output_from_tt.shape} != {torch_output_tensor.shape}"

    # Use PCC for comparison since this is a reduction operation
    assert_with_pcc(torch_output_tensor, torch_output_from_tt, 0.99)

    logger.info("✓ Trace mode fast_reduce_nc test passed with correct output")
