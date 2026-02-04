# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """PyTorch reference implementation of RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return weight * x


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "norm_type, hidden_size, shard_grid, shard_width",
    [
        ("q_norm", 1536, (8, 2), 96),  # Q norm: [1, 1, 32, 1536], width sharded 8x2 [32, 96]
        ("kv_norm", 512, (8, 2), 32),  # KV norm: [1, 1, 32, 512], width sharded 8x2 [32, 32]
    ],
    ids=["q_norm", "kv_norm"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 550912,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_rms_norm_trace_mode(
    device,
    batch_size,
    norm_type,
    hidden_size,
    shard_grid,
    shard_width,
    warmup_iters,
    num_iters,
):
    """
    Test the RMS normalization operations from lines 1141 and 1146 of mla1d.py with trace mode.

    Two RMS norm operations are used in the MLA decode forward pass:
    1. Q norm (line 1141): normalizes query features [1, 1, 32, 1536]
    2. KV norm (line 1146): normalizes key-value nope features [1, 1, 32, 512]

    Both use width-sharded memory layout with 8x2 core grid.

    Configuration:
    - Q norm: input [1, 1, 32, 1536], width sharded 8x2 with shard shape [32, 96]
    - KV norm: input [1, 1, 32, 512], width sharded 8x2 with shard shape [32, 32]
    - Epsilon: 1e-6 (default RMS norm epsilon)
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Compute config: LoFi (MathFidelity.LoFi)
    """
    torch.manual_seed(0)

    input_shape = [1, 1, batch_size, hidden_size]
    epsilon = 1e-6

    # Create random tensors for input and weights
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Weight tensor needs to be shaped as tile width sticks: [1, -1, TILE_SIZE]
    # This matches the format used in rms_norm.py convert_weights line 42
    torch_weight_1d = torch.ones(hidden_size, dtype=torch.bfloat16)
    torch_weight_tensor_for_reference = torch_weight_1d.reshape(1, 1, 1, hidden_size)

    # Reshape weight for ttnn: [1, 1, num_tiles, TILE_SIZE]
    num_tiles = hidden_size // ttnn.TILE_SIZE
    torch_weight_tensor_reshaped = torch_weight_1d.reshape(1, 1, num_tiles, ttnn.TILE_SIZE)

    # Golden output using reference RMS norm
    torch_output_tensor = rms_norm_reference(torch_input_tensor, torch_weight_tensor_for_reference, epsilon)

    # Create ttnn tensors
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor_reshaped,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Setup WIDTH_SHARDED memory config matching mla1d.py
    shard_height = batch_size
    num_cores_y, num_cores_x = shard_grid

    shard_grid_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1),
            )
        }
    )

    shard_shape = [shard_height, shard_width]
    shard_spec = ttnn.ShardSpec(shard_grid_set, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    width_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Convert input to width sharded
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, width_sharded_mem_config)

    # Configure RMS norm program config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Get program config based on sharded memory layout
    activation_grid_bounding_box_size = shard_grid_set.bounding_box().grid_size()
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=activation_grid_bounding_box_size,
        subblock_w=1,
        block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
        block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
        inplace=False,
    )

    # Compile run
    logger.info(f"Compiling RMS norm operation: {norm_type}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Shard grid: {shard_grid}")
    logger.info(f"  Shard shape: {shard_shape}")

    tt_output_tensor = ttnn.rms_norm(
        tt_input_tensor,
        epsilon=epsilon,
        weight=tt_weight_tensor,
        program_config=program_config,
        memory_config=width_sharded_mem_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.rms_norm(
            tt_input_tensor,
            epsilon=epsilon,
            weight=tt_weight_tensor,
            program_config=program_config,
            memory_config=width_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.rms_norm(
            tt_input_tensor,
            epsilon=epsilon,
            weight=tt_weight_tensor,
            program_config=program_config,
            memory_config=width_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
        )
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

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_layout(tt_output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert (
        torch_output_from_tt.shape == torch_output_tensor.shape
    ), f"Shape mismatch: {torch_output_from_tt.shape} != {torch_output_tensor.shape}"

    # Use PCC for comparison (RMS norm can have some numerical differences)
    assert_with_pcc(torch_output_tensor, torch_output_from_tt, 0.99)

    logger.info(f"✓ Trace mode {norm_type} test passed with correct output")
