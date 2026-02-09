# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.deepseek_v3.utils.config_dataclass import SliceConfig
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("num_heads_local", [16])
@pytest.mark.parametrize(
    "slice_type, start_offset, output_size",
    [
        ("q_nope_slice", 0, 128),  # tt_q_nope: slice [0:128]
        ("q_rope_slice", 128, 64),  # tt_q_rope: slice [128:192]
    ],
    ids=["q_nope_slice", "q_rope_slice"],
)
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
def test_deepseek_v3_mla_slice_q_rope_nope_trace_mode(
    device,
    batch_size,
    num_heads_local,
    warmup_iters,
    num_iters,
    qk_rope_head_dim,
    slice_type,
    start_offset,
    output_size,
):
    """
    Test the slice operations from lines 1251 and 1253 of mla1d.py with trace mode.

    Configuration:
    - Input shape: [32, 1, 16, 192]
    - Output shapes vary by slice type:
      - q_nope_slice: [32, 1, 16, 128]
      - q_rope_slice: [32, 1, 16, 64]
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Memory: L1 interleaved
    """
    # For single device
    compute_grid_size = device.compute_with_storage_grid_size()  # e.g., CoreCoord(x=8, y=8) or (x=12, y=8)

    # Same calculation as before
    q_rope_shape = (1, USERS_PER_ROW, num_heads_local, qk_rope_head_dim)  # (1, 32, 16, 64)
    q_rope_shard_height = nearest_y(q_rope_shape[2], ttnn.TILE_SIZE)  # nearest_y(16, 32) = 32
    q_rope_shard_width = q_rope_shape[3]  # 64
    q_rope_num_cores = q_rope_shape[1]  # 32

    # Use single device's grid_size instead of mesh_device's
    q_rope_core_grid = ttnn.num_cores_to_corerangeset(
        q_rope_num_cores,  # 32 cores
        compute_grid_size,  # Single device grid (e.g., 8x8 = 64 cores available)
        row_wise=True,
    )

    # Same slice config
    q_rope_slice_config = SliceConfig(
        memory_config=ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),  # (32, 64)
            core_grid=q_rope_core_grid,  # 32 cores on single device
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    )
    torch.manual_seed(0)

    input_shape = [32, 1, 16, 192]
    output_shape = [32, 1, 16, output_size]

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output: slice along last dimension
    end_offset = start_offset + output_size
    torch_output_tensor = torch_input_tensor[:, :, :, start_offset:end_offset]

    # Create ttnn tensor
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Configure slice operation matching mla1d.py lines 1117-1126
    start_indices = [0, 0, 0, start_offset]
    end_indices = [batch_size, 1, 16, start_offset + output_size]
    if slice_type == "q_rope_slice":
        slice_config = q_rope_slice_config
    else:
        slice_config = {
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        }

    # Compile run
    logger.info(f"Compiling slice operation: {slice_type}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Start indices: {start_indices}")
    logger.info(f"  End indices: {end_indices}")
    logger.info(f"  Output shape: {output_shape}")

    tt_output_tensor = ttnn.slice(tt_input_tensor, start_indices, end_indices, **slice_config)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.slice(tt_input_tensor, start_indices, end_indices, **slice_config)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.slice(tt_input_tensor, start_indices, end_indices, **slice_config)
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
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert (
        torch_output_from_tt.shape == torch_output_tensor.shape
    ), f"Shape mismatch: {torch_output_from_tt.shape} != {torch_output_tensor.shape}"

    assert_equal(torch_output_tensor, torch_output_from_tt)

    logger.info(f"✓ Trace mode {slice_type} test passed with correct output")
