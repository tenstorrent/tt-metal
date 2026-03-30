import pytest
import torch
import math
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost

from tests.ttnn.utils_for_testing import assert_equal, assert_allclose


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, output_shape",
    [
        (
            "wo_tilize",
            [1, 1, 32, 16384],
            [1, 1, 32, 16384],
        ),
    ],
    ids=["wo_tilize"],
)
@pytest.mark.parametrize("memory_config_type", ["interleaved", "width_sharded"])
@pytest.mark.parametrize("warmup_iters", [10])
# @pytest.mark.parametrize("num_iters", [100000])  # OLD: capture all iterations in one trace
@pytest.mark.parametrize("trace_iters", [1000])  # NEW: Iterations captured in trace
@pytest.mark.parametrize("num_trace_execs", [1000])  # NEW: Number of times to execute the trace (total = 1M iterations)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 5627904,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_tilize_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    output_shape,
    memory_config_type,
    warmup_iters,
    # num_iters,  # OLD
    trace_iters,  # NEW
    num_trace_execs,  # NEW
):
    """
    Test the tilize operation from mla1d.py with trace mode.

    This operation converts from ROW_MAJOR layout to TILE_LAYOUT:
    - wo_tilize (line 1903): [1, 1, 32, 16384] ROW_MAJOR → [1, 1, 32, 16384] TILE_LAYOUT
      Context: After all_gather in decode path, converts v_out before wo matmul
      Input: L1 ROW_MAJOR (from all_gather)
      Output: L1 TILE_LAYOUT

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Input: L1 memory, ROW_MAJOR
    - Output: L1 memory, TILE_LAYOUT
    - Memory config: INTERLEAVED or WIDTH_SHARDED (parameterized)
      - INTERLEAVED: no sharding
      - WIDTH_SHARDED: width-wise sharding configuration as defined by the test's shard_spec
    """
    torch.manual_seed(2003)

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output - same shape, just layout conversion
    torch_output_tensor = torch_input_tensor.clone()

    # Verify expected output shape
    assert (
        list(torch_output_tensor.shape) == output_shape
    ), f"Output shape mismatch: {list(torch_output_tensor.shape)} != {output_shape}"

    # Configure memory config based on type
    if memory_config_type == "width_sharded":
        # WIDTH_SHARDED: grid=[{x:1,y:0}-{x:4,y:1}], shape=[32, 2048], ROW_MAJOR
        # Avoid column 0 which contains dispatch cores when dispatch_core_axis=COL
        # 8 cores (4x2 grid), each processes 16384/8 = 2048 width
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 1))}),
            [32, 2048],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            [32, 1024],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )
    else:
        # INTERLEAVED
        input_memory_config = ttnn.L1_MEMORY_CONFIG
        output_memory_config = ttnn.L1_MEMORY_CONFIG

    # Create ttnn tensor with L1 memory config in ROW_MAJOR layout
    # This simulates the output from all_gather which is in ROW_MAJOR
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config,
    )

    tt_output_tensor = ttnn.tilize(tt_input_tensor, memory_config=output_memory_config)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.tilize(tt_input_tensor, memory_config=output_memory_config)
        # No deallocate inside trace - tensor is reused automatically
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace with trace_iters iterations (not all num_trace_execs)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(trace_iters):
        tt_output_tensor = ttnn.tilize(tt_input_tensor, memory_config=output_memory_config)
        # No deallocate inside trace - tensor is reused automatically
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace num_trace_execs times (total iterations = trace_iters * num_trace_execs)
    total_iters = trace_iters * num_trace_execs
    print(
        f"Stress testing: {trace_iters} iters/trace × {num_trace_execs} executions = {total_iters:,} total iterations"
    )

    signpost("start")
    profiler.start("main")
    for exec_num in range(num_trace_execs):
        ttnn.execute_trace(device, trace_id, blocking=False)

        # Progress indicator every 100 executions
        if (exec_num + 1) % 100 == 0:
            ttnn.synchronize_device(device)
            completed_iters = (exec_num + 1) * trace_iters
            print(
                f"  Progress: {exec_num + 1}/{num_trace_execs} executions ({completed_iters:,}/{total_iters:,} iterations)"
            )

    ttnn.synchronize_device(device)
    ttnn.release_trace(device, trace_id)
    profiler.end("main")
    signpost("stop")

    print(f"✓ Stress test completed: {total_iters:,} iterations executed without hanging")

    # Cleanup
    ttnn.deallocate(tt_output_tensor)
    ttnn.deallocate(tt_input_tensor)
