# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost

from tests.ttnn.utils_for_testing import assert_equal

shapes = [[[1, 1, 32, 32]], [[3, 1, 320, 384]], [[1, 1, 128, 7328]]]


@pytest.mark.parametrize(
    "input_shapes",
    shapes,
)
@pytest.mark.parametrize(
    "tilize_args",
    (
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "use_multicore": False,
        },
    ),
)
def test_tilize_test(input_shapes, tilize_args, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test("tilize", input_shapes, datagen_func, comparison_func, device, tilize_args)


@pytest.mark.parametrize("shape", [(64, 128), (512, 512)])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_fp32_truncation(device, shape, use_multicore):
    torch.manual_seed(2005)
    input_a = torch.full(shape, 1.9908e-05, dtype=torch.float32)
    # Use the fixture-provided device directly
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.tilize(input_tensor, use_multicore=use_multicore)
    output_tensor = ttnn.to_torch(input_tensor)
    assert torch.allclose(input_a, output_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[32, 256 * 64]])
@pytest.mark.parametrize("shard_shape", [[32, 256]])
@pytest.mark.parametrize(
    "shard_core_grid", [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})]
)  # 64 cores in an 8x8 grid
def test_tilize_row_major_to_width_sharded(device, dtype, tensor_shape, shard_shape, shard_core_grid):
    torch.manual_seed(42)
    torch.set_printoptions(sci_mode=False)

    shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create test data with sequential values from 1 to n for debugging
    for _ in range(30):
        input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)

        # Convert to ttnn tensor with row major layout and width sharding
        input_ttnn_tensor = ttnn.from_torch(
            input_torch_tensor,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_memory_config,
        )
        ttnn_output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config)
        output_torch_tensor = ttnn.to_torch(ttnn_output_tensor)

        assert torch.equal(input_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize("input_shape", [(32, 15936), (160, 5210112)])
def test_run_tilize_large_row_input(device, input_shape):
    orig_shape = input_shape

    input = torch.randn(orig_shape, dtype=torch.bfloat16)
    halos = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    halos_tile = ttnn.to_layout(halos, layout=ttnn.TILE_LAYOUT)
    halos_rm = ttnn.to_layout(halos_tile, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.to_torch(halos_rm)
    assert_equal(input, output)


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
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 1671168,
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
    num_iters,
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
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.tilize(tt_input_tensor, memory_config=output_memory_config)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace with signposts
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
