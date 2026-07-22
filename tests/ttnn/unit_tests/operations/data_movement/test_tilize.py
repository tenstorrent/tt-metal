# SPDX-FileCopyrightText: © 2023-2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math

import ttnn

from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost

from tests.ttnn.utils_for_testing import assert_equal, assert_allclose, assert_with_pcc
from models.common.utility_functions import skip_for_slow_dispatch, run_for_blackhole

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
    shape = input_shapes[0]
    torch_input = (torch.rand(shape) * 200 - 100).to(torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=tilize_args["dtype"][0],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=tilize_args["input_mem_config"][0],
    )
    tt_output = ttnn.tilize(
        tt_input, memory_config=tilize_args["output_mem_config"], use_multicore=tilize_args["use_multicore"]
    )
    torch_output = ttnn.to_torch(tt_output)

    assert_equal(torch_input, torch_output)


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


@pytest.mark.parametrize(
    "tensor_shape, num_cores",
    [
        ([1, 1, 128, 64], 4),
        ([1, 1, 256, 128], 8),
        ([1, 1, 64, 64], 2),
        ([1, 1, 128, 64], 2),  # shard_height=64: exercises multi-tile-row tile_start_id math
        ([1, 1, 256, 128], 4),  # shard_height=64: multi-tile-row with wider tensor
    ],
)
def test_tilize_height_sharded_to_interleaved(device, tensor_shape, num_cores):
    torch.manual_seed(99)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_shape = [tensor_shape[-2] // num_cores, tensor_shape[-1]]
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_input = torch.rand(tensor_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem_config
    )
    tt_output = ttnn.tilize(tt_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert_equal(torch_input, ttnn.to_torch(tt_output))


@pytest.mark.parametrize("api", ["tilize", "to_layout"])
def test_tilize_width_sharded_to_interleaved(device, api):
    torch.manual_seed(42)

    tensor_shape = [32, 256]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        [32, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    output_memory_config = ttnn.L1_MEMORY_CONFIG

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    if api == "tilize":
        output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)
    else:
        output_tensor = ttnn.to_layout(input_ttnn_tensor, ttnn.TILE_LAYOUT, memory_config=output_memory_config)

    assert output_tensor.layout == ttnn.TILE_LAYOUT
    assert output_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert_equal(input_torch_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, output_shard_shape",
    [
        # output_shard_shape=None means reuse shard_shape for the output nd_shard_spec
        ([4, 128, 128], [2, 64, 64], None),
        ([3, 160, 160], [2, 64, 64], None),
        ([5, 4, 160, 160], [2, 3, 64, 96], None),
        ([23, 96, 160], [4, 64, 96], None),  # uneven input sharding with cliff cores
        # Different output nd_shard_spec (last 2 dims tile-aligned for tilized output)
        ([4, 128, 128], [2, 64, 64], [1, 64, 128]),
        ([3, 160, 160], [2, 64, 64], [1, 64, 96]),  # uneven output: 160 % 96 != 0
        ([5, 4, 160, 160], [2, 3, 64, 96], [3, 2, 96, 64]),  # uneven output: 5 % 3, 160 % 96, 160 % 64
        ([23, 96, 160], [4, 64, 96], [6, 64, 64]),  # uneven output: 23 % 6, 160 % 64. With cliff cores
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize("input_is_nd_sharded", [True, False])
@pytest.mark.parametrize("output_is_nd_sharded", [True, False])
@pytest.mark.parametrize("use_multicore", [True, False])
def test_tilize_nd_sharded(
    device,
    dtype,
    tensor_shape,
    shard_shape,
    output_shard_shape,
    shard_core_grid,
    input_is_nd_sharded,
    output_is_nd_sharded,
    use_multicore,
):
    if not output_is_nd_sharded and output_shard_shape is not None:
        pytest.skip("output_shard_shape only applies when output is nd-sharded")

    torch.manual_seed(42)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    effective_output_shard_shape = output_shard_shape if output_shard_shape is not None else shard_shape
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=effective_output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    if input_is_nd_sharded:
        input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=input_nd_shard_spec)
        if not use_multicore:
            pytest.skip("Singlecore is not supported for sharded input")
    else:
        input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    if output_is_nd_sharded:
        output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    else:
        output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)

    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    ttnn_output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=use_multicore)
    output_torch_tensor = ttnn.to_torch(ttnn_output_tensor)

    assert_equal(input_torch_tensor, output_torch_tensor)


# Legacy-sharded TILE output requires shard height and width to be multiples of 32 (tile size).
# Only (tensor_shape, grid) where height_shard, width_shard, and block_shard dims are all % 32 == 0.
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        (
            [4, 128, 128],
            [2, 64, 64],
        ),  # 2D (512,128): height_shard (128,128), width_shard (512,32), block (256,64) all tile-aligned
        ([8, 64, 128], [2, 32, 64]),  # 2D (512,128): same as above
        (
            [7, 128, 128],
            [2, 64, 96],
        ),  # Uneven ND sharding: dim 0 has 7 % 2 != 0 -> cliff core; 2D (896,128) still tile-aligned
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})],
)
@pytest.mark.parametrize(
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_tilize_nd_sharded_to_legacy_sharded(
    device, dtype, tensor_shape, shard_shape, shard_core_grid, output_memory_layout
):
    """tilize: ND-sharded row-major input -> legacy-sharded (HEIGHT/WIDTH/BLOCK) tile output."""
    torch.manual_seed(42)

    num_shard_cores = shard_core_grid.num_cores()
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    height_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
    )
    shard_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": height_shard_shape},
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": width_shard_shape},
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {"shard_grid": shard_core_grid, "shard_shape": block_shard_shape},
    }
    layout_info = shard_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        layout_info["shard_grid"], layout_info["shard_shape"], ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    input_torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    ttnn_output_tensor = ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    output_torch_tensor = ttnn.to_torch(ttnn_output_tensor)

    assert_equal(input_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize("input_shape", [(32, 15936), (160, 5210112)])
def test_run_tilize_large_row_input(device, input_shape):
    orig_shape = input_shape

    input = torch.randn(orig_shape, dtype=torch.bfloat16)
    halos = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    halos_tile = ttnn.to_layout(halos, layout=ttnn.TILE_LAYOUT)
    halos_rm = ttnn.to_layout(halos_tile, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.to_torch(halos_rm)
    assert_equal(input, output)


@pytest.mark.parametrize("shape", [(1, 7168, 2304)])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("torch_dtype", [torch.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_from_torch_conversion_deep_seek_mc_large_number_of_pages_per_row(
    device, shape, ttnn_dtype, torch_dtype, layout
):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(shape, dtype=torch_dtype)

    dram_cores = device.dram_grid_size().x

    if shape[2] % dram_cores != 0:
        pytest.skip(f"Shape width {shape[2]} is not evenly divisible by {dram_cores} DRAM cores")

    # Calculate shard_shape by dividing last dimension by number of DRAM cores
    shard_shape = (shape[1], shape[2] // dram_cores)

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )

    ttnn_output_tensor = ttnn.tilize(ttnn_input_tensor)

    ttnn.synchronize_device(device)

    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
    assert_allclose(torch_input_tensor, ttnn_output_tensor, atol=4e-3, rtol=4e-3)


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
            "trace_region_size": 8 * 1024 * 1024,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
@skip_for_slow_dispatch()
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


# Regression coverage for https://github.com/tenstorrent/tt-metal/issues/45331.
# Calls ttnn.tilize directly on a width-sharded DRAM ROW_MAJOR input that
# previously sent the op into TilizeMultiCoreDefaultProgramFactory, whose
# full-row CB allocation (~5.5 MB) exceeded the 1.5 MB L1 per-core budget on
# Wormhole. The ttnn::tilize wrapper now reroutes such cases via interleaved
# DRAM so TilizeMultiCoreBlockProgramFactory (bounded CBs) is used instead.
# The assertions are intentionally minimal: this test exists to catch a crash
# regression, not to validate full numerical fidelity of tilize.
@pytest.mark.parametrize(
    "shard_shape",
    [
        (2048, 3584),
    ],
)
def test_tilize_width_sharded_dram_input_45331(device, shard_shape):
    torch.manual_seed(0)
    # Width-shard across every DRAM bank the device exposes (12 on Wormhole,
    # 8 on Blackhole). Hardcoding 12 cores would request a DRAM bank that does
    # not exist on Blackhole and crash in the allocator before tilize runs.
    num_cores = device.dram_grid_size().x
    shard_height, shard_width = shard_shape
    tensor_shape = (shard_height, shard_width * num_cores)
    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_dram_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )

    tt_rm = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sharded_dram_cfg,
        device=device,
    )

    tt_tile = ttnn.tilize(tt_rm, memory_config=sharded_dram_cfg, use_multicore=True)

    assert tt_tile.layout == ttnn.TILE_LAYOUT
    torch_out = ttnn.to_torch(tt_tile)
    assert torch.equal(torch_tensor, torch_out), "tilize round-trip mismatch"


def test_tilize_width_sharded_dram_input_to_l1_sharded_output_49107(device):
    torch.manual_seed(0)
    num_cores = device.dram_grid_size().x
    shard_shape = (32, 64)
    tensor_shape = (shard_shape[0], shard_shape[1] * num_cores)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    tt_input = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_mem_config,
        device=device,
    )
    tt_output = ttnn.tilize(tt_input, memory_config=output_mem_config, use_multicore=True)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert tt_output.memory_config().buffer_type == ttnn.BufferType.L1
    assert_equal(torch_tensor, ttnn.to_torch(tt_output))


@pytest.mark.parametrize(
    "shard_type,shard_shape",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (512, 64)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (32, 128)),
    ],
    ids=["height_sharded_dram", "width_sharded_dram"],
)
def test_tilize_dram_backed_sharded_input(device, shard_type, shard_shape):
    torch.manual_seed(0)
    num_cores = 4
    shard_h, shard_w = shard_shape
    if shard_type == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        tensor_shape = (shard_h * num_cores, shard_w)
    else:
        tensor_shape = (shard_h, shard_w * num_cores)
    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    dram_sharded_cfg = ttnn.MemoryConfig(shard_type, ttnn.BufferType.DRAM, shard_spec)

    tt_rm = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram_sharded_cfg,
        device=device,
    )

    tt_tile = ttnn.tilize(tt_rm, memory_config=dram_sharded_cfg, use_multicore=True)
    assert tt_tile.layout == ttnn.TILE_LAYOUT
    torch_out = ttnn.to_torch(tt_tile)
    assert torch.equal(torch_tensor, torch_out), "tilize round-trip mismatch on DRAM-backed sharded input"


# fp8 is a valid tile INPUT (it tilizes to any float TILE output) though ROW_MAJOR-only as a dtype. Even
# tile-widths only (odd fp8 widths hit a separate reader NoC bug); golden is the host-quantized fp8 source.
@run_for_blackhole()
@pytest.mark.parametrize(
    "out_dtype,min_pcc",
    [(ttnn.float32, 0.9999), (ttnn.bfloat16, 0.9999), (ttnn.bfloat8_b, 0.999), (ttnn.bfloat4_b, 0.98)],
    ids=["out_fp32", "out_bf16", "out_bfp8", "out_bfp4"],
)
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 32, 64, 512)], ids=["small", "wide"])
def test_tilize_fp8_input(device, shape, out_dtype, min_pcc):
    torch.manual_seed(0)
    torch_input = torch.randn(*shape, dtype=torch.float32)
    golden = torch_input.to(torch.float8_e4m3fn).to(torch.float32)  # match device fp8 quantization
    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = ttnn.tilize(tt_in, dtype=out_dtype)
    assert tt_out.layout == ttnn.TILE_LAYOUT and tt_out.dtype == out_dtype
    assert_with_pcc(golden, ttnn.to_torch(tt_out).float(), min_pcc)


def _make_fp32_precision_tensor(shape):
    """Build a fp32 tensor with values that require full 23-bit mantissa precision.

    These values are NOT representable in bfloat16 (7-bit mantissa) or TF32
    (10-bit mantissa).  Any silent truncation will cause torch.equal to fail.
    """
    torch.manual_seed(35303)
    t = torch.randn(shape, dtype=torch.float32)
    precision_values = torch.tensor(
        [
            1.0000001192092896,  # smallest fp32 > 1.0  (bit 0x3F800001)
            -1.0000001192092896,
            0.693147182464599609,
            -0.693147182464599609,
            3.14159265358979,  # pi at full fp32 precision
            -3.14159265358979,
            16777217.0,  # 2^24 + 1, not representable in bf16
            -16777217.0,
            8388609.0,  # 2^23 + 1
            -8388609.0,
            1.41421353816986084,
            -1.41421353816986084,
            1.1920928955078125e-07,  # machine epsilon for fp32
            -1.1920928955078125e-07,
            1.9908e-05,  # regression value from issue #39310
            -1.9908e-05,
            1234567.125,  # large with fractional low bits
            -1234567.125,
            1.17549435e-38,  # smallest positive normal fp32
            -1.17549435e-38,
            3.40282347e38,  # FLT_MAX
            -3.40282347e38,
            0.333333343267440796,  # 1/3 at full fp32 precision
            -0.333333343267440796,
            2.7182817459106445,  # e at full fp32 precision
            -2.7182817459106445,
        ],
        dtype=torch.float32,
    )
    n = min(precision_values.numel(), t.shape[-1])
    t.view(-1, t.shape[-1])[0, :n] = precision_values[:n]
    return t


@pytest.mark.parametrize("shape", [(32, 32), (64, 128), (256, 512)])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_fp32_lossless(device, shape, use_multicore):
    """Tilize must preserve fp32 values with perfect bitwise equality."""
    input_tensor = _make_fp32_precision_tensor(shape)

    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.tilize(tt_input, use_multicore=use_multicore)
    output_tensor = ttnn.to_torch(tt_output)

    assert torch.equal(input_tensor, output_tensor)


@pytest.mark.parametrize("shape", [(32, 32), (128, 256)])
def test_tilize_untilize_fp32_roundtrip(device, shape):
    """Full tilize -> untilize round-trip must be bit-exact for fp32."""
    input_tensor = _make_fp32_precision_tensor(shape)

    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_tiled = ttnn.tilize(tt_input)
    tt_rm = ttnn.untilize(tt_tiled)
    output_tensor = ttnn.to_torch(tt_rm)

    assert torch.equal(input_tensor, output_tensor)


@pytest.mark.parametrize("shape", [(48, 80), (33, 65)])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_with_zero_padding_fp32_lossless(device, shape, use_multicore):
    """tilize_with_zero_padding must preserve fp32 data region exactly."""
    input_tensor = _make_fp32_precision_tensor(shape)

    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.tilize_with_zero_padding(tt_input, use_multicore=use_multicore)
    output_tensor = ttnn.to_torch(tt_output)

    H, W = shape
    assert torch.equal(input_tensor, output_tensor[:H, :W])


@pytest.mark.parametrize("shape", [(48, 80), (33, 65)])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_with_val_padding_fp32_lossless(device, shape, use_multicore):
    """tilize_with_val_padding must preserve fp32 data region exactly."""
    input_tensor = _make_fp32_precision_tensor(shape)
    H, W = shape
    padded_H = math.ceil(H / 32) * 32
    padded_W = math.ceil(W / 32) * 32

    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.tilize_with_val_padding(tt_input, [padded_H, padded_W], 1.0075, use_multicore=use_multicore)
    output_tensor = ttnn.to_torch(tt_output)

    assert torch.equal(input_tensor, output_tensor[:H, :W])


def test_tilize_fp32_lossless_via_to_layout(device):
    """to_layout (host→device tilize path) must be bit-exact for fp32."""
    input_tensor = _make_fp32_precision_tensor((64, 128))

    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.to_layout(tt_input, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(tt_output)

    assert torch.equal(input_tensor, output_tensor)


@pytest.mark.parametrize(
    "memory_layout, tensor_shape, grid_shape",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [1, 1, 128, 64], (4, 1)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, [1, 1, 32, 128], (1, 4)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, [1, 1, 128, 128], (2, 2)),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype, min_pcc",
    [
        (ttnn.float32, ttnn.float32, 1.0),
        (ttnn.bfloat16, ttnn.bfloat8_b, 0.999),
    ],
    ids=["fp32_in_fp32_out", "bf16_in_bfp8_out"],
)
def test_tilize_sharded_fp32_llk_acc(device, memory_layout, tensor_shape, grid_shape, in_dtype, out_dtype, min_pcc):
    torch.manual_seed(5)
    n_y, n_x = grid_shape
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_x - 1, n_y - 1))})
    H, W = tensor_shape[-2], tensor_shape[-1]
    num_cores = n_y * n_x
    if memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_shape = [H // num_cores, W]
    elif memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        shard_shape = [H, W // num_cores]
    else:
        shard_shape = [H // n_y, W // n_x]
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_cfg = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)
    torch_input = torch.rand(tensor_shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    tt_output = ttnn.tilize(tt_input, memory_config=mem_cfg, dtype=out_dtype)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert_with_pcc(torch_input, ttnn.to_torch(tt_output).float(), min_pcc)


@pytest.mark.parametrize(
    "memory_layout, tensor_shape, shard_shape, grid_shape",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [1, 1, 256, 64], [64, 64], (1, 4)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, [1, 1, 32, 256], [32, 64], (1, 4)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, [1, 1, 128, 128], [64, 64], (2, 2)),
    ],
)
def test_tilize_col_major_orientation(device, memory_layout, tensor_shape, shard_shape, grid_shape):
    torch.manual_seed(7)
    n_y, n_x = grid_shape
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_x - 1, n_y - 1))})
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    mem_cfg = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)
    torch_input = torch.rand(tensor_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    tt_output = ttnn.tilize(tt_input, memory_config=mem_cfg)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert_equal(torch_input, ttnn.to_torch(tt_output))


@pytest.mark.parametrize(
    "tensor_shape, num_cores",
    [
        ([1, 1, 32, 64], 1),
        ([1, 1, 64, 64], 2),
        ([1, 1, 128, 64], 4),
        ([1, 1, 256, 128], 8),
        ([1, 1, 512, 64], 8),
        ([1, 1, 96, 128], 3),
        ([1, 1, 192, 256], 6),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tilize_height_sharded_shapes(device, tensor_shape, num_cores, dtype):
    torch.manual_seed(42)
    H, W = tensor_shape[-2], tensor_shape[-1]
    shard_h = H // num_cores
    if shard_h == 0 or shard_h % 32 != 0:
        pytest.skip(f"shard_height={shard_h} not tile-aligned")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_h, W], ttnn.ShardOrientation.ROW_MAJOR)
    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.rand(tensor_shape, dtype=torch_dtype)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    tt_output = ttnn.tilize(tt_input, memory_config=mem_cfg)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert_equal(torch_input, ttnn.to_torch(tt_output))


@pytest.mark.parametrize(
    "tensor_shape, num_cores",
    [
        # Basic shapes
        ([1, 1, 32, 64], 2),
        ([1, 1, 32, 128], 4),
        ([1, 1, 64, 256], 4),
        ([1, 1, 128, 512], 8),
        # Multi-tile shard height
        ([1, 1, 64, 128], 4),
        ([1, 1, 96, 256], 4),
        ([1, 1, 128, 256], 4),
        # Wide tensors split across many cores
        ([1, 1, 32, 1024], 8),
        ([1, 1, 64, 2048], 8),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tilize_width_sharded_shapes(device, tensor_shape, num_cores, dtype):
    torch.manual_seed(42)
    H, W = tensor_shape[-2], tensor_shape[-1]
    shard_w = W // num_cores
    if shard_w == 0 or shard_w % 32 != 0:
        pytest.skip(f"shard_width={shard_w} not tile-aligned for this shape/num_cores combo")
    if H % 32 != 0:
        pytest.skip(f"shard_height={H} not tile-aligned")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [H, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.rand(tensor_shape, dtype=torch_dtype)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    tt_output = ttnn.tilize(tt_input, memory_config=mem_cfg)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert_equal(torch_input, ttnn.to_torch(tt_output))


@pytest.mark.parametrize(
    "tensor_shape, grid_shape",
    [
        # Square grids
        ([1, 1, 64, 64], (2, 2)),
        ([1, 1, 128, 128], (2, 2)),
        ([1, 1, 256, 256], (4, 4)),
        ([1, 1, 128, 256], (2, 4)),
        ([1, 1, 256, 128], (4, 2)),
        # Rectangular grids
        ([1, 1, 64, 256], (2, 4)),
        ([1, 1, 256, 64], (4, 2)),
        ([1, 1, 128, 512], (2, 4)),
        ([1, 1, 512, 128], (4, 2)),
        # Larger
        ([1, 1, 512, 512], (4, 4)),
        ([1, 1, 256, 512], (4, 4)),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tilize_block_sharded_shapes(device, tensor_shape, grid_shape, dtype):
    torch.manual_seed(42)
    n_y, n_x = grid_shape
    H, W = tensor_shape[-2], tensor_shape[-1]
    shard_h, shard_w = H // n_y, W // n_x
    if shard_h % 32 != 0 or shard_w % 32 != 0:
        pytest.skip(f"shard ({shard_h},{shard_w}) not tile-aligned")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_x - 1, n_y - 1))})
    shard_spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.rand(tensor_shape, dtype=torch_dtype)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    tt_output = ttnn.tilize(tt_input, memory_config=mem_cfg)
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert_equal(torch_input, ttnn.to_torch(tt_output))


@pytest.mark.parametrize(
    "tensor_shape, shard_layout",
    [
        ([1, 1, 128, 256], None),  # Interleaved input/output.
        ([1, 1, 32, 1024], ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ([1, 1, 1024, 32], ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ([1, 1, 256, 256], ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
@pytest.mark.parametrize(
    "tile_shape",
    [(16, 32), (8, 32), (4, 32), (2, 32), (1, 32)],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tilize_row_major_to_tiny_tile(device, tensor_shape, shard_layout, tile_shape, dtype):
    """Tilize a ROW_MAJOR input (interleaved or sharded) directly into a tiny (non-32x32) tile shape."""
    torch.manual_seed(42)
    tile_h, tile_w = tile_shape
    H, W = tensor_shape[-2], tensor_shape[-1]
    shard_h, shard_w = 32, 32
    assert H % tile_h == 0 and W % tile_w == 0, "tensor dims must be divisible by tile dims"

    if shard_layout is None:
        mem_cfg = None
    elif shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        assert W == shard_w, "height-sharded shard width must match tensor width"
        assert H % shard_h == 0, "tensor height must be divisible by shard height"
        num_cores = H // shard_h
        grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
        shard_shape = [shard_h, shard_w]
    elif shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        assert H == shard_h, "width-sharded shard height must match tensor height"
        assert W % shard_w == 0, "tensor width must be divisible by shard width"
        num_cores = W // shard_w
        grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
        shard_shape = [shard_h, shard_w]
    else:
        assert shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        assert H % shard_h == 0 and W % shard_w == 0, "tensor dims must be divisible by shard dims"
        n_y = H // shard_h
        n_x = W // shard_w
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_x - 1, n_y - 1))})
        shard_shape = [shard_h, shard_w]

    if shard_layout is not None:
        shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        mem_cfg = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_input = torch.rand(tensor_shape, dtype=torch_dtype)

    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    tt_output = ttnn.tilize(tt_input, tile=ttnn.Tile(list(tile_shape)), memory_config=mem_cfg)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    if shard_layout is not None:
        assert tt_output.memory_config().memory_layout == shard_layout
    assert_equal(torch_input, ttnn.to_torch(tt_output))


@pytest.mark.parametrize(
    "tensor_shape, shard_layout",
    [
        # Interleaved input/output.
        ([1, 1, 128, 256], None),
        ([1, 1, 64, 256], None),
        ([1, 1, 64, 128], None),
        ([1, 1, 16, 128], None),
        # Sharded input/output (invokes the sharded retile factory).
        ([1, 1, 32, 1024], ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ([1, 1, 1024, 32], ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ([1, 1, 256, 256], ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
@pytest.mark.parametrize("input_tile_shape", [(32, 32), (16, 32), (8, 32), (4, 32), (2, 32), (1, 32)])
@pytest.mark.parametrize("output_tile_shape", [(32, 32), (16, 32), (8, 32), (4, 32), (2, 32), (1, 32)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_tilize_retile(device, tensor_shape, shard_layout, input_tile_shape, output_tile_shape, dtype):
    """Retile an already-tiled input into a different tile shape (invokes the retile factory)."""
    torch.manual_seed(42)
    torch_input = torch.rand(tensor_shape, dtype=torch.bfloat16)

    if input_tile_shape[0] == output_tile_shape[0]:
        pytest.skip("Input and output tile shapes are the same")

    # Build a (possibly sharded) already-tiled input using the source tile shape.
    mem_cfg = None
    if shard_layout is not None:
        # Use a 32x32 shard so the shard height is divisible by every tile height under test,
        # keeping both the input and output tilings tile-aligned within each shard.
        H, W = tensor_shape[-2], tensor_shape[-1]
        shard_h, shard_w = 32, 32
        if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            num_cores = H // shard_h
            grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
        elif shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            num_cores = W // shard_w
            grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
        else:
            assert shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            n_y, n_x = H // shard_h, W // shard_w
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_x - 1, n_y - 1))})
        shard_spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        mem_cfg = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile(list(input_tile_shape)),
        memory_config=mem_cfg,
    )
    assert tt_input.layout == ttnn.TILE_LAYOUT

    # Re-tilize into a different tile shape; input and output tile shapes differ.
    tt_output = ttnn.tilize(tt_input, tile=ttnn.Tile(list(output_tile_shape)), memory_config=mem_cfg)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    if shard_layout is not None:
        assert tt_output.memory_config().memory_layout == shard_layout
    assert_equal(torch_input, ttnn.to_torch(tt_output))
