# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.utility_functions import is_wormhole_b0, is_wormhole_b0, is_blackhole
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32


# TODO (7735): Switch to new interleaved_to_sharded with sharded_mem_config input and re-enable BLOCK sharded tests
@pytest.mark.parametrize(
    "input_shape, shard_scheme, shard_size",
    [
        ([1, 1, 100352, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64)),
        ([1, 1, 128, 50176], ttnn.TensorMemoryLayout.WIDTH_SHARDED, (128, 512)),
        pytest.param(
            [1, 1, 100352, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (2048, 32),
            marks=pytest.mark.xfail(
                reason="7735: Switch to new interleaved_to_sharded with sharded_mem_config input and re-enable BLOCK sharded tests"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_tile(
    device,
    input_shape,
    shard_size,
    shard_scheme,
    shard_orientation,
    input_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    input_size = torch.Size(input_shape)
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    x = torch.arange(input_size.numel()).reshape(input_size).bfloat16().float()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            input_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttnn.BufferType.L1,
            ),
        )
    )

    yt = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=output_dtype
    )

    zt = ttnn.sharded_to_interleaved(
        yt,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
        output_dtype=input_dtype,
    )

    tt_og = xt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    tt_got_back = zt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    if input_dtype == output_dtype:
        passing, output = comp_equal(tt_og, tt_got_back)
    else:
        passing, output = comp_pcc(tt_og, tt_got_back, 0.999)
    logger.info(output)

    assert passing


# TODO (7735): Switch to new interleaved_to_sharded with sharded_mem_config input and re-enable BLOCK sharded tests
@pytest.mark.parametrize(
    "input_shape, shard_scheme, shard_size, num_cores",
    [
        ([1, 1, 100352, 64], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), 98),
        ([1, 1, 128, 50176], ttnn.TensorMemoryLayout.WIDTH_SHARDED, (128, 512), 98),
        pytest.param(
            [1, 1, 100352, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (2048, 32),
            98,
            marks=pytest.mark.xfail(
                reason="7735: switch to new interleaved_to_sharded with sharded_mem_config input and re-enable block sharded tests"
            ),
        ),
        ([1, 1, 32, 40], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (32, 40), 1),
        pytest.param(
            [2, 64, 64, 320],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (1024, 40),
            64,
            marks=pytest.mark.xfail(
                reason="7735: switch to new interleaved_to_sharded with sharded_mem_config input and re-enable block sharded tests"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_sharded_rm(
    device,
    input_shape,
    shard_size,
    shard_scheme,
    shard_orientation,
    num_cores,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    input_size = torch.Size(input_shape)
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    x = torch.arange(input_size.numel()).reshape(input_size).bfloat16().float()

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(
        device,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        ),
    )

    yt = ttnn.interleaved_to_sharded(xt, grid_size, shard_size, shard_scheme, shard_orientation)

    zt = ttnn.sharded_to_interleaved(
        yt,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to_torch()

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(tt_og, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[100352, 98], [25088, 98]])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_untilize(H, num_cores, in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    N = 1
    C = 1
    W = 64
    if out_sharded and not in_sharded and H == 100352:
        pytest.skip("Unsupported config for sharding")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [H // num_cores, W],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    yt = ttnn.untilize(
        xt,
        memory_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    if dtype == ttnn.bfloat16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("H, num_cores", [[25088, 98]])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_tilize(H, num_cores, output_dtype, device, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    N = 1
    C = 1
    W = 64

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(
        device,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    )

    yt = ttnn.interleaved_to_sharded(
        xt,
        grid_size,
        [H // num_cores, W],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    yt_tilized = ttnn.tilize(
        yt,
        memory_config=ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        ),
        use_multicore=True,
        dtype=output_dtype,
    )

    zt = ttnn.sharded_to_interleaved(
        yt_tilized,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    )

    tt_got_back = zt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    if output_dtype == ttnn.bfloat16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="WH ND hang, see issue #4392")
@pytest.mark.parametrize("M", [127 * 32])
@pytest.mark.parametrize("K", [1 * 32])
@pytest.mark.parametrize("N", [1 * 32])
@pytest.mark.parametrize("num_cores", [64])
def test_height_sharded_matmul_1d_padding(device, M, K, N, num_cores):
    grid_size = device.compute_with_storage_grid_size()
    if num_cores > (grid_size.x * grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {grid_size}")
    grid_size = (8, 8)
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    height_shard_spec = [2 * 32, 32]  # [2, 1] in tiles

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttnn.bfloat16)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttnn.bfloat16)

    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        height_shard_spec,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=None,
        program_config=program_config,
        memory_config=sharded_mem_config,
        dtype=ttnn.bfloat16,
    )

    output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1
    tt_out = tt2torch_tensor(output_t)
    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="WH ND hang, see issue #4392")
@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M, num_cores", [[25088, 98], [50176, 98]])
@pytest.mark.parametrize("K, N", [[64, 64], [64, 256], [256, 64], [256, 128]])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_1d_in1(
    device,
    in0_sharded,
    out_sharded,
    M,
    K,
    N,
    num_cores,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // num_cores, K],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(12, 9),
        in0_block_w=K // 32,
        out_subblock_h=8 // (N // 32),
        out_subblock_w=N // 32,
        per_core_M=M // 32 // num_cores,
        per_core_N=N // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("H, num_cores", [[64, 64]])
@pytest.mark.parametrize("num_slices", [2])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["out_BFLOAT16", "out_BFLOAT8_B"],
)
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_sharded_partial_op(
    device,
    H,
    num_cores,
    num_slices,
    activations_dtype,
    output_dtype,
    async_mode,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    device.enable_async(async_mode)
    grid_size = (8, 8)
    in0_shape = [1, 1, H, 64]
    W = in0_shape[-1]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.ones(in0_shape).bfloat16().float()
    out_initial = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_initial, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    height_shard_spec = [H // 2, W]

    for slice_index in range(num_slices):
        in0_t_slice = ttnn.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            height_shard_spec,
            num_slices,
            slice_index,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        ttnn.sharded_to_interleaved_partial(
            in0_t_slice,
            out_tt_tensor,
            num_slices,
            slice_index,
            memory_config=interleaved_mem_config,
        )

    pt_out = in0

    tt_out = ttnn.to_torch(out_tt_tensor)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("H, W, num_cores", [[32 * 32, 16 * 32, 64]])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["out_BFLOAT16", "out_BFLOAT8_B"],
)
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_block_sharded_partial_op(
    device, H, W, num_cores, activations_dtype, output_dtype, async_mode, function_level_defaults, use_program_cache
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    device.enable_async(async_mode)
    grid_size = (8, 8)
    in0_shape = [1, 1, H, W]
    W = in0_shape[-1]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    out_initial = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_initial, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    block_shard_spec = [2 * 32, 2 * 32]
    num_slices = 2

    for slice_index in range(num_slices):
        in0_t_slice = ttnn.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            block_shard_spec,
            num_slices,  # num_slices
            slice_index,  # slice_index
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        ttnn.sharded_to_interleaved_partial(
            in0_t_slice,
            out_tt_tensor,
            num_slices,
            slice_index,
            memory_config=interleaved_mem_config,
        )

    pt_out = in0

    tt_out = ttnn.to_torch(out_tt_tensor)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("num_cores", [64, 1], ids=["multi_core", "single_core"])
@pytest.mark.parametrize("in0_height_sharded", [True, False], ids=["in0_height_sharded", "in0_dram_interleaved"])
@pytest.mark.parametrize("out_height_sharded", [True, False], ids=["out_height_sharded", "out_dram_interleaved"])
@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "not_in_place"])
def test_bcast_hw(device, num_cores, in0_height_sharded, out_height_sharded, in_place):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    if in0_height_sharded != out_height_sharded:
        pytest.skip(f"Currently bcast hw op supports sharding if both inputs and outputs are sharded")

    scalar_shape = [1, 1, 1, 1]
    in0_shape = [1, 1, num_cores * 32, 128]
    height_shard_spec = [32, 128]

    torch_scalar = torch.randn(scalar_shape).bfloat16().float()
    torch_in0 = torch.randn(in0_shape).bfloat16().float()

    height_sharded_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG

    tt_scalar_dram = ttnn.from_torch(
        torch_scalar, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_in0_dram = ttnn.from_torch(
        torch_in0, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if out_height_sharded:
        out_mem_config = height_sharded_memory_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    if in0_height_sharded:
        compute_with_storage_grid_size = device.compute_with_storage_grid_size()
        device_grid_size = ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)

        tt_in0_height_sharded = ttnn.to_memory_config(
            tt_in0_dram,
            ttnn.create_sharded_memory_config(
                height_shard_spec,
                core_grid=device_grid_size,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )

        tt_out = ttnn.multiply(
            tt_in0_height_sharded,
            tt_scalar_dram,
            memory_config=out_mem_config,
        )
        tt_in0_height_sharded.deallocate()
    else:
        tt_out = ttnn.multiply(tt_in0_dram, tt_scalar_dram, memory_config=out_mem_config)

    if out_height_sharded:
        tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)

    # Reference is out and input dram interleaved
    tt_out_ref = ttnn.multiply(
        tt_in0_dram,
        tt_scalar_dram,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_in0_dram.deallocate()

    tt_out_torch = ttnn.to_torch(tt_out)
    tt_ref_torch = ttnn.to_torch(tt_out_ref)

    passing, output = comp_pcc(tt_out_torch, tt_ref_torch)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("H, W, num_cores, num_slices", [[4 * 32, 32 * 32, 64, 2]])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["out_BFLOAT16", "out_BFLOAT8_B"],
)
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_width_sharded_partial_op(
    device,
    H,
    W,
    num_cores,
    num_slices,
    activations_dtype,
    output_dtype,
    async_mode,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    device.enable_async(async_mode)
    grid_size = (8, 8)
    in0_shape = [1, 1, H, W]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    out_initial = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_initial, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    width_shard_spec = [H // num_slices, 1 * 32]

    for slice_index in range(num_slices):
        in0_t_slice = ttnn.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            width_shard_spec,
            num_slices,
            slice_index,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        ttnn.sharded_to_interleaved_partial(
            in0_t_slice,
            out_tt_tensor,
            num_slices,
            slice_index,
            memory_config=interleaved_mem_config,
        )

    pt_out = in0

    tt_out = ttnn.to_torch(out_tt_tensor)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True])
@pytest.mark.parametrize("in1_sharded", [True])
@pytest.mark.parametrize("out_sharded", [True])
@pytest.mark.parametrize("H, num_cores", [[128 * 32, 64]])
@pytest.mark.parametrize("num_slices", [2])
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["inputs_BFLOAT16", "inputs_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["out_BFLOAT16", "out_BFLOAT8_B"],
)
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_partial_sharded_op_binary(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    H,
    num_cores,
    num_slices,
    activations_dtype,
    output_dtype,
    async_mode,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    device.enable_async(async_mode)
    grid_size = (8, 8)
    in0_shape = [1, 1, H, 96]
    in1_shape = in0_shape
    W = in0_shape[-1]

    if out_sharded and not in0_sharded and not in1_sharded and H == 64:
        pytest.skip("Unsupported sharding config")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    out_values = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    out_tt_tensor = torch2tt_tensor(
        out_values, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype
    )

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config
    height_shard_spec = [32, W]

    for slice_index in range(num_slices):
        in0_t_slice = ttnn.interleaved_to_sharded_partial(
            in0_t,
            grid_size,
            height_shard_spec,
            num_slices,
            slice_index,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        in1_t_slice = ttnn.interleaved_to_sharded_partial(
            in1_t,
            grid_size,
            height_shard_spec,
            num_slices,
            slice_index,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        sliced_tensor = ttnn.add(in0_t_slice, in1_t_slice, memory_config=output_mem_config, dtype=output_dtype)
        ttnn.sharded_to_interleaved_partial(
            sliced_tensor, out_tt_tensor, num_slices, slice_index, memory_config=interleaved_mem_config
        )

    pt_out = in0 + in1

    tt_out = ttnn.to_torch(out_tt_tensor)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("H, num_cores", [[25088, 98]])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_binary(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    H,
    num_cores,
    activations_dtype,
    output_dtype,
    function_level_defaults,
):
    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, H, 64]
    in1_shape = in0_shape
    W = in0_shape[-1]

    if out_sharded and not in0_sharded and not in1_sharded and H == 25088:
        pytest.skip("Unsupported sharding config")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [H // num_cores, W],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    if in1_sharded:
        in1_t = ttnn.interleaved_to_sharded(
            in1_t,
            grid_size,
            [H // num_cores, W],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    output_t = ttnn.add(in0_t, in1_t, memory_config=output_mem_config, dtype=output_dtype)
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 + in1

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def test_sharded_program_cache(device, use_program_cache, function_level_defaults):
    grid_size = device.compute_with_storage_grid_size()
    num_cores = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 25088
    W = 64
    x = torch.ones((N, C, H, W)).bfloat16().float()
    x2 = torch.zeros((N, C, H, W)).bfloat16().float()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttnn.BufferType.L1,
            ),
        )
    )

    yt = ttnn.interleaved_to_sharded(
        xt,
        grid_size,
        [H // num_cores, W],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    zt = ttnn.sharded_to_interleaved(
        yt,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    )

    xt2 = (
        ttnn.Tensor(
            x2.reshape(-1).tolist(),
            x2.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttnn.BufferType.L1,
            ),
        )
    )

    yt2 = ttnn.interleaved_to_sharded(
        xt2,
        grid_size,
        [H // num_cores, W],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    zt2 = ttnn.sharded_to_interleaved(
        yt2,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    )
    zt = ttnn.sharded_to_interleaved(
        yt,
        ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        ),
    )

    tt_og = xt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_og2 = xt2.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    tt_got_back = zt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_got_back2 = zt2.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    eq = torch.equal(tt_og, tt_got_back)
    assert eq
    eq = torch.equal(tt_og2, tt_got_back2)
    assert eq


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("K", [256, 512])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_2d(
    device,
    in0_sharded,
    out_sharded,
    M,
    N,
    K,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (8, 5)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=10,
        per_core_N=4,
        transpose_mcast=False,
        fused_activation=None,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_interleaved"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_interleaved"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_interleaved"])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_2d_in0_height_sharded_in1_width_sharded(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    activations_dtype,
    weights_dtype,
    output_dtype,
    function_level_defaults,
):
    M = 6 * 32
    N = 12 * 32
    K = 2 * 32

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (6, 6)
    compute_grid_size = device.compute_with_storage_grid_size()

    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_block_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    # Generate the tensor
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            ttnn.CoreCoord(1, grid_size[0]),
            [M // grid_size[0], K],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    if in1_sharded:
        in1_t = ttnn.interleaved_to_sharded(
            in1_t,
            ttnn.CoreCoord(grid_size[1], 1),
            [K, N // grid_size[1]],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=M // (32 * grid_size[0]),
        per_core_N=N // (32 * grid_size[1]),
        transpose_mcast=False,
        fused_activation=None,
    )
    output_mem_config = sharded_block_mem_config if out_sharded else interleaved_mem_config
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )

    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [1600])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_2d_transposed(
    device,
    in0_sharded,
    out_sharded,
    M,
    N,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    K = 256
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=5,
        per_core_N=4,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def test_resharded_binary_to_matmul(device, function_level_defaults):
    grid_size_binary = device.compute_with_storage_grid_size()
    num_cores_binary = 98
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores_binary > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores_binary} cores to run this test but core grid is {compute_grid_size}")
    grid_size_matmul = (10, 8)
    if grid_size_matmul[0] > compute_grid_size.x or grid_size_matmul[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size_matmul} grid size to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, 6272, 512]
    in1_shape = in0_shape
    weight_shape = [1, 1, 512, 256]
    bias_shape = [1, 1, 1, 256]
    H = in0_shape[-2]
    W = in0_shape[-1]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    height_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    block_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    weight = torch.randn(weight_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config)
    weight_t = torch2tt_tensor(weight, device, tt_memory_config=interleaved_mem_config)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config)[0]

    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    in1_t = ttnn.interleaved_to_sharded(
        in1_t,
        grid_size_binary,
        [H // num_cores_binary, W],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    output_binary_t = ttnn.add(in0_t, in1_t, memory_config=interleaved_mem_config)
    output_binary_t = ttnn.interleaved_to_sharded(
        output_binary_t,
        grid_size_matmul,
        [math.ceil((H // 32) / grid_size_matmul[0]) * 32, W // grid_size_matmul[1]],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size_matmul,
        in0_block_w=2,
        out_subblock_h=5,
        out_subblock_w=1,
        per_core_M=20,
        per_core_N=1,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_matmul_t = ttnn.linear(
        output_binary_t,
        weight_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=block_sharded_mem_config,
    )
    output_matmul_t = ttnn.sharded_to_interleaved(output_matmul_t, interleaved_mem_config)

    tt_out = tt2torch_tensor(output_matmul_t)

    pt_out = (in0 + in1) @ weight

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_untilize_padded_shard(in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 6272
    W = 256

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    yt = ttnn.untilize(
        xt,
        memory_config=out_mem_config,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    if dtype == ttnn.bfloat16:
        passing, output = comp_equal(x, tt_got_back)
    else:
        passing, output = comp_pcc(x, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_binary_padded_shard(
    in_sharded, out_sharded, activations_dtype, output_dtype, device, function_level_defaults
):
    grid_size = (10, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 1568
    W = 1024

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.ones((N, C, H, W)).bfloat16()
    y = torch.ones((N, C, H, W)).bfloat16() * 2

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            activations_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    yt = (
        ttnn.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            activations_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )
        yt = ttnn.interleaved_to_sharded(
            yt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    zt = ttnn.add(xt, yt, memory_config=out_mem_config, dtype=output_dtype)

    if out_sharded:
        zt = ttnn.sharded_to_interleaved(
            zt,
            interleaved_mem_config,
        )

    tt_got_back = zt.cpu().to_torch()

    passing, output = comp_equal(x + y, tt_got_back)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [False], ids=["out_unsharded"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_block_sharded_untilize_with_unpadding(in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (7, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N = 1
    C = 1
    H = 416
    W = 512

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [
                math.ceil((xt.get_legacy_shape()[-2] // 32) / grid_size[0]) * 32,
                xt.get_legacy_shape()[-1] // grid_size[1],
            ],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    yt = ttnn.untilize_with_unpadding(
        xt,
        output_tensor_end=ttnn.Shape([0, 0, 391, 511]),
        memory_config=out_mem_config,
    )

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = x[..., :392, :512]

    if dtype == ttnn.bfloat16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)

    logger.info(output)

    assert passing


@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize(
    "shape, output_H, out_sharded",
    [
        [(8, 1, 32, 2048), 1, True],
        [(1, 1, 32, 1024), 8, False],
        [(16, 1, 32, 2048), 1, True],
        [(1, 1, 32, 1024), 16, False],
    ],
    ids=[
        "batched_8_shape_out_sharded",
        "unbatched_8_shape_out_interleaved",
        "batched_16_shape_out_sharded",
        "unbatched_16_shape_out_interleaved",
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_width_sharded_untilize_with_unpadding(
    shape, output_H, in_sharded, out_sharded, dtype, device, function_level_defaults
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N, C, H, W = shape

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    yt = ttnn.untilize_with_unpadding(
        xt,
        output_tensor_end=ttnn.Shape([N - 1, C - 1, output_H - 1, W - 1]),
        memory_config=out_mem_config,
    )

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to_torch()

    y = x[..., :output_H, :]
    if dtype == ttnn.bfloat16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("input_shape", [[8, 1, 49, 2048], [1, 1, 8, 2048], [16, 1, 49, 2048], [1, 1, 16, 2048]])
@pytest.mark.parametrize("sharding_config", [(True, True), (False, False)], ids=["both_sharded", "both_interleaved"])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_tilize_with_val_padding(input_shape, sharding_config, output_dtype, device, function_level_defaults):
    grid_size = (8, 4)
    in_sharded, out_sharded = sharding_config
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    N, C, H, W = input_shape

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.arange(N * C * H * W).reshape((N, C, H, W)).bfloat16()

    xt = ttnn.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(
        device,
        interleaved_mem_config,
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    yt = ttnn.tilize_with_val_padding(
        xt,
        ttnn.Shape([N, C, roundup32(H), W]),
        1.0,
        memory_config=out_mem_config,
        dtype=output_dtype,
        use_multicore=True,
    )

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    y = torch.nn.functional.pad(x, [0, 0, 0, roundup32(H) - H], "constant", 1.0)

    if output_dtype == ttnn.bfloat16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.parametrize("in_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_reduce_h(N, in_sharded, out_sharded, dtype, device, function_level_defaults):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    C = 1
    H = 64
    W = 2048

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    out_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    x = torch.randn((N, C, H, W)).bfloat16()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    if in_sharded:
        xt = ttnn.interleaved_to_sharded(
            xt,
            grid_size,
            [N * C * H, W // (grid_size[0] * grid_size[1])],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    yt = ttnn.max(xt, 2, memory_config=out_mem_config)

    if out_sharded:
        yt = ttnn.sharded_to_interleaved(
            yt,
            interleaved_mem_config,
        )

    tt_got_back = yt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()[:, :, :1, :]

    y = torch.max(x, 2, True)[0]

    if dtype == ttnn.bfloat16:
        passing, output = comp_equal(y, tt_got_back)
    else:
        passing, output = comp_pcc(y, tt_got_back, 0.999)
    logger.info(output)

    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize("M", [32, 64])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("K", [2048, 4096])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_sharded_matmul_1d_in0(
    device,
    in0_sharded,
    out_sharded,
    M,
    K,
    N,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    grid_size = (8, 4)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    if activations_dtype != weights_dtype and is_wormhole_b0():
        pytest.skip("WH does not work with mixed precision")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, K // num_cores],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // 32,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(output)
    assert passing


# Have at least one example of 1d matmul with in1 mcasted that runs on WH
def test_sharded_matmul_1d_in1_wormhole(device, function_level_defaults):
    M = 4096
    K = 64
    N = 256
    grid_size = (8, 4)
    num_cores = grid_size[0] * grid_size[1]
    dtype = ttnn.bfloat16

    grid_size = device.compute_with_storage_grid_size()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=dtype)[0]

    output_mem_config = sharded_mem_config

    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // num_cores, K],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // 32 // num_cores,
        per_core_N=N // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=dtype,
    )
    output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)
    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_sharded", [True, False], ids=["in0_sharded", "in0_unsharded"])
@pytest.mark.parametrize("in1_sharded", [True, False], ids=["in1_sharded", "in1_unsharded"])
@pytest.mark.parametrize("out_sharded", [True, False], ids=["out_sharded", "out_unsharded"])
@pytest.mark.parametrize(
    "B, H, M, K, N, out_subblock_h, out_subblock_w",
    [[12, 16, 384, 64, 384, 1, 6], [12, 16, 384, 384, 64, 4, 2]],
)
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
def test_sharded_matmul_no_mcast(
    device,
    in0_sharded,
    in1_sharded,
    out_sharded,
    B,
    H,
    M,
    K,
    N,
    out_subblock_h,
    out_subblock_w,
    activations_dtype,
    function_level_defaults,
):
    grid_size = (12, 8)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    in0_shape = [B, H, M, K]
    in1_shape = [B, H, K, N]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [B * H * M // num_cores, K],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )
    if in1_sharded:
        in1_t = ttnn.interleaved_to_sharded(
            in1_t,
            grid_size,
            [B * H * K // num_cores, N],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=B * H * M // num_cores // 32,
        per_core_N=N // 32,
    )

    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("in0_shape, grid_size", [([12, 16, 384, 64], (12, 8)), ([1, 32, 32, 64], (1, 4))])
@pytest.mark.parametrize("in0_sharded, out_sharded", [[True, True], [False, False]], ids=["sharded", "unsharded"])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
def test_sharded_concat_heads(
    device,
    in0_shape,
    grid_size,
    in0_sharded,
    out_sharded,
    activations_dtype,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")
    num_cores = grid_size[0] * grid_size[1]
    B = in0_shape[0]
    num_heads = in0_shape[1]
    seq_len = in0_shape[2]
    head_dim = in0_shape[3]

    in0_shape = [B, num_heads, seq_len, head_dim]

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [B * num_heads * seq_len // num_cores, head_dim],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    output_t = ttnn.experimental.nlp_concat_heads(
        in0_t,
        memory_config=output_mem_config,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = torch.transpose(in0, -3, -2).reshape([B, 1, seq_len, num_heads * head_dim])

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


def run_reshard_test(
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_shape,
    input_shard_orientation,
    input_sharding_scheme,
    output_shard_grid,
    output_shard_shape,
    output_shard_orientation,
    output_sharding_scheme,
    tt_dtype,
):
    compute_grid = ttnn.CoreCoord(input_shard_grid[0], input_shard_grid[1])
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), compute_grid)})

    compute_grid = ttnn.CoreCoord(output_shard_grid[0], output_shard_grid[1])
    output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), compute_grid)})
    output_shard_spec = ttnn.ShardSpec(output_shard_grid, output_shard_shape, output_shard_orientation, False)
    output_mem_config = ttnn.MemoryConfig(output_sharding_scheme, ttnn.BufferType.L1, output_shard_spec)
    if input_layout == ttnn.ROW_MAJOR_LAYOUT and tt_dtype == ttnn.bfloat8_b:
        pytest.skip("Illegal layout/dtype config")

    dram_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    torch_tensor = torch.randn(input_shape).bfloat16()
    tt_tensor_sharded = ttnn.Tensor(torch_tensor, tt_dtype).to(input_layout)
    tt_tensor_sharded = tt_tensor_sharded.to(device, dram_memory_config)
    tt_tensor_sharded = ttnn.interleaved_to_sharded(
        tt_tensor_sharded,
        input_shard_grid,
        input_shard_shape,
        input_sharding_scheme,
        input_shard_orientation,
        output_dtype=tt_dtype,
    )

    tt_tensor_reshard = ttnn.reshard(tt_tensor_sharded, output_mem_config)

    tt_tensor_interleaved = ttnn.sharded_to_interleaved(
        tt_tensor_reshard,
        dram_memory_config,
    )

    tt_tensor_interleaved = tt_tensor_interleaved.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    torch_tensor_after_round_trip = tt_tensor_interleaved.to_torch()

    return torch_tensor, torch_tensor_after_round_trip


@pytest.mark.parametrize(
    "input_shape, shard_scheme",
    [
        ([1, 1, 128, 256], ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ([1, 1, 128, 256], ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ([1, 1, 128, 256], ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_sharded_to_from_l1(device, input_shape, shard_scheme, shard_orientation):
    input_dtype = ttnn.bfloat16
    output_dtype = ttnn.bfloat16

    assert input_shape[-2] % 32 == 0
    assert input_shape[-1] % 32 == 0
    if shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        grid_x = input_shape[-2] // 32
        grid_y = 1
        shard_shape = [input_shape[-2] // grid_x, input_shape[-1] // grid_y]
    elif shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        grid_x = input_shape[-1] // 32
        grid_y = 1
        shard_shape = [input_shape[-2] // grid_y, input_shape[-1] // grid_x]
    elif shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        grid_x = input_shape[-1] // 32
        grid_y = input_shape[-2] // 32
        shard_shape = [input_shape[-2] // grid_y, input_shape[-1] // grid_x]

        if shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
            grid_x, grid_y = grid_y, grid_x
    else:
        assert False, f"Unsupported {shard_scheme}"

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    shard_halo = False
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
    mem_config = ttnn.MemoryConfig(shard_scheme, ttnn.BufferType.L1, shard_spec)

    volume = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
    golden = torch.arange(volume).reshape(input_shape).bfloat16()
    ttl_golden = ttnn.Tensor(golden.reshape(-1).tolist(), golden.shape, input_dtype, ttnn.ROW_MAJOR_LAYOUT)

    ## TEST to/from ##
    ttl_device = ttl_golden.to(device, mem_config)
    result = ttl_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    passing, output = comp_equal(result, golden)
    assert passing


@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
        ttnn.bfloat4_b,
    ],
)
@pytest.mark.parametrize("y", [256, 512, 1024, 2048])
def test_interleaved_2_sharded_L1(device, dtype, y):
    input_dtype = dtype
    shard_scheme = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    compute_grid_size = device.compute_with_storage_grid_size()
    if 64 > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {64} cores to run this test but core grid is {compute_grid_size}")

    if y == 2048 and dtype == ttnn.float32 and not is_wormhole_b0():
        pytest.skip(f"Can fit this case on GS")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )

    x = torch.randn((1, 1, y, 144 * 32)).bfloat16().float()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            input_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    yt = ttnn.interleaved_to_sharded(xt, shard_grid, (y // 8, 18 * 32), shard_scheme, ttnn.ShardOrientation.ROW_MAJOR)


@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
        ttnn.bfloat4_b,
    ],
)
@pytest.mark.parametrize("y", [256, 512, 1024, 2048])
def test_interleaved_2_sharded_DRAM(device, dtype, y):
    input_dtype = dtype
    shard_scheme = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    compute_grid_size = device.compute_with_storage_grid_size()
    if 64 > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {64} cores to run this test but core grid is {compute_grid_size}")

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    x = torch.randn((1, 1, y, 144 * 32)).bfloat16().float()

    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            input_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(
            device,
            interleaved_mem_config,
        )
    )

    yt = ttnn.interleaved_to_sharded(xt, shard_grid, (y // 8, 18 * 32), shard_scheme, ttnn.ShardOrientation.ROW_MAJOR)
