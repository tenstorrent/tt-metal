# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from loguru import logger
from typing import Union, Tuple

import torch
import torch.nn as nn
import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout

TILE_WIDTH = 32


def get_shard_grid_from_num_cores(device, ncores: Union[int, Tuple[int, int]]) -> ttnn.CoreRangeSet:
    device_grid = device.compute_with_storage_grid_size()
    max_grid_size = (device_grid.y, device_grid.x)
    if isinstance(ncores, int):
        if ncores % max_grid_size[1] == 0:
            core_grid = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
            grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
            return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        else:
            if ncores < max_grid_size[1]:
                core_grid = ttnn.CoreGrid(y=1, x=ncores)
                grid_coord = ttnn.CoreCoord(core_grid.x - 1, 0)
                return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            else:
                core_grid_1 = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
                core_grid_2 = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
                grid_coord_1 = ttnn.CoreCoord(core_grid_1.x - 1, core_grid_1.y - 1)
                grid_coord_2 = ttnn.CoreCoord(core_grid_2.x - 1, core_grid_2.y - 1)
                return ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord_1),
                        ttnn.CoreRange(ttnn.CoreCoord(0, grid_coord_2.y), grid_coord_2),
                    }
                )
    elif isinstance(ncores, tuple):
        ncores_h, ncores_w = ncores
        assert ncores_h <= max_grid_size[0]
        assert ncores_w <= max_grid_size[1]
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(ncores_w - 1, ncores_h - 1),
                )
            }
        )
    else:
        raise ValueError("Invalid ncores")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 640, 16, 16],
        [2, 1280, 16, 16],
        [2, 640, 16, 16],
        [1, 256, 28, 28],
        [1, 512, 14, 14],
        [1, 64, 32, 32],
        [2, 32, 64, 64],
        [1, 128, 32, 32],
        [1, 64, 64, 64],
        [2, 64, 32, 32],
        [1, 32, 96, 96],
        [1, 96, 32, 32],
        [1, 32, 80, 32],
    ],
)
@pytest.mark.parametrize("scale_h", [2, 3])
@pytest.mark.parametrize("scale_w", [2, 3])
@pytest.mark.parametrize("memory_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("run_twice", [False])
def test_upsample_nearest_interleaved(device, input_shapes, scale_h, scale_w, memory_layout, run_twice):
    batch_size, num_channels, height, width = input_shapes
    torch.manual_seed(0)

    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    tt_input = input.permute(0, 2, 3, 1)
    input_tensor = ttnn.from_torch(tt_input, device=device, layout=memory_layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if input_tensor.padded_shape != input_tensor.shape and memory_layout == ttnn.TILE_LAYOUT:
        pytest.skip("Disabled until different logical and padded shapes are supported for TILE_LAYOUT")

    scale_factor = (scale_h, scale_w)
    torch_upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = torch_upsample(input)

    scale_factor = (scale_h, scale_w)

    output_tensor = ttnn.upsample(input_tensor, scale_factor)

    if run_twice:
        ttnn.deallocate(output_tensor, True)
        output_tensor = ttnn.upsample(input_tensor, scale_factor)

    output_tensor = ttnn.to_torch(output_tensor)

    torch_result = torch_result.permute(0, 2, 3, 1)
    pcc_passed, pcc_message = assert_with_pcc(torch_result, output_tensor)
    logger.info(pcc_message)
    allclose = torch.allclose(output_tensor, torch_result)
    isclose = torch.all(torch.isclose(output_tensor, torch_result))
    isequal = torch.equal(output_tensor, torch_result)
    assert allclose
    assert isclose
    assert isequal


def upsample_multicore_common(
    device,
    input_shape,
    scale_h,
    scale_w,
    shard_strategy,
    shard_orientation,
    mode=None,
    core_range=None,
    math_fidelity=None,
    math_approx_mode=None,
    run_twice=False,
):
    ## input shape is N C H W
    batch_size, num_channels, height, width = input_shape
    torch.manual_seed(0)
    input = torch.randn(input_shape, dtype=torch.bfloat16)

    ## golden reference using torch
    scale_factor = (scale_h, scale_w)
    if mode == "bilinear":
        torch_upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
    else:
        torch_upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = torch_upsample(input)

    ## permute to N H W C, which is what the upsample op expects
    tt_input = input.permute(0, 2, 3, 1)

    num_bytes = 2  ## only BFLOAT16 is supported

    ## calculate ncores, corresponding grid_size and in_shard_shape based on the input_shape
    ncores = None
    shard_grid = None
    device_grid = device.compute_with_storage_grid_size()
    max_grid_size = (device_grid.y, device_grid.x)
    if core_range != None:
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(core[0][0], core[0][1]), ttnn.CoreCoord(core[1][0], core[1][1]))
                for core in core_range
            }
        )
        if shard_strategy == ttnn.ShardStrategy.BLOCK:
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
                ncores = (core_range[0][1][1] - core_range[0][0][1] + 1, core_range[0][1][0] - core_range[0][0][0] + 1)
            elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
                ncores = (core_range[0][1][0] - core_range[0][0][0] + 1, core_range[0][1][1] - core_range[0][0][1] + 1)
        elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
            ncores = shard_grid.num_cores()
        else:
            raise ValueError("Invalid shard strategy")

    else:
        if shard_strategy == ttnn.ShardStrategy.HEIGHT:
            max_nshards = min(batch_size * height * width, max_grid_size[0] * max_grid_size[1])
            nshards = max_nshards
            while nshards > 0:
                if batch_size * height * width % nshards == 0:
                    break
                nshards -= 1
            ncores = nshards
        elif shard_strategy == ttnn.ShardStrategy.BLOCK:
            max_nshards_h = min(batch_size * height * width, max_grid_size[0])  ## height along NHW
            max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C
            ## find nshards_h along NHW
            nshards_h = max_nshards_h
            while nshards_h > 0:
                if batch_size * height % nshards_h == 0:
                    break
                nshards_h -= 1
            ## find nshards_w along C
            nshards_w = max_nshards_w
            while nshards_w > 0:
                ## make sure: 1. nshards_w divides num_channels, and 2. shard_shape[1] is aligned to 32B
                if num_channels % nshards_w == 0 and math.ceil(num_channels * num_bytes / nshards_w) % TILE_WIDTH == 0:
                    break
                nshards_w -= 1
            if nshards_w == 0 or nshards_h == 0:
                pytest.skip("nshards_h or nshards_w is 0")

            ncores = (nshards_h, nshards_w)
        shard_grid = get_shard_grid_from_num_cores(device, ncores)

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED

    ## input shard
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        shard_height = math.ceil(batch_size * height * width / ncores[0])
        shard_width = math.ceil(num_channels / ncores[1])
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        shard_height = math.ceil(batch_size * height * width / ncores)
        shard_width = num_channels
    shard_shape = (shard_height, shard_width)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)

    scale_factor = (scale_h, scale_w)
    input_tensor = ttnn.from_torch(tt_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
    if mode == "bilinear":
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=False,
        )
        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor,
            mode="bilinear",
            compute_kernel_config=compute_kernel_config,
        )
        if run_twice:
            ttnn.deallocate(output_tensor, True)
            output_tensor = ttnn.upsample(
                input_tensor,
                scale_factor,
                mode="bilinear",
                compute_kernel_config=compute_kernel_config,
            )
    else:
        output_tensor = ttnn.upsample(input_tensor, scale_factor)
        if run_twice:
            ttnn.deallocate(output_tensor, True)
            output_tensor = ttnn.upsample(input_tensor, scale_factor)
    output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    return (torch_result, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 1280, 4, 4],  # 256x256
        [2, 640, 16, 16],
        [2, 1280, 8, 8],  # 512x512
        [2, 1280, 16, 16],
        [1, 64, 132, 10],
        [1, 32, 8, 8],
        [2, 640, 32, 32],
        # some random shapes
        [1, 32, 5, 4],
        [3, 32, 4, 4],
        [5, 64, 5, 5],
        [1, 128, 5, 8],
        [1, 32, 5, 4],
        [1, 64, 128, 17],
        [1, 64, 132, 19],
        [1, 8, 28, 28],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("scale_h", [2, 3])
@pytest.mark.parametrize("scale_w", [2, 3])
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("run_twice", [False])
def test_upsample_multicore(device, input_shape, scale_h, scale_w, shard_strategy, shard_orientation, run_twice):
    if (shard_strategy == ttnn.ShardStrategy.BLOCK) and (shard_orientation == ttnn.ShardOrientation.COL_MAJOR):
        pytest.skip("Disabled until illegal shard configs are fixed (#17795)")

    (torch_result, output_tensor) = upsample_multicore_common(
        device,
        input_shape,
        scale_h,
        scale_w,
        shard_strategy,
        shard_orientation,
        core_range=None,
        run_twice=run_twice,
    )
    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)

    isequal = torch.equal(output_tensor, torch_result)

    assert isequal


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 192, 12, 12],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("scale_h", [2, 3])
@pytest.mark.parametrize("scale_w", [2])
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "core_range",
    [
        [((1, 1), (6, 6))],
        [((2, 2), (5, 5))],
        [((1, 1), (3, 3)), ((4, 4), (6, 6))],
        [((2, 2), (4, 5)), ((5, 3), (7, 6))],
    ],
)
@pytest.mark.parametrize("run_twice", [False])
def test_upsample_multicore_corerange(
    device, input_shape, scale_h, scale_w, shard_strategy, shard_orientation, core_range, run_twice
):
    if (shard_strategy == ttnn.ShardStrategy.BLOCK) and (shard_orientation == ttnn.ShardOrientation.COL_MAJOR):
        pytest.skip("Disabled until illegal shard configs are fixed (#17795)")

    if (len(core_range) != 1) and (shard_strategy == ttnn.ShardStrategy.BLOCK):
        pytest.skip("illegal core range for BLOCK strategy")

    (torch_result, output_tensor) = upsample_multicore_common(
        device,
        input_shape,
        scale_h,
        scale_w,
        shard_strategy,
        shard_orientation,
        core_range=core_range,
        run_twice=run_twice,
    )
    ## compare the results
    torch_result = torch_result.permute(0, 2, 3, 1)

    isequal = torch.equal(output_tensor, torch_result)

    assert isequal


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, num_channels, height, width, scale_h, scale_w",
    (
        (1, 1280, 8, 8, 2, 2),
        (1, 1280, 8, 8, 3, 3),
        (7, 32, 7, 13, 5, 3),
        (1, 256, 4, 4, 3, 5),
        (1, 256, 32, 32, 4, 4),
        (2, 32, 128, 128, 2, 2),
        (7, 64, 64, 64, 2, 2),
        (1, 32, 24, 11, 2, 2),
        (3, 32, 43, 17, 3, 7),
        (4, 64, 13, 65, 3, 5),
        (1, 192, 33, 33, 3, 4),
        (1, 288, 33, 33, 4, 3),
        (1, 32, 10, 11, 2, 2),
        (1, 256, 2, 2, 2, 2),
        (1, 256, 2, 2, 3, 3),
        (1, 256, 32, 32, 5, 4),
        (1, 256, 128, 128, 3, 1),
        (1, 72, 8, 8, 5, 7),
        (1, 288, 8, 8, 2, 4),
        (1, 1024, 8, 8, 2, 2),
        (1, 256, 28, 28, 3, 2),
        (1, 512, 14, 14, 2, 2),
        (2, 32, 16, 16, 2, 2),
        (4, 64, 48, 48, 3, 3),
        (64, 32, 4, 4, 2, 2),
    ),
)
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("math_approx_mode", [False])
@pytest.mark.parametrize("run_twice", [False])
def test_bilinear_multi_core(
    device,
    batch_size,
    num_channels,
    height,
    width,
    scale_h,
    scale_w,
    shard_strategy,
    math_fidelity,
    math_approx_mode,
    run_twice,
):
    TILE_WIDTH = 32

    num_channels_padded = num_channels
    if num_channels % TILE_WIDTH != 0:
        num_channels_padded = num_channels + (TILE_WIDTH - num_channels % TILE_WIDTH)

    input_shape = [batch_size, num_channels_padded, height, width]
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    torch_result, output_tensor = upsample_multicore_common(
        device,
        input_shape,
        scale_h,
        scale_w,
        shard_strategy,
        shard_orientation,
        mode="bilinear",
        math_fidelity=math_fidelity,
        math_approx_mode=math_approx_mode,
        run_twice=run_twice,
    )

    torch_result = torch_result.permute(0, 2, 3, 1)
    torch_result = torch_result[:, :, :, 0:num_channels]
    output_tensor = output_tensor[:, :, :, 0:num_channels]
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_result, output_tensor, pcc=0.999)
    allclose = torch.allclose(output_tensor, torch_result, atol=1e-1, rtol=5e-2)

    logger.info(pcc_msg)

    assert allclose
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "scale_h, scale_w",
    (
        (2, 2),
        (4, 4),
    ),
)
@pytest.mark.parametrize(
    "batch_size, channels, height, width, core_grid, shard_height, shard_width, shard_strategy",
    (
        (
            1,
            32,
            14,
            2,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                }
            ),
            16,
            32,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            1,
            128,
            132,
            20,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(1, 5)),
                }
            ),
            64,
            128,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            1,
            64,
            14,
            2,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                }
            ),
            16,
            32,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            1,
            128,
            13,
            13,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
                }
            ),
            3,
            128,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
    ),
)
@pytest.mark.parametrize("run_twice", [False])
def test_nearest_upsample_with_uneven_input_shards(
    device,
    batch_size,
    channels,
    height,
    width,
    scale_h,
    scale_w,
    core_grid,
    shard_height,
    shard_width,
    shard_strategy,
    run_twice,
):
    if device.core_grid.x * device.core_grid.y < core_grid.num_cores():
        pytest.skip("Not enough cores for specified core grid")

    assert (
        shard_height * core_grid.num_cores() > batch_size * height * width
    ), "Expected all test cases in this test suite to contain uneven shards (i.e. physical size > logical size)"
    if shard_strategy == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        assert shard_width == channels, "Shard width must match number of input channels when height sharding"

    input_shape = [batch_size, channels, height, width]
    input = torch.randn(input_shape, dtype=torch.bfloat16)
    input_nhw_c = input.permute(0, 2, 3, 1)

    input_shard_shape = (shard_height, shard_width)

    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(shard_strategy, ttnn.BufferType.L1, input_shard_spec)

    tt_input_tensor = ttnn.from_torch(input_nhw_c, device=device, memory_config=input_mem_config)
    output_tensor = ttnn.upsample(tt_input_tensor, (scale_h, scale_w), mode="nearest")

    if run_twice:
        ttnn.deallocate(output_tensor, True)
        output_tensor = ttnn.upsample(tt_input_tensor, (scale_h, scale_w), mode="nearest")

    output_tensor = ttnn.to_torch(output_tensor)

    upsample = nn.Upsample(scale_factor=(scale_h, scale_w), mode="nearest")
    torch_result = upsample(input)
    torch_result = torch_result.permute(0, 2, 3, 1)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_result, output_tensor, pcc=0.99999)
    allclose = torch.allclose(output_tensor, torch_result, atol=1e-1, rtol=1e-1)
    logger.info(pcc_msg)

    assert allclose
    assert passing


@pytest.mark.parametrize(
    "input_shape, scale_factor_h, scale_factor_w",
    [
        # Basic fractional upscaling (NCHW format: [N, C, H, W])
        ([1, 64, 8, 8], 1.5, 1.5),
        ([1, 128, 16, 16], 1.25, 1.25),
        ([1, 32, 8, 8], 2.5, 2.5),
        # Asymmetric float scales
        ([1, 64, 8, 16], 1.5, 2.0),
        ([1, 128, 16, 8], 2.0, 1.5),
        # Downscaling
        ([1, 64, 16, 16], 0.5, 0.5),
        ([1, 128, 32, 32], 0.75, 0.75),
        # Mixed upscale/downscale
        ([1, 64, 8, 16], 2.0, 0.5),
        ([1, 128, 16, 8], 0.5, 2.0),
        # Typical ML shapes
        ([1, 64, 28, 28], 2.5, 2.5),
    ],
)
def test_upsample_nearest_float_interleaved(device, input_shape, scale_factor_h, scale_factor_w):
    """Test upsample with float scale factors using interleaved memory."""
    torch.manual_seed(0)

    # Input shape is NCHW, create tensor and permute to NHWC for ttnn
    input_nchw = torch.randn(input_shape, dtype=torch.bfloat16)
    input_nhwc = input_nchw.permute(0, 2, 3, 1)

    # PyTorch reference (uses NCHW)
    torch_result_nchw = nn.functional.interpolate(
        input_nchw, scale_factor=(scale_factor_h, scale_factor_w), mode="nearest"
    )
    torch_result = torch_result_nchw.permute(0, 2, 3, 1)

    input_tensor = ttnn.from_torch(
        input_nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    output_tensor = ttnn.upsample(input_tensor, [scale_factor_h, scale_factor_w], mode="nearest")
    output_torch = ttnn.to_torch(output_tensor)

    assert list(output_torch.shape) == list(
        torch_result.shape
    ), f"Shape mismatch: expected {list(torch_result.shape)}, got {list(output_torch.shape)}"

    is_equal = torch.equal(output_torch, torch_result)
    if not is_equal:
        pcc_passed, pcc_message = assert_with_pcc(torch_result, output_torch, pcc=0.9999)
        logger.info(pcc_message)
        assert pcc_passed, f"PCC check failed: {pcc_message}"
