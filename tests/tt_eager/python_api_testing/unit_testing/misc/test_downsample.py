# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from loguru import logger

import ttnn
from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
    create_conv_bias_tensor,
    create_conv_weight_tensor_special_padding,
)
from models.utility_functions import skip_for_blackhole
import torch


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, stride_h, stride_w, num_cores, grid_size, height_sharded",
    (
        # (10, 64, 64, 16, 16, 2, 2, 20, (10,2), False),
        # (10, 64, 64, 16, 16, 1, 1, 20, (10,2), False),
        # (8, 64, 64, 56, 56, 1, 1, 98, (12,9), True),
        # (8, 256, 256, 56, 56, 2, 2, 98, (12, 9), True),
        # (8, 512, 512, 28, 28, 2, 2, 80, (10, 8), False),
        (8, 1024, 1024, 14, 14, 2, 2, 56, (8, 7), False),
        # (16, 256, 256, 56, 56, 2, 2, 98, (12, 9), True),
        # (16, 512, 512, 28, 28, 2, 2, 80, (11, 8), False),
        # (16, 1024, 1024, 14, 14, 2, 2, 56, (9, 8), False),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_run_downsample(
    device,
    use_program_cache,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    stride_h,
    stride_w,
    num_cores,
    grid_size,
    height_sharded,
    dtype,
):
    if batch_size > 8 and dtype != ttnn.bfloat8_b:
        pytest.skip("Batch > 8 must be run fully bfp8")
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    assert input_channels % 32 == 0
    assert output_channels % 32 == 0
    assert stride_h == stride_w

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    torch.manual_seed(0)
    a_activation_shape = [batch_size, input_channels, input_height, input_width]
    A_pyt = torch.normal(mean=0, std=0.1, size=a_activation_shape).bfloat16()

    b_weights_shape = [output_channels, input_channels, 1, 1]
    B_pyt = torch.normal(mean=0, std=0.1, size=b_weights_shape).bfloat16()

    output_height = math.ceil(input_height / stride_h)
    output_width = math.ceil(input_width / stride_w)

    conv_output_shape = [batch_size, output_height, output_width, output_channels]

    # Convert NCHW to NHWC shape
    A_pyt_nhwc = torch.permute(A_pyt, (0, 2, 3, 1))
    A_pyt_nhwc = A_pyt_nhwc.reshape(1, 1, batch_size * input_height * input_width, input_channels)
    # for i in range(2):
    #    for j in range(32):
    #        logger.info(f"A_pyt_nhwc_2d[{i}][{j}]={A_pyt_nhwc[0][0][i][j]}")
    # logger.info("A_pyt_nhwc_2d[32][0]=", A_pyt_nhwc[0][0][32][0])
    a_activation_shape_nhwc = [batch_size, input_height, input_width, input_channels]
    A_cl_host = ttnn.experimental.view(
        ttnn.Tensor(A_pyt_nhwc, dtype), 1, 1, batch_size * input_height * input_width, input_channels
    )
    num_cores_height_slices = num_cores if height_sharded else grid_size[0]
    input_shape = [1, 1, _nearest_y(batch_size * input_height * input_width, 32), input_channels]
    A_cl_host = A_cl_host.pad(input_shape, (0, 0, 0, 0), 0.0)
    A_interleaved = A_cl_host.to(ttnn.TILE_LAYOUT).to(
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    assert A_interleaved.shape.with_tile_padding()[0] == 1 and A_interleaved.shape.with_tile_padding()[1] == 1

    # image flattened params
    input_2d_height = A_interleaved.shape.with_tile_padding()[2]
    input_2d_width = A_interleaved.shape.with_tile_padding()[3]
    input_2d_height_padded = _nearest_y(input_2d_height, num_cores_height_slices * 32)
    input_shard_height = (int)(input_2d_height_padded / num_cores_height_slices)
    output_2d_height_padded = _nearest_y(batch_size * output_height * output_width, num_cores_height_slices * 32)
    output_shard_height = (int)(output_2d_height_padded / num_cores_height_slices)
    logger.debug(f"input_2d_height={input_2d_height}")
    logger.debug(f"input_2d_width={input_2d_width}")
    sharded_memory_layout = (
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharded else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    )
    sharded_memory_orientation = ttnn.ShardOrientation.ROW_MAJOR if height_sharded else ttnn.ShardOrientation.COL_MAJOR
    input_shard_width = input_2d_width if height_sharded else ((int)(input_2d_width / grid_size[1]))
    logger.debug(f"grid_size={grid_size}")
    logger.debug(f"shard_memory_layout={sharded_memory_layout}")
    logger.debug(f"input_shard_height={input_shard_height}, input_shard_width={input_shard_width}")

    A_sharded = ttnn.interleaved_to_sharded(
        A_interleaved,
        grid_size,
        [input_shard_height, input_shard_width],
        sharded_memory_layout,
        sharded_memory_orientation,
    )
    # Prepare weights for simple matmul
    B_tiled_host = create_conv_weight_tensor(B_pyt, output_channels, input_channels, 1, 1, 1, 1)
    B_tiled = B_tiled_host.to(device)

    # downsample golden output using maxpool
    out_golden = torch.nn.functional.max_pool2d(A_pyt, 1, stride=stride_h)
    out_golden_2d_nhwc = torch.permute(out_golden, (0, 2, 3, 1)).reshape(
        1, 1, batch_size * output_height * output_width, input_channels
    )

    downsample_params = [batch_size, input_height, input_width, stride_h, stride_w]
    sharded_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    # Run downsample op
    A_downampled_sharded = ttnn.downsample(A_sharded, downsample_params, dtype=dtype)
    A_downsampled = ttnn.sharded_to_interleaved(
        A_downampled_sharded,
        ttnn.L1_MEMORY_CONFIG,
    )
    out = A_downsampled
    out_shape = [1, 1, _nearest_y(batch_size * output_height * output_width, 32), input_channels]
    assert out_shape == list(out.shape.with_tile_padding())
    out_shape_unpadded = [1, 1, batch_size * output_height * output_width, input_channels]
    assert out_shape_unpadded == list(out.shape)
    out = ttnn.format_output_tensor(out, out.shape, device, ttnn.ROW_MAJOR_LAYOUT)
    out = out.cpu()

    out_debug = out
    out_debug = out_debug.to_torch().float()

    # DEBUG
    # for i in range(16):
    #     for j in range(input_2d_width):
    #         logger.debug(f"out_golden_2d_nhwc[{i}][{j}]={out_golden_2d_nhwc[0][0][i][j]}")

    # for i in range(16):
    #     for j in range(input_2d_width):
    #         logger.debug(f"out_result_2d_nhwc[{i}][{j}]={out_debug[0][0][i][j]}")

    num_errors = 0
    core_idx = 0
    start_i = core_idx * output_shard_height
    end_i = start_i + output_shard_height
    for i in range(start_i, end_i):
        for j in range(input_shard_width):
            calculated = out_golden_2d_nhwc[0][0][i][j]
            golden = out_debug[0][0][i][j]
            atol_delta = torch.abs(golden - calculated).item()
            rtol_delta = torch.abs(golden - calculated) / torch.abs(calculated)
            if dtype == ttnn.bfloat8_b:
                fail = atol_delta > 0.1
            else:
                fail = atol_delta > 0.1 or rtol_delta > 0.1
            if fail:
                if num_errors < 10:
                    logger.debug(
                        f"Bad value at {i} (sharded index {i - start_i}), {j} with ATOL={atol_delta} and RTOL={rtol_delta}"
                    )
                    logger.debug(f"    result={calculated}, golden={golden}")
                num_errors += 1
                # if (num_errors >= 10):
                #     assert False
    logger.debug(f"Num errors: {num_errors}")

    out = ttnn.experimental.view(out, batch_size, output_height, output_width, input_channels)
    assert out.get_layout() == ttnn.ROW_MAJOR_LAYOUT

    # Copy output to host and convert tt tensor to pytorch tensor
    out_result = out.to_torch().float()
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    # logger.debug (f'OUTPUT: {out_result}')
    # logger.debug (f'GOLDEN: {out_golden}')

    if dtype == ttnn.bfloat8_b:
        passing, output_info = comp_allclose_and_pcc(
            out_golden, out_result, rtol=0, atol=4e-3, pcc=0.9999
        )  # For LowFi we need 0.99976
    else:
        passing, output_info = comp_equal(out_golden, out_result)
    assert passing
