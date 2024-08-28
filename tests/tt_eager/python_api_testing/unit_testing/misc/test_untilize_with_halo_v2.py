# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy
import ttnn
from ttnn.operations.conv.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    validate_input_padded_tensor_and_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
    construct_utwh_output_shards,
    validate_utwh_output_shards_and_req_conv_input_shard_start_end,
    validate_tensor_metadata,
    validate_untilize_with_halo_kernel_configs,
)
from ttnn.operations.conv.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
    validate_conv_sharded_input_top_left_indices,
    validate_max_pool_sharded_input_top_left_indices,
)
from ttnn.operations.conv.tt_py_untilize_with_halo import (
    TTPyUntilizeWithHalo,
    SlidingWindowOpParamsWithParallelConfig,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc, comp_pcc
from tt_lib.utils import _nearest_y
from loguru import logger


def plot_diff(vals, fid, nsticks, stick_len):
    import matplotlib.pyplot as plt

    plt.clf()
    plt.figure(figsize=(100, 50))
    plt.xticks(torch.arange(0, stick_len) + 0.5, range(0, stick_len))
    plt.yticks(torch.arange(0, nsticks) + 0.5, range(0, nsticks))
    # plt.grid()
    bool_vals = vals > 0
    plt.imshow(bool_vals, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    plt.savefig(f"diff_core_{fid}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


# conv params - output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "conv_params, batch_size, input_chw_shape, num_cores_nhw, grid_size, test_max_pool",
    (
        # ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), 8, (1, 8, 8), 1, False),
        # ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), 8, (1, 8, 8), 2, False),
        # ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), 8, (1, 8, 8), 1, False),
        # ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), 8, (1, 8, 8), 2, False),
        # resnet50 s1 convs
        (
            (64, 16, 4, 4, 1, 1, 0, 0, 1, 1),
            8,
            (16, 115, 115),
            98,
            (12, 9),
            False,
        ),  # first conv b8 - 98 cores for height slicing
        (
            (32, 32, 3, 3, 1, 1, 1, 1, 1, 1),
            8,
            (32, 56, 56),
            98,
            (12, 9),
            False,
        ),  # layer1 b8 - 98 cores for height slicing
        (
            (64, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            8,
            (64, 56, 56),
            98,
            (12, 9),
            False,
        ),  # layer1 b8 - 98 cores for height slicing
        (
            (128, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            8,
            (128, 28, 28),
            98,
            (12, 9),
            False,
        ),  # layer2 b8 - 98 cores for height slicing
        (
            (256, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            8,
            (256, 14, 14),
            10,
            (10, 8),
            False,
        ),  # layer3 b8 - 10 cores for height slicing
        (
            (512, 512, 3, 3, 1, 1, 1, 1, 1, 1),
            8,
            (512, 7, 7),
            7,
            (7, 8),
            False,
        ),  # layer4 b8 - 7 cores for height slicing
        (
            (64, 16, 4, 4, 1, 1, 0, 0, 1, 1),
            16,
            (16, 115, 115),
            98,
            (12, 9),
            False,
        ),  # first conv b16 - 98 cores for height slicing
        (
            (64, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            16,
            (64, 56, 56),
            98,
            (12, 9),
            False,
        ),  # layer1 b16 - 98 cores for height slicing
        (
            (128, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            16,
            (128, 28, 28),
            98,
            (12, 9),
            False,
        ),  # layer2 b16 - 98 cores for height slicing
        (
            (256, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            16,
            (256, 14, 14),
            11,
            (11, 8),
            False,
        ),  # layer3 b16 - 11 cores for height slicing
        (
            (512, 512, 3, 3, 1, 1, 1, 1, 1, 1),
            16,
            (512, 7, 7),
            9,
            (9, 8),
            False,
        ),  # layer4 b16 - 9 cores for height slicing
        (
            (64, 16, 4, 4, 1, 1, 0, 0, 1, 1),
            20,
            (16, 115, 115),
            98,
            (12, 9),
            False,
        ),  # first conv b20 - 98 cores for height slicing
        (
            (64, 64, 3, 3, 1, 1, 1, 1, 1, 1),
            20,
            (64, 56, 56),
            98,
            (12, 9),
            False,
        ),  # layer1 b20 - 98 cores for height slicing
        (
            (128, 128, 3, 3, 1, 1, 1, 1, 1, 1),
            20,
            (128, 28, 28),
            98,
            (12, 9),
            False,
        ),  # layer2 b20 - 98 cores for height slicing
        (
            (256, 256, 3, 3, 1, 1, 1, 1, 1, 1),
            20,
            (256, 14, 14),
            12,
            (12, 8),
            False,
        ),  # layer3 b20 - 12 cores for height slicing
        (
            (512, 512, 3, 3, 1, 1, 1, 1, 1, 1),
            20,
            (512, 7, 7),
            11,
            (11, 8),
            False,
        ),  # layer4 b20 - 11 cores for height slicing
        # resnet50 s2 convs
        (
            (128, 128, 3, 3, 2, 2, 1, 1, 1, 1),
            8,
            (128, 56, 56),
            98,
            (12, 9),
            False,
        ),  # layer2 b8 - 98 cores for height slicing
        (
            (256, 256, 3, 3, 2, 2, 1, 1, 1, 1),
            8,
            (256, 28, 28),
            10,
            (10, 8),
            False,
        ),  # layer3 b8 - 10 cores for height slicing
        (
            (512, 512, 3, 3, 2, 2, 1, 1, 1, 1),
            8,
            (512, 14, 14),
            7,
            (7, 8),
            False,
        ),  # layer4 b8 - 7 cores for height slicing
        (
            (128, 128, 3, 3, 2, 2, 1, 1, 1, 1),
            16,
            (128, 56, 56),
            98,
            (12, 9),
            False,
        ),  # layer2 b16 - 98 cores for height slicing
        (
            (256, 256, 3, 3, 2, 2, 1, 1, 1, 1),
            16,
            (256, 28, 28),
            11,
            (11, 8),
            False,
        ),  # layer3 b16 - 11 cores for height slicing
        (
            (512, 512, 3, 3, 2, 2, 1, 1, 1, 1),
            16,
            (512, 14, 14),
            9,
            (9, 8),
            False,
        ),  # layer3 b16 - 9 cores for height slicing
        (
            (128, 128, 3, 3, 2, 2, 1, 1, 1, 1),
            20,
            (128, 56, 56),
            98,
            (12, 9),
            False,
        ),  # layer2 b20 - 98 cores for height slicing
        (
            (256, 256, 3, 3, 2, 2, 1, 1, 1, 1),
            20,
            (256, 28, 28),
            12,
            (12, 8),
            False,
        ),  # layer3 b20 - 12 cores for height slicing
        (
            (512, 512, 3, 3, 2, 2, 1, 1, 1, 1),
            20,
            (512, 14, 14),
            11,
            (11, 8),
            False,
        ),  # layer3 b20 - 11 cores for height slicing
        # resnet50 maxpool
        ((64, 64, 3, 3, 2, 2, 1, 1, 1, 1), 8, (64, 112, 112), 98, (12, 9), True),
        ((64, 64, 3, 3, 2, 2, 1, 1, 1, 1), 16, (64, 112, 112), 98, (12, 9), True),
        ((64, 64, 3, 3, 2, 2, 1, 1, 1, 1), 20, (64, 112, 112), 98, (12, 9), True),
    ),
)
@pytest.mark.parametrize(
    "skip_untilize",
    (
        False,
        True,
    ),
)
def test_generate_all_configs_and_references(
    device, conv_params, batch_size, input_chw_shape, num_cores_nhw, grid_size, test_max_pool, skip_untilize
):
    pytest.skip("Requires TT_METAL_CLEAR_L1=1 to pass")
    assert len(conv_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]

    if test_max_pool and batch_size > 16:
        pytest.skip(f"Skipping maxpool config with batch_size = {batch_size} due to mem limitations")

    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    torch.set_printoptions(threshold=10000, edgeitems=50, linewidth=400)  ##, sci_mode=False)

    # Construct conv inputs and filters and run pytorch conv for golden reference
    # unpadded raw tensor
    input_tensor = []
    assert len(input_chw_shape) == 3
    input_c, input_h, input_w = input_chw_shape
    assert input_c == input_channels
    input_nchw_shape = [batch_size, input_c, input_h, input_w]
    input_volume = numpy.prod(input_nchw_shape)
    input_nhw_size = batch_size * input_h * input_w
    conv_output_h = ((int)((input_h + (2 * pad_h) - filter_h) / stride_h)) + 1
    conv_output_w = ((int)((input_w + (2 * pad_w) - filter_w) / stride_w)) + 1
    conv_output_nhw_size = batch_size * conv_output_h * conv_output_w

    input_size_to_shard_evenly = _nearest_y(input_nhw_size, num_cores_nhw * 32)
    untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
    output_size_to_shard_evenly = _nearest_y(conv_output_nhw_size, num_cores_nhw * 32)
    conv_output_shard_height = (int)(output_size_to_shard_evenly / num_cores_nhw)

    logger.info(f"untilize with halo input shard height={untilize_with_halo_input_shard_height}")
    logger.info(f"conv_output_shard_height={conv_output_shard_height}")
    logger.info(f"grid_size={grid_size}")
    logger.info(f"num_cores_nhw={num_cores_nhw}")

    # Initialize tensor with data

    # # Inserting sequential integer data
    # for val in range(1, input_volume + 1):
    #     input_tensor.append(val % 3136)
    # input_pyt_tensor = torch.tensor(input_tensor, dtype=torch.bfloat16)
    input_pyt_tensor = torch.rand(input_volume, dtype=torch.bfloat16)
    input_pyt_tensor = torch.reshape(input_pyt_tensor, input_nchw_shape)
    # Pad channels to nearest 32
    # input_pyt_tensor = torch.nn.functional.pad(input_pyt_tensor, (0, 0, 0, 0, 0, _nearest_y(input_c, 32) - input_c))
    input_nchw_shape = list(input_pyt_tensor.shape)
    input_padded_c = input_nchw_shape[1]
    input_padded_width = input_w + 2 * pad_w
    input_padded_height = input_h + 2 * pad_h
    # Generate following configs by tracing conv -
    logger.info("Trace conv and generate follwing configs - pad_metadata and data_top_left_indices.")
    pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
        conv_params, input_nchw_shape
    )
    # # print("Data top left indices - ", data_top_left_indices)
    # # print("Pad meta data -", pad_metadata)

    logger.info("Generate input tensor")
    input_padded_pyt_tensor = torch.nn.functional.pad(input_pyt_tensor, (pad_w, pad_w, pad_h, pad_h), value=0)
    input_padded_pyt_tensor = input_padded_pyt_tensor.permute(0, 2, 3, 1)
    input_padded_tensor = input_padded_pyt_tensor.reshape(-1).tolist()
    # run trace conv reference to validate pad_metadata and data_top_left_indices

    # Generate more configs -
    logger.info(
        "Decompose conv into shards and generate the required conv input shard start/end stick indices and tensor metadata."
    )
    req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
        data_top_left_indices,
        pad_metadata,
        input_padded_width,
        conv_output_shard_height,
        untilize_with_halo_input_shard_height,
        num_cores_nhw,
        filter_h,
        filter_w,
    )
    # print("req_conv_input_shard_start_end-", req_conv_input_shard_start_end)
    # print("tensor_metadata-", tensor_metadata)
    logger.info("Construct reference utwh output shards")
    input_nchw_padded_shape = [input_nchw_shape[0], input_nchw_shape[1], input_padded_height, input_padded_width]
    golden_untilize_with_halo_output_shards = construct_utwh_output_shards(
        input_padded_tensor, input_nchw_padded_shape, req_conv_input_shard_start_end
    )
    # golden_untilize_with_halo_output_shards shape = 3d ls[# of shards, shard height=utwh_output_nhw_shard, shard width=output_c]

    # On device test
    num_cores_w, num_cores_h = grid_size
    num_cores_c = 1

    is_block_sharded = num_cores_nhw == num_cores_w
    sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        window_h=filter_h,
        window_w=filter_w,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        num_cores_h=num_cores_h,
        num_cores_w=num_cores_w,
        num_cores_nhw=num_cores_nhw,
    )
    # construct op object and set op configs
    halo_reader_patterns_cache = {}
    tt_py_untilize_with_halo_op = TTPyUntilizeWithHalo(device, sliding_window_op_params, halo_reader_patterns_cache)

    input_pyt_tensor = torch.reshape(
        torch.permute(input_pyt_tensor, [0, 2, 3, 1]), [1, 1, batch_size * input_h * input_w, input_padded_c]
    )
    # print(f"INPUT SHAPE: {input_pyt_tensor.shape}")

    if input_c < 32 and not skip_untilize:
        ## for input_c < 32, always need to pad when not skipping untilize, so skip
        pytest.skip("Skipping first conv tests when untilize is skipped.")

    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    untilize_with_halp_input_tt_tensor = ttnn.Tensor(input_pyt_tensor, ttnn.bfloat16)
    if skip_untilize:
        ## no need to pad, just construct the tensor in RM
        untilize_with_halp_input_tt_tensor = untilize_with_halp_input_tt_tensor.to(ttnn.ROW_MAJOR_LAYOUT)
    else:
        ## pad to tile size first, then convert to TILE
        input_padded_to_tile_shape = [
            1,
            1,
            _nearest_y(batch_size * input_h * input_w, num_cores_nhw * 32),
            input_padded_c,
        ]
        untilize_with_halp_input_tt_tensor = untilize_with_halp_input_tt_tensor.pad(
            input_padded_to_tile_shape, (0, 0, 0, 0), 0
        ).to(ttnn.TILE_LAYOUT)
    ## move input to device
    untilize_with_halp_input_tt_tensor = untilize_with_halp_input_tt_tensor.to(device, memory_config)

    # untilize_with_halp_input_tt_tensor = ttl.tensor.permute(untilize_with_halp_input_tt_tensor, (0, 2, 3, 1))
    # untilize_with_halp_input_tt_tensor = ttnn.reshape_on_device(untilize_with_halp_input_tt_tensor, batch_size, 1, input_h * input_w, input_c)
    grid_size_binary = device.compute_with_storage_grid_size()

    logger.info(f"GRID SIZE BINARY: {grid_size_binary}")

    if is_block_sharded:
        num_cores_c = num_cores_h
        assert input_padded_c % num_cores_c == 0
        untilize_with_halp_input_tt_tensor = ttnn.interleaved_to_sharded(
            untilize_with_halp_input_tt_tensor,
            grid_size,  ## need to pass in actual grid size for block sharded
            [input_size_to_shard_evenly // num_cores_nhw, input_padded_c // num_cores_c],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )
    else:
        untilize_with_halp_input_tt_tensor = ttnn.interleaved_to_sharded(
            untilize_with_halp_input_tt_tensor,
            grid_size_binary,
            [input_size_to_shard_evenly // num_cores_nhw, input_padded_c],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    # Run forward
    untilize_with_halo_output_tt_tensor = tt_py_untilize_with_halo_op(untilize_with_halp_input_tt_tensor)

    # Move the output tensor to host for comparison against golden
    untilize_with_halo_output_pyt_tensor = untilize_with_halo_output_tt_tensor.cpu().to_torch()

    ## make each golden shard same size as max shard size
    max_out_shard_nsticks = 0
    out_shard_nsticks_per_core = {}
    i = 0
    for _, (start, end) in req_conv_input_shard_start_end:
        size = end - start + 1
        out_shard_nsticks_per_core[i] = size
        if max_out_shard_nsticks < size:
            max_out_shard_nsticks = size
        i += 1
    for i in range(len(golden_untilize_with_halo_output_shards)):
        start, end = req_conv_input_shard_start_end[i][1]
        pad_size = max_out_shard_nsticks - (end - start + 1)
        pad_vec = numpy.full([pad_size, input_padded_c], 0)
        golden_untilize_with_halo_output_shards[i] = numpy.append(
            golden_untilize_with_halo_output_shards[i], pad_vec, axis=0
        )

    ## flatten
    golden_untilize_with_halo_output = [
        item
        for sublist_outer in golden_untilize_with_halo_output_shards
        for sublist in sublist_outer
        for item in sublist
    ]
    golden_untilize_with_halo_output_pyt_tensor = torch.Tensor(golden_untilize_with_halo_output)

    untilize_with_halo_output_pyt_tensor = torch.reshape(untilize_with_halo_output_pyt_tensor, (-1,))

    # plot_diff(torch.abs(torch.reshape(golden_untilize_with_halo_output_pyt_tensor, [max_out_shard_nsticks, num_cores_nhw * input_padded_c]) - torch.reshape(untilize_with_halo_output_pyt_tensor, [max_out_shard_nsticks, num_cores_nhw * input_padded_c])), 0, max_out_shard_nsticks, num_cores_nhw * input_padded_c)

    if is_block_sharded:
        pass_status = True
        golden_untilize_with_halo_output_pyt_tensor = torch.reshape(
            golden_untilize_with_halo_output_pyt_tensor, [-1, input_padded_c]
        )
        shard_size_c = input_padded_c // num_cores_c
        for i in range(num_cores_nhw):
            for j in range(num_cores_c):
                output_shard = torch.reshape(
                    untilize_with_halo_output_tt_tensor.extract_shard(ttnn.CoreCoord(i, j)).to_torch(),
                    [max_out_shard_nsticks, shard_size_c],
                )
                golden_shard = golden_untilize_with_halo_output_pyt_tensor[
                    i * max_out_shard_nsticks : (i + 1) * max_out_shard_nsticks,
                    j * shard_size_c : (j + 1) * shard_size_c,
                ]
                passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(
                    golden_shard,
                    output_shard,
                    rtol=1e-1,
                    atol=1e-3,
                    pcc=0.9999,
                )
                logger.trace(f"Core ({i, j}), Passing={passing_allclose_and_pcc}, Output={output_info}")
                pass_status = pass_status and passing_allclose_and_pcc

    else:  ## height sharding
        for i in range(num_cores_nhw):
            for j in range(num_cores_c):
                output_shard = untilize_with_halo_output_pyt_tensor[
                    i * max_out_shard_nsticks * input_padded_c : (i + 1) * max_out_shard_nsticks * input_padded_c
                ]
                golden_shard = golden_untilize_with_halo_output_pyt_tensor[
                    i * max_out_shard_nsticks * input_padded_c : (i + 1) * max_out_shard_nsticks * input_padded_c
                ]
                passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(
                    golden_shard,
                    output_shard,
                    rtol=1e-1,
                    atol=1e-3,
                    pcc=0.9999,
                )
                logger.trace(f"Core {i}, Passing={passing_allclose_and_pcc}, Output={output_info}")
                # ## for debugging:
                # if i >= 0:
                #     output_shard = torch.reshape(torch.Tensor(output_shard), (-1, 32))[0 : out_shard_nsticks_per_core[i]]
                #     golden_shard = torch.reshape(torch.Tensor(golden_shard), (-1, 32))[0 : out_shard_nsticks_per_core[i]]
                #     print(f"CORE {i}:")
                #     print(f"OUTPUT: {output_shard}")
                #     print(f"GOLDEN: {golden_shard}")
                # #     diff = torch.abs(golden_shard - output_shard)
                # #     plot_diff(diff, i, out_shard_nsticks_per_core[i], input_padded_c)

    # Clear the cache map
    halo_reader_patterns_cache.clear()

    if not is_block_sharded:
        passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(
            golden_untilize_with_halo_output_pyt_tensor,
            untilize_with_halo_output_pyt_tensor,
            rtol=1e-1,
            atol=1e-3,
            pcc=0.9999,
        )
        logger.debug(f"Passing={passing_allclose_and_pcc}")
        logger.debug(f"Output info={output_info}")
        passing_pcc, _ = comp_pcc(
            golden_untilize_with_halo_output_pyt_tensor, untilize_with_halo_output_pyt_tensor, pcc=0.999
        )
        assert passing_pcc
    else:
        logger.info(f"TODO: enable full tensor comparison once the tensor transfer ordering is fixed!")
        assert pass_status
