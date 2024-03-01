# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import tt_lib as ttl
from tt_lib.utils import (
    blocked_mm_with_conv_act,
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from models.utility_functions import print_diff_argmax, is_close, comp_pcc
from tests.tt_eager.python_api_testing.conv.pytorch_conv_tb import (
    TestLevel,
    generate_conv_tb_with_pytorch_golden,
    generate_conv_tb,
)
from tests.tt_eager.python_api_testing.conv.conv_unit_test_utils import (
    create_conv_act_tensor,
    create_conv_weight_tensor,
    create_conv_weight_tensor_special_padding,
)

import torch


def conv_activation_reader(
    C,
    conv_input_height,
    conv_input_width,
    conv_output_height,
    conv_output_width,
    kernel_width,
    kernel_height,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    act_block_width_datums_unpadded,
    act_block_width_datums,
    act_block_height_datums,
    weight_block_width_datums,
    num_blocks_act_w,
    num_blocks_act_h,
    num_blocks_weight_w,
):
    address_maps = []
    assert act_block_width_datums_unpadded == C * kernel_width
    assert num_blocks_act_w == kernel_height
    assert act_block_width_datums % C == 0
    assert conv_input_height == conv_input_width
    c1 = _nearest_y(conv_input_height, 2)
    c2 = c1 - conv_input_height
    # assert(c2 % 2 == 0)

    out_h = 0
    out_w = 0
    out_h_start = 0
    out_w_start = 0

    for nbh in range(0, num_blocks_act_h):
        for nbr in range(0, num_blocks_weight_w):
            in_h_offset_within_kernel_window = 0
            for nbw in range(0, num_blocks_act_w):
                out_h = out_h_start
                out_w = out_w_start
                address_map_this_group = []
                dst_l1_addr = 0
                for bh in range(0, act_block_height_datums):
                    in_h_offset = out_h * stride_h
                    in_w_offset = out_w * stride_w  # stride 1 or 2.. compile time args - also conv input width
                    in_w_offset_within_kernel_window = 0
                    for bw in range(0, (int)(act_block_width_datums_unpadded / C)):  # constant argument
                        dram_src_address = 0
                        read_size = C
                        pad = False
                        if out_h < conv_output_height:
                            in_h = in_h_offset + in_h_offset_within_kernel_window
                            in_w = in_w_offset + in_w_offset_within_kernel_window
                            # print("in_h ", in_h)
                            # print("in_w ", in_w)
                            if (
                                in_h < pad_h
                                or in_w < pad_w
                                or in_h >= (conv_input_height + pad_h)
                                or in_w >= (conv_input_width + pad_w)
                            ):
                                # pad 0s in l1
                                pad = True
                            else:
                                # read one channel from dram multi bank - row_id = channel_id
                                in_h_raw = in_h - pad_h
                                in_w_raw = in_w - pad_w
                                channel_id = (in_h_raw * conv_input_width) + in_w_raw
                                # following formula to do the multiplication with power of 2
                                # channel_id = (in_h_raw * c1) - (in_h_raw * c2) + in_w_raw
                                dram_src_address = channel_id * C  # noc_async_read
                        else:
                            # pad 0s for now but do nothing in reader kernel - let garbage rows be in l1
                            pad = True
                        address_map_this_group.append(dram_src_address)  # src address unused if pad is true
                        address_map_this_group.append(dst_l1_addr)
                        address_map_this_group.append(read_size)
                        address_map_this_group.append(pad)
                        dst_l1_addr += read_size
                        in_w_offset_within_kernel_window += 1
                    # pad 0s for block padding on the right side of block.. only first conv since C%32 != 0.. ifdef with compile time arg
                    pad_size = act_block_width_datums - act_block_width_datums_unpadded
                    if pad_size > 0:  # can be skipped .. ifdef out
                        address_map_this_group.append(0)  # src address unused if pad is true
                        address_map_this_group.append(dst_l1_addr)
                        address_map_this_group.append(pad_size)
                        address_map_this_group.append(pad)
                        dst_l1_addr += pad_size
                    if out_w < conv_output_width - 1:
                        out_w += 1
                    else:
                        out_h += 1
                        out_w = 0
                address_maps.append(address_map_this_group)
                in_h_offset_within_kernel_window += 1
        out_h_start = out_h
        out_w_start = out_w

    # combine address maps for all groups into one buffer with size information in the beginning of each group
    # need this for validation
    address_map_full = []
    num_groups = len(address_maps)
    address_map_full.append(num_groups)
    for g in range(0, num_groups):
        address_map_full.append(len(address_maps[g]))
        address_map_full.extend(address_maps[g])
    return address_map_full


@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        (64, 32, 10, 10, 7, 7, 2, 2, 3, 3),
        # resnet 50 first conv. Takes ~3 mins to validate
        # (64, 3, 224, 224, 7, 7, 2, 2, 3, 3),
    ),
)
def test_run_conv_as_large_matmul_cpu(K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w):
    a_activation_shape = [1, C, H, W]
    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    b_weights_shape = [K, C, R, S]
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()

    # Parameters defining block dims
    act_block_width_datums = _nearest_32(_nearest_y(C, 16) * S)
    act_block_width_datums_unpadded = _nearest_y(C, 16) * S
    act_block_height_datums = 4 * 32
    weight_block_width_datums = 2 * 32
    act_block_w = (int)(act_block_width_datums / 32)
    act_block_h = (int)(act_block_height_datums / 32)
    weight_block_w = (int)(weight_block_width_datums / 32)
    weight_block_h = act_block_w
    OH = ((int)((H - R + 2 * pad_h) / stride_h)) + 1
    OW = ((int)((W - S + 2 * pad_w) / stride_w)) + 1
    mm_output_shape = [
        1,
        1,
        _nearest_y(OH * OW, 32 * act_block_h),
        _nearest_y(K, 32 * weight_block_w),
    ]

    # Prepare activations
    A_cl = create_conv_act_tensor(A_pyt, 1, C, H, W)
    # Prepare weights
    B_tiled = create_conv_weight_tensor_special_padding(B_pyt, K, C, R, S, weight_block_h, weight_block_w)

    # Call DTX pass to transform A

    matrix_activation_h_tiles = (int)(_nearest_y(OH * OW, act_block_height_datums) / 32)
    matrix_weight_w_tiles = (int)(_nearest_y(K, weight_block_width_datums) / 32)
    matrix_activation_w_tiles = (int)(_nearest_y(_nearest_y(C, 16) * R * S, act_block_width_datums) / 32)

    num_blocks_act_w = (int)(matrix_activation_w_tiles / act_block_w)
    num_blocks_act_h = (int)(matrix_activation_h_tiles / act_block_h)
    num_blocks_weight_w = (int)(matrix_weight_w_tiles / weight_block_w)

    # Calling dtx pass to compute weight address map
    (act_address_map_, weight_address_map) = ttl.dtx.conv_transform(
        [_nearest_y(C, 16), H, W],
        [_nearest_y(K, weight_block_width_datums), _nearest_y(C, 16), R, S],
        [R, S, stride_h, stride_w, pad_h, pad_w],
        act_block_height_datums,
        act_block_width_datums,
        weight_block_width_datums,
        num_blocks_act_h,
        num_blocks_weight_w,
        1,
        True,
    )

    # Calling new activation reader function to compute activation address map
    act_address_map = conv_activation_reader(
        _nearest_y(C, 16),
        H,
        W,
        OH,
        OW,
        R,
        S,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        act_block_width_datums_unpadded,
        act_block_width_datums,
        act_block_height_datums,
        weight_block_width_datums,
        num_blocks_act_w,
        num_blocks_act_h,
        num_blocks_weight_w,
    )

    # Run host side CPU function which uses address maps to read activation and weights
    out_pytorch = blocked_mm_with_conv_act(
        A_cl.buffer(),
        B_tiled.buffer(),
        act_address_map,
        weight_address_map,
        num_blocks_act_h,
        num_blocks_act_w,
        num_blocks_weight_w,
        act_block_h,
        act_block_w,
        weight_block_w,
    )

    assert list(out_pytorch.shape) == mm_output_shape
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), 0:K]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert list(out_tr.shape) == [1, 1, K, (OH * OW)]
    out_result = out_tr.reshape([1, K, OH, OW])

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    assert out_result.shape == out_golden.shape
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc
