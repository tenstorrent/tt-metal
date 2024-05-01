# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy

from tt_lib._internal.comparison_funcs import comp_equal


def generate_sliding_window_op_sharded_input_top_left_indices(
    data_top_left_indices, conv_shard_start_end, pad_tile=False, pad_last_core=False
):
    # data_top_left_indices point to the global input tensor (padded)
    # conv_shard_start_end has the start and end index (inclusive) for each shard in the global input tensor
    # generate local indices (top left position in the sliding window) in the conv sharded input
    conv_sharded_input_top_left_indices = []
    for item in conv_shard_start_end:
        conv_output_shard_start, conv_output_shard_end = item[0]
        conv_input_shard_start, conv_input_shard_end = item[1]
        # sanity check to see that the first element in the input shard is at the top left position of sliding window
        assert conv_output_shard_start < len(data_top_left_indices)
        assert conv_input_shard_start == data_top_left_indices[conv_output_shard_start]
        local_top_left_indices = [
            index - data_top_left_indices[conv_output_shard_start]
            for index in data_top_left_indices[conv_output_shard_start : conv_output_shard_end + 1]
        ]
        conv_sharded_input_top_left_indices.append(local_top_left_indices)

    if pad_tile:
        # Pad indices for last core if not equal to other cores
        for i in range(len(conv_sharded_input_top_left_indices)):
            tile_size = 32
            extend = len(conv_sharded_input_top_left_indices[i]) % tile_size
            if extend != 0:
                conv_sharded_input_top_left_indices[i].extend([0] * (tile_size - extend))

    if pad_last_core:
        # Pad indices for last core if not equal to other cores
        indices_length_per_core = len(conv_sharded_input_top_left_indices[0])
        conv_sharded_input_top_left_indices[-1].extend(
            [0] * (indices_length_per_core - len(conv_sharded_input_top_left_indices[-1]))
        )

    return conv_sharded_input_top_left_indices


def validate_conv_sharded_input_top_left_indices(
    conv_input_shards,
    input_padded_width,
    filter_pyt_tensor,
    out_golden_pyt_tensor,
    conv_sharded_input_top_left_indices,
):
    filter_k = filter_pyt_tensor.size()[0]
    filter_c = filter_pyt_tensor.size()[1]
    filter_h = filter_pyt_tensor.size()[2]
    filter_w = filter_pyt_tensor.size()[3]

    output_n = out_golden_pyt_tensor.size()[0]
    output_c = out_golden_pyt_tensor.size()[1]
    output_h = out_golden_pyt_tensor.size()[2]
    output_w = out_golden_pyt_tensor.size()[3]
    assert output_c == filter_k
    # permute filter tensor to be channels last - kchw --> khwc
    filter_pyt_tensor_khwc = torch.permute(filter_pyt_tensor, (0, 2, 3, 1))
    # permute output golden pytorch tensor from nchw to cnhw shape
    out_golden_pyt_tensor_cnhw = torch.permute(out_golden_pyt_tensor, (1, 0, 2, 3))
    # reshape cnhw to 2d shape = [c, nhw]
    out_golden_pyt_tensor_cnhw = torch.reshape(out_golden_pyt_tensor_cnhw, (output_c, output_n * output_h * output_w))
    conv_output_shard_start = 0
    for shard_idx, local_top_left_indices in enumerate(conv_sharded_input_top_left_indices):
        conv_shard_output = []
        assert shard_idx < len(conv_input_shards)
        conv_input_shard = conv_input_shards[shard_idx]
        output_shard_size = len(local_top_left_indices)
        conv_output_shard_end = conv_output_shard_start + output_shard_size
        for k in range(filter_k):
            for local_output_idx, local_input_top_left_idx in enumerate(local_top_left_indices):
                start_window_row_idx = local_input_top_left_idx
                conv_input_window = []
                for fh in range(filter_h):
                    for fw in range(filter_w):
                        assert start_window_row_idx + fw < len(conv_input_shard)
                        conv_input_window.append(conv_input_shard[start_window_row_idx + fw, :])
                    start_window_row_idx += input_padded_width
                output_val = numpy.dot(
                    numpy.array(conv_input_window).flatten(), filter_pyt_tensor_khwc[k, :, :, :].reshape(-1).tolist()
                )
                conv_shard_output.append(output_val)

        output_pyt_shard = torch.tensor(conv_shard_output).reshape((filter_k, output_shard_size))
        # compare output shard with golden output pytorch tensor
        assert (
            output_pyt_shard.size()
            == out_golden_pyt_tensor_cnhw[:, conv_output_shard_start:conv_output_shard_end].size()
        )
        # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
        # print("out_shard=", output_pyt_shard)
        passing_pcc, output_pcc = comp_equal(
            out_golden_pyt_tensor_cnhw[:, conv_output_shard_start:conv_output_shard_end], output_pyt_shard
        )
        # print("Passing=", passing_pcc)
        # print("Output pcc=", output_pcc)
        assert passing_pcc
        conv_output_shard_start += output_shard_size
    assert conv_output_shard_start == output_n * output_h * output_w


def validate_max_pool_sharded_input_top_left_indices(
    pool_input_shards,
    input_padded_width,
    pool_window_h,
    pool_window_w,
    out_golden_pyt_tensor,
    pool_sharded_input_top_left_indices,
):
    output_n = out_golden_pyt_tensor.size()[0]
    output_c = out_golden_pyt_tensor.size()[1]
    output_h = out_golden_pyt_tensor.size()[2]
    output_w = out_golden_pyt_tensor.size()[3]
    # permute output golden pytorch tensor from nchw to cnhw shape
    out_golden_pyt_tensor_cnhw = torch.permute(out_golden_pyt_tensor, (1, 0, 2, 3))
    # reshape cnhw to 2d shape = [c, nhw]
    out_golden_pyt_tensor_cnhw = torch.reshape(out_golden_pyt_tensor_cnhw, (output_c, output_n * output_h * output_w))
    pool_output_shard_start = 0
    for shard_idx, local_top_left_indices in enumerate(pool_sharded_input_top_left_indices):
        assert shard_idx < len(pool_input_shards)
        pool_input_shard = pool_input_shards[shard_idx]
        pool_shard_output = []
        output_shard_size = len(local_top_left_indices)
        pool_output_shard_end = pool_output_shard_start + output_shard_size
        for out_c in range(output_c):
            for local_output_idx, local_input_top_left_idx in enumerate(local_top_left_indices):
                start_window_row_idx = local_input_top_left_idx
                pool_input_window = []
                for fh in range(pool_window_h):
                    for fw in range(pool_window_w):
                        assert start_window_row_idx + fw < len(pool_input_shard)
                        pool_input_window.append(pool_input_shard[start_window_row_idx + fw][out_c])
                    start_window_row_idx += input_padded_width
                max_val = max(pool_input_window)
                pool_shard_output.append(max_val)
        output_pyt_shard = torch.tensor(pool_shard_output).reshape((output_c, output_shard_size))
        # compare output shard with golden output pytorch tensor
        assert (
            output_pyt_shard.size()
            == out_golden_pyt_tensor_cnhw[:, pool_output_shard_start:pool_output_shard_end].size()
        )
        # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
        # print("out_shard=", output_pyt_shard)
        passing_pcc, output_pcc = comp_equal(
            out_golden_pyt_tensor_cnhw[:, pool_output_shard_start:pool_output_shard_end], output_pyt_shard
        )
        # print("Passing=", passing_pcc)
        # print("Output pcc=", output_pcc)
        assert passing_pcc
        pool_output_shard_start += output_shard_size
    assert pool_output_shard_start == output_n * output_h * output_w
