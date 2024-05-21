# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from loguru import logger

from tt_lib._internal.comparison_funcs import comp_equal


def trace_conv_to_generate_data_top_left_indices_and_pad_metadata(conv_params, input_nchw_shape):
    assert len(conv_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]
    assert dilation == 1 and groups == 1
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = [input_nchw_shape[i] for i in range(4)]

    # image 1 data
    # 1  2  3  4  5  6  7  8
    # 9  10 11 12 13 14 15 16
    # 17 18 19 20 21 22 23 24
    # 25 26 27 28 29 30 31 32
    # image 2 data
    # 33 34 35 36 37 38 39 40
    # 41 42 43 44 45 46 47 48
    # 49 50 51 52 53 54 55 56
    # 57 58 59 60 61 62 63 64

    # Concatenated image data from above
    # Inserted padding above and between and on the sides of the images (pad = 1)
    # 0  0  0  0  0  0  0  0  0 0
    # 0  1  2  3  4  5  6  7  8 0
    # 0  9 10 11 12 13 14 15 16 0
    # 0 17 18 19 20 21 22 23 24 0
    # 0 25 26 27 28 29 30 31 32 0
    # 0  0  0  0  0  0  0  0  0 0
    # 0  0  0  0  0  0  0  0  0 0
    # 0 33 34 35 36 37 38 39 40 0
    # 0 41 42 43 44 45 46 47 48 0
    # 0 49 50 51 52 53 54 55 56 0
    # 0 57 58 59 60 61 62 63 64 0
    # 0  0  0  0  0  0  0  0  0 0

    # We encode above shown padded tensor into pad_metadata (list of boolean - true if padding location)
    # pad_meta_data: [true, true, ..., false, ...]
    index = 0
    padded_input_h = input_h + (2 * pad_h)
    padded_input_w = input_w + (2 * pad_w)
    pad_metadata = np.full(input_n * padded_input_h * padded_input_w, False, dtype=bool)
    for n in range(input_n):
        for h in range(padded_input_h):
            for w in range(padded_input_w):
                if h < pad_h or h >= (input_h + pad_h) or w < pad_w or w >= (input_w + pad_w):
                    pad_metadata[index] = True
                index += 1

    # TODO: add support for dilation > 1
    output_h = ((int)((padded_input_h - filter_h) / stride_h)) + 1
    output_w = ((int)((padded_input_w - filter_w) / stride_w)) + 1
    # generate a list of input indices corresponding to the top left position of sliding window
    # the index refers to the location in the padded tensor
    # data_top_left_indices = []
    index = 0
    data_top_left_indices = np.full(input_n * output_h * output_w, 0, dtype=int)
    for n in range(input_n):
        for oh in range(output_h):
            for ow in range(output_w):
                ih = oh * stride_h
                iw = ow * stride_w
                channel_idx = (n * padded_input_h * padded_input_w) + (ih * padded_input_w) + iw
                data_top_left_indices[index] = channel_idx
                index += 1
    return pad_metadata.tolist(), data_top_left_indices.tolist()


def validate_input_padded_tensor_and_data_top_left_indices_and_pad_metadata(
    input_padded_tensor,
    input_nchw_shape,
    pad_h,
    pad_w,
    filter_pyt_tensor,
    out_golden_pyt_tensor,
    pad_metadata,
    data_top_left_indices,
):
    input_n, input_c, input_h, input_w = input_nchw_shape
    filter_k, filter_c, filter_h, filter_w = list(filter_pyt_tensor.size())
    assert input_c == filter_c

    # permute filter tensor to be channels last - kchw --> khwc
    filter_pyt_tensor_khwc = torch.permute(filter_pyt_tensor, (0, 2, 3, 1))

    input_padded_width = input_w + (2 * pad_w)
    input_padded_height = input_h + (2 * pad_h)
    input_padded_volume = input_n * input_c * input_padded_height * input_padded_width
    assert len(input_padded_tensor) == input_padded_volume
    input_padded_pyt_tensor_nhwc = torch.tensor(input_padded_tensor).reshape(
        [input_n * input_padded_height, input_padded_width, input_c]
    )
    output_tensor = []
    # run conv over padded tensor using data_top_left_indices
    for k in range(filter_k):
        for i in data_top_left_indices:
            i_bh = (int)(i / input_padded_width)
            i_w = (int)(i % input_padded_width)
            output_tensor.append(
                torch.dot(
                    input_padded_pyt_tensor_nhwc[i_bh : i_bh + filter_h, i_w : i_w + filter_w, :].reshape(-1),
                    filter_pyt_tensor_khwc[k, :, :, :].reshape(-1),
                )
            )

    output_pyt_tensor = torch.tensor(output_tensor)
    assert np.prod(output_pyt_tensor.size()) == np.prod(out_golden_pyt_tensor.size())
    # permute output golden pytorch tensor from nchw to cnhw shape
    out_golden_pyt_tensor_cnhw = torch.permute(out_golden_pyt_tensor, (1, 0, 2, 3))
    # compare to pytorch
    passing_pcc, output_pcc = comp_equal(out_golden_pyt_tensor_cnhw.reshape(-1), output_pyt_tensor.reshape(-1))
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc


def decompose_conv_into_shards_and_generate_tensor_metadata(
    data_top_left_indices,
    pad_metadata,
    input_padded_w,
    conv_output_shard_height,
    unpadded_input_shard_height,
    num_cores,
    filter_h,
    filter_w,
    act_reshard_num_cores=0,
    input_nhw_height=0,
):
    req_conv_input_shard_start_end = []  # start and end indices refer to global padded input tensor
    conv_output_start_stick = 0
    for core_id in range(num_cores):
        if conv_output_start_stick >= len(data_top_left_indices):
            print("core_id=", core_id)
            print("conv_output_start_stick=", conv_output_start_stick)
            print("len(data_top_left_indices)=", len(data_top_left_indices))
            print("conv_output_shard_height=", conv_output_shard_height)
        assert conv_output_start_stick < len(data_top_left_indices)
        req_conv_input_shard_start_stick = data_top_left_indices[conv_output_start_stick]
        conv_output_end_stick = min(conv_output_start_stick + conv_output_shard_height, len(data_top_left_indices)) - 1
        req_conv_input_shard_end_stick = data_top_left_indices[conv_output_end_stick]
        halo_with_pad_nsticks = ((filter_h - 1) * input_padded_w) + filter_w - 1
        req_conv_input_shard_end_stick += halo_with_pad_nsticks
        req_conv_input_shard_start_end.append(
            (
                (conv_output_start_stick, conv_output_end_stick),
                (req_conv_input_shard_start_stick, req_conv_input_shard_end_stick),
            )
        )
        conv_output_start_stick += conv_output_shard_height

    remap = lambda a, b: (a, b)
    if act_reshard_num_cores != 0:
        assert input_nhw_height != 0
        assert (
            input_nhw_height % act_reshard_num_cores == 0
        ), f"{input_nhw_height} {act_reshard_num_cores} {num_cores} {unpadded_input_shard_height}"
        act_unpadded_input_shard_height = input_nhw_height // act_reshard_num_cores

        def _remap(cid, lid):
            idx = cid * unpadded_input_shard_height + lid
            return (idx // act_unpadded_input_shard_height, idx % act_unpadded_input_shard_height)

        remap = _remap

    tensor_metadata = []
    unpadded_input_shard_local_idx = 0
    core_id = 0
    for padded_input_tensor_idx in range(len(pad_metadata)):
        pad_stick = pad_metadata[padded_input_tensor_idx]
        if pad_stick:
            tensor_metadata.append((True, 0, 0))
        else:
            # sanity check
            assert core_id < num_cores, f"{core_id} {num_cores}"
            assert unpadded_input_shard_local_idx < unpadded_input_shard_height
            tensor_metadata.append((False, *remap(core_id, unpadded_input_shard_local_idx)))
            unpadded_input_shard_local_idx += 1
            if unpadded_input_shard_local_idx == unpadded_input_shard_height:
                unpadded_input_shard_local_idx = 0
                core_id += 1
    assert len(tensor_metadata) == len(pad_metadata)
    return req_conv_input_shard_start_end, tensor_metadata


def construct_utwh_output_shards(
    # Padded input tensor
    input_padded_tensor,
    # Padded input tensor shape
    input_nchw_padded_shape,
    # config to construct shards
    req_conv_input_shard_start_end,
):
    # reshape input padded tensor to 2d shape - [nhw, c]
    assert len(input_nchw_padded_shape) == 4
    input_n, input_c, input_padded_height, input_padded_width = [input_nchw_padded_shape[i] for i in range(4)]
    input_2d_padded_tensor = np.reshape(
        input_padded_tensor, (input_n * input_padded_height * input_padded_width, input_c)
    )
    utwh_output_shards = []
    for item in req_conv_input_shard_start_end:
        req_conv_input_shard_start, req_conv_input_shard_end = item[1]
        req_conv_input_shard_size = req_conv_input_shard_end - req_conv_input_shard_start + 1
        assert req_conv_input_shard_size <= 65535  # max uint16 value
        utwh_output_shards.append(input_2d_padded_tensor[req_conv_input_shard_start : req_conv_input_shard_end + 1, :])
    return utwh_output_shards


def validate_utwh_output_shards_and_req_conv_input_shard_start_end(
    # Padded input tensor shape
    input_nchw_padded_shape,
    # Filter pytorch tensor
    filter_pyt_tensor,
    # Conv golden output tensor to compare against
    out_golden_pyt_tensor,
    # Input indices corresponding to top left position of sliding window. Used to perform conv operation.
    data_top_left_indices,
    # validate utwh output shards
    utwh_output_shards,
    # Validate this config -
    req_conv_input_shard_start_end,
):
    filter_k = filter_pyt_tensor.size()[0]
    filter_c = filter_pyt_tensor.size()[1]
    filter_h = filter_pyt_tensor.size()[2]
    filter_w = filter_pyt_tensor.size()[3]

    output_n = out_golden_pyt_tensor.size()[0]
    output_c = out_golden_pyt_tensor.size()[1]
    output_h = out_golden_pyt_tensor.size()[2]
    output_w = out_golden_pyt_tensor.size()[3]
    assert len(data_top_left_indices) == output_n * output_h * output_w
    assert len(input_nchw_padded_shape) == 4
    input_n, input_c, input_padded_height, input_padded_width = [input_nchw_padded_shape[i] for i in range(4)]
    assert filter_c == input_c
    assert output_n == input_n
    assert output_c == filter_k

    # permute filter tensor to be channels last - kchw --> khwc
    filter_pyt_tensor_khwc = torch.permute(filter_pyt_tensor, (0, 2, 3, 1))

    # Perform conv on input shards one at a time, and compare against output. Use data_top_left_indices (global) to perform the conv operation.
    output_stick_global = 0
    for input_shard_idx, item in enumerate(req_conv_input_shard_start_end):
        assert input_shard_idx < len(utwh_output_shards)
        conv_output_shard_start, conv_output_shard_end = item[0]
        req_conv_input_shard_start, req_conv_input_shard_end = item[1]
        # sanity check that the first item in the shard is at the top left position of sliding window
        assert output_stick_global < len(data_top_left_indices)
        assert req_conv_input_shard_start == data_top_left_indices[output_stick_global]
        output_shard = []
        output_shard_size = conv_output_shard_end - conv_output_shard_start + 1
        for k in range(filter_k):
            output_stick = output_stick_global
            for o in range(output_shard_size):
                assert output_stick < len(data_top_left_indices)
                input_top_left_position_stick = data_top_left_indices[output_stick]
                assert input_top_left_position_stick >= req_conv_input_shard_start
                input_shard_stick_local_idx = input_top_left_position_stick - req_conv_input_shard_start
                conv_input_window = []
                for fh in range(filter_h):
                    for fw in range(filter_w):
                        assert input_shard_stick_local_idx + fw < len(utwh_output_shards[input_shard_idx])
                        conv_input_window.append(
                            utwh_output_shards[input_shard_idx][input_shard_stick_local_idx + fw, :]
                        )
                    input_shard_stick_local_idx += input_padded_width
                output_val = np.dot(
                    np.array(conv_input_window).flatten(), filter_pyt_tensor_khwc[k, :, :, :].reshape(-1).tolist()
                )
                output_shard.append(output_val)
                output_stick += 1
        output_stick_global = output_stick
        output_pyt_shard = torch.tensor(output_shard).reshape((filter_k, output_shard_size))
        # compare output shard with golden output pytorch tensor
        # permute output golden pytorch tensor from nchw to cnhw shape
        out_golden_pyt_tensor_cnhw = torch.permute(out_golden_pyt_tensor, (1, 0, 2, 3))
        # reshape cnhw to 2d shape = [c, nhw]
        out_golden_pyt_tensor_cnhw = torch.reshape(
            out_golden_pyt_tensor_cnhw, (output_c, output_n * output_h * output_w)
        )
        assert (
            output_pyt_shard.size()
            == out_golden_pyt_tensor_cnhw[:, conv_output_shard_start : conv_output_shard_end + 1].size()
        )
        # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
        # print("out_shard=", output_pyt_shard)
        passing_pcc, output_pcc = comp_equal(
            out_golden_pyt_tensor_cnhw[:, conv_output_shard_start : conv_output_shard_end + 1], output_pyt_shard
        )
        # print("Passing=", passing_pcc)
        # print("Output pcc=", output_pcc)
        assert passing_pcc

    return


def validate_tensor_metadata(
    input_tensor,
    input_nchw_shape,
    input_shard_size,
    tensor_metadata,
    req_conv_input_shard_start_end,
    golden_conv_input_shards,
):
    # input tensor is unpadded
    # Permute input tensor from nchw shape to nhwc shape and reshape to 2d shape - [nhw, c]
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = [input_nchw_shape[i] for i in range(4)]
    input_nhw_size = input_n * input_h * input_w
    input_tensor = np.reshape(input_tensor, input_nchw_shape)
    input_tensor_nhwc = np.transpose(input_tensor, (0, 2, 3, 1))
    input_tensor_nhwc = np.reshape(input_tensor_nhwc, (input_n * input_h * input_w, input_c))
    # construct unpadded input tensor shards
    unpadded_input_tensor_shards = []
    num_shards = len(req_conv_input_shard_start_end)
    unpadded_input_tensor_shard_start = 0
    for i in range(num_shards):
        unpadded_input_tensor_shard_end = min(unpadded_input_tensor_shard_start + input_shard_size, input_nhw_size)
        assert unpadded_input_tensor_shard_start < len(input_tensor_nhwc) and unpadded_input_tensor_shard_end <= len(
            input_tensor_nhwc
        )
        unpadded_input_tensor_shards.append(
            input_tensor_nhwc[unpadded_input_tensor_shard_start:unpadded_input_tensor_shard_end, :]
        )
        unpadded_input_tensor_shard_start += input_shard_size
    # Validate tensor_metadata
    # Construct conv input shard using tensor_metadata and req_conv_input_shard_start_end indices. Then, compare against golden conv input shards
    conv_input_shards = []
    assert len(req_conv_input_shard_start_end) == len(golden_conv_input_shards)
    for shard_idx, item in enumerate(req_conv_input_shard_start_end):
        conv_input_shard = []
        req_conv_input_shard_start = item[1][0]
        req_conv_input_shard_end = item[1][1]
        for idx in range(req_conv_input_shard_start, req_conv_input_shard_end + 1):
            assert idx < len(tensor_metadata)
            pad = tensor_metadata[idx][0]
            if pad:
                conv_input_shard.append([0] * input_c)
            else:
                core_id = tensor_metadata[idx][1]
                core_local_idx = tensor_metadata[idx][2]
                assert core_id < len(unpadded_input_tensor_shards)
                assert core_local_idx < len(unpadded_input_tensor_shards[core_id])
                conv_input_shard.append(unpadded_input_tensor_shards[core_id][core_local_idx, :])
        assert (conv_input_shard == golden_conv_input_shards[shard_idx]).all()
    return unpadded_input_tensor_shards


# Makes all sublists the same length, optionally tile aligns too
def align_up_2d_python_list(list2d: list, extend_value, align_granularity=0):
    assert type(list2d) is list
    if len(list2d) == 0:
        return list2d
    assert type(list2d[0]) is list
    max_len = 0
    for l in list2d:
        max_len = max(len(l), max_len)
    if align_granularity > 0:
        align_amount = max_len % align_granularity
        if align_amount > 0:
            max_len += align_granularity - align_amount
    for l in list2d:
        extend_amount = max_len - len(l)
        if extend_amount > 0:
            l.extend([extend_value] * extend_amount)


def generate_untilize_with_halo_kernel_configs(
    tensor_metadata: list,
    resharded_start_and_end: list,
    core_id_to_physical_coord=lambda core_id: (0, core_id),
    remote_read=False,
):
    ncores = len(resharded_start_and_end)

    per_core_gather_data = {}
    pad_local = 0xFFFF  # uint16_t max index means pad

    def run_length_encode(l, src, dst, is_pad):
        if len(l) > 0:
            src_start, dst_start, length = l[-3], l[-2], l[-1]
            # src index is always 0 if is_pad, so we only need to RLE the dst
            if (src == (src_start + length) or is_pad) and dst == (dst_start + length):
                l[-1] = length + 1
                return False
        l.extend([src, dst, 1])
        return True

    ## NOTE: assuming the core_id's are contiguous
    for core_id in np.arange(ncores):
        dst_global_start_idx, dst_global_end_idx = resharded_start_and_end[core_id][1]

        for dst_global_idx in np.arange(dst_global_start_idx, dst_global_end_idx + 1):
            dst_core_id = core_id
            dst_local_idx = dst_global_idx - dst_global_start_idx
            is_pad, src_core_id, src_local_idx = tensor_metadata[dst_global_idx]
            if is_pad:
                assert src_local_idx == 0
                src_core_id = pad_local
                dst_core_id = core_id
            if (src_core_id, dst_core_id) not in per_core_gather_data:
                per_core_gather_data[(src_core_id, dst_core_id)] = []
            assert src_local_idx < 0xFFFF, "Index overflows uint16_t storage type"
            assert dst_local_idx < 0xFFFF, "Index overflows uint16_t storage type"
            run_length_encode(per_core_gather_data[(src_core_id, dst_core_id)], src_local_idx, dst_local_idx, is_pad)

    padding_config = []
    local_config = []
    remote_config = []

    for core_id in range(ncores):
        padding_config.append([])
        local_config.append([])
        remote_config.append([])

    # print("per_core_gather_data", per_core_gather_data)

    for core_key, core_data in per_core_gather_data.items():
        src_core_id, dst_core_id = core_key

        # Padding Encoding: [dst_idx0, num_elems0, dst_idx1, num_elems1, ...]
        # Local/Remote encoding: [dst_core_id0, num_elems0, ...G0..., dst_core_id1, num_elems1, ...G1..., ...]
        is_padding = src_core_id == pad_local
        is_local = dst_core_id == src_core_id
        is_remote = not is_padding and not is_local

        if is_padding:
            del core_data[0::3]
            padding_config[dst_core_id].extend(core_data)
        elif is_local:
            noc_x, noc_y = core_id_to_physical_coord(dst_core_id)
            local_config[src_core_id].extend([noc_x, noc_y, len(core_data)])
            local_config[src_core_id].extend(core_data)
        elif remote_read:
            assert is_remote
            noc_x, noc_y = core_id_to_physical_coord(src_core_id)
            remote_config[dst_core_id].extend([noc_x, noc_y, len(core_data)])
            remote_config[dst_core_id].extend(core_data)
        else:
            assert is_remote
            noc_x, noc_y = core_id_to_physical_coord(dst_core_id)
            remote_config[src_core_id].extend([noc_x, noc_y, len(core_data)])
            remote_config[src_core_id].extend(core_data)

    # NULL plug
    for core_id in range(ncores):
        padding_config[core_id].extend([0, 0])
        local_config[core_id].extend([0, 0, 0])
        remote_config[core_id].extend([0, 0, 0])

    align_up_2d_python_list(padding_config, 0, align_granularity=2)
    align_up_2d_python_list(local_config, 0, align_granularity=2)
    align_up_2d_python_list(remote_config, 0, align_granularity=2)

    # print("padding_config", padding_config)
    # print("local_config", local_config)
    # print("remote_config", remote_config)

    max_out_nsticks_per_core = max(
        [
            resharded_start_and_end[core_id][1][1] - resharded_start_and_end[core_id][1][0] + 1
            for core_id in range(ncores)
        ]
    )

    return padding_config, local_config, remote_config, max_out_nsticks_per_core


def validate_untilize_with_halo_kernel_configs(
    golden,
    input_tensor_shards,
    resharded_start_and_end,
    padding_config,
    local_config,
    remote_config,
    max_out_nsticks_per_core,
    physical_coord_to_core_id=lambda x, y: y,
):
    ## using the kernel configs, construct the resulting resharding for each core
    ncores = len(resharded_start_and_end)
    assert len(input_tensor_shards) == ncores
    assert len(golden) == ncores
    input_c = len(golden[0][0])
    max_size = 0
    for _, dst in resharded_start_and_end:
        start = dst[0]
        end = dst[1]
        size = end - start + 1
        max_size = size if max_size < size else max_size
    pad_val = 0

    def copy_sticks(reshards, input_tensor_shards, config, src_core_id):
        i = 0
        length = 1
        while length > 0:
            noc_x = config[i + 0]
            noc_y = config[i + 1]
            length = config[i + 2]
            assert noc_x == 0, "Validation assumes noc_x is always 0"
            dst_core_id = physical_coord_to_core_id(noc_x, noc_y)
            i += 3
            for j in range(0, length, 3):
                src_local_idx = config[i + j + 0]
                dst_local_idx = config[i + j + 1]
                nsticks = config[i + j + 2]
                for k in range(nsticks):
                    reshards[dst_core_id][dst_local_idx + k] = input_tensor_shards[src_core_id][src_local_idx + k]
            i += length

    reshards = {}
    for core in np.arange(ncores):
        dst_range = resharded_start_and_end[core][1]
        curr_size = dst_range[1] - dst_range[0] + 1
        reshards[core] = np.zeros([curr_size, input_c], dtype=int)

    for core in np.arange(ncores):
        core_padding_config = padding_config[core]
        core_local_config = local_config[core]
        core_remote_config = remote_config[core]

        for base_dst_idx, nsticks in zip(core_padding_config[0::2], core_padding_config[1::2]):
            for dst_idx in range(base_dst_idx, base_dst_idx + nsticks):
                reshards[core][dst_idx] = [pad_val] * input_c
                dst_idx += 1

        copy_sticks(reshards, input_tensor_shards, core_local_config, core)
        copy_sticks(reshards, input_tensor_shards, core_remote_config, core)

    assert max_out_nsticks_per_core == max([len(golden[core]) for core in range(ncores)])
    for core in np.arange(ncores):
        # print(f'OUTPUT CORE {core}: {reshards[core]}')
        # print(f'GOLDEN CORE {core}: {golden[core]}')
        assert (reshards[core] == golden[core]).all()
