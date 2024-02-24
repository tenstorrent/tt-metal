# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def construct_2d_padded_tensor_list(input_tensor, input_nchw_shape, pad_metadata, pad_val: torch.int16 = 0x0):
    if pad_val == 0xF7FF:
        pad_val = -1.03e34  ## TODO: how to do this in python properly???
    # Construct the padded tensor using pad_metadata
    input_tensor_idx = 0
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = [input_nchw_shape[i] for i in range(4)]
    # Permute input tensor from nchw shape to nhwc shape
    input_tensor_nchw = np.reshape(input_tensor, input_nchw_shape)
    input_tensor_nhwc = np.transpose(input_tensor_nchw, (0, 2, 3, 1))
    input_tensor_nhwc = np.reshape(input_tensor_nhwc, (np.prod(input_nchw_shape)))

    # input_padded_tensor = np.full(len(pad_metadata)*input_c, pad_val, dtype=float)
    input_padded_tensor = np.full(len(pad_metadata) * input_c, pad_val, dtype=type(input_tensor_nhwc[0]))
    index = 0
    for i in range(len(pad_metadata)):
        for c in range(input_c):
            if not pad_metadata[i]:
                assert input_tensor_idx < len(input_tensor_nhwc)
                input_padded_tensor[index] = input_tensor_nhwc[input_tensor_idx]
                input_tensor_idx += 1
            index += 1

    return input_padded_tensor.tolist()


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

    padded_input_h = input_h + (2 * pad_h)
    padded_input_w = input_w + (2 * pad_w)
    pad_metadata = []
    for n in range(input_n):
        for h in range(padded_input_h):
            for w in range(padded_input_w):
                if h < pad_h or h >= (input_h + pad_h) or w < pad_w or w >= (input_w + pad_w):
                    pad_metadata.append(True)
                else:
                    pad_metadata.append(False)

    # TODO: add support for dilation > 1
    output_h = ((int)((padded_input_h - filter_h) / stride_h)) + 1
    output_w = ((int)((padded_input_w - filter_w) / stride_w)) + 1
    # generate a list of input indices corresponding to the top left position of sliding window
    # the index refers to the location in the padded tensor
    data_top_left_indices = []
    for n in range(input_n):
        for oh in range(output_h):
            for ow in range(output_w):
                ih = oh * stride_h
                iw = ow * stride_w
                channel_idx = (n * padded_input_h * padded_input_w) + (ih * padded_input_w) + iw
                data_top_left_indices.append(channel_idx)

    return pad_metadata, data_top_left_indices


def construct_input_padded_tensor(input_pyt_tensor, pad_metadata, pad_val: torch.int16 = 0x0):
    return construct_2d_padded_tensor_list(
        input_pyt_tensor.reshape(-1).tolist(), list(input_pyt_tensor.size()), pad_metadata, pad_val
    )


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
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
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

    tensor_metadata = []
    unpadded_input_shard_local_idx = 0
    core_id = 0
    for padded_input_tensor_idx in range(len(pad_metadata)):
        pad_stick = pad_metadata[padded_input_tensor_idx]
        if pad_stick:
            tensor_metadata.append((pad_stick, 0, 0))
        else:
            # sanity check
            assert core_id < num_cores
            assert unpadded_input_shard_local_idx < unpadded_input_shard_height
            tensor_metadata.append((pad_stick, core_id, unpadded_input_shard_local_idx))
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


NEIGHBORHOOD_DIST = 2  ## ll, l, r, rr


## Function to generate the untilize with halo writer kernel config using the tensor metadata and required shard start/end information.
##
## Inputs:  1. tensor_metadata:             [(is_pad, src_core_id, src_local_idx), ...], size = padded tensor size
##                                              NOTE: (src_core_id, src_local_idx) == src_global_idx
##          2. resharded_start_and_end:     [(req_shard_start, req_shard_end), ...], size = num cores
##
## Outputs: 1. local_data_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##          2. local_pad_start_and_size:    [[(dst_start, size), ...], ...], size = num cores
##          3. neighbor data config:            NOTE: currently NEIGHBORHOOD_DIST = 2. Can be generalized.
##              1. ll_send_start_and_size:  [[(dst_start, size), ...], ...], size = num cores
##              2. l_send_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##              3. r_send_start_and_size:   [[(dst_start, size), ...], ...], size = num cores
##              4. rr_send_start_and_size:  [[(dst_start, size), ...], ...], size = num cores
def generate_untilize_with_halo_kernel_configs(tensor_metadata: list, resharded_start_and_end: list):
    # print(f'tensor metadata: {tensor_metadata}')
    ncores = len(resharded_start_and_end)

    ## data :: { core -> [
    ##              [],    ## ll
    ##              [],    ## l
    ##              [],    ## local
    ##              [],    ## r
    ##              [],    ## rr
    ##          ]}
    core_neighbor_data = {}
    core_pad_start_and_size = {}
    core_src_local_start_idx = {}  ## {core -> [ ll, l, local, r, rr ]}

    ## NOTE: assuming the core_id's are contiguous
    for dst_core_id in np.arange(ncores):
        ## generate the config for dst_core_id using the input metadata

        dst_global_start_idx, dst_global_end_idx = resharded_start_and_end[dst_core_id][1]

        core_pad_start_and_size[dst_core_id] = []

        curr_segment_size = 0
        is_curr_segment_pad = None
        curr_segment_src_core_id = None
        curr_segment_dst_start_idx = None
        curr_segment_neighbor_idx = None

        for dst_global_idx in np.arange(dst_global_start_idx, dst_global_end_idx + 1):
            dst_local_idx = dst_global_idx - dst_global_start_idx
            is_pad, src_core_id, src_local_idx = tensor_metadata[dst_global_idx]

            if is_pad:  ## pad stick
                if curr_segment_size > 0 and is_curr_segment_pad:
                    ## current segment is padding
                    curr_segment_size += 1
                else:
                    if curr_segment_size > 0:
                        ## current segment is data, a new pad segment starts here
                        ## finish off the data seg first
                        if curr_segment_src_core_id not in core_neighbor_data:
                            core_neighbor_data[curr_segment_src_core_id] = []
                            for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                                core_neighbor_data[curr_segment_src_core_id].append([])
                        core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                            (curr_segment_dst_start_idx, curr_segment_size)
                        )
                    else:
                        ## there is no current segment
                        pass
                    ## start new pad segment
                    is_curr_segment_pad = True
                    curr_segment_size = 1
                    curr_segment_dst_start_idx = dst_local_idx

            else:  ## data stick
                ## the neighbor core of dst_core_id this data stick is coming from (src_core_id): ll, l, local, r or rr
                neighbor_idx = NEIGHBORHOOD_DIST + (dst_core_id - src_core_id)
                assert neighbor_idx >= 0 and neighbor_idx < 2 * NEIGHBORHOOD_DIST + 1

                if curr_segment_size > 0:
                    if curr_segment_src_core_id == src_core_id:
                        ## this data stick belong to the same src core as current segment
                        ## if the curr segment is also data, then it is contiguous
                        ## else, this is new data segment after a pad break
                        if not is_curr_segment_pad:
                            ## contiguous data stick
                            curr_segment_size += 1
                        else:
                            ## curr segment is padding, and a new data segment starts here
                            ## finish off the pad segment first (always local only)
                            core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
                            ## start the new data segment
                            is_curr_segment_pad = False
                            curr_segment_size = 1
                            curr_segment_dst_start_idx = dst_local_idx
                            curr_segment_src_core_id = src_core_id
                            curr_segment_neighbor_idx = neighbor_idx
                            if curr_segment_src_core_id not in core_src_local_start_idx:
                                core_src_local_start_idx[curr_segment_src_core_id] = [-1] * (2 * NEIGHBORHOOD_DIST + 1)
                            if core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] < 0:
                                core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] = src_local_idx
                    else:
                        if not is_curr_segment_pad:
                            ## this data stick belongs to a different src core than the current data segment
                            ## first finish the current data segment
                            if curr_segment_src_core_id not in core_neighbor_data:
                                core_neighbor_data[curr_segment_src_core_id] = []
                                for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                                    core_neighbor_data[curr_segment_src_core_id].append([])
                            core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                                (curr_segment_dst_start_idx, curr_segment_size)
                            )
                        else:
                            ## current segment is padding, finish it off
                            core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
                        ## start the new data segment
                        is_curr_segment_pad = False
                        curr_segment_size = 1
                        curr_segment_dst_start_idx = dst_local_idx
                        curr_segment_src_core_id = src_core_id
                        curr_segment_neighbor_idx = neighbor_idx
                        if curr_segment_src_core_id not in core_src_local_start_idx:
                            core_src_local_start_idx[curr_segment_src_core_id] = [-1] * (2 * NEIGHBORHOOD_DIST + 1)
                        if core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] < 0:
                            core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] = src_local_idx
                else:
                    ## there is no current segment, create new data segment
                    is_curr_segment_pad = False
                    curr_segment_size = 1
                    curr_segment_dst_start_idx = dst_local_idx
                    curr_segment_src_core_id = src_core_id
                    curr_segment_neighbor_idx = neighbor_idx
                    if curr_segment_src_core_id not in core_src_local_start_idx:
                        core_src_local_start_idx[curr_segment_src_core_id] = [-1] * (2 * NEIGHBORHOOD_DIST + 1)
                    if core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] < 0:
                        core_src_local_start_idx[curr_segment_src_core_id][neighbor_idx] = src_local_idx

        ## finish off the remaining last segment, if any
        if curr_segment_size > 0:
            if is_curr_segment_pad:
                ## padding segment
                core_pad_start_and_size[dst_core_id].append((curr_segment_dst_start_idx, curr_segment_size))
            else:
                ## data segment
                if curr_segment_src_core_id not in core_neighbor_data:
                    core_neighbor_data[curr_segment_src_core_id] = []
                    for i in np.arange(2 * NEIGHBORHOOD_DIST + 1):
                        core_neighbor_data[curr_segment_src_core_id].append([])
                core_neighbor_data[curr_segment_src_core_id][curr_segment_neighbor_idx].append(
                    (curr_segment_dst_start_idx, curr_segment_size)
                )
    ll_data_start_and_size = []
    l_data_start_and_size = []
    local_data_start_and_size = []
    r_data_start_and_size = []
    rr_data_start_and_size = []
    local_pad_start_and_size = []
    src_local_start_idx = []
    local_pad_nsegments_per_core = []
    ll_data_nsegments_per_core = []
    l_data_nsegments_per_core = []
    local_data_nsegments_per_core = []
    r_data_nsegments_per_core = []
    rr_data_nsegments_per_core = []
    max_ll_data_nsegments_across_cores = 0
    max_l_data_nsegments_across_cores = 0
    max_local_data_nsegments_across_cores = 0
    max_r_data_nsegments_across_cores = 0
    max_rr_data_nsegments_across_cores = 0
    max_local_pad_nsegments_across_cores = 0

    for i in range(ncores):
        ll_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST - 2])
        ll_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 2]))
        max_ll_data_nsegments_across_cores = max(
            max_ll_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 2])
        )

        l_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST - 1])
        l_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 1]))
        max_l_data_nsegments_across_cores = max(
            max_l_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST - 1])
        )

        local_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST])
        local_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST]))
        max_local_data_nsegments_across_cores = max(
            max_local_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST])
        )

        r_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST + 1])
        r_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 1]))
        max_r_data_nsegments_across_cores = max(
            max_r_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 1])
        )

        rr_data_start_and_size.append(core_neighbor_data[i][NEIGHBORHOOD_DIST + 2])
        rr_data_nsegments_per_core.append(len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 2]))
        max_rr_data_nsegments_across_cores = max(
            max_rr_data_nsegments_across_cores, len(core_neighbor_data[i][NEIGHBORHOOD_DIST + 2])
        )

        local_pad_start_and_size.append(core_pad_start_and_size[i])
        local_pad_nsegments_per_core.append(len(core_pad_start_and_size[i]))
        max_local_pad_nsegments_across_cores = max(
            max_local_pad_nsegments_across_cores, len(core_pad_start_and_size[i])
        )

        # print(f'{core_src_local_start_idx[i]}')
        src_local_start_idx.append(core_src_local_start_idx[i])

    # Pad all config arrays to max nsegments since it needs to be sharded equally across cores
    # Also, flatten the list of tuples
    for i in range(ncores):
        ll_data_start_and_size[i].extend(
            [(0, 0)] * (max_ll_data_nsegments_across_cores - ll_data_nsegments_per_core[i])
        )
        ll_data_start_and_size[i] = [item for tuple_item in ll_data_start_and_size[i] for item in tuple_item]
        l_data_start_and_size[i].extend([(0, 0)] * (max_l_data_nsegments_across_cores - l_data_nsegments_per_core[i]))
        l_data_start_and_size[i] = [item for tuple_item in l_data_start_and_size[i] for item in tuple_item]
        local_data_start_and_size[i].extend(
            [(0, 0)] * (max_local_data_nsegments_across_cores - local_data_nsegments_per_core[i])
        )
        local_data_start_and_size[i] = [item for tuple_item in local_data_start_and_size[i] for item in tuple_item]
        r_data_start_and_size[i].extend([(0, 0)] * (max_r_data_nsegments_across_cores - r_data_nsegments_per_core[i]))
        r_data_start_and_size[i] = [item for tuple_item in r_data_start_and_size[i] for item in tuple_item]
        rr_data_start_and_size[i].extend(
            [(0, 0)] * (max_rr_data_nsegments_across_cores - rr_data_nsegments_per_core[i])
        )
        rr_data_start_and_size[i] = [item for tuple_item in rr_data_start_and_size[i] for item in tuple_item]
        local_pad_start_and_size[i].extend(
            [(0, 0)] * (max_local_pad_nsegments_across_cores - local_pad_nsegments_per_core[i])
        )
        local_pad_start_and_size[i] = [item for tuple_item in local_pad_start_and_size[i] for item in tuple_item]

    # for core_id in range(ncores):
    #     print(f'Core {core_id}: {resharded_start_and_end[core_id][1][1] - resharded_start_and_end[core_id][1][0] + 1}')

    max_out_nsticks_per_core = max(
        [
            resharded_start_and_end[core_id][1][1] - resharded_start_and_end[core_id][1][0] + 1
            for core_id in range(ncores)
        ]
    )
    return (
        local_data_start_and_size,
        local_pad_start_and_size,
        ll_data_start_and_size,
        l_data_start_and_size,
        r_data_start_and_size,
        rr_data_start_and_size,
        src_local_start_idx,
        local_data_nsegments_per_core,
        local_pad_nsegments_per_core,
        ll_data_nsegments_per_core,
        l_data_nsegments_per_core,
        r_data_nsegments_per_core,
        rr_data_nsegments_per_core,
        max_out_nsticks_per_core,
    )


def validate_untilize_with_halo_kernel_configs(
    golden,
    input_tensor_shards,
    resharded_start_and_end,
    local_data_start_and_size,
    local_pad_start_and_size,
    ll_send_start_and_size,
    l_send_start_and_size,
    r_send_start_and_size,
    rr_send_start_and_size,
    src_local_start_idx,
    local_data_nsegments_per_core,
    local_pad_nsegments_per_core,
    ll_data_nsegments_per_core,
    l_data_nsegments_per_core,
    r_data_nsegments_per_core,
    rr_data_nsegments_per_core,
    max_out_nsticks_per_core,
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

    reshards = {}
    for core in np.arange(ncores):
        dst_range = resharded_start_and_end[core][1]
        curr_size = dst_range[1] - dst_range[0] + 1
        reshards[core] = np.zeros([curr_size, input_c], dtype=int)

    # print (f'RESHARD: {resharded_start_and_end}')
    for core in np.arange(ncores):
        local_data = local_data_start_and_size[core]
        local_pad = local_pad_start_and_size[core]
        ll_data = ll_send_start_and_size[core]
        l_data = l_send_start_and_size[core]
        r_data = r_send_start_and_size[core]
        rr_data = rr_send_start_and_size[core]
        src_start_idx = src_local_start_idx[core]
        ## local pad
        for local_pad_segment_idx in range(0, local_pad_nsegments_per_core[core] * 2, 2):
            dst_start = local_pad[local_pad_segment_idx]
            size = local_pad[local_pad_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core][dst_idx] = [pad_val] * input_c
                dst_idx += 1

        ## local data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST]
        for local_data_segment_idx in range(0, local_data_nsegments_per_core[core] * 2, 2):
            dst_start = local_data[local_data_segment_idx]
            size = local_data[local_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core][dst_idx] = input_tensor_shards[core][src_idx, :]  ## TODO: make global
                src_idx += 1
                dst_idx += 1

        ## push ll_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST - 2]
        for ll_data_segment_idx in range(0, ll_data_nsegments_per_core[core] * 2, 2):
            dst_start = ll_data[ll_data_segment_idx]
            size = ll_data[ll_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core - 2][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

        ## push l_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST - 1]
        for l_data_segment_idx in range(0, l_data_nsegments_per_core[core] * 2, 2):
            dst_start = l_data[l_data_segment_idx]
            size = l_data[l_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core - 1][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

        ## push r_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST + 1]
        for r_data_segment_idx in range(0, r_data_nsegments_per_core[core] * 2, 2):
            dst_start = r_data[r_data_segment_idx]
            size = r_data[r_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core + 1][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

        ## push rr_data
        src_idx = src_start_idx[NEIGHBORHOOD_DIST + 2]
        for rr_data_segment_idx in range(0, rr_data_nsegments_per_core[core] * 2, 2):
            dst_start = rr_data[rr_data_segment_idx]
            size = rr_data[rr_data_segment_idx + 1]
            dst_idx = dst_start
            while dst_idx < dst_start + size:
                reshards[core + 2][dst_idx] = input_tensor_shards[core][src_idx, :]
                src_idx += 1
                dst_idx += 1

    assert max_out_nsticks_per_core == max([len(golden[core]) for core in range(ncores)])
    for core in np.arange(ncores):
        # print(f'OUTPUT CORE {core}: {reshards[core]}')
        # print(f'GOLDEN CORE {core}: {golden[core]}')
        assert (reshards[core] == golden[core]).all()
