import torch
import numpy
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def construct_2d_padded_tensor_list(input_tensor, pad_metadata):
    # Construct the padded tensor using pad_metadata
    input_padded_tensor = []
    input_tensor_idx = 0

    for i in range(len(pad_metadata)):
        if pad_metadata[i]:
            input_padded_tensor.append(0)
        else:
            assert input_tensor_idx < len(input_tensor)
            input_padded_tensor.append(input_tensor[input_tensor_idx])
            input_tensor_idx += 1
    return input_padded_tensor


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


def validate_data_top_left_indices_and_pad_medata(
    input_pyt_tensor, filter_pyt_tensor, out_golden_pyt_tensor, pad_metadata, data_top_left_indices, conv_params
):
    assert len(conv_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]
    input_n, input_c, input_h, input_w = list(input_pyt_tensor.size())
    # TODO: add sanity checks to validata filter_pyt_tensor shape
    assert input_c == 1  # Ref done for channel size = 1

    input_padded_width = input_w + (2 * pad_w)
    input_padded_height = input_h + (2 * pad_h)
    input_padded_volume = input_n * input_padded_height * input_padded_width
    assert len(pad_metadata) == input_padded_volume
    input_padded_tensor = construct_2d_padded_tensor_list(input_pyt_tensor.reshape(-1).tolist(), pad_metadata)
    assert len(input_padded_tensor) == input_padded_volume
    input_padded_pyt_tensor = torch.tensor(input_padded_tensor).reshape(
        [1, input_n * input_padded_height, input_padded_width]
    )
    output_tensor = []
    # run conv over padded tensor using data_top_left_indices
    for i in data_top_left_indices:
        i_bh = (int)(i / input_padded_width)
        i_w = (int)(i % input_padded_width)
        output_tensor.append(
            torch.dot(
                input_padded_pyt_tensor[:, i_bh : i_bh + filter_h, i_w : i_w + filter_w].reshape(-1),
                filter_pyt_tensor.reshape(-1),
            )
        )

    output_pyt_tensor = torch.tensor(output_tensor)
    assert numpy.prod(output_pyt_tensor.size()) == numpy.prod(out_golden_pyt_tensor.size())
    output_pyt_tensor = torch.reshape(output_pyt_tensor, out_golden_pyt_tensor.size())

    # compare to pytorch
    passing_pcc, output_pcc = comp_equal(out_golden_pyt_tensor, output_pyt_tensor)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc

    return input_padded_tensor


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


def validate_required_conv_input_sharded_start_end(
    # Padded input tensor
    input_padded_tensor,
    # 2d padded tensor width
    input_padded_width,  # need this to get offset within sliding window
    # Filter pytorch tensor
    filter_pyt_tensor,
    # Conv golden output tensor to compare against
    out_golden_pyt_tensor,
    # Input indices corresponding to top left position of sliding window. Used to perform conv operation.
    data_top_left_indices,
    # Validate this config -
    req_conv_input_shard_start_end,
):
    filter_h = filter_pyt_tensor.size()[2]
    filter_w = filter_pyt_tensor.size()[3]
    assert filter_pyt_tensor.size()[0] == 1 and filter_pyt_tensor.size()[1] == 1
    assert len(data_top_left_indices) == numpy.prod(list(out_golden_pyt_tensor.size()))

    # Validate req_conv_input_shard_start_end. First, generate conv input shards
    conv_input_shards = []
    for item in req_conv_input_shard_start_end:
        req_conv_input_shard_start, req_conv_input_shard_end = item[1]
        req_conv_input_shard_size = req_conv_input_shard_end - req_conv_input_shard_start + 1
        assert req_conv_input_shard_size <= 65535  # max uint16 value
        conv_input_shards.append(input_padded_tensor[req_conv_input_shard_start : req_conv_input_shard_end + 1])
    # Perform conv on input shards one at a time, and compare against output. Use data_top_left_indices (global) to perform the conv operation.
    output_stick = 0
    for input_shard_idx, item in enumerate(req_conv_input_shard_start_end):
        assert input_shard_idx < len(conv_input_shards)
        conv_output_shard_start, conv_output_shard_end = item[0]
        req_conv_input_shard_start, req_conv_input_shard_end = item[1]
        # sanity check that the first item in the shard is at the top left position of sliding window
        assert output_stick < len(data_top_left_indices)
        assert req_conv_input_shard_start == data_top_left_indices[output_stick]
        output_shard = []
        output_shard_size = conv_output_shard_end - conv_output_shard_start + 1
        for o in range(output_shard_size):
            assert output_stick < len(data_top_left_indices)
            input_top_left_position_stick = data_top_left_indices[output_stick]
            assert input_top_left_position_stick >= req_conv_input_shard_start
            input_shard_stick_local_idx = input_top_left_position_stick - req_conv_input_shard_start
            conv_input_window = []
            for fh in range(filter_h):
                for fw in range(filter_w):
                    assert input_shard_stick_local_idx + fw < len(conv_input_shards[input_shard_idx])
                    conv_input_window.append(conv_input_shards[input_shard_idx][input_shard_stick_local_idx + fw])
                input_shard_stick_local_idx += input_padded_width
            output_val = numpy.dot(conv_input_window, filter_pyt_tensor.reshape(-1).tolist())
            output_shard.append(output_val)
            output_stick += 1
        # compare output shard with golden output pytorch tensor
        output_pyt_shard = torch.tensor(output_shard)
        assert (
            output_pyt_shard.size()
            == out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1].size()
        )
        # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
        # print("out_shard=", output_pyt_shard)
        passing_pcc, output_pcc = comp_equal(
            out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1], output_pyt_shard
        )
        # print("Passing=", passing_pcc)
        # print("Output pcc=", output_pcc)
        assert passing_pcc

    # We have validated conv_input_shards, return it so it can be used for supsequent references as golden reference
    return conv_input_shards


def validate_tensor_metadata(
    input_tensor, input_shard_size, tensor_metadata, req_conv_input_shard_start_end, golden_conv_input_shards
):
    # input tensor is unpadded
    # construct unpadded input tensor shards
    unpadded_input_tensor_shards = []
    num_shards = len(req_conv_input_shard_start_end)
    unpadded_input_tensor_shard_start = 0
    for i in range(num_shards):
        unpadded_input_tensor_shard_end = min(unpadded_input_tensor_shard_start + input_shard_size, len(input_tensor))
        assert unpadded_input_tensor_shard_start < len(input_tensor) and unpadded_input_tensor_shard_end <= len(
            input_tensor
        )
        unpadded_input_tensor_shards.append(
            input_tensor[unpadded_input_tensor_shard_start:unpadded_input_tensor_shard_end]
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
                conv_input_shard.append(0)
            else:
                core_id = tensor_metadata[idx][1]
                core_local_idx = tensor_metadata[idx][2]
                assert core_id < len(unpadded_input_tensor_shards)
                assert core_local_idx < len(unpadded_input_tensor_shards[core_id])
                conv_input_shard.append(unpadded_input_tensor_shards[core_id][core_local_idx])
        # print("golden_conv_input_shard=", golden_conv_input_shards[shard_idx])
        # print("conv_input_shard=", conv_input_shard)
        assert conv_input_shard == golden_conv_input_shards[shard_idx]
