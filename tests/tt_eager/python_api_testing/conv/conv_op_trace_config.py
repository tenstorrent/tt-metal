# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


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
    output_h = ((int)(padded_input_h - filter_h / stride_h)) + 1
    output_w = ((int)(padded_input_w - filter_w / stride_w)) + 1
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


def traced_conv_reference(pad_metadata, data_top_left_indices, conv_params, input_nchw_shape):
    assert len(conv_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]
    # unpadded tensor
    input_tensor = []
    assert len(input_nchw_shape) == 4
    input_n, input_c, input_h, input_w = input_nchw_shape
    assert input_c == 1  # Ref done for channel size = 1
    input_volume = numpy.prod(input_nchw_shape)

    # Initialize tensor with data
    # Inserting sequential integer data
    for val in range(1, input_volume + 1):
        input_tensor.append(val)
    input_pyt_tensor = torch.tensor(input_tensor)
    input_pyt_tensor = torch.reshape(input_pyt_tensor, input_nchw_shape)

    # Construct the padded tensor using pad_metadata
    input_padded_tensor = []
    input_padded_width = input_w + (2 * pad_w)
    input_padded_height = input_h + (2 * pad_h)
    input_padded_volume = input_n * input_padded_height * input_padded_width
    input_tensor_idx = 0
    assert len(pad_metadata) == input_padded_volume
    for i in range(input_padded_volume):
        if pad_metadata[i]:
            input_padded_tensor.append(0)
        else:
            input_padded_tensor.append(input_tensor[input_tensor_idx])
            input_tensor_idx += 1

    assert len(input_padded_tensor) == input_padded_volume
    input_padded_pyt_tensor = torch.tensor(input_padded_tensor).reshape(
        [1, input_n * input_padded_height, input_padded_width]
    )
    filter_volume = filter_h * filter_w
    # Initializing filters with all 1s
    filter_pyt_tensor = torch.full((1, 1, filter_h, filter_w), 1)

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
    # run conv pytorch
    out_golden_pyt_tensor = torch.nn.functional.conv2d(
        input_pyt_tensor, filter_pyt_tensor, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )
    assert numpy.prod(output_pyt_tensor.size()) == numpy.prod(out_golden_pyt_tensor.size())
    output_pyt_tensor = torch.reshape(output_pyt_tensor, out_golden_pyt_tensor.size())

    # compare to pytorch
    passing_pcc, output_pcc = comp_equal(out_golden_pyt_tensor, output_pyt_tensor)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc

    return
