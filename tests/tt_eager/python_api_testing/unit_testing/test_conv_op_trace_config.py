import pytest
from tt_lib.fused_ops.conv import conv_op_trace
import torch
import numpy
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def traced_conv_reference(data_indices, data_start_size, pad_start_size, conv_params, input_nchw_shape):
    # reconstruct the padded tensor from the 2 lists
    # run the traced conv op on the padded tensor

    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]
    # unpadded tensor
    input_tensor = []
    i_b, i_c, i_h, i_w = input_nchw_shape
    input_volume = numpy.prod(input_nchw_shape)
    for val in range(1, input_volume + 1):
        input_tensor.append(val)
    input_pyt_tensor = torch.tensor(input_tensor)
    input_pyt_tensor = torch.reshape(input_pyt_tensor, input_nchw_shape)

    # special padded tensor with only 1 row of padding between images
    input_padded_tensor = []
    input_padded_width = i_w + (2 * pad_w)
    input_padded_height_batched = (i_b * i_h) + (2 * pad_h) + ((i_b - 1) * pad_h)
    input_padded_volume = i_c * input_padded_height_batched * input_padded_width
    pad_start_size_idx = 0
    data_start_size_idx = 0
    pad_remaining = pad_start_size_idx < len(pad_start_size)
    data_remaining = data_start_size_idx < len(data_start_size)
    input_tensor_idx = 0
    while pad_remaining or data_remaining:
        pad_next = False
        if pad_remaining and data_remaining:
            assert pad_start_size[pad_start_size_idx][0] != data_start_size[data_start_size_idx][0]
            if pad_start_size[pad_start_size_idx][0] < data_start_size[data_start_size_idx][0]:
                pad_next = True
        elif pad_remaining:
            pad_next = True
        if pad_next:
            assert pad_remaining
            input_padded_tensor.extend([0] * pad_start_size[pad_start_size_idx][1] * i_c)
            pad_start_size_idx += 1
        else:
            assert data_remaining
            input_padded_tensor.extend(
                input_tensor[input_tensor_idx : input_tensor_idx + data_start_size[data_start_size_idx][1]]
            )
            input_tensor_idx += data_start_size[data_start_size_idx][1]
            data_start_size_idx += 1
        pad_remaining = pad_start_size_idx < len(pad_start_size)
        data_remaining = data_start_size_idx < len(data_start_size)

    assert len(input_padded_tensor) == input_padded_volume
    input_padded_pyt_tensor = torch.tensor(input_padded_tensor).reshape(
        [1, input_padded_height_batched, input_padded_width]
    )
    filter_volume = i_c * filter_h * filter_w
    filter_pyt_tensor = torch.full((1, i_c, filter_h, filter_w), 1)

    output_tensor = []
    # run conv over padded tensor using data_indices
    for i in data_indices:
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
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc

    return


# conv params - output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
@pytest.mark.parametrize(
    "conv_params, input_nhwc_shape",
    (
        ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), (8, 8, 8, 1)),
        ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), (8, 8, 8, 1)),
    ),
    # (((1, 1, 4, 4, 1, 1, 0, 0, 1, 1), (8, 115, 115, 1)),),
)
def test_run_op_trace_config(conv_params, input_nhwc_shape):
    data_indices, data_start_size, pad_start_size = conv_op_trace(conv_params, input_nhwc_shape)
    print("Data indices - ", data_indices)
    print("Data start size -", data_start_size)
    print("Pad start size - ", pad_start_size)
    input_nchw_shape = [input_nhwc_shape[0], input_nhwc_shape[3], input_nhwc_shape[1], input_nhwc_shape[2]]
    # run trace conv reference
    traced_conv_reference(data_indices, data_start_size, pad_start_size, conv_params, input_nchw_shape)
