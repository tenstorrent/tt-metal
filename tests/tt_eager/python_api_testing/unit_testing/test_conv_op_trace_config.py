import pytest
from tt_lib.fused_ops.conv import conv_op_trace


# conv params - output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
@pytest.mark.parametrize(
    "conv_params, input_nhwc_shape",
    (((32, 32, 4, 4, 1, 1, 0, 0, 1, 1), (8, 115, 115, 32)),),
)
def test_run_op_trace_config(conv_params, input_nhwc_shape):
    data_indices, data_start_size, pad_start_size = conv_op_trace(conv_params, input_nhwc_shape)
    print("Data indices - ", data_indices)
    print("Data start size -", data_start_size)
    print("Pad start size - ", pad_start_size)
