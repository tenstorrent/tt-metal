import pytest
import torch
import numpy
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    validate_data_top_left_indices_and_pad_medata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
    validate_required_conv_input_sharded_start_end,
    validate_tensor_metadata,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
    validate_conv_sharded_input_top_left_indices,
    validate_max_pool_sharded_input_top_left_indices,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.generate_kernel_configs import generate_kernel_configs

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc
from tt_lib.utils import _nearest_y


# conv params - output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
@pytest.mark.parametrize(
    "conv_params, batch_size, input_chw_shape, num_cores, test_max_pool",
    (
        ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), 8, (1, 8, 8), 1, False),
        ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), 8, (1, 8, 8), 2, False),
        ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), 8, (1, 8, 8), 1, False),
        ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), 8, (1, 8, 8), 2, False),
        # resnet50 s1 convs
        ((1, 1, 4, 4, 1, 1, 1, 1, 1, 1), 8, (1, 115, 115), 98, False),  # first conv b8 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 8, (1, 56, 56), 98, False),  # layer1 b8 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 8, (1, 28, 28), 98, False),  # layer2 b8 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 8, (1, 14, 14), 10, False),  # layer3 b8 - 10 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 8, (1, 7, 7), 7, False),  # layer4 b8 - 7 cores for height slicing
        ((1, 1, 4, 4, 1, 1, 1, 1, 1, 1), 16, (1, 115, 115), 98, False),  # first conv b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 56, 56), 98, False),  # layer1 b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 28, 28), 98, False),  # layer2 b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 14, 14), 11, False),  # layer3 b16 - 11 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 7, 7), 9, False),  # layer4 b16 - 9 cores for height slicing
        ((1, 1, 4, 4, 1, 1, 1, 1, 1, 1), 20, (1, 115, 115), 98, False),  # first conv b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 20, (1, 56, 56), 98, False),  # layer1 b20 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 20, (1, 28, 28), 98, False),  # layer2 b20 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 20, (1, 14, 14), 12, False),  # layer3 b20 - 12 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 20, (1, 7, 7), 11, False),  # layer4 b20 - 11 cores for height slicing
        # resnet50 s2 convs
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 8, (1, 56, 56), 98, False),  # layer2 b8 - 98 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 8, (1, 28, 28), 10, False),  # layer3 b8 - 10 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 8, (1, 14, 14), 7, False),  # layer4 b8 - 7 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 16, (1, 56, 56), 98, False),  # layer2 b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 16, (1, 28, 28), 11, False),  # layer3 b16 - 11 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 16, (1, 14, 14), 9, False),  # layer3 b16 - 9 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 20, (1, 56, 56), 98, False),  # layer2 b20 - 98 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 20, (1, 28, 28), 12, False),  # layer3 b20 - 12 cores for height slicing
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 20, (1, 14, 14), 11, False),  # layer3 b20 - 11 cores for height slicing
        # resnet50 maxpool
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 8, (1, 112, 112), 98, True),
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 16, (1, 112, 112), 98, True),
        ((1, 1, 3, 3, 2, 2, 1, 1, 1, 1), 20, (1, 112, 112), 98, True),
    ),
)
def test_generate_all_configs_and_references(conv_params, batch_size, input_chw_shape, num_cores, test_max_pool):
    assert len(conv_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]

    # Construct conv inputs and filters and run pytorch conv for golden reference
    # unpadded raw tensor
    input_tensor = []
    assert len(input_chw_shape) == 3
    input_c, input_h, input_w = input_chw_shape
    input_nchw_shape = [batch_size, input_c, input_h, input_w]
    assert input_c == 1  # Ref done for channel size = 1
    input_volume = numpy.prod(input_nchw_shape)
    assert output_channels == 1
    conv_output_h = ((int)((input_h + (2 * pad_h) - filter_h) / stride_h)) + 1
    conv_output_w = ((int)((input_w + (2 * pad_w) - filter_w) / stride_w)) + 1
    conv_output_volume = batch_size * conv_output_h * conv_output_w

    input_size_to_shard_evenly = _nearest_y(input_volume, num_cores * 32)
    untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores)
    output_size_to_shard_evenly = _nearest_y(conv_output_volume, num_cores)
    conv_output_shard_height = (int)(output_size_to_shard_evenly / num_cores)

    print("untilize with halo input shard height=", untilize_with_halo_input_shard_height)
    print("conv_output_shard_height=", conv_output_shard_height)

    # Initialize tensor with data
    # Inserting sequential integer data
    for val in range(1, input_volume + 1):
        input_tensor.append(val)
    input_pyt_tensor = torch.tensor(input_tensor)
    input_pyt_tensor = torch.reshape(input_pyt_tensor, input_nchw_shape)
    # Initializing filters with all 1s
    filter_pyt_tensor = torch.full((1, 1, filter_h, filter_w), 1)
    # run conv pytorch
    out_golden_pyt_tensor = torch.nn.functional.conv2d(
        input_pyt_tensor, filter_pyt_tensor, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )
    input_padded_width = input_w + 2 * pad_w
    # Generate following configs by tracing conv -
    print("Trace conv and generate follwing configs - pad_metadata and data_top_left_indices.")
    pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
        conv_params, input_nchw_shape
    )
    # print("Data top left indices - ", data_top_left_indices)
    # print("Pad meta data -", pad_metadata)

    # run trace conv reference to validate pad_metadata and data_top_left_indices
    print("Validate pad_metadata and data_top_left_indices.")
    input_padded_tensor = validate_data_top_left_indices_and_pad_medata(
        input_pyt_tensor, filter_pyt_tensor, out_golden_pyt_tensor, pad_metadata, data_top_left_indices, conv_params
    )

    # Generate more configs -
    print(
        "Decompose conv into shards and generate the required conv input shard start/end stick indices and tensor metadata."
    )
    req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
        data_top_left_indices,
        pad_metadata,
        input_padded_width,
        conv_output_shard_height,
        untilize_with_halo_input_shard_height,
        num_cores,
        filter_h,
        filter_w,
    )
    # print("req_conv_input_shard_start_end-", req_conv_input_shard_start_end)
    # print("tensor_metadata-", tensor_metadata)
    print("Validate required conv input shard start/end stick indices")
    golden_input_shards = validate_required_conv_input_sharded_start_end(
        input_padded_tensor,
        input_padded_width,
        filter_pyt_tensor,
        out_golden_pyt_tensor,
        data_top_left_indices,
        req_conv_input_shard_start_end,
    )

    print("Validate tensor metadata")
    validate_tensor_metadata(
        input_tensor,
        untilize_with_halo_input_shard_height,
        tensor_metadata,
        req_conv_input_shard_start_end,
        golden_input_shards,
    )

    # Generate and validate the final untilize with halo configs here (TODO Abhinav)
    print(f"Generate untilize with halo kernel configs")
    # print(f'tensor metadata: {tensor_metadata}')
    print(f"req shards start and end: {req_conv_input_shard_start_end}")
    local_data, local_pad, ll_data, l_data, r_data, rr_data = generate_kernel_configs(
        tensor_metadata, req_conv_input_shard_start_end
    )
    print(f"local data:     {local_data}")
    print(f"local pad:      {local_pad}")
    print(f"ll data:        {ll_data}")
    print(f"l data:         {l_data}")
    print(f"r data:         {r_data}")
    print(f"rr data:        {rr_data}")

    # Generate sliding window op config -
    print("Generate sliding window op configs - top left positioned indices for input shards")
    sliding_window_op_sharded_input_top_left_indices = generate_sliding_window_op_sharded_input_top_left_indices(
        data_top_left_indices, req_conv_input_shard_start_end
    )

    if not test_max_pool:
        print("Validate conv_sharded_input_top_left_indices")
        validate_conv_sharded_input_top_left_indices(
            golden_input_shards,
            input_padded_width,
            filter_pyt_tensor,
            out_golden_pyt_tensor,
            sliding_window_op_sharded_input_top_left_indices,
        )
    else:
        print("Validate pool_sharded_input_top_left_indices")
        # run max pool pytorch to get golden output
        assert filter_h == filter_w and stride_h == stride_w and pad_h == pad_w
        pool_out_golden_pyt_tensor = torch.nn.MaxPool2d(
            filter_h,
            stride=stride_h,
            padding=pad_h,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )(input_pyt_tensor.float())

        validate_max_pool_sharded_input_top_left_indices(
            golden_input_shards,
            input_padded_width,
            filter_h,
            filter_w,
            pool_out_golden_pyt_tensor,
            sliding_window_op_sharded_input_top_left_indices,
        )
