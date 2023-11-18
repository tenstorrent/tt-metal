import torch
import numpy
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def generate_sliding_window_op_sharded_input_top_left_indices(data_top_left_indices, conv_shard_start_end):
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
    return conv_sharded_input_top_left_indices


def validate_conv_sharded_input_top_left_indices(
    conv_input_shards,
    input_padded_width,
    filter_pyt_tensor,
    output_golden_pyt_tensor,
    conv_sharded_input_top_left_indices,
):
    filter_k = filter_pyt_tensor.size()[0]
    filter_c = filter_pyt_tensor.size()[1]
    filter_h = filter_pyt_tensor.size()[2]
    filter_w = filter_pyt_tensor.size()[3]
    assert filter_k == 1 and filter_c == 1

    conv_output = []  # the whole conv output, we append to this list in order of shards
    for shard_idx, local_top_left_indices in enumerate(conv_sharded_input_top_left_indices):
        assert shard_idx < len(conv_input_shards)
        conv_input_shard = conv_input_shards[shard_idx]
        for local_output_idx, local_input_top_left_idx in enumerate(local_top_left_indices):
            start_window_row_idx = local_input_top_left_idx
            conv_input_window = []
            for fh in range(filter_h):
                for fw in range(filter_w):
                    assert start_window_row_idx + fw < len(conv_input_shard)
                    conv_input_window.append(conv_input_shard[start_window_row_idx + fw])
                start_window_row_idx += input_padded_width
            output_val = numpy.dot(conv_input_window, filter_pyt_tensor.reshape(-1).tolist())
            conv_output.append(output_val)

    output_pyt_tensor = torch.tensor(conv_output)
    assert output_pyt_tensor.size() == output_golden_pyt_tensor.reshape(-1).size()
    # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
    # print("out_shard=", output_pyt_shard)
    passing_pcc, output_pcc = comp_equal(output_golden_pyt_tensor.reshape(-1), output_pyt_tensor)
    # print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc


def validate_max_pool_sharded_input_top_left_indices(
    pool_input_shards,
    input_padded_width,
    pool_window_h,
    pool_window_w,
    output_golden_pyt_tensor,
    pool_sharded_input_top_left_indices,
):
    pool_output = []  # the whole conv output, we append to this list in order of shards
    for shard_idx, local_top_left_indices in enumerate(pool_sharded_input_top_left_indices):
        assert shard_idx < len(pool_input_shards)
        pool_input_shard = pool_input_shards[shard_idx]
        for local_output_idx, local_input_top_left_idx in enumerate(local_top_left_indices):
            start_window_row_idx = local_input_top_left_idx
            pool_input_window = []
            for fh in range(pool_window_h):
                for fw in range(pool_window_w):
                    assert start_window_row_idx + fw < len(pool_input_shard)
                    pool_input_window.append(pool_input_shard[start_window_row_idx + fw])
                start_window_row_idx += input_padded_width
            max_val = max(pool_input_window)
            pool_output.append(max_val)

    output_pyt_tensor = torch.tensor(pool_output)
    assert output_pyt_tensor.size() == output_golden_pyt_tensor.reshape(-1).size()
    # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
    # print("out_shard=", output_pyt_shard)
    passing_pcc, output_pcc = comp_equal(output_golden_pyt_tensor.reshape(-1), output_pyt_tensor)
    # print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
