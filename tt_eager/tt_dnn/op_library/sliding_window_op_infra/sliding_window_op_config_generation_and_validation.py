import torch
import numpy
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def generate_sliding_window_op_sharded_input_top_left_indices(data_top_left_indices, conv_shard_start_end):
    # data_top_left_indices point to the global input tensor
    # conv_shard_start_end has the start and end index (inclusive) for each shard in the global input tensor
    # generate local indices (top left position in the sliding window) in the conv sharded input
    conv_sharded_input_top_left_indices = []
    for item in conv_shard_start_end:
        conv_output_shard_start, conv_output_shard_end = item[0]
        conv_input_shard_start, conv_input_shard_end = item[1]
        local_top_left_indices = []
        # sanity check to see that the first element in the input shard is at the top left position of sliding window
        assert conv_output_shard_start < len(data_top_left_indices)
        assert conv_input_shard_start == data_top_left_indices[conv_output_shard_start]
        for output_idx in range(conv_output_shard_start, conv_output_shard_end + 1):
            assert output_idx < len(data_top_left_indices)
            assert conv_input_shard_start <= data_top_left_indices[output_idx]
            local_top_left_indices.append(data_top_left_indices[output_idx] - conv_input_shard_start)
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
            out_val = 0
            for fh in range(filter_h):
                for fw in range(filter_w):
                    assert start_window_row_idx + fw < len(conv_input_shard)
                    out_val += conv_input_shard[start_window_row_idx + fw] * filter_pyt_tensor[0][0][fh][fw]
                start_window_row_idx += input_padded_width
            conv_output.append(out_val)

    output_pyt_tensor = torch.tensor(conv_output)
    assert output_pyt_tensor.size() == output_golden_pyt_tensor.reshape(-1).size()
    # print("out_golden_shard=", out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1])
    # print("out_shard=", output_pyt_shard)
    passing_pcc, output_pcc = comp_equal(output_golden_pyt_tensor.reshape(-1), output_pyt_tensor)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
