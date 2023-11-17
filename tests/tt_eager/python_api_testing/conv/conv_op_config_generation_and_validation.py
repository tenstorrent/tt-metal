def generate_conv_sharded_input_top_left_indices(data_top_left_indices, conv_shard_start_end):
    # data_top_left_indices point to the global input tensor
    # conv_shard_start_end has the start and end index (inclusive) for each shard in the global input tensor
    # generate local indices (top left position in the sliding window) in the conv sharded input
    conv_sharded_input_top_left_indices = []
    for shard_idx, item in conv_shard_start_end:
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
