import torch
import numpy
from tests.tt_eager.python_api_testing.conv.conv_op_trace_config import construct_2d_padded_tensor_list
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


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
    assert len(data_top_left_indices) == numpy.prod(list(out_golden_pyt_tensor.size()))

    # Validate req_conv_input_shard_start_end. First, generate conv input shards
    conv_input_shards = []
    for item in req_conv_input_shard_start_end:
        req_conv_input_shard_start, req_conv_input_shard_end = item[1]
        conv_input_shards.append(input_padded_tensor[req_conv_input_shard_start : req_conv_input_shard_end + 1])
    # Perform conv on input shards one at a time, and compare against output. Use data_top_left_indices (global) to perform the conv operation.
    output_stick = 0
    input_shard_idx = 0
    for item in req_conv_input_shard_start_end:
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
            output_val = 0
            assert input_top_left_position_stick >= req_conv_input_shard_start
            input_shard_stick_local_idx = input_top_left_position_stick - req_conv_input_shard_start
            for fh in range(filter_h):
                for fw in range(filter_w):
                    assert input_shard_stick_local_idx + fw < len(conv_input_shards[input_shard_idx])
                    output_val += (
                        conv_input_shards[input_shard_idx][input_shard_stick_local_idx + fw]
                        * filter_pyt_tensor[0][0][fh][fw]
                    )
                input_shard_stick_local_idx += input_padded_width
            output_shard.append(output_val)
            output_stick += 1
        # compare output shard with golden output pytorch tensor
        output_pyt_shard = torch.tensor(output_shard)
        assert (
            output_pyt_shard.size()
            == out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1].size()
        )
        passing_pcc, output_pcc = comp_equal(
            out_golden_pyt_tensor.reshape(-1)[conv_output_shard_start : conv_output_shard_end + 1], output_pyt_shard
        )
        print("Passing=", passing_pcc)
        print("Output pcc=", output_pcc)
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
        assert conv_input_shard == golden_conv_input_shards[shard_idx]
