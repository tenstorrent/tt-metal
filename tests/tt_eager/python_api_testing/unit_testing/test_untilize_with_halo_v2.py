import pytest
import torch
import numpy
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    construct_input_padded_tensor,
    validate_input_padded_tensor_and_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
    construct_utwh_output_shards,
    validate_utwh_output_shards_and_req_conv_input_shard_start_end,
    validate_tensor_metadata,
    generate_untilize_with_halo_kernel_configs,
    validate_untilize_with_halo_kernel_configs,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
    validate_conv_sharded_input_top_left_indices,
    validate_max_pool_sharded_input_top_left_indices,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_untilize_with_halo import TTPyUntilizeWithHalo
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc, comp_pcc
from tt_lib.utils import _nearest_y
import tt_lib as ttl


def plot_diff(vals, fid, nsticks, stick_len):
    import matplotlib.pyplot as plt

    plt.clf()
    plt.figure(figsize=(100, 50))
    plt.xticks(torch.arange(0, stick_len) + 0.5, range(0, stick_len))
    plt.yticks(torch.arange(0, nsticks) + 0.5, range(0, nsticks))
    plt.grid()
    bool_vals = vals > 0
    plt.imshow(bool_vals, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    plt.savefig(f"diff_core_{fid}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


# conv params - output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups
@pytest.mark.parametrize(
    "conv_params, batch_size, input_chw_shape, num_cores, test_max_pool",
    (
        # ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), 8, (1, 8, 8), 1, False),
        # ((1, 1, 2, 2, 1, 1, 0, 0, 1, 1), 8, (1, 8, 8), 2, False),
        # ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), 8, (1, 8, 8), 1, False),
        # ((1, 1, 2, 2, 1, 1, 1, 1, 1, 1), 8, (1, 8, 8), 2, False),
        # resnet50 s1 convs
        ((32, 32, 4, 4, 1, 1, 0, 0, 1, 1), 8, (32, 115, 115), 98, False),  # first conv b8 - 98 cores for height slicing
        ((32, 32, 3, 3, 1, 1, 1, 1, 1, 1), 8, (32, 56, 56), 98, False),  # layer1 b8 - 98 cores for height slicing
        ((64, 64, 3, 3, 1, 1, 1, 1, 1, 1), 8, (64, 56, 56), 98, False),  # layer1 b8 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 8, (1, 28, 28), 98, False),  # layer2 b8 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 8, (1, 14, 14), 10, False),  # layer3 b8 - 10 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 8, (1, 7, 7), 7, False),  # layer4 b8 - 7 cores for height slicing
        ((1, 1, 4, 4, 1, 1, 0, 0, 1, 1), 16, (1, 115, 115), 98, False),  # first conv b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 56, 56), 98, False),  # layer1 b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 28, 28), 98, False),  # layer2 b16 - 98 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 14, 14), 11, False),  # layer3 b16 - 11 cores for height slicing
        ((1, 1, 3, 3, 1, 1, 1, 1, 1, 1), 16, (1, 7, 7), 9, False),  # layer4 b16 - 9 cores for height slicing
        ((1, 1, 4, 4, 1, 1, 0, 0, 1, 1), 20, (1, 115, 115), 98, False),  # first conv b16 - 98 cores for height slicing
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
def test_generate_all_configs_and_references(
    device, conv_params, batch_size, input_chw_shape, num_cores, test_max_pool
):
    assert len(conv_params) == 10
    output_channels, input_channels, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation, groups = [
        conv_params[i] for i in range(10)
    ]

    torch.set_printoptions(threshold=10000, edgeitems=50, linewidth=400)  ##, sci_mode=False)

    # Construct conv inputs and filters and run pytorch conv for golden reference
    # unpadded raw tensor
    input_tensor = []
    assert len(input_chw_shape) == 3
    input_c, input_h, input_w = input_chw_shape
    assert input_c == input_channels
    input_nchw_shape = [batch_size, input_c, input_h, input_w]
    input_volume = numpy.prod(input_nchw_shape)
    input_nhw_size = batch_size * input_h * input_w
    conv_output_h = ((int)((input_h + (2 * pad_h) - filter_h) / stride_h)) + 1
    conv_output_w = ((int)((input_w + (2 * pad_w) - filter_w) / stride_w)) + 1
    conv_output_nhw_size = batch_size * conv_output_h * conv_output_w

    input_size_to_shard_evenly = _nearest_y(input_nhw_size, num_cores * 32)
    untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores)
    output_size_to_shard_evenly = _nearest_y(conv_output_nhw_size, num_cores * 32)
    conv_output_shard_height = (int)(output_size_to_shard_evenly / num_cores)

    print("untilize with halo input shard height=", untilize_with_halo_input_shard_height)
    print("conv_output_shard_height=", conv_output_shard_height)

    # Initialize tensor with data

    # # Inserting sequential integer data
    # for val in range(1, input_volume + 1):
    #     input_tensor.append(val % 3136)
    # input_pyt_tensor = torch.tensor(input_tensor, dtype=torch.bfloat16)
    input_pyt_tensor = torch.rand(input_volume, dtype=torch.bfloat16)
    input_pyt_tensor = torch.reshape(input_pyt_tensor, input_nchw_shape)

    # filter_pyt_tensor = torch.full((output_channels, input_channels, filter_h, filter_w), 1., dtype=torch.bfloat16)
    filter_pyt_tensor = torch.rand((output_channels, input_channels, filter_h, filter_w), dtype=torch.bfloat16)
    # run conv pytorch
    out_golden_pyt_tensor = torch.nn.functional.conv2d(
        input_pyt_tensor, filter_pyt_tensor, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )
    # print(f'output golden pyt tensor: {out_golden_pyt_tensor}')
    input_padded_width = input_w + 2 * pad_w
    input_padded_height = input_h + 2 * pad_h
    # Generate following configs by tracing conv -
    print("Trace conv and generate follwing configs - pad_metadata and data_top_left_indices.")
    pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
        conv_params, input_nchw_shape
    )
    # # print("Data top left indices - ", data_top_left_indices)
    # # print("Pad meta data -", pad_metadata)

    # run trace conv reference to validate pad_metadata and data_top_left_indices
    print("Construct input padded tensor")
    input_padded_tensor = construct_input_padded_tensor(input_pyt_tensor, pad_metadata)
    # print (f'input_padded_tensor: {input_padded_tensor}')

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
    print("Construct reference utwh output shards")
    input_nchw_padded_shape = [batch_size, input_c, input_padded_height, input_padded_width]
    golden_untilize_with_halo_output_shards = construct_utwh_output_shards(
        input_padded_tensor, input_nchw_padded_shape, req_conv_input_shard_start_end
    )

    # On device test
    sliding_window_op_params = [
        (stride_h, stride_w),
        (pad_h, pad_w),
        (filter_h, filter_w),
        (batch_size, input_h, input_w),
        num_cores,
    ]
    # Assume height sharding
    num_cores_width = 12
    num_cores_height = num_cores // num_cores_width
    core_range_1 = ttl.tensor.CoreRange(
        ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(num_cores_width - 1, num_cores_height - 1)
    )
    num_cores_last = num_cores % num_cores_width
    core_range_2 = None
    if num_cores_last > 0:
        core_range_2 = ttl.tensor.CoreRange(
            ttl.tensor.CoreCoord(0, num_cores_height), ttl.tensor.CoreCoord(num_cores_last - 1, num_cores_height)
        )
        shard_grid = ttl.tensor.CoreRangeSet({core_range_1, core_range_2})
    else:
        shard_grid = ttl.tensor.CoreRangeSet({core_range_1})

    # construct op object and set op configs
    tt_py_untilize_with_halo_op = TTPyUntilizeWithHalo(device, sliding_window_op_params, shard_grid)

    input_pyt_tensor = torch.reshape(
        torch.permute(input_pyt_tensor, [0, 2, 3, 1]), [1, 1, batch_size * input_h * input_w, input_c]
    )
    print(f"INPUT SHAPE: {input_pyt_tensor.shape}")

    memory_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    untilize_with_halp_input_tt_tensor = (
        ttl.tensor.Tensor(input_pyt_tensor, ttl.tensor.DataType.BFLOAT16)
        .to(ttl.tensor.Layout.TILE)
        .to(device, memory_config)
    )
    # untilize_with_halp_input_tt_tensor = ttl.tensor.permute(untilize_with_halp_input_tt_tensor, (0, 2, 3, 1))
    # untilize_with_halp_input_tt_tensor = ttl.tensor.reshape(untilize_with_halp_input_tt_tensor, batch_size, 1, input_h * input_w, input_c)
    grid_size_binary = device.compute_with_storage_grid_size()
    untilize_with_halp_input_tt_tensor = ttl.tensor.interleaved_to_sharded(
        untilize_with_halp_input_tt_tensor,
        grid_size_binary,
        [input_size_to_shard_evenly // num_cores, input_c],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )
    # Run forward
    untilize_with_halo_output_tt_tensor = tt_py_untilize_with_halo_op(untilize_with_halp_input_tt_tensor)

    # Compare against golden untilize with halo output
    untilize_with_halo_output_pyt_tensor = untilize_with_halo_output_tt_tensor.cpu().to_torch()
    # print(f"OUTPUT: {untilize_with_halo_output_pyt_tensor}")

    ## make each golden shard same size as max shard size
    max_out_shard_nsticks = 0
    out_shard_nsticks_per_core = {}
    i = 0
    for _, (start, end) in req_conv_input_shard_start_end:
        size = end - start + 1
        out_shard_nsticks_per_core[i] = size
        if max_out_shard_nsticks < size:
            max_out_shard_nsticks = size
        i += 1
    print(f"MAX_OUT_SHARD_NSTICKS: {max_out_shard_nsticks}")
    print(f"OUT_SHARD_NSTICKS_PER_CORE: {out_shard_nsticks_per_core}")
    for i in range(len(golden_untilize_with_halo_output_shards)):
        start, end = req_conv_input_shard_start_end[i][1]
        pad_size = max_out_shard_nsticks - (end - start + 1)
        pad_vec = numpy.full([pad_size, input_c], 0)
        # print(f"{golden_untilize_with_halo_output_shards[i].shape}")
        golden_untilize_with_halo_output_shards[i] = numpy.append(
            golden_untilize_with_halo_output_shards[i], pad_vec, axis=0
        )
        # print(f"{golden_untilize_with_halo_output_shards[i].shape}")
    golden_untilize_with_halo_output = [
        item
        for sublist_outer in golden_untilize_with_halo_output_shards
        for sublist in sublist_outer
        for item in sublist
    ]
    golden_untilize_with_halo_output_pyt_tensor = torch.Tensor(golden_untilize_with_halo_output)

    # print(f'GOLDEN SHAPE: {golden_untilize_with_halo_output_pyt_tensor.shape}')
    # print(f'OUTPUT SHAPE: {untilize_with_halo_output_pyt_tensor.shape}')

    # print(f'OUTPUT SHAPE: {untilize_with_halo_output_pyt_tensor.shape}')
    untilize_with_halo_output_pyt_tensor = torch.reshape(untilize_with_halo_output_pyt_tensor, (-1,))
    # print(f'OUTPUT SHAPE: {untilize_with_halo_output_pyt_tensor.shape}')
    # print(f"GOLDEN: {golden_untilize_with_halo_output_pyt_tensor}")
    # print(f"GOLDEN: {torch.sum(golden_untilize_with_halo_output_pyt_tensor)}")
    # print(f"OUTPUT: {untilize_with_halo_output_pyt_tensor}")
    # print(f"OUTPUT: {torch.sum(untilize_with_halo_output_pyt_tensor)}")

    for i in range(len(golden_untilize_with_halo_output_shards)):
        core_x = i % 12
        core_y = i // 12
        output_shard = untilize_with_halo_output_pyt_tensor[
            i * max_out_shard_nsticks * input_c : (i + 1) * max_out_shard_nsticks * input_c
        ]
        golden_shard = golden_untilize_with_halo_output_pyt_tensor[
            i * max_out_shard_nsticks * input_c : (i + 1) * max_out_shard_nsticks * input_c
        ]
        print(
            f"Core {i} ({core_x},{core_y}), GOLDEN sum = {torch.sum(golden_shard)}, OUTPUT sum = {torch.sum(output_shard)}"
        )
        passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(
            golden_shard,
            output_shard,
            rtol=1e-1,
            atol=1e-3,
            pcc=0.9999,
        )
        print(f"Core {i}, Passing={passing_allclose_and_pcc}, Output={output_info}")
        if i > 100:
            output_shard = torch.reshape(torch.Tensor(output_shard), (-1, 32))[0 : out_shard_nsticks_per_core[i]]
            golden_shard = torch.reshape(torch.Tensor(golden_shard), (-1, 32))[0 : out_shard_nsticks_per_core[i]]
            print(f"CORE {i}:")
            print(f"OUTPUT: {output_shard}")
            print(f"GOLDEN: {golden_shard}")
            diff = torch.abs(golden_shard - output_shard)
            plot_diff(diff, i, out_shard_nsticks_per_core[i], input_c)

    passing_allclose_and_pcc, output_info = comp_allclose_and_pcc(
        golden_untilize_with_halo_output_pyt_tensor,
        untilize_with_halo_output_pyt_tensor,
        rtol=1e-1,
        atol=1e-3,
        pcc=0.9999,
    )
    print("Passing=", passing_allclose_and_pcc)
    print("Output info=", output_info)
    passing_pcc, _ = comp_pcc(
        golden_untilize_with_halo_output_pyt_tensor, untilize_with_halo_output_pyt_tensor, pcc=0.999
    )
    assert passing_pcc
