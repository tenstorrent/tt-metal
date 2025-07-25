import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "dims, cores",
    [
        [[2, 100, 100, 32], 64],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_halo_device_operation(dims, cores, device, reset_seeds):
    layout = ttnn.TILE_LAYOUT
    input_dtype = ttnn.bfloat8_b
    core_grid = device.compute_with_storage_grid_size()
    if core_grid.x * core_grid.y < cores:
        pytest.skip(
            "Skipping test_slice_height_sharded_for_conv2d as device does not have enough Tensix cores. Needs %d, but device has %d"
            % (cores, core_grid.x * core_grid.y)
        )
    core_range = ttnn.num_cores_to_corerangeset(cores, core_grid, True)
    filter_height = 3
    filter_width = 3
    input_channels = dims[3]
    output_channels = dims[3]
    conv_weight_shape = (output_channels, input_channels, filter_height, filter_width)
    torch_input = torch.randint(-10, 10, dims).to(dtype=torch.bfloat16)
    # torch_input = torch.ones(dims).to(dtype=torch.bfloat16)
    # torch_input = torch.tensor(range(dims[2])).reshape(1,1,dims[2],1).broadcast_to(dims).to(dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input, device=device, layout=layout, dtype=input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    torch_input_nchw = torch_input.permute(0, 3, 1, 2)
    torch_weight_tensor = torch.randint(-10, 10, conv_weight_shape).to(dtype=torch.bfloat16)
    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor)

    parallel_config = ttnn.SlidingWindowParallelConfig(
        grid=core_range,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
        ttnn_input.padded_shape, parallel_config, 32
    )
    ref = torch.nn.functional.conv2d(
        torch_input_nchw,
        torch_weight_tensor,
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
    )
    print("Input Padded Shape = ", ttnn_input.padded_shape)
    print("Memory Config = ", memory_config)
    ttnn_input = ttnn.to_memory_config(ttnn_input, memory_config)
    print(ttnn_input.padded_shape, ttnn_input.memory_config())
    [tt_output_tensor_on_device, [out_height, out_width], [d_w, d_b]] = ttnn.conv2d(
        input_tensor=ttnn_input,
        weight_tensor=tt_weight_tensor,
        device=device,
        kernel_size=(filter_height, filter_width),
        out_channels=output_channels,
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        dilation=(1, 1),
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat8_b,
    )
    out = tt_output_tensor_on_device.cpu().to_torch()
    ref = torch.permute(ref, (0, 2, 3, 1))

    passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=0.99)
    assert passing, pcc_msg
