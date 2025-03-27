import pytest
import torch
import ttnn

from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [
        # nchw
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize(
    "channel_padding",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "is_sharded",
    [
        True,
        False,
    ],
)
def test_conv_padded(device, shape, channel_padding, is_sharded):
    # Create and reorder torch tensor
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)

    torch_bias = torch.rand((1, 1, 1, 32), dtype=torch.bfloat16)
    torch_weights = torch.rand((32, 3, 7, 7), dtype=torch.bfloat16)

    bias = ttnn.from_torch(torch_bias)
    weights = ttnn.from_torch(torch_weights)

    # Calculate shard spec and create memory config
    n, h, w, c = torch_input_tensor.shape
    core_grid = ttnn.CoreGrid(y=8, x=8)
    num_cores = core_grid.x * core_grid.y
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )
    shard_spec = ttnn.ShardSpec(
        shard_grid, ((n * w * h + num_cores - 1) // num_cores, channel_padding), ttnn.ShardOrientation.ROW_MAJOR
    )

    input_mem_config = None
    if is_sharded:
        # sharded case fails
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
    else:
        # interleaved works as expected
        input_mem_config = ttnn.L1_MEMORY_CONFIG

    # Pad tensor and send to device
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_padded = ttnn.pad(input_tensor, [n, h, w, channel_padding], [0, 0, 0, 0], 0)

    input_device = ttnn.to_device(input_padded, device, input_mem_config)

    # Setup conv
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=False,
        deallocate_activation=True,
        reallocate_halo_output=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        output_layout=ttnn.Layout.TILE,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    conv_kwargs = {
        "in_channels": input_device.shape[3],
        "out_channels": input_device.padded_shape[3],
        "batch_size": input_device.shape[0],
        "input_height": input_device.shape[1],
        "input_width": input_device.shape[2],
        "kernel_size": (7, 7),
        "stride": (4, 4),
        "padding": (3, 3),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    # Call conv
    [output_tensor, [_out_height, _out_width], [out_weights, out_bias]] = ttnn.conv2d(
        input_tensor=input_device,
        weight_tensor=weights,
        bias_tensor=bias,
        **conv_kwargs,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
