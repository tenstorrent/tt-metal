# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import infer_ttnn_module_args

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    ChannelSliceStrategyConfiguration,
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    HeightSliceStrategyConfiguration,
    MaxPool2dConfiguration,
    ShardedStrategyConfiguration,
    TtConv2d,
    TtMaxPool2d,
    TtUpsample,
    UpsampleConfiguration,
    WidthShardedStrategyConfiguration,
    WidthSliceStrategyConfiguration,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

DEVICE_PARAMS = {"l1_small_size": 32768}
PCC_THRESHOLD = 0.999

INPUT_SIZES = [(8, 8), (16, 8)]
SLICE_INPUT_SIZES = [(256, 128)]

CHANNEL_CONFIGS = [
    {"in_channels": 8, "out_channels": 16},
    {"in_channels": 16, "out_channels": 16},
]

SLICE_CHANNEL_CONFIGS = [
    {"in_channels": 64, "out_channels": 128},
    {"in_channels": 128, "out_channels": 64},
]

SLICE_CONFIGS = [
    {"type": "no_slice", "num_slices": 0},
    {"type": "height_slice", "num_slices": 2},
    {"type": "width_slice", "num_slices": 2},
    {"type": "channel_slice", "num_slices": 4},
]

BATCH_SIZES = [1, 2]

KERNEL_CONFIGS = [{"kernel_size": 3, "padding": 1}, {"kernel_size": 5, "padding": 2}]

POOL_CONFIGS = [
    {"kernel_size": 2, "padding": 0, "ceil_mode": True},
    {"kernel_size": 4, "padding": 0, "ceil_mode": False},
]

UPSAMPLE_CONFIGS = [
    {"scale_factor": 2, "mode": "bilinear"},
    {"scale_factor": (2, 3), "mode": "nearest"},
    {"scale_factor": 4, "mode": "nearest"},
]


def create_conv2d_input_tensor(configuration: Conv2dConfiguration):
    shape = (configuration.batch_size, configuration.in_channels, configuration.input_height, configuration.input_width)
    nchw = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(nchw, (0, 2, 3, 1))
    nhwc = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return nchw, nhwc


def create_pool2d_input_tensor(configuration: MaxPool2dConfiguration):
    shape = (configuration.batch_size, configuration.channels, configuration.input_height, configuration.input_width)
    nchw = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(nchw, (0, 2, 3, 1)).reshape(
        1, 1, configuration.batch_size * configuration.input_height * configuration.input_width, configuration.channels
    )
    nhwc = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return nchw, nhwc


def create_upsample_input_tensor(configuration: UpsampleConfiguration):
    shape = (configuration.batch_size, configuration.channels, configuration.input_height, configuration.input_width)
    nchw = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(nchw, (0, 2, 3, 1))
    nhwc = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return nchw, nhwc


@pytest.mark.parametrize(
    "num_cores, rows, cols",
    [
        (
            1,
            1,
            1,
        ),
        (
            1,
            8,
            8,
        ),
        (
            4,
            8,
            8,
        ),
        (
            63,
            8,
            8,
        ),
    ],
)
def test_num_cores_to_core_grid(num_cores, rows, cols):
    core_grid = ShardedStrategyConfiguration.get_core_grid_from_num_cores(num_cores, rows, cols)
    assert (
        core_grid.num_cores() == num_cores
    ), f"Expected number of cores to match (was {core_grid.num_cores()} but expected {num_cores})"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES)  # Use first 2 sizes for faster tests
@pytest.mark.parametrize("channel_config", CHANNEL_CONFIGS)  # Use first 2 channel configs
@pytest.mark.parametrize("batch_size", BATCH_SIZES)  # Use first 2 batch sizes
@pytest.mark.parametrize("kernel_config", KERNEL_CONFIGS)  # Use first 2 kernel configs
@pytest.mark.parametrize(
    "sharding_strategy",
    [
        AutoShardedStrategyConfiguration(),
        HeightShardedStrategyConfiguration(),
        WidthShardedStrategyConfiguration(),
        BlockShardedStrategyConfiguration(),
    ],
)
def test_conv2d(input_size, channel_config, batch_size, kernel_config, sharding_strategy, device):
    input_height, input_width = input_size
    in_channels, out_channels = channel_config["in_channels"], channel_config["out_channels"]
    kernel_size, padding = kernel_config["kernel_size"], kernel_config["padding"]

    configuration = Conv2dConfiguration.with_random_weights(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
        sharding_strategy=sharding_strategy,
    )

    weight, bias = configuration.weight, configuration.bias
    torch_input_tensor, ttnn_input_tensor = create_conv2d_input_tensor(configuration)

    layer = TtConv2d(configuration, device)

    ttnn_output_tensor = layer(ttnn_input_tensor)
    torch_output_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        ttnn.to_torch(weight),
        ttnn.to_torch(bias).reshape(-1) if bias is not None else None,
        padding=configuration.padding,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.out_channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES)  # Use first 2 sizes
@pytest.mark.parametrize("channels", [16, 8])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)  # Use first 2 batch sizes
@pytest.mark.parametrize("pool_config", POOL_CONFIGS)
def test_pool2d(input_size, channels, batch_size, pool_config, device):
    input_height, input_width = input_size
    kernel_size, padding, ceil_mode = pool_config["kernel_size"], pool_config["padding"], pool_config["ceil_mode"]

    configuration = MaxPool2dConfiguration(
        input_height,
        input_width,
        channels,
        batch_size=batch_size,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
        ceil_mode=ceil_mode,
    )

    torch_input_tensor, ttnn_input_tensor = create_pool2d_input_tensor(configuration)

    layer = TtMaxPool2d(configuration, device)

    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = layer(ttnn_input_tensor)

    torch_output_tensor = torch.nn.functional.max_pool2d(
        torch_input_tensor,
        kernel_size=configuration.kernel_size,
        stride=configuration.stride,
        padding=configuration.padding,
        dilation=configuration.dilation,
        ceil_mode=configuration.ceil_mode,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", [(224, 224), (32, 32)])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_downblock(input_size, batch_size, device):
    input_height, input_width = input_size
    sharding_strategy = HeightShardedStrategyConfiguration(act_block_h_override=64, reshard_if_not_optimal=False)

    configurations = [
        Conv2dConfiguration.with_random_weights(
            input_height=input_height,
            input_width=input_width,
            in_channels=16,
            out_channels=32,
            batch_size=batch_size,
            kernel_size=(3, 3),
            padding=(1, 1),
            sharding_strategy=sharding_strategy,
        ),
        Conv2dConfiguration.with_random_weights(
            input_height=input_height,
            input_width=input_width,
            in_channels=32,
            out_channels=32,
            batch_size=batch_size,
            kernel_size=(3, 3),
            padding=(1, 1),
            sharding_strategy=sharding_strategy,
        ),
        MaxPool2dConfiguration(
            input_height, input_width, 32, batch_size=batch_size, kernel_size=(2, 2), padding=(0, 0), stride=(2, 2)
        ),
    ]

    torch_input_tensor, ttnn_input_tensor = create_conv2d_input_tensor(configurations[0])

    downblock = [
        TtConv2d(configurations[0], device),
        TtConv2d(configurations[1], device),
        TtMaxPool2d(configurations[2], device),
    ]

    x = ttnn_input_tensor
    for layer in downblock:
        x = layer(x)
    ttnn_output_tensor = x

    weight0, bias0 = ttnn.to_torch(configurations[0].weight), ttnn.to_torch(configurations[0].bias)
    weight1, bias1 = ttnn.to_torch(configurations[1].weight), ttnn.to_torch(configurations[1].bias)

    x = torch_input_tensor
    x = torch.nn.functional.conv2d(x, weight0, bias0.reshape(-1), padding=(1, 1))
    x = torch.nn.functional.conv2d(x, weight1, bias1.reshape(-1), padding=(1, 1))
    torch_output_tensor = torch.nn.functional.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configurations[0].batch_size, output_height, output_width, configurations[2].channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("channels", [16, 8])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("upsample_config", UPSAMPLE_CONFIGS)
def test_upsample(input_size, channels, batch_size, upsample_config, device):
    input_height, input_width = input_size
    scale_factor, mode = upsample_config["scale_factor"], upsample_config["mode"]

    # Skip bilinear upsampling if channels is not divisible by 32
    if mode == "bilinear":
        if channels % 32 != 0:
            pytest.skip(f"Bilinear upsampling requires channels ({channels}) to be divisible by 32")

    configuration = UpsampleConfiguration(
        input_height=input_height,
        input_width=input_width,
        channels=channels,
        batch_size=batch_size,
        scale_factor=scale_factor,
        mode=mode,
    )

    torch_input_tensor, ttnn_input_tensor = create_upsample_input_tensor(configuration)

    layer = TtUpsample(configuration, device)

    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_output_tensor = layer(ttnn_input_tensor)

    torch_output_tensor = torch.nn.functional.interpolate(
        torch_input_tensor,
        scale_factor=configuration.scale_factor,
        mode=configuration.mode,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES[:1])
@pytest.mark.parametrize("channel_config", CHANNEL_CONFIGS[:1])
@pytest.mark.parametrize("batch_size", BATCH_SIZES[:1])
@pytest.mark.parametrize(
    "torch_layer_config",
    [
        {"kernel_size": 3, "stride": 2, "padding": 1, "groups": 1, "bias": True},
        {"kernel_size": 5, "stride": 1, "padding": 2, "groups": 1, "bias": False},
    ],
)
def test_conv2d_configuration_from_torch_layer(input_size, channel_config, batch_size, torch_layer_config, device):
    input_height, input_width = input_size
    in_channels, out_channels = channel_config["in_channels"], channel_config["out_channels"]

    torch_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **torch_layer_config)

    configuration = Conv2dConfiguration.from_torch(
        torch_layer=torch_layer, input_height=input_height, input_width=input_width, batch_size=batch_size
    )

    assert configuration.in_channels == torch_layer.in_channels
    assert configuration.out_channels == torch_layer.out_channels
    assert configuration.kernel_size == torch_layer.kernel_size
    assert configuration.stride == torch_layer.stride
    assert configuration.padding == torch_layer.padding
    assert configuration.groups == torch_layer.groups

    assert_with_pcc(ttnn.to_torch(configuration.weight), torch_layer.weight.data, 1.0)
    if torch_layer.bias is not None:
        assert_with_pcc(ttnn.to_torch(configuration.bias).reshape(-1), torch_layer.bias.data, 1.0)
    else:
        assert configuration.bias is None

    configuration.validate_weights()

    tt_layer = TtConv2d(configuration, device)

    torch_input = torch.randn(batch_size, in_channels, input_height, input_width)
    ttnn_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn_output = tt_layer(ttnn_input)
    torch_output = torch_layer(torch_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    output_height, output_width = torch_output.shape[-2:]
    ttnn_output_torch = ttnn_output_torch.reshape(batch_size, output_height, output_width, out_channels).permute(
        0, 3, 1, 2
    )

    assert_with_pcc(torch_output, ttnn_output_torch, PCC_THRESHOLD)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES[:1])
@pytest.mark.parametrize("channels", [16, 8])
@pytest.mark.parametrize("batch_size", BATCH_SIZES[:1])
@pytest.mark.parametrize(
    "torch_layer_config",
    [
        {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1, "ceil_mode": False},
        {"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "ceil_mode": True},
    ],
)
def test_pool2d_configuration_from_torch_layer(input_size, channels, batch_size, torch_layer_config, device):
    input_height, input_width = input_size

    torch_layer = torch.nn.MaxPool2d(**torch_layer_config)

    configuration = MaxPool2dConfiguration.from_torch(
        torch_layer=torch_layer,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        channels=channels,
    )

    assert configuration.kernel_size == (torch_layer.kernel_size, torch_layer.kernel_size)
    assert configuration.stride == (torch_layer.stride, torch_layer.stride)
    assert configuration.padding == (torch_layer.padding, torch_layer.padding)

    tt_layer = TtMaxPool2d(configuration, device)

    torch_input = torch.randn(batch_size, channels, input_height, input_width)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, -1, channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    ttnn_output = tt_layer(ttnn_input)
    torch_output = torch_layer(torch_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    output_height, output_width = torch_output.shape[-2:]
    ttnn_output_torch = ttnn_output_torch.reshape(batch_size, output_height, output_width, channels).permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output_torch, PCC_THRESHOLD)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES[:1])
@pytest.mark.parametrize("channels", [16])
@pytest.mark.parametrize("batch_size", BATCH_SIZES[:1])
@pytest.mark.parametrize(
    "torch_layer_config",
    [
        {"scale_factor": 2, "mode": "nearest"},
        {"scale_factor": (2, 3), "mode": "nearest"},
        {"scale_factor": 4, "mode": "nearest"},
    ],
)
def test_upsample_configuration_from_torch_layer(input_size, channels, batch_size, torch_layer_config, device):
    input_height, input_width = input_size

    torch_layer = torch.nn.Upsample(**torch_layer_config)

    configuration = UpsampleConfiguration.from_torch(
        torch_layer=torch_layer,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        channels=channels,
    )

    assert configuration.scale_factor == (
        torch_layer.scale_factor
        if isinstance(torch_layer.scale_factor, tuple)
        else (torch_layer.scale_factor, torch_layer.scale_factor)
    )
    assert configuration.mode == torch_layer.mode

    tt_layer = TtUpsample(configuration, device)

    torch_input = torch.randn(batch_size, channels, input_height, input_width)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    ttnn_output = tt_layer(ttnn_input)
    torch_output = torch_layer(torch_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    output_height, output_width = torch_output.shape[-2:]
    ttnn_output_torch = ttnn_output_torch.reshape(batch_size, output_height, output_width, channels).permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output_torch, PCC_THRESHOLD)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("channel_config", CHANNEL_CONFIGS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize(
    "torch_layer_config",
    [
        {"kernel_size": 3, "stride": 2, "padding": 1, "groups": 1, "bias": True},
        {"kernel_size": 5, "stride": 1, "padding": 2, "groups": 1, "bias": False},
    ],
)
def test_conv2d_configuration_from_model_args(input_size, channel_config, batch_size, torch_layer_config, device):
    input_height, input_width = input_size
    in_channels, out_channels = channel_config["in_channels"], channel_config["out_channels"]

    torch_input = torch.randn(batch_size, in_channels, input_height, input_width)
    ttnn_input = ttnn.from_torch(torch_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **torch_layer_config)

    parameters = infer_ttnn_module_args(model=torch_layer, run_model=lambda _: torch_layer(torch_input), device=None)

    weight, bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(
        torch_layer.weight.data, torch_layer.bias.data if torch_layer.bias is not None else None
    )
    configuration = Conv2dConfiguration.from_model_args(parameters, weights=weight, bias=bias)

    assert configuration.in_channels == torch_layer.in_channels
    assert configuration.out_channels == torch_layer.out_channels
    assert configuration.kernel_size == torch_layer.kernel_size
    assert configuration.stride == torch_layer.stride
    assert configuration.padding == torch_layer.padding
    assert configuration.groups == torch_layer.groups

    assert_with_pcc(ttnn.to_torch(configuration.weight), torch_layer.weight.data, 1.0)
    if torch_layer.bias is not None:
        assert_with_pcc(ttnn.to_torch(configuration.bias).reshape(-1), torch_layer.bias.data, 1.0)
    else:
        assert configuration.bias is None

    configuration.validate_weights()

    tt_layer = TtConv2d(configuration, device)

    ttnn_output = tt_layer(ttnn_input)
    torch_output = torch_layer(torch_input)

    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    output_height, output_width = torch_output.shape[-2:]
    ttnn_output_torch = ttnn_output_torch.reshape(batch_size, output_height, output_width, out_channels).permute(
        0, 3, 1, 2
    )

    assert_with_pcc(torch_output, ttnn_output_torch, PCC_THRESHOLD)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_conv2d_slicing_validation_errors(device):
    """Test that validation errors are raised correctly for Conv2D slicing"""

    # Test channel slicing with non-divisible channels
    with pytest.raises(ValueError, match="Channel slicing requires num_slices > 1"):
        ChannelSliceStrategyConfiguration(num_slices=1)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_pool2d_slicing_validation_errors(device):
    """Test that validation errors are raised correctly for MaxPool2d slicing"""

    # Test that height/width slicing raises error
    with pytest.raises(ValueError, match="Height and Width slicing are not supported"):
        MaxPool2dConfiguration(
            input_height=16,
            input_width=16,
            channels=16,
            batch_size=1,
            kernel_size=(2, 2),
            slice_strategy=HeightSliceStrategyConfiguration(num_slices=2),
        )

    with pytest.raises(ValueError, match="Height and Width slicing are not supported"):
        MaxPool2dConfiguration(
            input_height=16,
            input_width=16,
            channels=16,
            batch_size=1,
            kernel_size=(2, 2),
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=2),
        )

    # Test that channels not divisible by num_slices raises error
    with pytest.raises(ValueError, match="must be divisible by number of slices"):
        MaxPool2dConfiguration(
            input_height=16,
            input_width=16,
            channels=15,  # Not divisible by 4
            batch_size=1,
            kernel_size=(2, 2),
            slice_strategy=ChannelSliceStrategyConfiguration(num_slices=4),
        )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_upsample_slicing_validation_errors(device):
    """Test that validation errors are raised correctly for Upsample configuration"""

    # Test that height/width slicing raises error
    with pytest.raises(ValueError, match="Height and Width slicing are not supported"):
        UpsampleConfiguration(
            input_height=16,
            input_width=16,
            channels=16,
            batch_size=1,
            scale_factor=2,
            slice_strategy=HeightSliceStrategyConfiguration(num_slices=2),
        )

    with pytest.raises(ValueError, match="Height and Width slicing are not supported"):
        UpsampleConfiguration(
            input_height=16,
            input_width=16,
            channels=16,
            batch_size=1,
            scale_factor=2,
            slice_strategy=WidthSliceStrategyConfiguration(num_slices=2),
        )

    # Test that channels not divisible by num_slices raises error
    with pytest.raises(ValueError, match="must be divisible by number of slices"):
        UpsampleConfiguration(
            input_height=16,
            input_width=16,
            channels=15,  # Not divisible by 4
            batch_size=1,
            scale_factor=2,
            slice_strategy=ChannelSliceStrategyConfiguration(num_slices=4),
        )

    # Test invalid mode
    with pytest.raises(ValueError, match="Mode must be one of"):
        UpsampleConfiguration(
            input_height=16,
            input_width=16,
            channels=16,
            batch_size=1,
            scale_factor=2,
            mode="invalid_mode",
        )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", SLICE_INPUT_SIZES)
@pytest.mark.parametrize("channels", SLICE_CHANNEL_CONFIGS)  # Must be divisible by num_slices for channel slicing
@pytest.mark.parametrize("batch_size", [1])  # Only use batch size 1 for slicing tests
@pytest.mark.parametrize(
    "slice_config",
    SLICE_CONFIGS,
)
def test_conv2d_slicing_strategies(input_size, channels, batch_size, slice_config, device):
    """Test Conv2D with different slicing strategies"""
    input_height, input_width = input_size
    in_channels = channels["in_channels"]
    out_channels = channels["out_channels"]

    # Create slice strategy based on config
    if slice_config["type"] == "no_slice":
        slice_strategy = None
    elif slice_config["type"] == "height_slice":
        slice_strategy = HeightSliceStrategyConfiguration(num_slices=slice_config["num_slices"])
    elif slice_config["type"] == "width_slice":
        slice_strategy = WidthSliceStrategyConfiguration(num_slices=slice_config["num_slices"])
    elif slice_config["type"] == "channel_slice":
        slice_strategy = ChannelSliceStrategyConfiguration(num_slices=slice_config["num_slices"])
    else:
        raise ValueError(f"Unknown slice type: {slice_config['type']}")

    configuration = Conv2dConfiguration.with_random_weights(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=(3, 3),
        padding=(1, 1),
        slice_strategy=slice_strategy,
    )

    torch_input_tensor, ttnn_input_tensor = create_conv2d_input_tensor(configuration)

    # Test with slicing
    layer = TtConv2d(configuration, device)

    # Verify that weight_slices are created for channel slicing
    assert hasattr(layer, "weight_slices"), "Layer should have weight_slices attribute"

    # For slicing tests, ensure input is on device with DRAM memory config
    if slice_config["type"] != "no_slice":
        ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if slice_config["type"] == "channel_slice":
        # For channel slicing, weight_slices should be populated after first call
        ttnn_output_tensor = layer(ttnn_input_tensor)
        # Note: weight_slices are populated during the first forward pass for channel slicing
    else:
        # For non-channel slicing (no_slice, height_slice, width_slice), no weight slices should be created
        ttnn_output_tensor = layer(ttnn_input_tensor)
        assert len(layer.weight_slices) == 0, "No weight slices should be created for non-channel slicing"

    # Reference implementation without slicing
    torch_output_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        ttnn.to_torch(configuration.weight),
        ttnn.to_torch(configuration.bias).reshape(-1) if configuration.bias is not None else None,
        stride=configuration.stride,
        padding=configuration.padding,
        dilation=configuration.dilation,
        groups=configuration.groups,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [N, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.out_channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", SLICE_INPUT_SIZES)
@pytest.mark.parametrize("channels", SLICE_CHANNEL_CONFIGS)  # Must be divisible by num_slices for channel slicing
@pytest.mark.parametrize("batch_size", [1])  # Only use batch size 1 for slicing tests
@pytest.mark.parametrize("pool_config", POOL_CONFIGS)
@pytest.mark.parametrize(
    "slice_config",
    SLICE_CONFIGS,
)
def test_maxpool2d_slicing_strategies(input_size, channels, batch_size, pool_config, slice_config, device):
    """Test MaxPool2d with different slicing strategies"""
    input_height, input_width = input_size
    # For MaxPool2d, we use the out_channels from the channel config
    num_channels = channels["out_channels"]
    kernel_size, padding, ceil_mode = pool_config["kernel_size"], pool_config["padding"], pool_config["ceil_mode"]

    # Create slice strategy based on config
    if slice_config["type"] == "no_slice":
        slice_strategy = None
    elif slice_config["type"] == "channel_slice":
        slice_strategy = ChannelSliceStrategyConfiguration(num_slices=slice_config["num_slices"])
    elif slice_config["type"] in ["height_slice", "width_slice"]:
        pytest.skip(f"{slice_config['type']} slicing is not supported for MaxPool2d")
    else:
        raise ValueError(f"Unknown slice type: {slice_config['type']}")

    configuration = MaxPool2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        channels=num_channels,
        batch_size=batch_size,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
        ceil_mode=ceil_mode,
        slice_strategy=slice_strategy,
    )

    torch_input_tensor, ttnn_input_tensor = create_pool2d_input_tensor(configuration)

    # Test with slicing
    layer = TtMaxPool2d(configuration, device)

    # For slicing tests, ensure input is on device with DRAM memory config
    if slice_config["type"] != "no_slice":
        ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)

    ttnn_output_tensor = layer(ttnn_input_tensor)

    # Reference implementation without slicing
    torch_output_tensor = torch.nn.functional.max_pool2d(
        torch_input_tensor,
        kernel_size=configuration.kernel_size,
        stride=configuration.stride,
        padding=configuration.padding,
        dilation=configuration.dilation,
        ceil_mode=configuration.ceil_mode,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("input_size", SLICE_INPUT_SIZES)
@pytest.mark.parametrize("channels", SLICE_CHANNEL_CONFIGS)  # Must be divisible by num_slices for channel slicing
@pytest.mark.parametrize("batch_size", [1])  # Only use batch size 1 for slicing tests
@pytest.mark.parametrize("upsample_config", UPSAMPLE_CONFIGS)
@pytest.mark.parametrize(
    "slice_config",
    SLICE_CONFIGS,
)
def test_upsample_slicing_strategies(input_size, channels, batch_size, upsample_config, slice_config, device):
    """Test Upsample with different slicing strategies"""
    input_height, input_width = input_size
    # For Upsample, we use the out_channels from the channel config
    num_channels = channels["out_channels"]
    scale_factor, mode = upsample_config["scale_factor"], upsample_config["mode"]

    # Create slice strategy based on config
    if slice_config["type"] == "no_slice":
        slice_strategy = None
    elif slice_config["type"] == "channel_slice":
        slice_strategy = ChannelSliceStrategyConfiguration(num_slices=slice_config["num_slices"])
    elif slice_config["type"] in ["height_slice", "width_slice"]:
        pytest.skip(f"{slice_config['type']} slicing is not supported for Upsample")
    else:
        raise ValueError(f"Unknown slice type: {slice_config['type']}")

    # Skip bilinear upsampling if channels or channels per slice is not divisible by 32
    if mode == "bilinear":
        if num_channels % 32 != 0:
            pytest.skip(f"Bilinear upsampling requires channels ({num_channels}) to be divisible by 32")
        if slice_strategy is not None and isinstance(slice_strategy, ChannelSliceStrategyConfiguration):
            channels_per_slice = num_channels // slice_strategy.get_num_slices()
            if channels_per_slice % 32 != 0:
                pytest.skip(
                    f"Bilinear upsampling requires channels per slice ({channels_per_slice}) to be divisible by 32"
                )

    configuration = UpsampleConfiguration(
        input_height=input_height,
        input_width=input_width,
        channels=num_channels,
        batch_size=batch_size,
        scale_factor=scale_factor,
        mode=mode,
        slice_strategy=slice_strategy,
    )

    torch_input_tensor, ttnn_input_tensor = create_upsample_input_tensor(configuration)

    # Test with slicing
    layer = TtUpsample(configuration, device)

    # For slicing tests, ensure input is on device with DRAM memory config
    if slice_config["type"] != "no_slice":
        ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)

    ttnn_output_tensor = layer(ttnn_input_tensor)

    # Reference implementation without slicing
    torch_output_tensor = torch.nn.functional.interpolate(
        torch_input_tensor,
        scale_factor=configuration.scale_factor,
        mode=configuration.mode,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.channels)
        .permute(0, 3, 1, 2),
        PCC_THRESHOLD,
    )
