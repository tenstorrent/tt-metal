import pytest
import torch

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    Conv2dConfiguration,
    DeviceDescriptor,
    HeightShardedStrategyConfiguration,
    MaxPool2dConfiguration,
    TtConv2d,
    TtMaxPool2d,
    WidthShardedStrategyConfiguration,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_conv2d_input_tensor(configuration: Conv2dConfiguration):
    shape = (configuration.batch_size, configuration.in_channels, configuration.input_height, configuration.input_width)
    nchw = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(nchw, (0, 2, 3, 1))
    nhwc = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return nchw, nhwc


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 1024,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_height, input_width, in_channels, out_channels, batch_size, kernel_size, padding, sharding_strategy",
    (
        (32, 32, 16, 16, 4, 3, 1, AutoShardedStrategyConfiguration()),
        (64, 64, 32, 16, 2, 5, 0, HeightShardedStrategyConfiguration()),
        (16, 16, 64, 64, 1, 3, 1, WidthShardedStrategyConfiguration()),
        (128, 64, 16, 32, 1, 3, 1, BlockShardedStrategyConfiguration()),
    ),
)
def test_conv2d(
    input_height, input_width, in_channels, out_channels, batch_size, kernel_size, padding, sharding_strategy, device
):
    device_descriptor = DeviceDescriptor(device, (4, 4))

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

    layer = TtConv2d(configuration, device_descriptor)

    ttnn_output_tensor = layer(ttnn_input_tensor)
    torch_output_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        weight,
        bias.reshape(-1) if bias is not None else None,
        padding=configuration.padding,
    )

    output_height, output_width = torch_output_tensor.shape[-2:]  # [B, C, H, W]
    assert_with_pcc(
        torch_output_tensor,
        ttnn.to_torch(ttnn_output_tensor)
        .reshape(configuration.batch_size, output_height, output_width, configuration.out_channels)
        .permute(0, 3, 1, 2),
        0.999,
    )


def create_pool2d_input_tensor(configuration: MaxPool2dConfiguration):
    shape = (configuration.batch_size, configuration.channels, configuration.input_height, configuration.input_width)
    nchw = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(nchw, (0, 2, 3, 1)).reshape(
        1, 1, configuration.batch_size * configuration.input_height * configuration.input_width, configuration.channels
    )
    nhwc = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return nchw, nhwc


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 1024,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_height, input_width, channels, batch_size, kernel_size, padding, ceil_mode",
    (
        (32, 32, 16, 4, 2, 0, True),
        (128, 64, 8, 4, 4, 0, False),
    ),
)
def test_pool2d(input_height, input_width, channels, batch_size, kernel_size, padding, ceil_mode, device):
    device_descriptor = DeviceDescriptor(device, (4, 4))

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

    layer = TtMaxPool2d(configuration, device_descriptor)

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
        0.999,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 32768,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_height, input_width, batch_size",
    (
        (224, 224, 1),
        (32, 32, 4),
    ),
)
def test_downblock(input_height, input_width, batch_size, device):
    device_descriptor = DeviceDescriptor(device, (8, 8))
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
            input_height, input_width, 32, batch_size=batch_size, kernel_size=(2, 2), padding=(0, 0)
        ),
    ]

    torch_input_tensor, ttnn_input_tensor = create_conv2d_input_tensor(configurations[0])

    downblock = [
        TtConv2d(configurations[0], device_descriptor),
        TtConv2d(configurations[1], device_descriptor),
        TtMaxPool2d(configurations[2], device_descriptor),
    ]

    x = ttnn_input_tensor
    for layer in downblock:
        x = layer(x)
    ttnn_output_tensor = x
