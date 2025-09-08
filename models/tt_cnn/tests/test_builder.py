import pytest
import torch

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    Conv2dConfiguration,
    DeviceDescriptor,
    HeightShardedStrategyConfiguration,
    TtConv2d,
    WidthShardedStrategyConfiguration,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_torch_weight_and_bias(configuration: Conv2dConfiguration):
    weight_shape = (
        configuration.out_channels,
        configuration.in_channels // configuration.groups,
        configuration.kernel_size[0],
        configuration.kernel_size[1],
    )
    weight = torch.randn(weight_shape, dtype=torch.bfloat16).float()
    bias_shape = (
        1,
        1,
        1,
        configuration.out_channels,
    )
    bias = torch.randn(bias_shape, dtype=torch.bfloat16).float()
    return weight, bias


def create_input_tensor(configuration: Conv2dConfiguration):
    shape = (configuration.batch_size, configuration.in_channels, configuration.input_height, configuration.input_width)
    nchw = torch.randn(shape, dtype=torch.bfloat16).float()

    nhwc = torch.permute(nchw, (0, 2, 3, 1))
    nhwc = ttnn.from_torch(nhwc, layout=ttnn.ROW_MAJOR_LAYOUT)

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
def test_basic_conv2d(
    input_height, input_width, in_channels, out_channels, batch_size, kernel_size, padding, sharding_strategy, device
):
    device_descriptor = DeviceDescriptor(device, (4, 4))

    configuration = Conv2dConfiguration(
        input_height,
        input_width,
        in_channels,
        out_channels,
        batch_size=batch_size,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
        sharding_strategy=sharding_strategy,
    )

    weight, bias = create_torch_weight_and_bias(configuration)
    torch_input_tensor, ttnn_input_tensor = create_input_tensor(configuration)

    layer = TtConv2d(configuration, weight, bias, device_descriptor)

    ttnn_output_tensor = layer(ttnn_input_tensor)
    torch_output_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        weight,
        bias.reshape(-1),
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
