from models.experimental.mobileNetV3.tt.ttnn_squeezeExcitation import ttnn_SqueezeExcitation as SElayer
import ttnn
from typing import Union, Tuple, Optional
from functools import partial
from models.experimental.mobileNetV3.tt.utils import _create_conv_config_from_params, post_conv_reshape
from models.tt_cnn.tt.builder import TtConv2d

from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
        input_height: int,
        input_width: int,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.input_height = input_height
        self.input_width = input_width

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class Conv2dNormActivation:
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        activation_layer=ttnn.relu,
        dilation=1,
        groups=1,
        parameters=None,
        device=None,
        input_height=1,
        input_width=1,
    ):
        if activation_layer == ttnn.relu:
            self.activation_layer = None
            activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        else:
            self.activation_layer = activation_layer
            activation = None

        # Normalize integer parameters to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if padding == None:
            padding = (kernel_size[0] - 1) // 2 * dilation[0]

        # Normalize padding to tuple if it's an integer
        if isinstance(padding, int):
            padding = (padding, padding)

        if groups == 576:
            width_sharding = True
        else:
            width_sharding = None

        # self.conv = Conv(
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     parameters=parameters[0],
        #     activation=activation,
        #     width_sharding=width_sharding,
        # )
        ############################################33
        self.conv_config = _create_conv_config_from_params(
            # input_height=1,
            # input_width=1,
            input_height=input_height,
            input_width=input_width,
            # input_height=parameters[0]["weight"].shape[1],
            # input_width=parameters[0]["weight"].shape[2],
            in_channels=parameters[0]["weight"].shape[1] * groups,
            out_channels=parameters[0]["weight"].shape[0],
            kernel_size=kernel_size,
            batch_size=1,
            parameters=parameters[0],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=activation,
            sharding_strategy=AutoShardedStrategyConfiguration(),
        )
        self.conv = TtConv2d(self.conv_config, device)
        #################################################3

    def __call__(self, device, input_tensor, return_output_dim=True):
        # input_tensor = self.conv(device, input_tensor)
        [input_tensor, [_out_height, _out_width]] = self.conv(input_tensor, return_output_dim=True)
        input_tensor = post_conv_reshape(input_tensor, out_height=_out_height, out_width=_out_width)
        if self.activation_layer != None:
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor
        # return [input_tensor, [_out_height, _out_width]]


class ttnn_InvertedResidual:
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_layer=partial(SElayer, scale_activation=ttnn.hardsigmoid),
        parameters=None,
        device=None,
        # input_height=1,
        # input_width=1,
    ):
        super().__init__()
        input_height = cnf.input_height
        input_width = cnf.input_width
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = ttnn.hardswish if cnf.use_hs else ttnn.relu
        index = 0

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    kernel_size=1,
                    activation_layer=activation_layer,
                    parameters=parameters[index],
                    device=device,
                    input_height=input_height,
                    input_width=input_width,
                )
            )
            index += 1

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                activation_layer=activation_layer,
                parameters=parameters[index],
                device=device,
                input_height=input_height,
                input_width=input_width,
            )
        )
        index += 1
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(
                se_layer(cnf.expanded_channels, squeeze_channels, parameters=parameters[index], device=device)
            )
            index += 1

        # project
        layers.append(
            Conv2dNormActivation(
                kernel_size=1,
                activation_layer=None,
                parameters=parameters[index],
                device=device,
                input_height=input_height // stride,
                input_width=input_width // stride,
                # input_height=input_height  if cnf.expanded_channels != cnf.input_channels else input_height//2,
                # input_width=input_width  if cnf.expanded_channels != cnf.input_channels else input_width//2,
            )
        )

        self.block = layers
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def __call__(self, device, input):
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

        if self.use_res_connect:
            result += input
        return result

    # def __call__(self, device, input):
    #     for i, layer in enumerate(self.block):
    #         if i == 0:
    #             # result = layer(device, input, return_output_dim=True)
    #             [result, [_out_height, _out_width]] = layer(device, input, return_output_dim=True)
    #             result = post_conv_reshape(result, out_height=_out_height, out_width=_out_width)
    #         else:
    #             result = layer(device, result)
    #     if self.use_res_connect:
    #         result += input
    #     return result
