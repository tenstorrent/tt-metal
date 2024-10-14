from models.experimental.functional_mobilenetv3.ttnn.common import Conv
from models.experimental.functional_mobilenetv3.ttnn.ttnn_squeezeExcitation import ttnn_SqueezeExcitation as SElayer
import ttnn
from typing import Union, Tuple, Optional
from functools import partial
from torch import nn


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
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

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
    ):
        if activation_layer == ttnn.relu:
            self.activation_layer = None
            act = "relu"
        else:
            self.activation_layer = activation_layer
            act = ""
        if padding == None:
            padding = (kernel_size - 1) // 2 * dilation

        if groups == 576:
            width_sharding = True
        else:
            width_sharding = None
        self.conv = Conv(
            [stride, stride, padding, padding],
            parameters[0],
            activation=act,
            dilation=dilation,
            groups=groups,
            width_sharding=width_sharding,
        )

    def __call__(self, device, input_tensor):
        input_tensor = self.conv(device, input_tensor)
        if self.activation_layer != None:
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor


class ttnn_InvertedResidual:
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self, cnf: InvertedResidualConfig, se_layer=partial(SElayer, scale_activation=ttnn.hardsigmoid), parameters=None
    ):
        super().__init__()

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = ttnn.hardswish if cnf.use_hs else ttnn.relu
        index = 0

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(kernel_size=1, activation_layer=activation_layer, parameters=parameters[index])
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
            )
        )
        index += 1
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels, parameters=parameters[index]))
            index += 1

        # project
        layers.append(Conv2dNormActivation(kernel_size=1, activation_layer=None, parameters=parameters[index]))

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
