from typing import Union
import torch.nn as nn
import tt_lib
from models.ssd.tt.ssd_mobilenetv3_convlayer import (
    TtMobileNetV3ConvLayer,
)


class TtMobileNetV3InvertedResidual(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 1,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_activation: Union[bool, str] = False,
        activation="",
        state_dict=None,
        base_address="",
        device=None,
        host=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.host = host

        self.expand_1x1 = TtMobileNetV3ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_activation=use_activation,
            activation=activation,
            state_dict=state_dict,
            base_address=f"{base_address}.block.0",
            device=device,
            host=host,
        )

        self.conv_3x3 = TtMobileNetV3ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels,
            dilation=dilation,
            use_activation=use_activation,
            activation=activation,
            state_dict=state_dict,
            base_address=f"{base_address}.block.1",
            device=device,
            host=host,
        )

        self.reduce_1x1 = TtMobileNetV3ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_activation=False,
            state_dict=state_dict,
            base_address=f"{base_address}.block.2",
            device=device,
            host=host,
        )

    def forward(self, features: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return features
