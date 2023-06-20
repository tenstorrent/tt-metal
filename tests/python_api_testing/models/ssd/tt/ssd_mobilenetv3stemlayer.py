import torch
import torch.nn as nn
import tt_lib
from python_api_testing.models.ssd.tt.ssd_mobilenetv3convlayer import (
    TtMobileNetV3ConvLayer,
)


class TtMobileNetV3Stem(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        padding: int = 1,
        state_dict=None,
        base_address="",
        device=None,
        host=None,
    ) -> None:
        super().__init__()

        self.conv_3x3 = TtMobileNetV3ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activation=True,
            groups=expanded_channels,
            state_dict=state_dict,
            base_address=f"{base_address}.1.block.0",
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
            base_address=f"{base_address}.1.block.1",
            device=device,
            host=host,
        )

    def forward(self, features: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return features
