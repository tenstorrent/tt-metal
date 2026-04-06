# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
import ttnn

from models.experimental.ssd.tt.ssd_mobilenetv3_convlayer import (
    TtMobileNetV3ConvLayer,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_inverted_squeeze import (
    TtMobileNetV3InvertedSqueeze,
)


class TtMobileNetV3extract(nn.Module):
    def __init__(
        self,
        config,
        state_dict=None,
        base_address="",
        device=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.base_address_with_dot = "" if base_address == "" else f"{base_address}"
        in_channels = [
            672,
            80,
            80,
            80,
        ]
        out_channels = [
            80,
            80,
            80,
            480,
        ]
        fc_channels = [
            168,
            120,
            120,
        ]
        expand_channels = [
            672,  # 14
            480,  # 15
            480,  # 16
        ]
        kernel_size = [5, 5, 5, 1]
        strides = [2, 1, 1, 1]
        padding = [2, 2, 2, 0]
        self.layers = nn.ModuleList()

        use_activation = True
        activation = "HS"

        for i in range(3):
            self.block = TtMobileNetV3InvertedSqueeze(
                config,
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                expanded_channels=expand_channels[i],
                fc_channels=fc_channels[i],
                kernel_size=kernel_size[i],
                stride=strides[i],
                padding=padding[i],
                use_activation=use_activation,
                activation=activation,
                dilation=1,
                state_dict=state_dict,
                base_address=f"{self.base_address_with_dot}.{'1'}.{i}",
                prune_conv=True if i == 0 else False,
            )
            self.layers.append(self.block)

        self.block = TtMobileNetV3ConvLayer(
            config,
            in_channels=in_channels[-1],
            out_channels=out_channels[-1],
            kernel_size=kernel_size[-1],
            stride=strides[-1],
            padding=padding[-1],
            use_activation=True,
            activation="HS",
            state_dict=state_dict,
            base_address=f"{self.base_address_with_dot}.{'1'}.{'3'}",
            device=device,
        )

        self.layers.append(self.block)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
    ) -> ttnn.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)

        return hidden_states
