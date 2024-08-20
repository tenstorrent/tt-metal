# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import nn
import ttnn

from models.experimental.ssd.tt.ssd_mobilenetv3_convlayer import (
    TtMobileNetV3ConvLayer,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_stemlayer import (
    TtMobileNetV3Stem,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_inverted_residual import (
    TtMobileNetV3InvertedResidual,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_inverted_squeeze import (
    TtMobileNetV3InvertedSqueeze,
)


class TtMobileNetV3Features(nn.Module):
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
            3,
            16,
            16,
            24,
            24,
            40,
            40,
            40,
            80,
            80,
            80,
            80,
            112,
            112,
        ]
        out_channels = [
            16,
            16,
            24,
            24,
            40,
            40,
            40,
            80,
            80,
            80,
            80,
            112,
            112,
            672,
        ]
        fc_channels = [
            0,
            0,
            0,
            0,
            24,
            32,
            32,
            0,
            0,
            0,
            0,
            120,
            168,
            0,
        ]
        expand_channels = [
            0,
            16,
            64,  # 2
            72,  # 3
            72,  # 4
            120,  # 5
            120,  # 6
            240,  # 7
            200,  # 8
            184,  # 9
            184,  # 10
            480,  # 11
            672,  # 12
            0,  # 13
        ]
        sequeeze_excitation = [4, 5, 6, 11, 12, 13, 14, 15]
        kernel_size = [3, 1, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 1]
        strides = [2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1]
        padding = [1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0]
        self.layers = nn.ModuleList()

        for i in range(14):
            if i == 0 or i == 13:
                self.block = TtMobileNetV3ConvLayer(
                    config,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=strides[i],
                    padding=padding[i],
                    use_activation=True,
                    activation="HS",
                    state_dict=state_dict,
                    base_address=f"{self.base_address_with_dot}.{'0'}.{i}",
                    device=device,
                )

            elif i == 1:
                self.block = TtMobileNetV3Stem(
                    config,
                    in_channels=in_channels[i],
                    expanded_channels=expand_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=strides[i],
                    padding=padding[i],
                    state_dict=state_dict,
                    base_address=f"{self.base_address_with_dot}.{'0'}.{i}",
                    device=device,
                )

            elif 1 < i < 13:
                if i >= 7:
                    use_activation = True
                    activation = "HS"
                else:
                    use_activation = True
                    activation = None
                if i not in sequeeze_excitation:
                    self.block = TtMobileNetV3InvertedResidual(
                        config,
                        in_channels=in_channels[i],
                        out_channels=out_channels[i],
                        expanded_channels=expand_channels[i],
                        kernel_size=kernel_size[i],
                        stride=strides[i],
                        padding=padding[i],
                        use_activation=use_activation,
                        activation=activation,
                        dilation=1,
                        state_dict=state_dict,
                        base_address=f"{self.base_address_with_dot}.{'0'}.{i}",
                    )

                else:
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
                        base_address=f"{self.base_address_with_dot}.{'0'}.{i}",
                    )

            self.layers.append(self.block)

    def forward(
        self,
        pixel_values: ttnn.Tensor,
    ) -> ttnn.Tensor:
        for i, layer_module in enumerate(self.layers):
            if i == 0:
                hidden_states = layer_module(pixel_values)
            else:
                hidden_states = layer_module(hidden_states)

        return hidden_states
