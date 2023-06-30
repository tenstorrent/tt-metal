from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")

import torch
from torch import nn
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from models.ssd.tt.ssd_mobilenetv3_convlayer import (
    TtMobileNetV3ConvLayer,
)
from models.ssd.tt.ssd_mobilenetv3_stemlayer import (
    TtMobileNetV3Stem,
)
from models.ssd.tt.ssd_mobilenetv3_inverted_residual import (
    TtMobileNetV3InvertedResidual,
)
from models.ssd.tt.ssd_mobilenetv3_inverted_squeeze import (
    TtMobileNetV3InvertedSqueeze,
)
from models.ssd.tt.ssd_mobilenetv3_classifier import (
    TtClassifier,
)


class TtMobileNetV3Model(nn.Module):
    def __init__(
        self,
        config,
        state_dict=None,
        base_address="",
        device=None,
        host=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.host = host
        self.base_address_with_dot = "" if base_address == "" else f"{base_address}"

        in_channels = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160]
        out_channels = [24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160]
        fc_channels = [0, 0, 24, 32, 32, 0, 0, 0, 0, 120, 168, 168, 240, 240]
        expand_channels = [
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
            672,  # 13
            960,  # 14
            960,  # 15
        ]
        sequeeze_excitation = [4, 5, 6, 11, 12, 13, 14, 15]
        kernel_size = [3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]
        padding = [1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2]

        self.layer = nn.ModuleList()

        self.conv_layer = TtMobileNetV3ConvLayer(
            config,
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            use_activation=True,
            activation="HS",
            state_dict=state_dict,
            base_address=f"{self.base_address_with_dot}.0",
            device=device,
            host=host,
        )

        self.stem_layer = TtMobileNetV3Stem(
            config,
            in_channels=16,
            expanded_channels=16,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=1,
            state_dict=state_dict,
            base_address=f"{self.base_address_with_dot}.1",
            device=device,
            host=host,
        )

        for i in range(14):
            if i + 2 >= 7:
                use_activation = True
                activation = "HS"
            else:
                use_activation = False
                activation = None
            if i + 2 not in sequeeze_excitation:
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
                    base_address=f"{self.base_address_with_dot}.{i+2}",
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
                    base_address=f"{self.base_address_with_dot}.{i+2}",
                )

            self.layer.append(self.block)

        self.conv_1x1 = TtMobileNetV3ConvLayer(
            config,
            in_channels=in_channels[-1],
            out_channels=expand_channels[-1],
            kernel_size=1,
            stride=1,
            padding=1,
            use_activation=True,
            activation="HS",
            state_dict=state_dict,
            base_address=f"{self.base_address_with_dot}.16",
        )

        self.avgpool = fallback_ops.AdaptiveAvgPool2d(1)

        self.classifier = TtClassifier(state_dict=state_dict, device=device, host=host)

    def forward(
        self,
        pixel_values: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        hidden_states = self.conv_layer(pixel_values)
        hidden_states = self.stem_layer(hidden_states)

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        last_hidden_state = self.conv_1x1(hidden_states)

        avg_out = self.avgpool(last_hidden_state)

        flatten_in = tt_to_torch_tensor(avg_out, self.host)
        flatten = torch.flatten(flatten_in, start_dim=1)
        flatten_out = torch.unsqueeze(torch.unsqueeze(flatten, 0), 0)
        flatten = torch_to_tt_tensor_rm(flatten_out, self.device)

        classifier = self.classifier(flatten)

        return classifier
