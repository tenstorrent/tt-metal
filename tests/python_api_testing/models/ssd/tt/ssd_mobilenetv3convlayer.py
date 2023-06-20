from typing import Union
import torch
import torch.nn as nn

import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from python_api_testing.models.ssd.reference.ssd_utils import assign_weight_batchnorm

ACT_FN_1 = tt_lib.tensor.relu
ACT_FN_2 = nn.Hardswish()

class TtMobileNetV3ConvLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
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
        self.config = config
        self.device = device
        self.host = host
        self.activation_str = activation

        weight = state_dict[f"{base_address}.{0}.weight"]
        bias = None
        self.convolution = fallback_ops.Conv2d(
            weights=weight,
            biases=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
        )

        self.normalization = nn.BatchNorm2d(
            num_features=out_channels,
            eps=0.001,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )

        assign_weight_batchnorm(self.normalization, state_dict, f"{base_address}.1")

        if use_activation:
            if activation == "HS":
                self.activation = ACT_FN_2
            else:
                self.activation = ACT_FN_1
        else:
            self.activation = None

    def forward(self, features: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        features = self.convolution(features)
        features = tt_to_torch_tensor(features, self.host)
        if self.activation is None:
            features = self.normalization(features)
            features = torch_to_tt_tensor_rm(features, self.device)
            return features
        else:
            features = self.normalization(features)
        if self.activation is not None:
            if self.activation_str == "HS":
                features = self.activation(features)
                features = torch_to_tt_tensor_rm(features, self.device)
            else:
                features = torch_to_tt_tensor_rm(features, self.device)
                features = self.activation(features)

        return features
