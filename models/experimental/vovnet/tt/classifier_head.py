# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

import tt_lib.fallback_ops

from models.helper_funcs import Linear
from models.experimental.vovnet.tt.select_adaptive_pool2d import (
    TtSelectAdaptivePool2d,
)

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)


class TtClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: str = "avg",
        use_conv: bool = False,
        input_fmt: str = "NCHW",
        device=None,
        state_dict=None,
        base_address=None,
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
        """
        super(TtClassifierHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.use_conv = use_conv
        self.input_fmt = input_fmt
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address

        self.global_pool = TtSelectAdaptivePool2d(
            output_size=1, pool_type="Fast", flatten=True, input_fmt="NCHW", device=None
        )

        self.weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc.weight"], self.device, put_on_device=False)
        self.bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.fc.bias"], self.device, put_on_device=False)

        self.shape = (1, 1, 1, 1000)

        self.fc = Linear(self.in_features, self.num_classes, self.weight, self.bias)

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.fc(x)
        x = tt_lib.fallback_ops.reshape(x, *(self.shape))
        return x
