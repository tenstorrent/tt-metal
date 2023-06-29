from pathlib import Path
import sys
from typing import List

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

import torch.nn as nn

from python_api_testing.models.vovnet.tt.separable_conv_norm_act import (
    TtSeparableConvNormAct,
)

import tt_lib
from tt_lib.fallback_ops import fallback_ops


class TtSequentialAppendList(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 128,
        layer_per_block: int = 3,
        state_dict=None,
        base_address=None,
        groups: int = 128,
    ) -> None:
        super(TtSequentialAppendList, self).__init__()
        self.layer_per_block = layer_per_block
        self.state_dict = state_dict
        self.base_address = base_address

        self.mid_convs = []
        for i in range(layer_per_block):
            conv = TtSeparableConvNormAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
                bias=False,
                channel_multiplier=1.0,
                groups=in_channels,
                state_dict=state_dict,
                base_address=f"{self.base_address}.conv_mid.{i}",
                device=None,
                host=None,
            )
            self.mid_convs.append(conv)

    def forward(
        self, x: tt_lib.tensor.Tensor, concat_list: List[tt_lib.tensor.Tensor]
    ) -> tt_lib.tensor.Tensor:
        for i, module in enumerate(self.mid_convs):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = fallback_ops.concat(concat_list, dim=1)
        return x
