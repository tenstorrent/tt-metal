# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.functional_vovnet.tt.common import Conv


class TtConvNormAct:
    def __init__(
        self,
        # kernel_size: int = 1,
        stride: int = 1,
        padding: int = 1,
        # dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        apply_act=True,
        norm_kwargs=None,
        act_kwargs=None,
        state_dict=None,
        base_address=None,
        device=None,
        torch_model=None,
        # parameters = None,
        # input_params = None,
        split_conv=False,
    ) -> None:
        self.device = device
        self.conv = Conv(
            device=device,
            model=torch_model,
            # parameters,
            path=base_address,
            # input_params = input_params,
            conv_params=[stride, stride, padding, padding],
            activation="relu",
            # enable_split_reader=True,
            # enable_act_double_buffer=True,
            split_conv=split_conv,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv(x)
        return x
