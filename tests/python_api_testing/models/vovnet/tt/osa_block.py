import torch.nn as nn

from python_api_testing.models.vovnet.tt.conv_norm_act import TtConvNormAct
from python_api_testing.models.vovnet.tt.sequential_append_list import (
    TtSequentialAppendList,
)
from python_api_testing.models.vovnet.tt.effective_se_module import (
    TtEffectiveSEModule,
)

import tt_lib


class TtOsaBlock(nn.Module):
    def __init__(
        self,
        in_chs=1,
        mid_chs=64,
        out_chs=64,
        layer_per_block=3,
        residual=False,
        depthwise=True,
        base_address=None,
        state_dict=None,
        host=None,
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.host = host
        self.residual = residual
        self.depthwise = depthwise
        self.state_dict = state_dict

        next_in_chs = in_chs
        if self.depthwise and next_in_chs != mid_chs:
            assert not residual
            self.conv_reduction = TtConvNormAct(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                apply_act=True,
                norm_kwargs=None,
                act_kwargs=None,
                state_dict=state_dict,
                base_address=f"{base_address}.conv_reduction",
                device=self.device,
                host=self.host,
            )
        else:
            self.conv_reduction = None

        self.conv_mid = TtSequentialAppendList(
            layer_per_block=layer_per_block,
            state_dict=state_dict,
            base_address=f"{base_address}",
            in_channels=128,
            groups=128,
        )

        # feature aggregation
        next_in_chs = in_chs + layer_per_block * mid_chs
        self.conv_concat = TtConvNormAct(
            in_channels=448,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            apply_act=True,
            norm_kwargs=None,
            act_kwargs=None,
            state_dict=state_dict,
            base_address=f"{base_address}.conv_concat",
            device=device,
            host=host,
        )

        self.attn = TtEffectiveSEModule(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=False,
            state_dict=state_dict,
            base_address=f"{base_address}.attn",
            device=device,
            host=host,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        output = [x]
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)
        x = self.conv_mid(x, output)
        x = self.conv_concat(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.residual:
            x = x + output[0]
        return x
