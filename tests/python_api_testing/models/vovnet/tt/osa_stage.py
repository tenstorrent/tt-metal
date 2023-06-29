import torch.nn as nn

from python_api_testing.models.vovnet.tt.osa_block import TtOsaBlock

import tt_lib.fallback_ops


class TtOsaStage(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_per_stage: int = 1,
        layer_per_block: int = 3,
        downsample=True,
        residual=True,
        depthwise=False,
        base_address=None,
        device=None,
        host=None,
        state_dict=None,
    ):
        super(TtOsaStage, self).__init__()
        self.device = device
        self.base_address = f"{base_address}.blocks.0"
        self.state_dict = state_dict
        self.host = host
        if downsample:
            self.pool = tt_lib.fallback_ops.MaxPool2d(
                kernel_size=3, stride=2, ceil_mode=True
            )
        else:
            self.pool = None

        blocks = []
        for i in range(block_per_stage):
            last_block = i == block_per_stage - 1
            blocks += [
                TtOsaBlock(
                    in_chs=1,
                    mid_chs=128,
                    out_chs=128,
                    layer_per_block=3,
                    residual=False,
                    depthwise=True,
                    base_address=self.base_address,
                    state_dict=self.state_dict,
                    device=self.device,
                    host=self.host,
                )
            ]
            in_chs = out_chs
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        if self.pool is not None:
            x = self.pool(x)
        x = self.blocks(x)
        return x
