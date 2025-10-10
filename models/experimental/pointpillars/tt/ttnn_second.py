# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.pointpillars.tt.common import TtConv
from typing import Optional, Sequence


class TtSECOND:
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: Sequence[int] = [128, 128, 256],
        layer_nums: Sequence[int] = [3, 5, 5],
        layer_strides: Sequence[int] = [2, 2, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        init_cfg=None,
        pretrained: Optional[str] = None,
        parameters=None,
        device=None,
    ) -> None:
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]

        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                TtConv(
                    parameters=parameters["blocks"][i][0],
                    device=device,
                    input_params=[3, layer_strides[i], 1, out_channels[i], in_filters[i]],
                    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                    deallocate_activation=True if i == 0 else False,
                    halo=True if i == 0 else False,
                )
            ]
            for j in range(layer_num):
                block.append(
                    TtConv(
                        parameters=parameters["blocks"][i][(j + 1) * 3],
                        device=device,
                        input_params=[3, 1, 1, out_channels[i], out_channels[i]],
                        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                        deallocate_activation=True,
                    )
                )

            blocks.append(block)

        self.blocks = blocks

    def __call__(self, x):
        outs = []
        for i in range(len(self.blocks)):
            for j in range(len(self.blocks[i])):
                x = self.blocks[i][j](x)
            outs.append(x)
        return tuple(outs)
