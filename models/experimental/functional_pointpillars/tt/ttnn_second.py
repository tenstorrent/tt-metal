# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_pointpillars.tt.common import TtConv
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
                    activation="relu",
                    deallocate_activation=True,
                    change_shard=True,
                )
            ]
            for j in range(layer_num):
                block.append(
                    TtConv(
                        parameters=parameters["blocks"][i][(j + 1) * 3],
                        device=device,
                        input_params=[3, 1, 1, out_channels[i], out_channels[i]],
                        activation="relu",
                        change_shard=True,
                        deallocate_activation=True,
                    )
                )

            blocks.append(block)

        self.blocks = blocks

        # assert not (init_cfg and pretrained), "init_cfg and pretrained cannot be setting at the same time"
        # if isinstance(pretrained, str):
        #     warnings.warn("DeprecationWarning: pretrained is a deprecated, " 'please use "init_cfg" instead')
        #     self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        # else:
        #     self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def __call__(self, x):
        outs = []
        print(self.blocks)
        for i in range(len(self.blocks)):
            for j in range(len(self.blocks[i])):
                x, h, w = self.blocks[i][j](x)
            x = ttnn.reshape(x, (1, h, w, x.shape[-1]))  # bs=1
            outs.append(x)
        return tuple(outs)
