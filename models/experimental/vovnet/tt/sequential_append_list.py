# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from typing import List
import ttnn
from models.experimental.vovnet.tt.separable_conv_norm_act import (
    TtSeparableConvNormAct,
)

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from models.experimental.yolo_common.yolo_utils import concat


class TtSequentialAppendList:
    def __init__(self, layer_per_block: int = 3, base_address=None, parameters=None, device=None, lay_idx=1000) -> None:
        self.layer_per_block = layer_per_block
        self.base_address = base_address
        self.lay_idx = lay_idx
        self.mid_convs = []
        for idx, i in enumerate(range(layer_per_block)):
            conv = TtSeparableConvNormAct(
                stride=1,
                padding=1,
                parameters=parameters,
                base_address=f"{self.base_address}.conv_mid.{i}",
                device=device,
                lay_idx=f"{self.lay_idx}_{idx}",
            )
            self.mid_convs.append(conv)

    def forward(self, x: ttnn.Tensor, concat_list: List[ttnn.Tensor]) -> ttnn.Tensor:
        if use_signpost:
            signpost(header="sequential_append_list")

        for i, module in enumerate(self.mid_convs):
            if i == 0:
                concat_list.append(ttnn.to_layout(module.forward(x)[0], layout=ttnn.TILE_LAYOUT))
            else:
                concat_list.append(ttnn.to_layout(module.forward(concat_list[-1])[0], layout=ttnn.TILE_LAYOUT))

        if concat_list[0].shape[1] != 1:
            concat_list[0] = ttnn.reshape(
                concat_list[0],
                (
                    1,
                    1,
                    concat_list[0].shape[0] * concat_list[0].shape[1] * concat_list[0].shape[2],
                    concat_list[0].shape[-1],
                ),
            )

        x = concat(-1, False, *concat_list)
        del concat_list

        return x
