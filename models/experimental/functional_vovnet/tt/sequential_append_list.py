# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from typing import List
import ttnn
from models.experimental.functional_vovnet.tt.separable_conv_norm_act import (
    TtSeparableConvNormAct,
)


class TtSequentialAppendList:
    def __init__(self, layer_per_block: int = 3, base_address=None, parameters=None, device=None) -> None:
        self.layer_per_block = layer_per_block
        self.base_address = base_address

        self.mid_convs = []
        for i in range(layer_per_block):
            conv = TtSeparableConvNormAct(
                stride=1,
                padding=1,
                # torch_model=torch_model,
                parameters=parameters,
                base_address=f"{self.base_address}.conv_mid.{i}",
                device=device,
            )
            self.mid_convs.append(conv)

    def forward(self, x: ttnn.Tensor, concat_list: List[ttnn.Tensor]) -> ttnn.Tensor:
        for i, module in enumerate(self.mid_convs):
            if i == 0:
                concat_list.append(ttnn.to_layout(module.forward(x)[0], layout=ttnn.TILE_LAYOUT))
            else:
                concat_list.append(ttnn.to_layout(module.forward(concat_list[-1])[0], layout=ttnn.TILE_LAYOUT))

        x = ttnn.concat(concat_list, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        del concat_list
        return x
