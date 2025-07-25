# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.functional_vovnet.tt.conv_norm_act import TtConvNormAct
from models.experimental.functional_vovnet.tt.sequential_append_list import (
    TtSequentialAppendList,
)
from models.experimental.functional_vovnet.tt.effective_se_module import (
    TtEffectiveSEModule,
)


class TtOsaBlock:
    def __init__(
        self,
        base_address=None,
        device=None,
        parameters=None,
    ) -> None:
        super().__init__()
        self.device = device

        self.conv_reduction = TtConvNormAct(
            stride=1,
            padding=0,
            parameters=parameters,
            base_address=f"{base_address}.conv_reduction",
            device=self.device,
        )

        self.conv_mid = TtSequentialAppendList(base_address=f"{base_address}", parameters=parameters, device=device)

        self.conv_concat = TtConvNormAct(
            stride=1,
            padding=0,
            parameters=parameters,
            base_address=f"{base_address}.conv_concat",
            device=device,
        )

        self.attn = TtEffectiveSEModule(
            kernel_size=1,
            stride=1,
            padding=0,
            parameters=parameters,
            base_address=f"{base_address}.attn",
            device=device,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        output = [ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)]
        if self.conv_reduction is not None:
            x = self.conv_reduction.forward(x)
        x = self.conv_mid.forward(x[0], output)
        x = self.conv_concat.forward(x)[0]
        if self.attn is not None:
            x = self.attn.forward(x)
        del output
        return x
