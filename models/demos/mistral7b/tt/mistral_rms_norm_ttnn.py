# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import ttnn


class TtRMSNorm(nn.Module):
    def __init__(
        self,
        device,
        base_address,
        state_dict,
        model_config,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        self.model_config = model_config
        self.state_dict = state_dict

        rmsNorm_weight = f"{base_address}weight"

        self.weight = ttnn.from_torch(
            self.state_dict[rmsNorm_weight].unsqueeze(0).expand(32, -1),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
        # x = rms_decomp(x, self.weight, self.eps)
        return x
