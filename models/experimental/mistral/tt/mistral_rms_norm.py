# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import tt_lib
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


class TtRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        state_dict=None,
        device=None,
        base_address=None,
    ):
        super().__init__()
        self.eps = eps
        self.device = device
        weight = state_dict[f"{base_address}weight"]

        # converting to bfp8 reduces PCC
        self.weight = tt_lib.tensor.Tensor(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).reshape(-1).tolist(),
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        return tt_lib.tensor.rmsnorm(x, self.eps, self.weight)
