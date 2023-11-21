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
        base_address=None,
        tt_cache_path=None,
    ):
        super().__init__()
        self.eps = eps

        # bfp8 reduces PCC for so using weights in bfloat16
        self.weight = tt_lib.tensor.load_tensor(tt_cache_path + base_address + "weightDataType.BFLOAT16.bin")

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        return tt_lib.tensor.rmsnorm(x, self.eps, self.weight)
