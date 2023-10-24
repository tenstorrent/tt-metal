# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import tt_lib
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm

#TODO: use on-device RMSNorm
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
        self.weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}weight"], self.device)

    def _norm(self, x):
        pow_tensor = tt_lib.tensor.power(x, 2)
        pow_tensor = tt_to_torch_tensor(pow_tensor).mean(-1, keepdim=True) + self.eps
        pow_tensor = torch_to_tt_tensor_rm(pow_tensor, self.device, put_on_device=False)
        pow_tensor = tt_lib.tensor.sqrt(pow_tensor)
        pow_tensor = tt_lib.tensor.recip(pow_tensor)
        return tt_lib.tensor.bcast(x, pow_tensor, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.W)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        output = self._norm(x)
        return tt_lib.tensor.bcast(output, self.weight, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.H)
