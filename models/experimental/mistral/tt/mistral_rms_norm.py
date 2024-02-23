# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import tt_lib
from models.utility_functions import torch2tt_tensor


# Manual implementation of rmsnorm
def rms_decomp(x, norm_weight, eps):
    squared = tt_lib.tensor.pow(x, 2)
    # mean_squared = tt_lib.tensor.mean(squared, )
    sum_squared = tt_lib.tensor.reduce(squared, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.W, scaler=1.0)
    # Tensor is 1,1,32,1+31 now
    mean_squared = tt_lib.tensor.div_unary(sum_squared, x.shape()[-1])
    mean_squared_eps = tt_lib.tensor.add_unary(mean_squared, eps)
    rms = tt_lib.tensor.pow(mean_squared_eps, eps)
    rms_recip = tt_lib.tensor.recip(rms)
    normed_x = tt_lib.tensor.bcast(x, rms_recip, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W)
    norm_out = tt_lib.tensor.mul(normed_x, norm_weight)
    return norm_out


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

        self.weight = torch2tt_tensor(
            self.state_dict[rmsNorm_weight].unsqueeze(0).expand(32, -1),
            self.device,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = tt_lib.tensor.rmsnorm(x, self.eps, self.weight)
        # x = rms_decomp(x, self.weight, self.eps)
        return x
