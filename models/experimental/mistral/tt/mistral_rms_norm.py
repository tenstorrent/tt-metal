# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import tt_lib


class TtRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device=None,
        base_address=None,
        tt_cache_path=None,
        output_mem_config=None,
    ):
        super().__init__()
        self.eps = eps
        self.device = device
        self.output_mem_config = output_mem_config
        # bfp8 reduces PCC for so using weights in bfloat16
        self.weight = tt_lib.tensor.load_tensor(tt_cache_path + base_address + "weightDataType.BFLOAT16.bin")

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = tt_lib.tensor.rmsnorm(x, self.eps, self.weight)
        return x
