# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
from models.utility_functions import torch2tt_tensor


class TtMistralMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        model_config,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device

        self.model_config = model_config

        w1_str = f"{base_address}w1.weight"
        w2_str = f"{base_address}w2.weight"
        w3_str = f"{base_address}w3.weight"

        self.w1_weights = torch2tt_tensor(
            torch.transpose(
                self.state_dict[w1_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["FF1_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["FF1_MM_WEIGHTS_DTYPE"],
        )
        self.w2_weights = torch2tt_tensor(
            torch.transpose(
                self.state_dict[w2_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["FF2_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["FF2_MM_WEIGHTS_DTYPE"],
        )
        self.w3_weights = torch2tt_tensor(
            torch.transpose(
                self.state_dict[w3_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["FF3_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["FF3_MM_WEIGHTS_DTYPE"],
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """

        w1_out = tt_lib.operations.primary.matmul_1d(
            x,
            self.w1_weights,
            program_config=self.model_config["FF1_MM_PROGCFG"],  # specifies fused silu activation
            output_mem_config=self.model_config["FF1_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FF1_MM_OUTPUT_DTYPE"],
        )
        w1_out = tt_lib.tensor.silu(w1_out)

        w3_out = tt_lib.operations.primary.matmul_1d(
            x,
            self.w3_weights,
            program_config=self.model_config["FF3_MM_PROGCFG"],
            output_mem_config=self.model_config["FF3_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FF3_MM_OUTPUT_DTYPE"],
        )

        x.deallocate(True)

        w2_in = tt_lib.tensor.mul(w1_out, w3_out, output_mem_config=self.model_config["FF1_FF3_MUL_OUTPUT_MEMCFG"])

        # torch_x = torch.randn((1, 1, 32, 14336), dtype=torch.bfloat16)
        # w2_in = ttnn.from_torch(torch_x, layout=ttnn.TILE_LAYOUT, device=self.device).value

        # torch_w = torch.randn((1, 1, 14336, 4096), dtype=torch.bfloat16)
        # w = ttnn.from_torch(torch_w, layout=ttnn.TILE_LAYOUT, device=self.device).value

        w2_out = tt_lib.operations.primary.matmul_1d(
            w2_in,
            self.w2_weights,
            program_config=self.model_config["FF2_MM_PROGCFG"],
            output_mem_config=self.model_config["FF2_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FF2_MM_OUTPUT_DTYPE"],
        )

        return w2_out
