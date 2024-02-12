# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_reduce


class TtLlamaMLP(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        w1_str = f"{layer_name}.feed_forward.w1.weight"
        w2_str = f"{layer_name}.feed_forward.w2.weight"
        w3_str = f"{layer_name}.feed_forward.w3.weight"

        self.w1_list = []
        self.w3_list = []
        self.w2_list = []
        for i in range(self.num_devices):
            w1 = torch2tt_tensor(
                torch.chunk(
                    torch.transpose(
                        self.state_dict[w1_str],
                        -2,
                        -1,
                    ),
                    self.num_devices,
                    dim=-1,
                )[i],
                self.devices[i],
            )
            w3 = torch2tt_tensor(
                torch.chunk(
                    torch.transpose(
                        self.state_dict[w3_str],
                        -2,
                        -1,
                    ),
                    self.num_devices,
                    dim=-1,
                )[i],
                self.devices[i],
            )
            w2 = torch2tt_tensor(
                torch.chunk(
                    torch.transpose(
                        self.state_dict[w2_str],
                        -2,
                        -1,
                    ),
                    self.num_devices,
                    dim=-2,
                )[i],
                self.devices[i],
            )

            self.w1_list.append(w1)
            self.w3_list.append(w3)
            self.w2_list.append(w2)

    def forward(self, xs):
        w2_outputs = []
        for i in range(self.num_devices):
            x = xs[i]
            w1 = self.w1_list[i]
            w3 = self.w3_list[i]
            w2 = self.w2_list[i]
            w1_out = tt_lib.tensor.matmul(
                x,
                w1,
            )
            w1_sigmoid = tt_lib.tensor.silu(w1_out)

            w3_out = tt_lib.tensor.matmul(
                x,
                w3,
            )

            w2_in = tt_lib.tensor.mul(w1_sigmoid, w3_out)
            w2_out = tt_lib.tensor.matmul(
                w2_in,
                w2,
            )

            w2_outputs.append(w2_out)

        mlp_outputs = tt_all_reduce(w2_outputs)
        return mlp_outputs
