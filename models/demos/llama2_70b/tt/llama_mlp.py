# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_gather_torch


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

        H = 8 * 1024
        PADDED_H4 = 32 * 1024
        H4 = 28 * 1024
        padded_w1 = torch.zeros(H, PADDED_H4)
        padded_w3 = torch.zeros(H, PADDED_H4)
        padded_w2 = torch.zeros(PADDED_H4, H)
        padded_w1[:, :H4] = self.state_dict[w1_str].transpose(-2, -1)
        padded_w3[:, :H4] = self.state_dict[w3_str].transpose(-2, -1)
        padded_w2[:H4, :] = self.state_dict[w2_str].transpose(-2, -1)

        for i in range(self.num_devices):
            w1 = torch2tt_tensor(
                torch.chunk(
                    padded_w1,
                    self.num_devices,
                    dim=-1,
                )[i],
                self.devices[i],
                tt_memory_config=self.model_config["FF1_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FF1_MM_WEIGHTS_DTYPE"],
            )
            w3 = torch2tt_tensor(
                torch.chunk(
                    padded_w3,
                    self.num_devices,
                    dim=-1,
                )[i],
                self.devices[i],
                tt_memory_config=self.model_config["FF2_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FF2_MM_WEIGHTS_DTYPE"],
            )
            w2 = torch2tt_tensor(
                torch.chunk(
                    padded_w2,
                    self.num_devices,
                    dim=-1,
                )[i],
                self.devices[i],
                tt_memory_config=self.model_config["FF3_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FF3_MM_WEIGHTS_DTYPE"],
            )

            self.w1_list.append(w1)
            self.w3_list.append(w3)
            self.w2_list.append(w2)

    def forward(self, xs):
        w2_inputs = []
        w2_outputs = []
        for i in range(self.num_devices):
            # w1_out = tt_lib.tensor.matmul(
            w1_out = tt_lib.operations.primary.matmul_1d(
                xs[i], self.w1_list[i], fp32_dest_acc_en=True, packer_l1_acc=True
            )
            w1_sigmoid = tt_lib.tensor.silu(w1_out)

            # w3_out = tt_lib.tensor.matmul(
            w3_out = tt_lib.operations.primary.matmul_1d(
                xs[i], self.w3_list[i], fp32_dest_acc_en=True, packer_l1_acc=True
            )

            w2_in = tt_lib.tensor.mul(w1_sigmoid, w3_out)
            w2_inputs.append(w2_in)

        w2_in_replicated = tt_all_gather_torch(w2_inputs, dim=-1)

        for i in range(self.num_devices):
            # w2_out = tt_lib.tensor.matmul(
            w2_out = tt_lib.operations.primary.matmul_1d(
                w2_in_replicated[i], self.w2_list[i], fp32_dest_acc_en=True, packer_l1_acc=True
            )

            w2_outputs.append(w2_out)

        return w2_outputs
