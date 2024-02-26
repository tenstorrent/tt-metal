# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_gather, tt_all_gather_torch


class TtLlamaMLP_optimized(nn.Module):
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

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        # FOR BRINGUP! Inputs are interleaved so shard them
        x_sharded = [
            tt_lib.tensor.interleaved_to_sharded(t, sharded_mem_config=self.model_config["PADDED_LN_MLP_OUTPUT_MEMCFG"])
            for t in x
        ]
        x = x_sharded

        hidden_states = []

        for i in range(len(x)):
            w1_out = tt_lib.operations.primary.matmul_1d(
                x[i],
                self.w1_list[i],
                program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
                output_mem_config=self.model_config["FF1_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FF1_MM_OUTPUT_DTYPE"],
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            w3_out = tt_lib.operations.primary.matmul_1d(
                x[i],
                self.w3_list[i],
                program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
                output_mem_config=self.model_config["FF3_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FF3_MM_OUTPUT_DTYPE"],
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            x[i].deallocate(True)
            hidden_states.append(
                tt_lib.tensor.mul(w1_out, w3_out, output_mem_config=self.model_config["FF13_MUL_OUTPUT_MEMCFG"])
            )
            w1_out.deallocate(True)
            w3_out.deallocate(True)

        for i in range(len(hidden_states)):
            # Put w2_inputs in DRAM
            hidden_states[i] = tt_lib.tensor.sharded_to_interleaved(
                hidden_states[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
        breakpoint()
        hidden_states = tt_lib.tensor.all_gather(
            hidden_states,
            dim=3,
            num_links=1,
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )
        # hidden_states = tt_all_gather_torch(hidden_states, dim=-1)
        # Put AllGather results in L1
        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.tensor.interleaved_to_sharded(
                hidden_states[i], sharded_mem_config=self.model_config["PADDED_ALL_GATHER_OUTPUT_MEMCFG"]
            )

        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.operations.primary.matmul_1d(
                hidden_states[i],
                self.w2_list[i],
                program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
                output_mem_config=self.model_config["FF2_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FF2_MM_OUTPUT_DTYPE"],
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        # FOR BRINGUP! Outputs are sharded. Interleave them
        hidden_states_interleaved = [
            tt_lib.tensor.sharded_to_interleaved(t, output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            for t in hidden_states
        ]

        return hidden_states_interleaved
