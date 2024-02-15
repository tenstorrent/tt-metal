# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_gather, tt_all_reduce


class TtLlamaMLP_optimized(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        all_gather=True,
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
        self.w2_list = []
        self.w3_list = []

        self.all_gather = all_gather

        if self.all_gather:
            FF2_frac_dim = -1
        else:
            FF2_frac_dim = -2

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
                tt_memory_config=self.model_config["FF1_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FF1_MM_WEIGHTS_DTYPE"],
            )
            w2 = torch2tt_tensor(
                torch.chunk(
                    torch.transpose(
                        self.state_dict[w2_str],
                        -2,
                        -1,
                    ),
                    self.num_devices,
                    dim=FF2_frac_dim,
                )[i],
                self.devices[i],
                tt_memory_config=self.model_config["FF2_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FF2_MM_WEIGHTS_DTYPE"],
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
                tt_memory_config=self.model_config["FF3_MM_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["FF3_MM_WEIGHTS_DTYPE"],
            )
            self.w1_list.append(w1)
            self.w2_list.append(w2)
            self.w3_list.append(w3)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        if len(x) > 1:
            print("num devices:", self.num_devices)
            print("w1:0 shape:", self.w1_list[0].shape())
            print("x:0 shape:", x[0].shape())

        # FOR BRINGUP! Inputs are interleaved so shard them
        x_sharded = [
            tt_lib.tensor.interleaved_to_sharded(t, sharded_mem_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"])
            for t in x
        ]
        x = x_sharded

        hidden_states = []

        for i in range(len(x)):
            w1_out = tt_lib.operations.primary.matmul_1d(
                x[i],
                self.w1_list[i],
                program_config=self.model_config["FF1_MM_PROGCFG"],
                output_mem_config=self.model_config["FF1_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FF1_MM_OUTPUT_DTYPE"],
            )

            w3_out = tt_lib.operations.primary.matmul_1d(
                x[i],
                self.w3_list[i],
                program_config=self.model_config["FF3_MM_PROGCFG"],
                output_mem_config=self.model_config["FF3_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FF3_MM_OUTPUT_DTYPE"],
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
        hidden_states = tt_all_gather(hidden_states, dim=-1)
        # Put AllGather results in L1
        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.tensor.interleaved_to_sharded(
                hidden_states[i], sharded_mem_config=self.model_config["ALL_GATHER_OUTPUT_MEMCFG"]
            )

        for i in range(len(hidden_states)):
            print("w2_input shape:", hidden_states[i].shape())
            print("self.w2_list[i]:", self.w2_list[i].shape())
            hidden_states[i] = tt_lib.operations.primary.matmul_1d(
                hidden_states[i],
                self.w2_list[i],
                program_config=self.model_config["FF2_MM_PROGCFG"],
                output_mem_config=self.model_config["FF2_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FF2_MM_OUTPUT_DTYPE"],
            )

        # FOR BRINGUP! Outputs are sharded. Interleave them
        hidden_states_interleaved = [
            tt_lib.tensor.sharded_to_interleaved(t, output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            for t in hidden_states
        ]

        return hidden_states_interleaved

    # All Reduce forward pass
    # def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
    #     if len(x) > 1:
    #         print("num devices:", self.num_devices)
    #         print("w1:0 shape:", self.w1_list[0].shape())
    #         print("x:0 shape:", x[0].shape())
    #     w2_outputs = []
    #     for i in range(len(x)):
    #         w1_out = tt_lib.operations.primary.matmul_1d(
    #             x[i],
    #             self.w1_list[i],
    #             program_config=self.model_config["FF1_MM_PROGCFG"],
    #             output_mem_config=self.model_config["FF1_MM_OUTPUT_MEMCFG"],
    #             output_dtype=self.model_config["FF1_MM_OUTPUT_DTYPE"],
    #         )

    #         w3_out = tt_lib.operations.primary.matmul_1d(
    #             x[i],
    #             self.w3_list[i],
    #             program_config=self.model_config["FF3_MM_PROGCFG"],
    #             output_mem_config=self.model_config["FF3_MM_OUTPUT_MEMCFG"],
    #             output_dtype=self.model_config["FF3_MM_OUTPUT_DTYPE"],
    #         )

    #         x[i].deallocate(True)
    #         w2_in = tt_lib.tensor.mul(w1_out, w3_out, output_mem_config=self.model_config["FF13_MUL_OUTPUT_MEMCFG"])

    #         print("w2_input shape:", w2_in.shape())
    #         print("self.w2_list[i]:", self.w2_list[i].shape())

    #         w2_out = tt_lib.operations.primary.matmul_1d(
    #             w2_in,
    #             self.w2_list[i],
    #             program_config=self.model_config["FF2_MM_PROGCFG"],
    #             output_mem_config=self.model_config["FF2_MM_OUTPUT_MEMCFG"],
    #             output_dtype=self.model_config["FF2_MM_OUTPUT_DTYPE"],
    #         )
    #         w2_outputs.append(w2_out)

    #     for i in range(len(w2_outputs)):
    #         print("w2_output shape:", w2_outputs[i].shape())
    #     mlp_outputs = tt_all_reduce(w2_outputs, output_mem_config=self.model_config["ALL_REDUCE_OUTPUT_MEMCFG"])
    #     return mlp_outputs
