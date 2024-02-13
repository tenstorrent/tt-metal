# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_reduce


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
        self.w2_list = []
        self.w3_list = []

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
                    dim=-2,
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
        w2_outputs = []
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
            w2_in = tt_lib.tensor.mul(w1_out, w3_out, output_mem_config=self.model_config["FF13_MUL_OUTPUT_MEMCFG"])

            print("w2_in shape:", w2_in.shape())
            print("self.w2_list[i]:", self.w2_list[i].shape())

            w2_out = tt_lib.operations.primary.matmul_1d(
                w2_in,
                self.w2_list[i],
                program_config=self.model_config["FF2_MM_PROGCFG"],
                output_mem_config=self.model_config["FF2_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["FF2_MM_OUTPUT_DTYPE"],
            )
            w2_outputs.append(w2_out)

        for i in range(len(w2_outputs)):
            print("w2_out shape:", w2_outputs[i].shape())
        mlp_outputs = tt_all_reduce(w2_outputs, output_mem_config=self.model_config["ALL_REDUCE_OUTPUT_MEMCFG"])
        return mlp_outputs


# def tt_all_gather(tensors, dim=-1):
#     all_gathered_output = torch.cat(tensors, dim=dim)
#     simulated_all_gathered = [all_gathered_output for _ in range(len(tensors))]
#     return simulated_all_gathered


#     w2_inputs.append(w2_in)
# if len(w2_inputs) > 1:
#     for i in range(len(w2_inputs)):
#         w2_inputs[i] = tt_lib.tensor.sharded_to_interleaved(
#             w2_inputs[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
#         )

#     # w2_inputs = tt_all_gather(w2_inputs)
#     w2_inputs = tt_lib.tensor.all_gather(
#                 w2_inputs, dim=3, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
#                 )

#     for i in range(len(w2_in)):
#         w2_inputs[i] = tt_lib.tensor.interleaved_to_sharded(
#             w2_inputs[i], sharded_mem_config=self.model_config["MLP_ALL_GATHER_OUTPUT_MEMCFG"]
#         )

# for i in range(len(w2_inputs)):

#     w2_out = tt_lib.operations.primary.matmul_1d(
#         w2_inputs[i],
#         self.w2_list[i],
#         program_config=self.model_config["FF2_MM_PROGCFG"],
#         output_mem_config=self.model_config["FF2_MM_OUTPUT_MEMCFG"],
#         output_dtype=self.model_config["FF2_MM_OUTPUT_DTYPE"],
#     )
#     w2_outputs.append(w2_out)

# if len(w2_outputs) == 1:
#     w2_outputs = w2_outputs[0]

# return w2_outputs
