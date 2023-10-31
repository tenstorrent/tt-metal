# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F

import tt_lib
import tests.models.bloom.bloom_utils as bloom_utils
import tests.models.bloom.bloom_gelu_forward as bloom_gelu_forward
from models.utility_functions import pad_by_zero


# class BloomMLP(nn.Module):
#     def __init__(self, config: BloomConfig):
#         super().__init__()
#         hidden_size = config.hidden_size

#         self.pretraining_tp = config.pretraining_tp
#         self.slow_but_exact = config.slow_but_exact
#         self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
#         self.gelu_impl = BloomGelu()
#         self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
#         self.hidden_dropout = config.hidden_dropout

#     def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

#         if self.pretraining_tp > 1 and self.slow_but_exact:
#             intermediate_output = torch.zeros_like(residual)
#             slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
#             for i in range(self.pretraining_tp):
#                 intermediate_output = intermediate_output + F.linear(
#                     hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
#                     self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
#                 )
#         else:
#             intermediate_output = self.dense_4h_to_h(hidden_states)

#         output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

#         return output


class TtBloomMLP(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferStorage.L1)
        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.training = False
        self.use_tt_gelu = False

        # Get the weights
        self.tt_weight_mlp_h4h = state_dict[f"{base_address}.dense_h_to_4h.weight"]
        self.tt_weight_mlp_4hh = state_dict[f"{base_address}.dense_4h_to_h.weight"]

        # Transpose the weights
        self.tt_weight_mlp_h4h = torch.transpose(self.tt_weight_mlp_h4h, -1, -2)
        self.tt_weight_mlp_4hh = torch.transpose(self.tt_weight_mlp_4hh, -1, -2)

        # Push weights to Tt device
        self.tt_weight_mlp_h4h = bloom_utils.torch2tt_tensor(
            self.tt_weight_mlp_h4h, device
        )
        self.tt_weight_mlp_4hh = bloom_utils.torch2tt_tensor(
            self.tt_weight_mlp_4hh, device
        )

        # Load biases
        self.tt_bias_mlp_h4h = pad_by_zero(
            state_dict[f"{base_address}.dense_h_to_4h.bias"], device
        )[0]
        self.tt_bias_mlp_4hh = pad_by_zero(
            state_dict[f"{base_address}.dense_4h_to_h.bias"], device
        )[0]

        # self.gelu_impl = bloom_gelu_forward.tt_bloom_gelu_forward
        # self.gelu_impl = bloom_gelu_forward.bloom_gelu_forward

    def forward(self, hidden_states, residual, device):
        # h4h = self.dense_h_to_4h(hidden_states)
        h4h = bloom_utils.tt_matmul(hidden_states, self.tt_weight_mlp_h4h, device)
        h4h = tt_lib.tensor.bcast(
            h4h,
            self.tt_bias_mlp_h4h,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            self.mem_config,
        )

        if self.use_tt_gelu:
            hidden_states = bloom_gelu_forward.tt_bloom_gelu_forward(h4h, device)
        else:
            h4h = bloom_utils.tt2torch_tensor(h4h)
            hidden_states = bloom_gelu_forward.bloom_gelu_forward(h4h)
            hidden_states = bloom_utils.torch2tt_tensor(hidden_states, device)

        intermediate_output = bloom_utils.tt_matmul(
            hidden_states, self.tt_weight_mlp_4hh, device
        )
        intermediate_output = tt_lib.tensor.bcast(
            intermediate_output,
            self.tt_bias_mlp_4hh,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
            self.mem_config,
        )

        # Dropout is used in training only
        # intermediate_output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
        output = tt_lib.tensor.add(
            residual, intermediate_output, output_mem_config=self.mem_config
        )

        return output
