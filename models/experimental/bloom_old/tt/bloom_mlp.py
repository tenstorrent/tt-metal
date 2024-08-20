# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F

import ttnn
from fused_ops.linear import Linear as TtLinear

import models.experimental.bloom_old.bloom_utils as bloom_utils
import models.experimental.bloom_old.tt.bloom_gelu_forward as bloom_gelu_forward


class TtBloomMLP(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.training = False

        self.tt_weight_mlp_h4h = bloom_utils.tt_load_layer_weights(f"{base_address}.dense_h_to_4h.weight", state_dict)
        self.tt_bias_mlp_h4h = bloom_utils.tt_load_layer_weights(f"{base_address}.dense_h_to_4h.bias", state_dict)

        self.tt_weight_mlp_4hh = bloom_utils.tt_load_layer_weights(f"{base_address}.dense_4h_to_h.weight", state_dict)
        self.tt_bias_mlp_4hh = bloom_utils.tt_load_layer_weights(f"{base_address}.dense_4h_to_h.bias", state_dict)

        self.dense_h_to_4h = TtLinear(
            self.hidden_size,
            4 * self.hidden_size,
            self.tt_weight_mlp_h4h,
            self.tt_bias_mlp_h4h,
            device,
        )
        self.dense_4h_to_h = TtLinear(
            4 * self.hidden_size,
            self.hidden_size,
            self.tt_weight_mlp_4hh,
            self.tt_bias_mlp_4hh,
            device,
        )

        self.gelu_impl = bloom_gelu_forward.tt_bloom_gelu_forward

    def forward(self, hidden_states, residual, device):
        h4h = self.dense_h_to_4h(hidden_states)
        hidden_states = self.gelu_impl(h4h, device)
        intermediate_output = self.dense_4h_to_h(hidden_states)

        # Dropout is used in training only
        # intermediate_output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
        output = ttnn.add(residual, intermediate_output)

        return output
