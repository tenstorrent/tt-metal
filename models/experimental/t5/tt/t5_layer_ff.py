# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.t5.tt.t5_layer_norm import TtT5LayerNorm
from models.experimental.t5.tt.t5_dense_act_dense import TtT5DenseActDense
from models.experimental.t5.tt.t5_dense_gated_act_dense import TtT5DenseGatedActDense


class TtT5LayerFF(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        if "is_gated_act" in config and config["is_gated_act"]:
            self.DenseReluDense = TtT5DenseGatedActDense(config, state_dict, f"{base_address}.DenseReluDense", device)
        else:
            self.DenseReluDense = TtT5DenseActDense(config, state_dict, f"{base_address}.DenseReluDense", device)

        self.layer_norm = TtT5LayerNorm(config, state_dict, f"{base_address}.layer_norm", device)
        self.dropout = torch.nn.Dropout(config["dropout_rate"])

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        # hidden_states = hidden_states + self.dropout(forwarded_states)
        hidden_states = ttnn.add(hidden_states, forwarded_states)
        return hidden_states
