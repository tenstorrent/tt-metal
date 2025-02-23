# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
