# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
from typing import Optional


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)
        self.act = nn.SiLU()

    def forward(self, sample, condition=None):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample
