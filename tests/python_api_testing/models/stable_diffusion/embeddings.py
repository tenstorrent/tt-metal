from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


import tt_lib as ttl
from python_api_testing.models.stable_diffusion.utils import make_linear
from tt_lib.fallback_ops import fallback_ops


class TtTimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        state_dict=None,
        base_address="",
        host=None,
        device=None,
    ):
        super().__init__()

        weights = state_dict[f"{base_address}.linear_1.weight"]
        bias = state_dict[f"{base_address}.linear_1.bias"]
        self.linear_1 = make_linear(
            in_features=in_channels,
            out_features=time_embed_dim,
            weights=weights,
            bias=bias,
            device=device,
        )
        self.act = None
        if act_fn == "silu":
            # self.act = fallback_ops.silu
            self.act = ttl.tensor.silu
        elif act_fn == "mish":
            assert False, "tt does not support nn.Mish() yet"
            self.act = nn.Mish()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        weights = state_dict[f"{base_address}.linear_2.weight"]
        bias = state_dict[f"{base_address}.linear_2.bias"]
        self.linear_2 = make_linear(
            in_features=time_embed_dim,
            out_features=time_embed_dim_out,
            weights=weights,
            bias=bias,
            device=device,
        )

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample
