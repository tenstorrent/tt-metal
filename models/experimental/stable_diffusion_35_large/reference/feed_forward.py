# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention.py
class FeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        approximate: str = "none",
    ) -> None:
        super().__init__()

        inner_dim = 4 * dim

        self.net = torch.nn.ModuleList([])
        self.net.append(Gelu(dim, inner_dim, approximate=approximate))
        self.net.append(torch.nn.Identity())
        self.net.append(torch.nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/activations.py
class Gelu(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        # This is not needed anymore, but let's keep it for now to obtain the
        # same results as `diffusers`.
        if x.device.type == "mps":
            return torch.nn.functional.gelu(x.to(dtype=torch.float32), approximate=self.approximate).to(dtype=x.dtype)

        return torch.nn.functional.gelu(x, approximate=self.approximate)
