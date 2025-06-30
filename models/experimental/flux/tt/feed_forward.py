# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from . import utils
from .linear import Linear, LinearParameters
from .substate import substate

if TYPE_CHECKING:
    import torch


@dataclass
class FeedForwardParameters:
    in_proj: LinearParameters
    out_proj: LinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        linear_on_host: bool = False,
        mesh_sharded_input: bool = False,
    ) -> FeedForwardParameters:
        return cls(
            in_proj=LinearParameters.from_torch(
                substate(state, "net.0.proj"),
                dtype=dtype,
                device=device,
                on_host=linear_on_host,
                mesh_sharding_dim=0 if mesh_sharded_input else 1,
            ),
            out_proj=LinearParameters.from_torch(
                substate(state, "net.2"),
                dtype=dtype,
                device=device,
                on_host=linear_on_host,
                mesh_sharding_dim=0,
            ),
        )


class FeedForward:
    def __init__(self, parameters: FeedForwardParameters) -> None:
        super().__init__()

        self.in_proj = Linear(parameters.in_proj)
        self.out_proj = Linear(parameters.out_proj)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        utils.signpost("feedforward")

        x = self.in_proj.forward(x)
        # Turning on fast_and_approximate_mode leads to big changes in the generated image.
        # The image quality might still be okay.
        x = ttnn.gelu(x, fast_and_approximate_mode=False)

        return self.out_proj.forward(x)
