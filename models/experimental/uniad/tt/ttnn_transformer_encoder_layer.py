# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Union, Callable, Optional
from torch import Tensor
import torch.nn.functional as F

import ttnn


class TtTransformerEncoderLayer:
    __constants__ = ["norm_first"]

    def __init__(
        self,
        parameters,
        device,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.parameters = parameters
        self.device = device
        self.self_attn = nn.MultiheadAttention(
            # parameters.self_attn,
            # device,
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.self_attn.load_state_dict(parameters["self_attn"].state_dict())
        self.self_attn.eval()
        # Implementation of Feedforward model
        self.linear1 = ttnn.linear
        self.linear2 = ttnn.linear

        self.norm_first = norm_first
        self.norm1 = ttnn.layer_norm
        self.norm2 = ttnn.layer_norm

        # Legacy string support for activation function.
        self.activation = ttnn.relu

    def __call__(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + ttnn.from_torch(
                    self.self_attn(
                        ttnn.to_torch(x).to(dtype=torch.float),
                        ttnn.to_torch(x).to(dtype=torch.float),
                        ttnn.to_torch(x).to(dtype=torch.float),
                        attn_mask=src_mask,
                        key_padding_mask=src_key_padding_mask,
                        is_causal=is_causal,
                    )[0],
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                ),
                weight=self.parameters.norm1.weight,
                bias=self.parameters.norm1.bias,
            )
            x = self.norm2(x + self._ff_block(x), weight=self.parameters.norm2.weight, bias=self.parameters.norm2.bias)

        return x

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        y = self.linear1(x, self.parameters.linear1.weight, bias=self.parameters.linear1.bias, dtype=ttnn.bfloat16)
        x = self.linear2(
            self.activation(y), self.parameters.linear2.weight, bias=self.parameters.linear2.bias, dtype=ttnn.bfloat16
        )
        ttnn.deallocate(y)
        return x
