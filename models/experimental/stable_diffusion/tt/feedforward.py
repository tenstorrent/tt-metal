# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from typing import Optional

from tt_lib.fallback_ops import fallback_ops

import ttnn
from models.experimental.stable_diffusion.sd_utils import make_linear


class TtGEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        device=None,
        host=None,
        state_dict=None,
        base_address="mid_block.attentions.0.transformer_blocks.0.ff.net.0",
    ):
        super().__init__()
        self.device = device
        self.host = host

        weights = state_dict[f"{base_address}.proj.weight"]
        bias = state_dict[f"{base_address}.proj.bias"]
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        self.proj = make_linear(
            in_features=dim_in,
            out_features=dim_out * 2,
            weights=weights,
            bias=bias,
            device=device,
        )

    def gelu(self, gate: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.gelu(gate)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = self.proj(hidden_states)
        # hidden_states, gate = fallback_ops.chunk(hidden_states, 2, -1)
        hidden_states, gate = ttnn.split(hidden_states, 2, 3)
        act = self.gelu(gate)
        return ttnn.multiply(hidden_states, act)


class TtFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        device=None,
        host=None,
        state_dict=None,
        base_address="mid_block.attentions.0.transformer_blocks.0.ff",
    ):
        super().__init__()
        assert dropout == 0.0, "we do not support dropout"
        assert final_dropout == False, "we do not support dropout"
        assert activation_fn == "geglu", "except GEGLU, other activation functions are not supported."
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        if activation_fn == "geglu":
            act_fn = TtGEGLU(
                dim,
                inner_dim,
                device,
                host,
                state_dict,
                base_address=f"{base_address}.net.0",
            )
        else:
            assert False, "other activation ops are not implemented"

        self.act_fn = act_fn

        weights = state_dict[f"{base_address}.net.2.weight"]
        bias = state_dict[f"{base_address}.net.2.bias"]
        self.linear = make_linear(
            in_features=inner_dim,
            out_features=dim_out,
            weights=weights,
            bias=bias,
            device=device,
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = self.act_fn(hidden_states)
        return self.linear(hidden_states)
