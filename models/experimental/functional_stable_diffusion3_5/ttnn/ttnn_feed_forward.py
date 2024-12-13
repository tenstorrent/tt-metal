# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_gelu import ttnn_GELU


class ttnn_FeedForward:
    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        activation_fn: str = "geglu",
        inner_dim=None,
        bias: bool = True,
    ):
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = ttnn_GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = ttnn_GELU(dim, inner_dim, approximate="tanh", bias=bias)

        self.net = []

        self.net.append(act_fn)
        self.net.append(ttnn.linear)

    def __call__(self, hidden_states: ttnn.Tensor, parameters=None) -> ttnn.Tensor:
        for module in self.net:
            if module == ttnn.linear:
                hidden_states = module(hidden_states, parameters["net"][2]["weight"], bias=parameters["net"][2]["bias"])
            else:
                hidden_states = module(hidden_states, parameters=parameters["net"][0])
        return hidden_states
