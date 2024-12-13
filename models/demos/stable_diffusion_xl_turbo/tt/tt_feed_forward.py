# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def sd_geglu(
    hidden_states,
    parameters,
    device,
):
    x = ttnn.linear(
        hidden_states,
        parameters.proj.weight,
        bias=parameters.proj.bias,
    )
    x = ttnn.unsqueeze(x, 0)
    x = ttnn.geglu(x)
    x = ttnn.squeeze(x, 0)
    return x


def sd_feed_forward(
    hidden_states,
    parameters,
    device,
):
    hidden_states = sd_geglu(hidden_states, parameters.net[0], device)
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.net[2].weight,
        bias=parameters.net[2].bias,
    )
    return hidden_states
