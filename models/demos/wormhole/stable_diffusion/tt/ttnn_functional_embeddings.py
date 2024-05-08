# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def TtTimestepEmbedding(sample, parameters, act_fn: str = "silu"):
    sample = ttnn.matmul(sample, parameters.linear_1.weight)
    sample = ttnn.add(sample, parameters.linear_1.bias)

    act = None
    if act_fn == "silu":
        act = ttnn.silu
    elif act_fn == "mish":
        assert False, "ttnn does not support nn.Mist() yet"

    if act is not None:
        sample = act(sample)

    sample = ttnn.matmul(sample, parameters.linear_2.weight)
    sample = ttnn.add(sample, parameters.linear_2.bias)

    return sample
