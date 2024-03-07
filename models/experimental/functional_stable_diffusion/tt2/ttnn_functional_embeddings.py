# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtTimestepEmbedding:
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, sample, act_fn: str = "silu"):
        sample = ttnn.linear(
            sample,
            self.parameters.linear_1.weight,
            bias=self.parameters.linear_1.bias,
            core_grid=sample.device().core_grid,
        )

        act = None
        if act_fn == "silu":
            act = ttnn.silu
        elif act_fn == "mish":
            assert False, "ttnn does not support nn.Mist() yet"

        if act is not None:
            sample = act(sample)

        sample = ttnn.linear(
            sample,
            self.parameters.linear_2.weight,
            bias=self.parameters.linear_2.bias,
            core_grid=sample.device().core_grid,
        )

        return sample
