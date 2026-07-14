# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.uniad.tt.matmul_helpers import linear_flatten_batch


class TtFFN:
    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x

        # `linear_flatten_batch` folds all leading dims into M so the matmul
        # heuristic dispatches across many cores instead of being pinned to
        # 8 by a small penultimate dim. Covers both 4-D inputs like
        # (1, 901, 32, 256) and 3-D inputs like (2500, 1, 256).
        x = linear_flatten_batch(x, self.linear1_weight, bias=self.linear1_bias)
        x = ttnn.relu(x)
        x = linear_flatten_batch(x, self.linear2_weight, bias=self.linear2_bias)

        x = ttnn.add(x, identity)

        return x
