# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import ttnn

from models.experimental.audiox.tt.common import linear_weight, to_tt


class TtTimestepEmbedding:
    """TTNN port of the AudioX DiT timestep embedding (Fourier features + MLP)."""

    def __init__(self, mesh_device, state_dict: dict):
        sd = state_dict
        # FourierFeatures stores its weight as [fourier_dim/2, 1]. The forward pass
        # does `t @ weight.T`, so we precompute the transposed weight (shape [1, fourier_dim/2])
        # and use it directly with ttnn.linear.
        self.fourier_weight = to_tt(sd["fourier.weight"].transpose(0, 1).contiguous(), mesh_device)

        self.l1_w = to_tt(linear_weight(sd["linear1.weight"]), mesh_device)
        self.l1_b = to_tt(sd["linear1.bias"], mesh_device)
        self.l2_w = to_tt(linear_weight(sd["linear2.weight"]), mesh_device)
        self.l2_b = to_tt(sd["linear2.bias"], mesh_device)

    def __call__(self, t: ttnn.Tensor) -> ttnn.Tensor:
        # Ensure trailing feature dim of 1.
        if len(t.shape) == 1:
            t = ttnn.unsqueeze(t, -1)

        f = ttnn.linear(t, self.fourier_weight)
        f = ttnn.multiply(f, 2.0 * math.pi)
        cos_f = ttnn.cos(f)
        sin_f = ttnn.sin(f)
        fourier = ttnn.concat([cos_f, sin_f], dim=-1)

        h = ttnn.linear(fourier, self.l1_w, bias=self.l1_b)
        h = ttnn.silu(h)
        return ttnn.linear(h, self.l2_w, bias=self.l2_b)
