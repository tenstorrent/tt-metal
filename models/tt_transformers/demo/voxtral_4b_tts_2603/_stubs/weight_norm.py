# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `weight_norm`
(`audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.0`).

`torch._weight_norm(v, g, dim=0)`:

    w[o] = g[o] * v[o] / ||v[o]||_2

i.e. the L2 norm is per OUTPUT CHANNEL, over every other axis. `g` is `[out, 1, 1]` and `v` is
`[out, in, k]`.

Reduced in two stages (over `k`, then over `in`) rather than by flattening the tail axes into one.
`v` arrives as `[1024, 292, 3]` in TILE layout, where the trailing 3 is padded to a tile width of 32
and the 292 to 320, so a `[1024, 876]` reshape would have to move data across that padding; two
reductions plus an `[out, 1, 1]` broadcast stay inside the existing tiles.

Widened to float32 for the reduction: 876 squares are summed per channel. `g` is **signed** here --
23 of this layer's 1024 channels are negative -- so the verification identity is `||w|| == |g|`,
not `== g`; checked the wrong way it reads as a relative error of 2.0.
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    if int(torch_module.dim) != 0:
        raise NotImplementedError(
            f"weight_norm over dim {torch_module.dim} is not ported; only dim 0 "
            f"(per-output-channel)"
        )

    def weight_norm(weight_g, weight_v=None, **kwargs):
        if weight_v is None:
            raise ValueError("weight_norm needs both weight_g and weight_v")
        v = ttnn.typecast(weight_v, ttnn.float32)
        g = ttnn.typecast(weight_g, ttnn.float32)
        sq_sum = ttnn.sum(ttnn.sum(ttnn.multiply(v, v), dim=-1, keepdim=True), dim=-2, keepdim=True)
        return ttnn.multiply(v, ttnn.multiply(ttnn.rsqrt(sq_sum), g))

    return weight_norm
