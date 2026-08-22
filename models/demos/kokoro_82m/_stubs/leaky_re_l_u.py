# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `leaky_re_l_u` (hexgrad/Kokoro-82M, an `nn.LeakyReLU`).

`y = x if x >= 0 else negative_slope * x`. Native ttnn: `ttnn.leaky_relu`.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    slope = float(getattr(torch_module, "negative_slope", 0.2))

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(
                x.contiguous().float(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        return ttnn.leaky_relu(x, slope)

    return forward


def leaky_re_l_u(*args, **kwargs):
    raise RuntimeError(
        "leaky_re_l_u requires build(device, torch_module) to bind negative_slope; "
        "the bare callable has no configuration."
    )
