# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `vision_mlp` of Qwen/Qwen2-VL-7B-Instruct.

Reference: `visual.blocks.0.mlp`, a `VisionMlp`:

    return fc2(quick_gelu(fc1(x)))
    # quick_gelu(x) = x * sigmoid(1.702 * x)
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    fc1 = torch_module.fc1
    fc2 = torch_module.fc2

    def _w(t):
        return ttnn.from_torch(t.detach(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    fc1_w, fc1_b = _w(fc1.weight), _w(fc1.bias.reshape(1, -1))
    fc2_w, fc2_b = _w(fc2.weight), _w(fc2.bias.reshape(1, -1))

    def _linear(x, weight, bias=None):
        return ttnn.linear(x, weight, bias=bias, transpose_b=True, memory_config=_DRAM)

    def forward(x, *args, **kwargs):
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        h = _linear(x, fc1_w, fc1_b)
        h = ttnn.mul(h, ttnn.sigmoid(ttnn.mul(h, 1.702)))  # quick_gelu
        return _linear(h, fc2_w, fc2_b)

    return forward


def vision_mlp(*args, **kwargs):
    raise RuntimeError(
        "vision_mlp requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
