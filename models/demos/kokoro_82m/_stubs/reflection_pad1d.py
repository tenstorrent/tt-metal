# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `reflection_pad1d` (hexgrad/Kokoro-82M
`decoder.generator.reflection_pad`, an `nn.ReflectionPad1d`, padding=(1, 0)).

Reflection padding of a `[B, C, T]` tensor on the time axis: the `p_left`
prepended values are `x[p_left], ..., x[1]` (mirror around index 0, edge not
repeated); the `p_right` appended values are `x[T-2], ..., x[T-1-p_right]`.
Native ttnn: slice the reflected columns and `ttnn.concat` them around `x`.
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    pad = torch_module.padding
    if isinstance(pad, int):
        p_left = p_right = int(pad)
    else:
        p_left, p_right = int(pad[0]), int(pad[1])

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(
                x.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
            )
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        B, C, T = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])

        parts = []
        # left reflection: x[p_left], x[p_left-1], ..., x[1]
        for i in range(p_left, 0, -1):
            parts.append(ttnn.slice(x, [0, 0, i], [B, C, i + 1]))
        parts.append(x)
        # right reflection: x[T-2], x[T-3], ..., x[T-1-p_right]
        for i in range(1, p_right + 1):
            idx = T - 1 - i
            parts.append(ttnn.slice(x, [0, 0, idx], [B, C, idx + 1]))

        if len(parts) == 1:
            return x
        return ttnn.concat(parts, dim=2, memory_config=_DRAM)

    return forward


def reflection_pad1d(*args, **kwargs):
    raise RuntimeError(
        "reflection_pad1d requires build(device, torch_module) to bind the padding "
        "sizes; the bare callable has no configuration."
    )
