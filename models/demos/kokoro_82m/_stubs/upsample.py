# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `upsample` (hexgrad/Kokoro-82M
`decoder.generator.f0_upsamp`, a `nn.Upsample(scale_factor=300, mode='nearest')`).

Reference torch forward: `F.interpolate(x, scale_factor=300, mode='nearest')`
on a `[B, C, T]` tensor -> `[B, C, T*300]`, i.e. each frame is repeated 300
times along the time axis (captured IO: [1, 1, 50] -> [1, 1, 15000]).

Native ttnn: reshape to `[B, C, T, 1]` and broadcast-copy the trailing unit
axis to `scale` (multiply by a `[1, 1, 1, scale]` ones tensor), then reshape to
`[B, C, T*scale]`. This is exactly nearest-neighbour repeat without a 300-way
concat.
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG

HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    scale = int(round(float(getattr(torch_module, "scale_factor", 1.0))))
    mode = getattr(torch_module, "mode", "nearest")
    if mode != "nearest":
        raise RuntimeError(f"upsample native port supports mode='nearest' only (got {mode})")

    ones = ttnn.ones((1, 1, 1, scale), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _to_ttnn(x):
        if isinstance(x, ttnn.Tensor):
            if x.get_dtype() != ttnn.float32:
                x = ttnn.typecast(x, ttnn.float32)
            return x
        return ttnn.from_torch(
            x.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    def forward(x, *args, **kwargs):
        x = _to_ttnn(x)  # [B, C, T]
        B, C, T = [int(s) for s in x.shape]
        if scale == 1:
            return x
        xr = ttnn.reshape(x, [B, C, T, 1])
        xr = ttnn.multiply(xr, ones)  # broadcast last axis 1 -> scale
        return ttnn.reshape(xr, [B, C, T * scale])

    return forward


def upsample(*args, **kwargs):
    raise RuntimeError(
        "upsample requires build(device, torch_module) to read scale_factor; " "the bare callable has no configuration."
    )
