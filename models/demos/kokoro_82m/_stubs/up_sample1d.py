# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `up_sample1d` (hexgrad/Kokoro-82M
`predictor.F0.0.upsample`, a StyleTTS2/ISTFTNet `UpSample1d`).

Reference torch forward:

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        return F.interpolate(x, scale_factor=2, mode='nearest')

The `predictor.F0.0` block uses `upsample='none'`, so this is the identity
(captured IO confirms input == output == [1, 512, 25]). The nearest-neighbour
x2 path is implemented natively too (repeat along the time axis) for
completeness.
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG

HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    layer_type = getattr(torch_module, "layer_type", "none")

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
        if layer_type == "none":
            return x
        # nearest x2 along the time axis: duplicate each frame.
        B, C, T = x.shape
        xr = ttnn.reshape(x, [B, C, T, 1])
        xr = ttnn.concat([xr, xr], dim=3, memory_config=_DRAM)  # [B, C, T, 2]
        return ttnn.reshape(xr, [B, C, T * 2])

    return forward


def up_sample1d(*args, **kwargs):
    raise RuntimeError(
        "up_sample1d requires build(device, torch_module) to read layer_type; "
        "the bare callable has no configuration."
    )
