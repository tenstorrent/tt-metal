# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `pre_emphasis` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.speaker_encoder.torch_spec.0`, a
`TTS.encoder.models.base_encoder.PreEmphasis` (coefficient 0.97):

    x = F.pad(x.unsqueeze(1), (1, 0), "reflect")     # reflect-prepend x[1]
    return F.conv1d(x, filter=[-0.97, 1.0]).squeeze(1)

For a 2D `(B, T)` waveform this is the first-order pre-emphasis filter:

    y[t]   = x[t] - 0.97 * x[t-1]      for t >= 1
    y[0]   = x[0] - 0.97 * x[1]        (reflect boundary)

i.e. `y = x - 0.97 * shift`, where `shift = cat([x[:, 1:2], x[:, :T-1]], dim=1)`
is the reflect-shifted signal. Implemented natively in ttnn (float32) as a
slice/concat + scaled subtract — no learned parameters.
"""

from __future__ import annotations

import ttnn

_COEF = 0.97


def build(device, torch_module):
    """Return a native ttnn pre-emphasis forward closure (coefficient from the module)."""
    coef = float(getattr(torch_module, "coefficient", _COEF))

    def forward(x, *args, **kwargs):
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        n = int(x.shape[-1])

        # Build the reflect-shifted signal: [x[1], x[0], x[1], ..., x[T-2]].
        xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        first = ttnn.slice(xr, [0, 1], [1, 2])          # x[1]
        head = ttnn.slice(xr, [0, 0], [1, n - 1])       # x[0 : T-1]
        shift = ttnn.concat([first, head], dim=1)       # [1, T]
        shift = ttnn.to_layout(shift, ttnn.TILE_LAYOUT)

        return ttnn.subtract(x, ttnn.multiply(shift, coef))

    return forward


def pre_emphasis(*args, **kwargs):
    raise RuntimeError(
        "pre_emphasis requires build(device, torch_module) to bind the coefficient; "
        "use build(device, torch_module)."
    )
