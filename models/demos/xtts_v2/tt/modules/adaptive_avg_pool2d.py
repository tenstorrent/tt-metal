# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `adaptive_avg_pool2d` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.speaker_encoder.layer1.0.se.avg_pool`,
which is `torch.nn.AdaptiveAvgPool2d(output_size=1)` inside the speaker
encoder's squeeze-excitation block. It global-average-pools an NCHW
activation `[N, C, H, W]` down to `[N, C, 1, 1]` — i.e. the mean over the two
spatial dims (H and W). Captured shapes: in `[1, 32, 64, 301]` -> out
`[1, 32, 1, 1]`.

This is a pure reduction with no learned parameters, so `build` ignores the
torch module and returns a stateless native ttnn forward built from
`ttnn.mean`.
"""

from __future__ import annotations

import ttnn


def _adaptive_avg_pool2d(x):
    """Global average pool over the spatial dims of an NCHW ttnn tensor.

    `AdaptiveAvgPool2d(1)` averages over the two trailing spatial dims,
    keeping the reduced axes so the output rank matches
    (`[..., H, W] -> [..., 1, 1]`). We address them by NEGATIVE index
    (`dim=[-2, -1]`) so the same reduction is correct for the real 4D NCHW
    activation `[1, 32, 64, 301]` AND for a rank-3 `[C, H, W]` input — torch's
    own `AdaptiveAvgPool2d` likewise pools whatever the last two dims are.
    This mirrors the native SE-block global-average-pool pattern in
    `models/experimental/vovnet/tt/effective_se_module.py`; `keepdim=True`
    preserves the singleton spatial axes.
    """
    return ttnn.mean(x, dim=[-2, -1], keepdim=True)


def build(device, torch_module):
    """Return the native ttnn forward.

    `AdaptiveAvgPool2d` has no parameters, so `torch_module` is unused — it is
    accepted only to satisfy the harness's `build(device, torch_module)`
    contract. The returned callable takes the single ttnn input tensor.
    """

    def forward(x):
        return _adaptive_avg_pool2d(x)

    return forward


def adaptive_avg_pool2d(x, *args, **kwargs):
    """Bare module-level callable for harnesses that skip `build`."""
    return _adaptive_avg_pool2d(x)
