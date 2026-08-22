# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `s_conv_transpose1d` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.decoder.upsample_layers.1.0`, a
`vibevoice.modular.modular_vibevoice_tokenizer.SConvTranspose1d` — a thin
wrapper around `NormConvTranspose1d` (`self.convtr`, native port already
graduated as `norm_conv_transpose1d`) that additionally trims
`padding_total = kernel_size - stride` samples off the raw conv-transpose
output:

    y = self.convtr(x)                      # raw ConvTranspose1d + norm(=Identity)
    padding_right = ceil(padding_total * trim_right_ratio) if causal else padding_total // 2
    padding_left  = padding_total - padding_right
    y = y[:, :, padding_left : T - padding_right]   # unpad1d

This reuses `norm_conv_transpose1d.build()` for the conv-transpose math and
applies the same trim in ttnn via a time-axis slice.
"""

from __future__ import annotations

import math

import ttnn
from models.demos.vibevoice_1_5b._stubs.norm_conv_transpose1d import build as _build_norm_convtr


def build(device, torch_module):
    """Bind the trained conv-transpose weight/bias and return a native ttnn forward closure."""
    inner_forward = _build_norm_convtr(device, torch_module.convtr)

    padding_total = int(torch_module.padding_total)
    causal = bool(torch_module.causal)
    trim_right_ratio = float(torch_module.trim_right_ratio)

    if causal:
        padding_right = math.ceil(padding_total * trim_right_ratio)
    else:
        padding_right = padding_total // 2
    padding_left = padding_total - padding_right

    def forward(x, *args, **kwargs):
        y = inner_forward(x)  # [1, C_out, T_full] channels-first
        if padding_left + padding_right > 0:
            c_out = int(y.shape[1])
            t_full = int(y.shape[2])
            y = ttnn.slice(y, [0, 0, padding_left], [1, c_out, t_full - padding_right])
        return y

    return forward


def s_conv_transpose1d(*args, **kwargs):
    raise RuntimeError(
        "s_conv_transpose1d requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
