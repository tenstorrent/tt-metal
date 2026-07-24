# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `s_conv1d` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.encoder.downsample_layers.0.0`, a
`vibevoice.modular.modular_vibevoice_tokenizer.SConv1d` wrapping
`NormConv1d(nn.Conv1d(1, 32, kernel_size=7, stride=1, groups=1), norm='none')`
with `causal=True`, `pad_mode='constant'`:

    padding_total = (kernel_size - 1) * dilation - (stride - 1) = 6
    extra_padding = 0                         # stride==1 keeps output length == input length
    x = pad(x, (padding_total, extra_padding), mode='constant', value=0)   # left-pad only (causal)
    y = conv1d(x)                             # norm is Identity (norm='none')

SConv1d now owns ONLY the causal zero-padding; the unpadded stride-1 groups-1
convolution is delegated to the already-graduated child stub
`_stubs/norm_conv1d.build` (which takes/returns channels-first [1, C, T]). The
padding is applied to the channels-first input on the time axis before handing
the padded tensor to the child.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs._trace_pad import cached_zeros
from models.demos.vibevoice_1_5b._stubs.norm_conv1d import build as _build_norm_conv1d

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained conv weight/bias and return a native ttnn forward closure."""
    s = torch_module  # SConv1d
    conv = s.conv.conv  # nn.Conv1d
    if int(conv.stride[0]) != 1 or conv.groups != 1:
        raise RuntimeError(
            f"s_conv1d native port supports stride-1 groups-1 only (got stride={conv.stride[0]}, groups={conv.groups})"
        )
    if type(s.conv.norm).__name__ != "Identity":
        raise RuntimeError(f"s_conv1d native port assumes norm='none' (Identity), got {type(s.conv.norm).__name__}")

    pad_left = int(s.padding_total) if s.causal else int(s.padding_total) - int(s.padding_total) // 2
    pad_right = 0 if s.causal else int(s.padding_total) // 2

    # Compose the graduated child stub for the (unpadded) NormConv1d.
    conv_forward = _build_norm_conv1d(device, s.conv)
    _zc: dict = {}  # cached causal zero-pad buffers (trace-capture safe)

    def forward(x, *args, **kwargs):
        # x: [1, C_in, T] channels-first.
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)

        if pad_left > 0 or pad_right > 0:
            B = int(x.shape[0])
            C_in = int(x.shape[1])
            pieces = []
            if pad_left > 0:
                pieces.append(cached_zeros(_zc, (B, C_in, pad_left), ttnn.float32, ttnn.TILE_LAYOUT, device))
            pieces.append(x)
            if pad_right > 0:
                pieces.append(cached_zeros(_zc, (B, C_in, pad_right), ttnn.float32, ttnn.TILE_LAYOUT, device))
            x = ttnn.concat(pieces, dim=2, memory_config=_DRAM)  # pad on the time axis (channels-first)

        return conv_forward(x)  # norm_conv1d child: [1, C_in, T_pad] -> [1, C_out, T_out]

    return forward


def s_conv1d(*args, **kwargs):
    raise RuntimeError(
        "s_conv1d requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
