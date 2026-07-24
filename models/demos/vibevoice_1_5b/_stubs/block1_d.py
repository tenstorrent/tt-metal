# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `block1_d` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.encoder.stages.0.0`, a
`vibevoice.modular.modular_vibevoice_tokenizer.Block1D` — a ConvNeXt-style 1D
residual block used throughout the acoustic/semantic tokenizer encoders/decoders:

    residual = x                                   # [B, C, T] channels-first
    x = ConvRMSNorm(x)                              # normalize over C
    x = mixer(x)                                    # depthwise causal Conv1d, k=7
    x = x * gamma
    x = residual + x

    residual = x
    x = ConvRMSNorm(x)
    x = FFN(x.permute(0, 2, 1)).permute(0, 2, 1)    # Linear(C,4C) -> GELU -> Linear(4C,C)
    x = x * ffn_gamma
    x = residual + x

For this component: dim=32, mixer is a depthwise (`groups=dim`) causal Conv1d
with kernel_size=7, stride=1, dilation=1, causal left-padding=6.

This block now COMPOSES the already-graduated child stubs rather than inlining
their math: the token-mixer is delegated to `_stubs/convlayer.build` (channels-first
depthwise causal conv) and the position-wise MLP to `_stubs/f_f_n.build` (GELU MLP,
channel-last). Block1D itself only owns the ConvRMSNorm, the `gamma`/`ffn_gamma`
scales, the FFN channel permute, and the two residual adds.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs.convlayer import build as _build_convlayer
from models.demos.vibevoice_1_5b._stubs.f_f_n import build as _build_f_f_n

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained Block1D weights and return a native ttnn forward closure."""
    m = torch_module

    norm_w = m.norm.weight.detach().float()
    ffn_norm_w = m.ffn_norm.weight.detach().float()
    norm_eps = float(m.norm.eps)
    ffn_norm_eps = float(m.ffn_norm.eps)

    gamma = m.gamma.detach().float() if getattr(m, "gamma", None) is not None else None
    ffn_gamma = m.ffn_gamma.detach().float() if getattr(m, "ffn_gamma", None) is not None else None

    # Compose the graduated child stubs for the mixer (depthwise causal conv,
    # channels-first in/out) and the FFN (GELU MLP, channel-last in/out).
    mixer_forward = _build_convlayer(device, m.mixer)
    ffn_forward = _build_f_f_n(device, m.ffn)

    def _row(t_1d):
        return ttnn.from_torch(
            t_1d.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

    norm_weight = _row(norm_w)
    ffn_norm_weight = _row(ffn_norm_w)
    gamma_row = _row(gamma) if gamma is not None else None
    ffn_gamma_row = _row(ffn_gamma) if ffn_gamma is not None else None

    # Channels-last core of the mixer (depthwise causal conv) so the whole block
    # can run in [B, T, C] without transposing in/out around every sub-op.
    mixer_forward_tc = getattr(mixer_forward, "forward_tc", None)

    def _to_f32(t):
        return ttnn.typecast(t, ttnn.float32) if t.get_dtype() != ttnn.float32 else t

    def forward(x, *args, **kwargs):
        # The whole block runs channels-last [B, T, C]: RMSNorm (normalizes over C),
        # the depthwise mixer (taps broadcast as [1,1,C]), the gamma scales ([1,1,C]),
        # and the FFN all want channels-last, so we transpose ONCE on the way in and
        # ONCE on the way out instead of flipping channels-first<->last ~10x per block.
        x = _to_f32(x)
        x_tc = ttnn.transpose(x, 1, 2)  # [B, C, T] -> [B, T, C] (once)

        residual = x_tc
        h = ttnn.rms_norm(x_tc, epsilon=norm_eps, weight=norm_weight, memory_config=_DRAM)
        h = mixer_forward_tc(h)  # convlayer channels-last core: [B, T, C] -> [B, T, C]
        if gamma_row is not None:
            h = ttnn.mul(h, gamma_row, memory_config=_DRAM)
        x_tc = ttnn.add(residual, h, memory_config=_DRAM)

        residual = x_tc
        h = ttnn.rms_norm(x_tc, epsilon=ffn_norm_eps, weight=ffn_norm_weight, memory_config=_DRAM)
        h = ffn_forward(h)  # f_f_n is channels-last: [B, T, C] -> [B, T, C]
        if ffn_gamma_row is not None:
            h = ttnn.mul(h, ffn_gamma_row, memory_config=_DRAM)
        x_tc = ttnn.add(residual, h, memory_config=_DRAM)

        return ttnn.transpose(x_tc, 1, 2)  # [B, T, C] -> [B, C, T] (once)

    return forward


def block1_d(*args, **kwargs):
    raise RuntimeError(
        "block1_d requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
