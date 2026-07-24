# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `tokenizer_decoder` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.decoder`, a
`vibevoice.modular.modular_vibevoice_tokenizer.TokenizerDecoder`:

    for i in range(len(depths)):
        x = upsample_layers[i][0](x)        # SConv1d (i==0, the stem) or
                                             # SConvTranspose1d (i>0)
        for block in stages[i]:
            x = block(x)                    # Block1D
    x = norm(x)                             # Identity for this checkpoint
                                             # (disable_last_norm/layernorm config
                                             #  resolves to nn.Identity here)
    x = head(x)                             # SConv1d

This composes the already-graduated native ports for the constituent
components (`s_conv1d`, `s_conv_transpose1d`, `block1_d`) rather than
re-deriving their math — each of those `build(device, submodule)` closures
is a drop-in per-instance forward.
"""

from __future__ import annotations

from models.demos.vibevoice_1_5b._stubs.block1_d import build as _build_block1d
from models.demos.vibevoice_1_5b._stubs.s_conv1d import build as _build_s_conv1d
from models.demos.vibevoice_1_5b._stubs.s_conv_transpose1d import build as _build_s_conv_transpose1d


def build(device, torch_module):
    """Bind every child submodule's trained weights and return a native ttnn forward closure."""
    m = torch_module
    depths = list(m.depths)

    upsample_forwards = []
    for layer_seq in m.upsample_layers:
        layer = layer_seq[0]
        if type(layer).__name__ == "SConvTranspose1d":
            upsample_forwards.append(_build_s_conv_transpose1d(device, layer))
        else:
            upsample_forwards.append(_build_s_conv1d(device, layer))

    stage_forwards = [[_build_block1d(device, blk) for blk in stage] for stage in m.stages]

    if type(m.norm).__name__ != "Identity":
        raise RuntimeError(f"tokenizer_decoder native port assumes norm='none' (Identity), got {type(m.norm).__name__}")

    head_forward = _build_s_conv1d(device, m.head)

    def forward(x, *args, **kwargs):
        for i in range(len(depths)):
            x = upsample_forwards[i](x)
            for blk_fwd in stage_forwards[i]:
                x = blk_fwd(x)
        x = head_forward(x)
        return x

    return forward


def tokenizer_decoder(*args, **kwargs):
    raise RuntimeError(
        "tokenizer_decoder requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
