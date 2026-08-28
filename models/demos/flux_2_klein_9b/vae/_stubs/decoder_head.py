# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `decoder_head`.

`decoder_head` is a ROLE, not a distinct module. The reuse registry guessed a
transformer `lm_head` for it and the scaffold copied
`models/tt_transformers/tt/lm_head.py` here; that guess is wrong for an
autoencoder, which has no vocabulary projection. What the component actually
resolves to is recorded by the capture step in
`_captured/decoder_head/manifest.json`:

    submodule_path: "decoder"
    args:   [1, 32, 28, 28]  ->  output: [1, 3, 224, 224]

i.e. the very same `AutoencoderKLFlux2.decoder` that the `decoder` component
covers — the head that turns latents back into pixels. `_CANDIDATE_SUBMODULE_PATHS`
in `tests/pcc/test_decoder_head.py` lists `decoder` second, and the manifest path
is tried first, so the PCC test builds its golden from that module.

So the correct native port for this component IS the decoder port, and this stub
constructs the same tensor-parallel `TtDecoder`: every conv column-parallel over
its output channels with an `all_gather` on the channel dim, GroupNorm affine
params replicated, and the mid-block attention using the `attention` component's
own TP stub. See `_vae_blocks.py` for the derivation.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs.decoder import TtDecoder

_DECODER_ATTRS = ("conv_in", "mid_block", "up_blocks", "conv_norm_out", "conv_out")


class TtDecoderHead(TtDecoder):
    """The VAE decoder, built under the `decoder_head` component name."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is not None and not all(hasattr(torch_module, a) for a in _DECODER_ATTRS):
            raise RuntimeError(
                f"`decoder_head` resolved to a {type(torch_module).__name__}, which is not the "
                f"diffusers `Decoder` this port implements (expected attributes {_DECODER_ATTRS}). "
                f"Check `_CANDIDATE_SUBMODULE_PATHS` in tests/pcc/test_decoder_head.py."
            )
        super().__init__(device, torch_module)


def build(device, torch_module=None):
    return TtDecoderHead(device, torch_module)


def decoder_head(device, torch_module=None):
    return TtDecoderHead(device, torch_module)
