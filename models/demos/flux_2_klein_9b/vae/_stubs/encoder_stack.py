# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `encoder_stack`.

`encoder_stack` is a ROLE, not a distinct module. The reuse registry guessed a
Llama vision encoder for it and the scaffold copied
`models/tt_transformers/tt/multimodal/llama_vision_encoder.py` here; that guess
is wrong for an autoencoder, which has no transformer encoder stack. What the
component actually resolves to is recorded by the capture step in
`_captured/encoder_stack/manifest.json`:

    submodule_path: "encoder"
    args:   [1, 3, 224, 224]  ->  output: [1, 64, 28, 28]

i.e. the very same `AutoencoderKLFlux2.encoder` that the `encoder` component
covers — the convolutional stack that compresses pixels into latents.
`_CANDIDATE_SUBMODULE_PATHS` in `tests/pcc/test_encoder_stack.py` lists `encoder`
fourth, and the manifest path is tried first, so the PCC test builds its golden
from that module.

So the correct native port for this component IS the encoder port, and this stub
constructs the same tensor-parallel `TtEncoder`: every conv column-parallel over
its output channels with an `all_gather` on the channel dim, GroupNorm affine
params replicated, and the mid-block attention using the `attention` component's
own TP stub. See `_vae_blocks.py` for the derivation.
"""
from __future__ import annotations

from models.demos.flux_2_klein_9b.vae._stubs.encoder import TtEncoder

_ENCODER_ATTRS = ("conv_in", "down_blocks", "mid_block", "conv_norm_out", "conv_out")


class TtEncoderStack(TtEncoder):
    """The VAE encoder, built under the `encoder_stack` component name."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is not None and not all(hasattr(torch_module, a) for a in _ENCODER_ATTRS):
            raise RuntimeError(
                f"`encoder_stack` resolved to a {type(torch_module).__name__}, which is not the "
                f"diffusers `Encoder` this port implements (expected attributes {_ENCODER_ATTRS}). "
                f"Check `_CANDIDATE_SUBMODULE_PATHS` in tests/pcc/test_encoder_stack.py."
            )
        super().__init__(device, torch_module)


def build(device, torch_module=None):
    return TtEncoderStack(device, torch_module)


def encoder_stack(device, torch_module=None):
    return TtEncoderStack(device, torch_module)
