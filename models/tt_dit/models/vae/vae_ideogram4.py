# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4.0 "Flux2 KL autoencoder" decode path.

The reference autoencoder.py ships a state-dict converter from the diffusers
``AutoencoderKL`` layout, so the model IS a diffusers AutoencoderKL:
block_out_channels=(128, 256, 512, 512), latent_channels=32, layers_per_block=2,
GroupNorm(32), a single mid self-attention. The decode path is therefore
``post_quant_conv`` (1x1) followed by the standard diffusers Decoder — which the
SD3.5 port already implements as ``vae_sd35.VAEDecoder``. We reuse it wholesale and
only add the ``post_quant_conv`` (the reference Decoder applies it first; diffusers
keeps it in AutoencoderKL, not Decoder).

Inference needs decode only; the encoder is omitted.
"""

from __future__ import annotations

import ttnn

from ...layers.conv2d import Conv2d
from ...layers.module import Module
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from .vae_sd35 import VAEDecoder


class Ideogram4VAEDecoder(Module):
    def __init__(self, *, post_quant_conv: Conv2d, decoder: VAEDecoder) -> None:
        super().__init__()
        # The reference Decoder applies post_quant_conv (1x1, replicated) first; diffusers keeps
        # it in AutoencoderKL, not Decoder — so we hold it here and prepend it to the SD3.5 decoder.
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder

    @classmethod
    def from_torch(
        cls,
        akl,  # diffusers AutoencoderKL
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> "Ideogram4VAEDecoder":
        z_channels = akl.config.latent_channels
        post_quant_conv = Conv2d(
            z_channels, z_channels, kernel_size=1, mesh_device=mesh_device, ccl_manager=ccl_manager
        )
        post_quant_conv.load_torch_state_dict(akl.post_quant_conv.state_dict())
        # Delegate the decoder (dim reflection + weight load) to the shared SD3.5 VAEDecoder.from_torch.
        decoder = VAEDecoder.from_torch(
            akl.decoder, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        return cls(post_quant_conv=post_quant_conv, decoder=decoder)

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        """z: [B, H/8, W/8, z_channels] (NHWC, tile layout). Returns [B, H, W, out_channels]."""
        return self.decoder(self.post_quant_conv(z))
