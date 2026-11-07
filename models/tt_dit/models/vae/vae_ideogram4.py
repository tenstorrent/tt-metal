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

from ...layers.conv2d import Conv2d, configure_conv2d
from ...layers.module import Module
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from .vae_sd35 import VAEDecoder

# DRAM width slices per decoder conv shape, (input height, input width, in channels, out channels)
# -> num_slices, for the API's 14 output sizes and 512x512 on a BH Galaxy (4x8, TP4); shapes already in the shared
# Conv2d table are not repeated, and 1 runs the conv unsliced in L1. A fixed count, unlike ttnn's
# automatic choice, does not depend on the L1 free at call time, so the weights a conv prepares for a
# size always match how it runs there. The counts are the ones ttnn picks itself with width-only
# slicing (Conv2dSliceConfig(DRAM_WIDTH, num_slices=0)), measured per 3x3 conv on a Galaxy, except at
# the three tall 2K sizes, where ceil(w / 32) decodes 5-21% faster than ttnn's choice.
_SLICE_PARAMS: dict[tuple[int, int], dict[tuple[int, int, int, int], int]] = {
    (4, 8): {
        # 1280x720
        (90, 160, 32, 512): 1,
        (90, 160, 512, 512): 1,
        (180, 320, 512, 512): 2,
        (360, 640, 512, 512): 4,
        (360, 640, 512, 256): 4,
        (360, 640, 256, 256): 2,
        (720, 1280, 256, 256): 7,
        (720, 1280, 256, 128): 7,
        (720, 1280, 128, 128): 4,
        (720, 1280, 128, 3): 4,
        # 720x1280
        (160, 90, 32, 512): 1,
        (160, 90, 512, 512): 1,
        (320, 180, 512, 512): 2,
        (640, 360, 512, 512): 4,
        (640, 360, 512, 256): 4,
        (640, 360, 256, 256): 2,
        (1280, 720, 256, 256): 8,
        (1280, 720, 256, 128): 8,
        (1280, 720, 128, 128): 4,
        (1280, 720, 128, 3): 4,
        # 1152x864
        (108, 144, 32, 512): 1,
        (108, 144, 512, 512): 1,
        (216, 288, 512, 512): 2,
        (432, 576, 512, 512): 4,
        (432, 576, 512, 256): 4,
        (432, 576, 256, 256): 2,
        (864, 1152, 256, 256): 8,
        (864, 1152, 256, 128): 8,
        (864, 1152, 128, 128): 4,
        (864, 1152, 128, 3): 4,
        # 864x1152
        (144, 108, 32, 512): 1,
        (144, 108, 512, 512): 1,
        (288, 216, 512, 512): 2,
        (576, 432, 512, 512): 5,
        (576, 432, 512, 256): 5,
        (576, 432, 256, 256): 2,
        (1152, 864, 256, 256): 9,
        (1152, 864, 256, 128): 9,
        (1152, 864, 128, 128): 4,
        (1152, 864, 128, 3): 4,
        # 1248x832
        (104, 156, 32, 512): 1,
        (104, 156, 512, 512): 1,
        (208, 312, 512, 512): 2,
        (416, 624, 512, 512): 4,
        (416, 624, 512, 256): 4,
        (416, 624, 256, 256): 3,
        (832, 1248, 256, 256): 8,
        (832, 1248, 256, 128): 8,
        (832, 1248, 128, 128): 4,
        (832, 1248, 128, 3): 4,
        # 832x1248
        (156, 104, 32, 512): 1,
        (156, 104, 512, 512): 1,
        (312, 208, 512, 512): 2,
        (624, 416, 512, 512): 5,
        (624, 416, 512, 256): 5,
        (624, 416, 256, 256): 3,
        (1248, 832, 256, 256): 9,
        (1248, 832, 256, 128): 9,
        (1248, 832, 128, 128): 4,
        (1248, 832, 128, 3): 4,
        # 2048x1152
        (144, 256, 32, 512): 1,
        (576, 1024, 512, 512): 11,
        (576, 1024, 512, 256): 11,
        (576, 1024, 256, 256): 5,
        (1152, 2048, 256, 256): 22,
        (1152, 2048, 256, 128): 22,
        (1152, 2048, 128, 128): 8,
        (1152, 2048, 128, 3): 8,
        # 1152x2048 (tall 2K: ceil(w / 32))
        (256, 144, 32, 512): 5,
        (256, 144, 512, 512): 5,
        (512, 288, 512, 512): 9,
        (1024, 576, 512, 512): 18,
        (1024, 576, 512, 256): 18,
        (1024, 576, 256, 256): 18,
        (2048, 1152, 256, 256): 36,
        (2048, 1152, 256, 128): 36,
        (2048, 1152, 128, 128): 36,
        (2048, 1152, 128, 3): 36,
        # 2048x1536
        (192, 256, 32, 512): 1,
        (768, 1024, 512, 512): 16,
        (768, 1024, 512, 256): 16,
        (768, 1024, 256, 256): 7,
        (1536, 2048, 256, 256): 22,
        (1536, 2048, 256, 128): 22,
        (1536, 2048, 128, 128): 11,
        (1536, 2048, 128, 3): 11,
        # 1536x2048 (tall 2K: ceil(w / 32))
        (256, 192, 32, 512): 6,
        (256, 192, 512, 512): 6,
        (512, 384, 512, 512): 12,
        (1024, 768, 512, 512): 24,
        (1024, 768, 512, 256): 24,
        (1024, 768, 256, 256): 24,
        (2048, 1536, 256, 256): 48,
        (2048, 1536, 256, 128): 48,
        (2048, 1536, 128, 128): 48,
        (2048, 1536, 128, 3): 48,
        # 2048x1360
        (170, 256, 32, 512): 1,
        (170, 256, 512, 512): 1,
        (340, 512, 512, 512): 3,
        (680, 1024, 512, 512): 11,
        (680, 1024, 512, 256): 11,
        (680, 1024, 256, 256): 6,
        (1360, 2048, 256, 256): 22,
        (1360, 2048, 256, 128): 22,
        (1360, 2048, 128, 128): 11,
        (1360, 2048, 128, 3): 11,
        # 1360x2048 (tall 2K: ceil(w / 32))
        (256, 170, 32, 512): 6,
        (256, 170, 512, 512): 6,
        (512, 340, 512, 512): 11,
        (1024, 680, 512, 512): 22,
        (1024, 680, 512, 256): 22,
        (1024, 680, 256, 256): 22,
        (2048, 1360, 256, 256): 43,
        (2048, 1360, 256, 128): 43,
        (2048, 1360, 128, 128): 43,
        (2048, 1360, 128, 3): 43,
        # 512x512, the pipeline's default size (ceil(w / 32))
        (64, 64, 32, 512): 2,
        (64, 64, 512, 512): 2,
        (256, 256, 512, 256): 8,
        (256, 256, 256, 256): 8,
        (512, 512, 256, 128): 16,
        (512, 512, 128, 128): 16,
    },
}

# Output sizes (width, height) the table covers on each mesh shape: every decoder conv either has an
# entry above or is in the shared Conv2d table. Any other size is rejected on these meshes, because a
# conv with no slice config can run with too few slices and clash with resident L1 buffers.
SUPPORTED_SIZES: dict[tuple[int, int], frozenset[tuple[int, int]]] = {
    (4, 8): frozenset(
        {
            (512, 512),
            (1024, 1024),
            (1280, 720),
            (720, 1280),
            (1152, 864),
            (864, 1152),
            (1248, 832),
            (832, 1248),
            (2048, 2048),
            (2048, 1152),
            (1152, 2048),
            (2048, 1536),
            (1536, 2048),
            (2048, 1360),
            (1360, 2048),
        }
    ),
}


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
        vae = cls(post_quant_conv=post_quant_conv, decoder=decoder)
        # 14 output sizes, so many cached programs: keep their config tensors out of L1_SMALL.
        configure_conv2d(vae, slice_params=_SLICE_PARAMS, config_tensors_in_dram=True)
        return vae

    def check_output_size(self, width: int, height: int) -> None:
        """Raise ValueError if this mesh has a slice table and it does not cover the output size."""
        mesh_shape = tuple(self.post_quant_conv.mesh_device.shape)
        sizes = SUPPORTED_SIZES.get(mesh_shape)
        if sizes is not None and (width, height) not in sizes:
            msg = (
                f"Ideogram 4 VAE: output size {width}x{height} is not supported on mesh {mesh_shape}; "
                f"supported (width, height) sizes are {sorted(sizes)}"
            )
            raise ValueError(msg)

    def forward(self, z: ttnn.Tensor) -> ttnn.Tensor:
        """z: [B, H/8, W/8, z_channels] (NHWC, tile layout). Returns [B, H, W, out_channels]."""
        _, latent_h, latent_w, _ = z.shape
        self.check_output_size(latent_w * 8, latent_h * 8)
        return self.decoder(self.post_quant_conv(z))
