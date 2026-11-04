# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.2 32-channel KL autoencoder on the FLUX 3 Klein mesh: the decoder H-partitioned over one mesh axis and
tensor-parallel over the other, the encoder tensor-parallel and replicated."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

import ttnn

from ...layers.conv2d import Conv2d
from ...models.vae.encoder_flux2 import Flux2VaeEncoder
from ...models.vae.vae import VaeAttention
from ...models.vae.vae_flux2 import Flux2VaeDecoder
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .packing import latent_shape

SUBFOLDER = "vae"
SCALE = 8
LATENT_CHANNELS = 128
BLOCK_OUT_CHANNELS = (128, 256, 512, 512)

# DRAM slice counts per resolution level, keyed by (in_channels, out_channels), capped at width / 32
CONV_SLICES_PER_LEVEL = (
    {(3, 128): 16, (128, 128): 16, (128, 3): 8, (256, 128): 16, (256, 256): 16},
    {(128, 256): 8, (256, 256): 4, (512, 256): 16, (512, 512): 16},
    {(256, 512): 8, (512, 512): 8},
    {(512, 512): 4, (512, 64): 4, (32, 512): 1},
)
# the convs of the H-partitioned decoder per level; ttnn picks their slice count
PARTITIONED_DECODER_CONVS_PER_LEVEL = (
    ((256, 256), (256, 128), (128, 128), (128, 3)),
    ((512, 512), (512, 256), (256, 256)),
    ((512, 512),),
    ((32, 512), (512, 512)),
)


def register_conv_slices(mesh_shape: tuple[int, int], height: int, width: int, h_factor: int = 1) -> None:
    table = Conv2d.slice_params.setdefault(tuple(mesh_shape), dict(Conv2d.slice_default))
    h, w = height, width
    for level, partitioned in zip(CONV_SLICES_PER_LEVEL, PARTITIONED_DECODER_CONVS_PER_LEVEL):
        for (cin, cout), slices in level.items():
            # the table is sized for inputs up to 2048 wide; ttnn picks the count (0) for wider ones
            table.setdefault((h, w, cin, cout), 0 if width > 2048 else min(slices, max(1, -(-w // 32))))
        if h_factor > 1:
            # per-device input rows: the H shard plus one halo row on each side
            for cin, cout in partitioned:
                table.setdefault((h // h_factor + 2, w, cin, cout), 0)
        h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1


def _vae_kwargs(mesh_device, parallel_config, ccl_manager) -> dict:
    return dict(
        block_out_channels=BLOCK_OUT_CHANNELS,
        layers_per_block=2,
        z_channels=32,
        parallel_config=parallel_config,
        device=mesh_device,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )


def _configure(module) -> None:
    if isinstance(module, Conv2d):
        # keeps the sliding-window config tensors out of L1_SMALL at 2K and 4K
        module.conv_config_overrides["config_tensors_in_dram"] = True
    if isinstance(module, VaeAttention):
        module.local_queries = True
    for _, child in module.named_children():
        _configure(child)


class Flux3KleinVae:
    def __init__(
        self,
        snapshot: str | os.PathLike,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        dit_ccl_manager: CCLManager,
        dit_sp_axis: int,
        tp_axis: int = 0,
        h_axis: int = 1,
    ) -> None:
        self._snapshot = Path(snapshot)
        self._device = device
        self._ccl = ccl_manager
        self._dit_ccl = dit_ccl_manager
        self._dit_sp_axis = dit_sp_axis
        self._tp_axis = tp_axis
        self._h_axis = h_axis
        blob = Path(os.path.realpath(self._snapshot / SUBFOLDER / "diffusion_pytorch_model.safetensors")).name
        self._model_name = f"flux3_klein_vae_{blob[:16]}"
        # a conv prepares its weights for the first input shape it sees, so every shape gets its own instance
        self._decoders: dict[int, Flux2VaeDecoder] = {}
        self._encoders: dict[int, Flux2VaeEncoder] = {}

    def _load(self, module, parallel_config, subfolder: str):
        from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

        _configure(module)
        cache.load_model(
            module,
            model_name=self._model_name,
            subfolder=subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(self._device.shape),
            mesh_device=self._device,
            get_torch_state_dict=lambda: AutoencoderKLFlux2.from_pretrained(
                str(self._snapshot), subfolder=SUBFOLDER
            ).state_dict(),
        )
        return module

    def decoder(self, size: int) -> Flux2VaeDecoder:
        if size not in self._decoders:
            parallel_config = Flux2VaeParallelConfig.from_axes(
                self._device, tp_axis=self._tp_axis, h_axis=self._h_axis, w_axis=None
            )
            decoder = Flux2VaeDecoder(**_vae_kwargs(self._device, parallel_config, self._ccl))
            self._decoders[size] = self._load(decoder, parallel_config, SUBFOLDER)
        return self._decoders[size]

    def encoder(self, size: int) -> Flux2VaeEncoder:
        if size not in self._encoders:
            parallel_config = Flux2VaeParallelConfig.from_axes(
                self._device, tp_axis=self._tp_axis, h_axis=None, w_axis=None
            )
            encoder = Flux2VaeEncoder(**_vae_kwargs(self._device, parallel_config, self._ccl))
            self._encoders[size] = self._load(encoder, parallel_config, "vae_encoder")
        return self._encoders[size]

    def decode(self, state: ttnn.Tensor, *, size: int) -> ttnn.Tensor:
        """Decode the SP-sharded fp32 latent state of a size x size image to a replicated [1, size, size, 3] image."""
        h, w = latent_shape(size, size)
        decoder = self.decoder(size)
        register_conv_slices(tuple(self._device.shape), size, size, h_factor=self._device.shape[self._h_axis])

        z = ttnn.typecast(state, ttnn.bfloat16)
        gathered = self._dit_ccl.all_gather(z, dim=1, mesh_axis=self._dit_sp_axis, use_hyperparams=True)
        ttnn.deallocate(z)
        rows = ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(gathered)
        if rows.shape[1] != h * w:
            sliced = ttnn.slice(rows, [0, 0, 0], [1, h * w, LATENT_CHANNELS])
            ttnn.deallocate(rows)
            rows = sliced
        parted = ttnn.mesh_partition(rows, dim=1, cluster_axis=self._h_axis, memory_config=rows.memory_config())
        ttnn.deallocate(rows)
        z = decoder.preprocess_and_unpatchify(parted, height=size // SCALE, width=size // SCALE)
        ttnn.deallocate(parted)
        out = decoder.forward(z, traced=False)
        ttnn.deallocate(z)
        return out

    def encode(self, image) -> torch.Tensor:
        """`[h * w, 128]` bf16 latent tokens of a PIL image."""
        width, height = image.size
        register_conv_slices(tuple(self._device.shape), height, width)
        x = torch.from_numpy(np.asarray(image)).float() / 127.5 - 1.0
        x = tensor.from_torch(
            x[None].contiguous(),
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, None, None],
        )
        out = self.encoder(width).encode(x, height=height, width=width, traced=False)
        tokens = tensor.to_torch(out)[0]
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        return tokens.contiguous()

    def deallocate_weights(self) -> None:
        for module in (*self._decoders.values(), *self._encoders.values()):
            module.deallocate_weights()
