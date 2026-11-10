# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.2 32-channel KL autoencoder on the FLUX 3 Klein mesh: the decoder splits the image's rows over one mesh axis
and its columns over the other, the encoder is tensor-parallel (and H-sharded where the size allows)."""

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
from .weights_flux3_klein import checkpoint_id

SUBFOLDER = "vae"
VAE_WEIGHTS = "diffusion_pytorch_model.safetensors"
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


def register_conv_slices(
    mesh_shape: tuple[int, int], height: int, width: int, h_factor: int = 1, w_factor: int = 1
) -> None:
    table = Conv2d.slice_params.setdefault(tuple(mesh_shape), dict(Conv2d.slice_default))
    h, w = height, width
    for level, partitioned in zip(CONV_SLICES_PER_LEVEL, PARTITIONED_DECODER_CONVS_PER_LEVEL):
        for (cin, cout), slices in level.items():
            table.setdefault((h, w, cin, cout), min(slices, max(1, -(-w // 32))))
        if h_factor > 1:
            # per-device input rows (and columns): the shard plus one halo row / column on each side
            local_w = w // w_factor + 2 if w_factor > 1 else w
            for cin, cout in partitioned:
                table.setdefault((h // h_factor + 2, local_w, cin, cout), 0)
        h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1


class _AutoSliceTable(dict):
    """A conv slice table whose missing shapes get 0 (ttnn picks the DRAM width-slice count) instead of None."""

    def get(self, key, default=None):
        return super().get(key, 0)


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


# a decoder conv whose height-sharded config finds no valid DRAM slice (per-device input at 4096)
UNSHARDABLE_DECODER_CONVS = frozenset({(258, 2048, 512, 512)})


# L1 budget of a conv's full-K weights plus its activation block (the TP convs reach 1.33 MB)
CONV_FULL_K_L1_BYTES = 1_350_000
# smallest per-device input (pixels) whose convs split their output channels when the full-K weights do not fit
COUT_SPLIT_MIN_PIXELS = 100_000


def _full_inner_dim_fits(conv: Conv2d, out_channels: int | None = None, act_rows: int = 64) -> bool:
    cout = conv.out_channels if out_channels is None else out_channels
    return 2 * 9 * conv.in_channels * (cout + act_rows) <= CONV_FULL_K_L1_BYTES


def decoder_cache_subfolder(subfolder: str = SUBFOLDER) -> str:
    """The decoder's weight-cache subfolder: the local attention norm (norm_local, TILE) renames parameters."""
    return f"{subfolder}_attnlocal"


def _cout_split_for(conv: Conv2d) -> int:
    """The output-channel split of a conv whose full-K weights do not fit, 1 for none."""
    for n in (2, 4, 8, 16):
        if (
            conv.out_channels % n == 0
            and conv.out_channels // n >= 32
            and _full_inner_dim_fits(conv, conv.out_channels // n)
        ):
            return n
    return 1


def _configure(module, *, decoder: bool, spatial_2d: bool = False) -> None:
    if isinstance(module, Conv2d):
        # keeps the sliding-window config tensors out of L1_SMALL at 2K and 4K
        module.config_tensors_in_dram = True
        module.act_block_h_override = 64
        module.double_buffer = True
        fits = not spatial_2d or _full_inner_dim_fits(module)
        if decoder and spatial_2d and not fits:
            module.cout_split = _cout_split_for(module)
            module.split_min_pixels = COUT_SPLIT_MIN_PIXELS
        module.full_inner_dim = decoder and fits
        if decoder:
            # height-sharded activations; shapes that do not fit next to the resident DiT fall back per conv. With
            # full channels (2D layout) a core cannot hold a 512-input conv's weights (4.7 MB): split them over the
            # core columns instead (block sharding); those convs sit at the small latent and 1024 levels
            module.shard_layout = (
                ttnn.TensorMemoryLayout.BLOCK_SHARDED
                if spatial_2d and not fits
                else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            )
            module.no_shard = UNSHARDABLE_DECODER_CONVS
    if isinstance(module, VaeAttention):
        module.local_queries = True
    for _, child in module.named_children():
        _configure(child, decoder=decoder, spatial_2d=spatial_2d)


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
        self.h_axis = h_axis
        self._model_name = f"flux3_klein_vae_{checkpoint_id(self._snapshot / SUBFOLDER / VAE_WEIGHTS)}"
        # a conv prepares its weights for the first input shape it sees, so every shape gets its own instance
        self._decoders: dict[int, Flux2VaeDecoder] = {}
        self._encoders: dict[int, Flux2VaeEncoder] = {}

    def _load(self, module, parallel_config, subfolder: str):
        from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

        _configure(module, decoder=isinstance(module, Flux2VaeDecoder), spatial_2d=parallel_config.tp_parallel is None)
        if isinstance(module, Flux2VaeDecoder):
            subfolder = decoder_cache_subfolder(subfolder)
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
                self._device, tp_axis=None, h_axis=self.h_axis, w_axis=self._tp_axis
            )
            decoder = Flux2VaeDecoder(
                **_vae_kwargs(self._device, parallel_config, self._ccl), fused_spatial_decode=True
            )
            self._decoders[size] = self._load(decoder, parallel_config, SUBFOLDER)
        return self._decoders[size]

    def _encoder_h_sharded(self, height: int) -> bool:
        # H / h_factor rows per device must stay even through the three stride-2 downsamplers
        return height % (SCALE * self._device.shape[self.h_axis]) == 0

    def encoder(self, size: int) -> Flux2VaeEncoder:
        if size not in self._encoders:
            parallel_config = Flux2VaeParallelConfig.from_axes(
                self._device,
                tp_axis=self._tp_axis,
                h_axis=self.h_axis if self._encoder_h_sharded(size) else None,
                w_axis=None,
            )
            encoder = Flux2VaeEncoder(**_vae_kwargs(self._device, parallel_config, self._ccl))
            self._encoders[size] = self._load(encoder, parallel_config, "vae_encoder")
        return self._encoders[size]

    def decode(self, state: ttnn.Tensor, *, size: int) -> ttnn.Tensor:
        """Decode the SP-sharded fp32 latent state of a size x size image to a [1, size, size, 3] image H-sharded over
        `h_axis`."""
        h, w = latent_shape(size, size)
        decoder = self.decoder(size)
        register_conv_slices(
            tuple(self._device.shape),
            size,
            size,
            h_factor=self._device.shape[self.h_axis],
            w_factor=self._device.shape[self._tp_axis],
        )

        z = ttnn.typecast(state, ttnn.bfloat16)
        gathered = self._dit_ccl.all_gather(z, dim=1, mesh_axis=self._dit_sp_axis, use_hyperparams=True)
        ttnn.deallocate(z)
        rows = ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(gathered)
        if rows.shape[1] != h * w:
            sliced = ttnn.slice(rows, [0, 0, 0], [1, h * w, LATENT_CHANNELS])
            ttnn.deallocate(rows)
            rows = sliced
        parted = ttnn.mesh_partition(rows, dim=1, cluster_axis=self.h_axis, memory_config=rows.memory_config())
        ttnn.deallocate(rows)
        z = decoder.preprocess_and_unpatchify(parted, height=size // SCALE, width=size // SCALE)
        ttnn.deallocate(parted)
        # the columns over the TP axis; the output is gathered back to the H-partitioned contract
        rows = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(z)
        z = ttnn.mesh_partition(rows, dim=2, cluster_axis=self._tp_axis, memory_config=rows.memory_config())
        ttnn.deallocate(rows)
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        out = decoder.forward(z, traced=False, gather_hw=False)
        ttnn.deallocate(z)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return self._ccl.all_gather(out, dim=2, mesh_axis=self._tp_axis, use_hyperparams=True)

    def encode(self, image) -> torch.Tensor:
        """`[h * w, 128]` bf16 latent tokens of a PIL image."""
        width, height = image.size
        mesh_key = tuple(self._device.shape)
        register_conv_slices(mesh_key, height, width)
        sharded = height == width and self._encoder_h_sharded(height)
        x = torch.from_numpy(np.asarray(image)).float() / 127.5 - 1.0
        x = tensor.from_torch(
            x[None].contiguous(),
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, self.h_axis if sharded else None, None, None],
        )
        encoder = self.encoder(width)
        if sharded:
            # the per-device shapes of the H-sharded convs are not in the slice table: let ttnn pick (DRAM, auto)
            table = Conv2d.slice_params.get(mesh_key, Conv2d.slice_default)
            Conv2d.slice_params[mesh_key] = _AutoSliceTable(table)
            try:
                rows = encoder.forward(x, traced=False)
            finally:
                Conv2d.slice_params[mesh_key] = table
            full = self._ccl.all_gather(rows, dim=1, mesh_axis=self.h_axis, use_hyperparams=True)
            ttnn.deallocate(rows)
            out = encoder.normalize_and_patchify(full, height=height, width=width)
            ttnn.deallocate(full)
        else:
            out = encoder.encode(x, height=height, width=width, traced=False)
        tokens = tensor.to_torch(out)[0]
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        return tokens.contiguous()

    def deallocate_weights(self) -> None:
        for module in (*self._decoders.values(), *self._encoders.values()):
            module.deallocate_weights()
