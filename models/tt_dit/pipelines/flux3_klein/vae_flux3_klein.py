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
from ...models.vae.vae import VaeAttention
from ...models.vae.vae_flux2 import Flux2VaeDecoder
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .canvas_flux3_klein import CanvasMasks, CanvasVaeEncoder, canvas_for
from .packing import latent_shape, output_shape
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
    )


def token_rows_split(token_rows: int, h_factor: int) -> bool:
    """Whether the h token rows split evenly over the decoder's H axis; if not, the latent rows (2 per token row) do."""
    if 2 * token_rows % h_factor:
        raise ValueError(f"{2 * token_rows} latent rows do not split over {h_factor} devices")
    return token_rows % h_factor == 0


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


def decoder_cache_renames() -> list[tuple[str, str]]:
    """(regex, replacement) pairs from the base decoder's cached parameter paths to the ones this decoder loads."""
    return [(r"^(mid_block\.attentions\.\d+\.)norm\.", r"\1norm_local.")]


def decoder_cache_expected_files(base_names: list[str]) -> list[str]:
    """The parameter files this decoder asks the weight cache for, from the base decoder's listing: the renames, and
    the 2D statistics norm's (weight, bias, from_groups, to_groups) in place of the plain GroupNorm's mask."""
    import re

    out = []
    for name in base_names:
        new = name
        for pat, rep in decoder_cache_renames():
            new = re.sub(pat, rep, new)
        if re.match(r"^mid_block\.attentions\.\d+\.norm_local\.", new):
            stem, leaf = new.rsplit(".", 2)[0], new.rsplit(".", 2)[1]
            if leaf == "mask":
                continue
            out.append(new)
            if leaf == "weight":
                out.append(f"{stem}.from_groups.tensorbin")
                out.append(f"{stem}.to_groups.tensorbin")
            continue
        out.append(new)
    return sorted(out)


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
        self.ccl_manager = ccl_manager
        self._dit_ccl = dit_ccl_manager
        self._dit_sp_axis = dit_sp_axis
        self._tp_axis = tp_axis
        self.h_axis = h_axis
        self._model_name = f"flux3_klein_vae_{checkpoint_id(self._snapshot / SUBFOLDER / VAE_WEIGHTS)}"
        # a conv prepares its weights for the first input shape it sees, so every shape gets its own instance
        self._decoders: dict[tuple[int, int], Flux2VaeDecoder] = {}
        self._canvas_encoders: dict[tuple[int, int], CanvasVaeEncoder] = {}

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

    def decoder(self, size: int | tuple[int, int]) -> Flux2VaeDecoder:
        size = output_shape(size)
        if size not in self._decoders:
            parallel_config = Flux2VaeParallelConfig.from_axes(
                self._device, tp_axis=None, h_axis=self.h_axis, w_axis=self._tp_axis
            )
            decoder = Flux2VaeDecoder(
                **_vae_kwargs(self._device, parallel_config, self._ccl), fused_spatial_decode=True
            )
            self._decoders[size] = self._load(decoder, parallel_config, SUBFOLDER)
        return self._decoders[size]

    def canvas_encoder(self, canvas: tuple[int, int]) -> CanvasVaeEncoder:
        if canvas not in self._canvas_encoders:
            parallel_config = Flux2VaeParallelConfig.from_axes(
                self._device, tp_axis=self._tp_axis, h_axis=self.h_axis, w_axis=None
            )
            encoder = CanvasVaeEncoder(**_vae_kwargs(self._device, parallel_config, self._ccl))
            self._canvas_encoders[canvas] = self._load(encoder, parallel_config, "vae_encoder_canvas")
        return self._canvas_encoders[canvas]

    def encode_canvas(self, image) -> torch.Tensor:
        """`[h * w, 128]` bf16 latent tokens of a prepared reference (sides multiples of 16), encoded in the top-left
        corner of the smallest canvas that holds it."""
        width, height = image.size
        canvas = canvas_for(width, height)
        cw, ch = canvas
        mesh_key = tuple(self._device.shape)
        register_conv_slices(mesh_key, ch, cw)
        x = torch.zeros(1, ch, cw, 3)
        x[0, :height, :width] = torch.from_numpy(np.asarray(image)).float() / 127.5 - 1.0
        x = tensor.from_torch(
            x,
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, self.h_axis, None, None],
        )
        encoder = self.canvas_encoder(canvas)
        masks = CanvasMasks(self._device, canvas=canvas, image=(width, height), h_axis=self.h_axis)
        # the per-device shapes of the H-sharded convs are not in the slice table: let ttnn pick (DRAM, auto)
        table = Conv2d.slice_params.get(mesh_key, Conv2d.slice_default)
        Conv2d.slice_params[mesh_key] = _AutoSliceTable(table)
        try:
            rows = encoder.forward_canvas(x, masks)
        finally:
            Conv2d.slice_params[mesh_key] = table
        masks.deallocate()
        full = self._ccl.all_gather(rows, dim=1, mesh_axis=self.h_axis, use_hyperparams=True)
        ttnn.deallocate(rows)
        out = encoder.normalize_and_patchify(full, height=ch, width=cw)
        ttnn.deallocate(full)
        tokens = tensor.to_torch(out)[0]
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        tokens = tokens.reshape(ch // 16, cw // 16, -1)[: height // 16, : width // 16]
        return tokens.reshape(-1, tokens.shape[-1]).contiguous()

    def decode(self, state: ttnn.Tensor, *, size: int | tuple[int, int]) -> ttnn.Tensor:
        """Decode the SP-sharded fp32 latent state of an image of ``size`` (a square side or (width, height)) to a
        [1, height, width, 3] image H-sharded over `h_axis`."""
        width, height = output_shape(size)
        h, w = latent_shape(height, width)
        decoder = self.decoder(size)
        register_conv_slices(
            tuple(self._device.shape),
            height,
            width,
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
        if token_rows_split(h, self._device.shape[self.h_axis]):
            parted = ttnn.mesh_partition(rows, dim=1, cluster_axis=self.h_axis, memory_config=rows.memory_config())
            ttnn.deallocate(rows)
            z = decoder.preprocess_and_unpatchify(parted, height=height // SCALE, width=width // SCALE)
            ttnn.deallocate(parted)
            rows = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(z)
        else:
            # 144 token rows over 32 devices: unpatchify the gathered tokens, then split the 288 latent rows
            z = decoder.preprocess_and_unpatchify(rows, height=height // SCALE, width=width // SCALE, h_factor=1)
            ttnn.deallocate(rows)
            full = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(z)
            rows = ttnn.mesh_partition(full, dim=1, cluster_axis=self.h_axis, memory_config=full.memory_config())
            ttnn.deallocate(full)
        # the columns over the TP axis; the output is gathered back to the H-partitioned contract
        z = ttnn.mesh_partition(rows, dim=2, cluster_axis=self._tp_axis, memory_config=rows.memory_config())
        ttnn.deallocate(rows)
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        out = decoder.forward(z, traced=False, gather_hw=False)
        ttnn.deallocate(z)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return self._ccl.all_gather(out, dim=2, mesh_axis=self._tp_axis, use_hyperparams=True)

    def deallocate_weights(self) -> None:
        for module in (*self._decoders.values(), *self._canvas_encoders.values()):
            module.deallocate_weights()
