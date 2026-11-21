# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.2 32-channel KL autoencoder on the FLUX 3 Klein mesh: Flux2VaeDecoder splits the image's rows over one
mesh axis and its columns over the other, Flux2VaeEncoder is tensor-parallel and replicated on fixed reference
canvases (canvas_flux3_klein) and also splits the rows of the 1 MP square canvas over the decoder's row axis. One
instance of each serves every shape (a Conv2d keeps its prepared weights per input shape); their conv slicing is set
per instance."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

import ttnn

from ...layers.conv2d import Conv2dBlocking, configure_conv2d
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .canvas_flux3_klein import CanvasMasks, CanvasVaeEncoder, canvas_for
from .vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder

SUBFOLDER = "vae"
VAE_WEIGHTS = "diffusion_pytorch_model.safetensors"
SCALE = 8
PATCH = 2
LATENT_CHANNELS = 128
# A reference that fills the 1 MP square canvas (the served square reference, every square multi-reference canvas)
# runs the plain encoder with its rows split over h_axis: 0.063-0.067 s per reference, 0.26 s TP alone (226987 / 226769)
FULL_CANVAS = (1024, 1024)
BLOCK_OUT_CHANNELS = (128, 256, 512, 512)
Z_CHANNELS = 32

CONV_SLICES_PER_LEVEL = (
    {(3, 128): 16, (128, 128): 16, (128, 3): 8, (256, 128): 16, (256, 256): 16},
    {(128, 256): 8, (256, 256): 4, (512, 256): 16, (512, 512): 16},
    {(256, 512): 8, (512, 512): 8},
    {(512, 512): 4, (512, 64): 4, (32, 512): 1},
)
PARTITIONED_DECODER_CONVS_PER_LEVEL = (
    ((256, 256), (256, 128), (128, 128), (128, 3)),
    ((512, 512), (512, 256), (256, 256)),
    ((512, 512),),
    ((32, 512), (512, 512)),
)


def decoder_slice_params(sizes, h_factor: int, w_factor: int = 1) -> dict[tuple[int, int, int, int], int]:
    """The decoder's conv slice table for square outputs of each size (the partitioned convs: 0, ttnn picks)."""
    table = {}
    for size in sizes:
        h = w = size
        for level, partitioned in zip(CONV_SLICES_PER_LEVEL, PARTITIONED_DECODER_CONVS_PER_LEVEL):
            for (cin, cout), slices in level.items():
                table.setdefault((h, w, cin, cout), min(slices, max(1, ttnn.core.divup(w, 32))))
            if h_factor > 1:
                local_w = w // w_factor + 2 if w_factor > 1 else w
                for cin, cout in partitioned:
                    table.setdefault((h // h_factor + 2, local_w, cin, cout), 0)
            h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1
    return table


# Full-K weights plus the activation block a core can hold next to the resident DiT (the largest that fit: 1.33 MB)
CONV_FULL_K_L1_BYTES = 1_350_000
# Decoder convs whose double-buffered program clashes with the resident DiT's L1 (job 210339: 1024 x 1024, level 0)
DECODER_SINGLE_BUFFER_CONVS = frozenset({(130, 258, 256, 128)})
# Per-device input pixels from which a conv whose full-K weights do not fit runs as output-channel slices that do
OUT_CHANNEL_SPLIT_MIN_PIXELS = 100_000
# Encoder conv slices per mesh shape; ttnn picks the rest. These sit at its L1 threshold: 3 slices before the first
# render, 4 after (job 224624: 1 KB less L1 free), which compiled new programs on the second 2816x1536 encode
ENCODER_SLICE_PARAMS = {(4, 8): {(353, 1408, 256, 256): 4, (354, 1408, 256, 256): 4}}


def _full_k_fits(cin: int, cout: int) -> bool:
    return 2 * 9 * cin * (cout + 64) <= CONV_FULL_K_L1_BYTES


def _out_channel_splits(cin: int, cout: int) -> int:
    return next((n for n in (2, 4, 8, 16) if cout % n == 0 and cout // n >= 32 and _full_k_fits(cin, cout // n)), 1)


def decoder_conv_blocking(slice_table) -> dict[tuple[int, int, int, int], Conv2dBlocking]:
    """64 activation rows per core with double-buffered activations and weights; the full inner dimension and
    height-sharded activations where a 3 x 3 conv's full-K weights fit in L1. Where they do not, large inputs run as
    output-channel slices whose weights fit, small ones block-shard."""
    table = {}
    for key in slice_table:
        h, w, cin, cout = key
        fits = _full_k_fits(cin, cout)
        splits = 1 if fits or h * w < OUT_CHANNEL_SPLIT_MIN_PIXELS else _out_channel_splits(cin, cout)
        full_k = fits or splits > 1
        table[key] = Conv2dBlocking(
            act_block_h_override=64,
            double_buffer=key not in DECODER_SINGLE_BUFFER_CONVS,
            full_inner_dim=full_k,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED if full_k else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            out_channel_splits=splits,
        )
    return table


class Flux3KleinVae:
    def __init__(
        self,
        snapshot: str | os.PathLike,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        dit_ccl_manager: CCLManager,
        dit_sp_axis: int,
        sizes: tuple[int, ...],
        tp_axis: int = 0,
        h_axis: int = 1,
    ) -> None:
        self._snapshot = Path(snapshot)
        self._device = device
        self._ccl = ccl_manager
        self._dit_ccl = dit_ccl_manager
        self._dit_sp_axis = dit_sp_axis
        self.h_axis = h_axis
        self.w_axis = tp_axis
        model_name = f"flux3_klein_vae_{cache.checkpoint_id(self._snapshot / SUBFOLDER / VAE_WEIGHTS)}"
        mesh_shape = tuple(device.shape)
        common = {
            "block_out_channels": BLOCK_OUT_CHANNELS,
            "layers_per_block": 2,
            "z_channels": Z_CHANNELS,
            "device": device,
            "ccl_manager": ccl_manager,
        }
        torch_vae = None

        def state_dict():
            nonlocal torch_vae
            if torch_vae is None:
                from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

                torch_vae = AutoencoderKLFlux2.from_pretrained(str(self._snapshot), subfolder=SUBFOLDER)
            return torch_vae.state_dict()

        def load(module, parallel_config, subfolder: str):
            cache.load_model(
                module,
                model_name=model_name,
                subfolder=subfolder,
                parallel_config=parallel_config,
                mesh_shape=mesh_shape,
                mesh_device=device,
                get_torch_state_dict=state_dict,
            )
            return module

        decoder_parallel = Flux2VaeParallelConfig.from_axes(device, tp_axis=None, h_axis=h_axis, w_axis=tp_axis)
        self._decoder = Flux2VaeDecoder(
            parallel_config=decoder_parallel, use_conv3d=False, group_norm_2d=True, local_attention=True, **common
        )
        slice_table = decoder_slice_params(sizes, int(device.shape[h_axis]), int(device.shape[tp_axis]))
        configure_conv2d(
            self._decoder, slice_params={mesh_shape: slice_table}, shape_blocking=decoder_conv_blocking(slice_table)
        )
        load(self._decoder, decoder_parallel, SUBFOLDER)

        encoder_parallel = Flux2VaeParallelConfig.from_axes(device, tp_axis=tp_axis, h_axis=None, w_axis=None)
        self._encoder = CanvasVaeEncoder(parallel_config=encoder_parallel, **common)
        encoder_slices = ENCODER_SLICE_PARAMS.get(mesh_shape, {})
        configure_conv2d(self._encoder, slice_params={mesh_shape: encoder_slices}, replace=True)
        load(self._encoder, encoder_parallel, "vae_encoder_canvas")
        full_parallel = Flux2VaeParallelConfig.from_axes(device, tp_axis=tp_axis, h_axis=h_axis, w_axis=None)
        self._full_encoder = Flux2VaeEncoder(parallel_config=full_parallel, **common)
        configure_conv2d(self._full_encoder, slice_params={mesh_shape: {}}, replace=True)
        load(self._full_encoder, full_parallel, "vae_encoder_full")

        state = state_dict()
        self._bn_mean = state["bn.running_mean"].float()
        self._bn_std = torch.sqrt(state["bn.running_var"].float() + torch_vae.config.batch_norm_eps)

    def decode(self, state: ttnn.Tensor, *, size: int) -> ttnn.Tensor:
        """Decode the SP-sharded fp32 latent state of a size x size image to a [1, size, size, 3] image with its rows
        sharded over `h_axis` and its columns over `w_axis`."""
        h = w = size // (SCALE * PATCH)
        decoder = self._decoder

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
        rows = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(z)
        z = ttnn.mesh_partition(rows, dim=2, cluster_axis=self.w_axis, memory_config=rows.memory_config())
        ttnn.deallocate(rows)
        z = ttnn.to_layout(z, ttnn.TILE_LAYOUT)
        out = decoder.forward(z, traced=False, gather_hw=False)
        ttnn.deallocate(z)
        return out

    def encode(self, image) -> torch.Tensor:
        """`[h * w, 128]` bf16 latent tokens of a PIL image (sides multiples of 16, at most a reference canvas):
        the encoder on the image in the top-left corner of its canvas gives the latent mean, cropped to the image, then
        the 2x2 patchify and the BatchNorm on the host, as pipeline_flux2.py does."""
        width, height = image.size
        cw, ch = canvas_for(width, height)
        x = torch.zeros(1, ch, cw, 3)
        x[0, :height, :width] = torch.from_numpy(np.asarray(image)).float() / 127.5 - 1.0
        full = (width, height) == FULL_CANVAS
        x = tensor.from_torch(
            x,
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, self.h_axis if full else None, None, None],
        )
        if full:
            out = self._full_encoder.forward(x)
        else:
            masks = CanvasMasks(self._device, canvas=(cw, ch), image=(width, height))
            out = self._encoder.forward_canvas(x, masks)
            masks.deallocate()
        ttnn.deallocate(x)
        mean = tensor.to_torch(out).float()[:, : height // SCALE, : width // SCALE]
        ttnn.deallocate(out)
        b, h, w, c = mean.shape
        tokens = (
            mean.reshape(b, h // PATCH, PATCH, w // PATCH, PATCH, c).permute(0, 1, 3, 5, 2, 4).flatten(3).flatten(1, 2)
        )
        return ((tokens[0] - self._bn_mean) / self._bn_std).to(torch.bfloat16).contiguous()

    def deallocate_weights(self) -> None:
        self._decoder.deallocate_weights()
        self._encoder.deallocate_weights()
        self._full_encoder.deallocate_weights()
