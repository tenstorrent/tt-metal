# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.2 32-channel KL autoencoder on the FLUX 3 Klein mesh: Flux2VaeDecoder H-partitioned over one mesh
axis and tensor-parallel over the other, Flux2VaeEncoder tensor-parallel and replicated. One instance of each
serves every shape (a Conv2d keeps its prepared weights per input shape); their conv slicing is set per instance."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

import ttnn

from ...layers.conv2d import Conv2d, configure_conv2d
from ...layers.normalization import GroupNorm
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .vae import VaeAttention
from .vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder

SUBFOLDER = "vae"
VAE_WEIGHTS = "diffusion_pytorch_model.safetensors"
SCALE = 8
PATCH = 2
LATENT_CHANNELS = 128
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


def decoder_slice_params(sizes, h_factor: int) -> dict[tuple[int, int, int, int], int]:
    """The decoder's conv slice table for square outputs of each size (the H-partitioned convs: 0, ttnn picks)."""
    table = {}
    for size in sizes:
        h = w = size
        for level, partitioned in zip(CONV_SLICES_PER_LEVEL, PARTITIONED_DECODER_CONVS_PER_LEVEL):
            for (cin, cout), slices in level.items():
                table.setdefault((h, w, cin, cout), min(slices, max(1, ttnn.core.divup(w, 32))))
            if h_factor > 1:
                for cin, cout in partitioned:
                    table.setdefault((h // h_factor + 2, w, cin, cout), 0)
            h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1
    return table


def _auto_conv_slicing(module, mesh_shape: tuple[int, ...]) -> None:
    """ttnn picks the slicing of every conv of this instance (a reference can have any 16-px shape): an empty
    per-instance table for this mesh instead of the shared defaults, which are tuned for other models' shapes."""
    if isinstance(module, Conv2d):
        module.slice_params = {**Conv2d.slice_params, mesh_shape: {}}
    for _, child in module.named_children():
        _auto_conv_slicing(child, mesh_shape)


GROUP_NORM_MAX_BLOCK_ROWS = 64


def group_norm_out_blocks(norm: GroupNorm, batch: int, spatial: int, channels: int) -> int:
    """num_out_blocks for ``norm`` on a [batch, spatial, channels] input: -1 (the op's auto choice) unless that leaves
    more than GROUP_NORM_MAX_BLOCK_ROWS tile rows per block, else the smallest divisor of the per-core block height
    that does not."""
    grid = norm._op_core_grid(batch, spatial)
    virtual_rows = (grid.x // norm.num_virtual_cols) * grid.y
    block_h = batch * math.ceil(spatial / ttnn.TILE_SIZE) // virtual_rows
    target = max(
        1, spatial * channels // (256 * 256 * (grid.x // norm.num_virtual_cols) * norm.num_virtual_cols * grid.y)
    )
    auto = 1
    while auto < target and auto < 256:
        auto <<= 1
    if block_h <= GROUP_NORM_MAX_BLOCK_ROWS * auto:
        return -1
    low = max(auto, ttnn.core.divup(block_h, GROUP_NORM_MAX_BLOCK_ROWS))
    return next(d for d in range(low, block_h + 1) if block_h % d == 0)


def _bounded_group_norm_blocks(module) -> None:
    if isinstance(module, GroupNorm):
        forward = module.forward

        def bounded(x, num_out_blocks=-1, compute_kernel_config=None, _forward=forward, _norm=module):
            if num_out_blocks == -1:
                batch, height, width, channels = x.shape
                num_out_blocks = group_norm_out_blocks(_norm, batch, height * width, channels)
            return _forward(x, num_out_blocks=num_out_blocks, compute_kernel_config=compute_kernel_config)

        module.forward = bounded
    for _, child in module.named_children():
        _bounded_group_norm_blocks(child)


def _local_query_attention(module) -> None:
    if isinstance(module, VaeAttention):
        module.local_queries = True
    for _, child in module.named_children():
        _local_query_attention(child)


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
        self._h_axis = h_axis
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

        decoder_parallel = Flux2VaeParallelConfig.from_axes(device, tp_axis=tp_axis, h_axis=h_axis, w_axis=None)
        self._decoder = Flux2VaeDecoder(parallel_config=decoder_parallel, use_conv3d=False, **common)
        configure_conv2d(
            self._decoder, slice_params={mesh_shape: decoder_slice_params(sizes, int(device.shape[h_axis]))}
        )
        _local_query_attention(self._decoder)
        load(self._decoder, decoder_parallel, SUBFOLDER)

        encoder_parallel = Flux2VaeParallelConfig.from_axes(device, tp_axis=tp_axis, h_axis=None, w_axis=None)
        self._encoder = Flux2VaeEncoder(parallel_config=encoder_parallel, **common)
        _auto_conv_slicing(self._encoder, mesh_shape)
        _bounded_group_norm_blocks(self._encoder)
        load(self._encoder, encoder_parallel, "vae_encoder_main")

        state = state_dict()
        self._bn_mean = state["bn.running_mean"].float()
        self._bn_std = torch.sqrt(state["bn.running_var"].float() + torch_vae.config.batch_norm_eps)

    def decode(self, state: ttnn.Tensor, *, size: int) -> ttnn.Tensor:
        """Decode the SP-sharded fp32 latent state of a size x size image to a replicated [1, size, size, 3] image."""
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
        parted = ttnn.mesh_partition(rows, dim=1, cluster_axis=self._h_axis, memory_config=rows.memory_config())
        ttnn.deallocate(rows)
        z = decoder.preprocess_and_unpatchify(parted, height=size // SCALE, width=size // SCALE)
        ttnn.deallocate(parted)
        out = decoder.forward(z, traced=False)
        ttnn.deallocate(z)
        return out

    def encode(self, image) -> torch.Tensor:
        """`[h * w, 128]` bf16 latent tokens of a PIL image (sides multiples of 16): Flux2VaeEncoder gives the latent mean,
        then the 2x2 patchify and the BatchNorm on the host, as pipeline_flux2.py does."""
        x = torch.from_numpy(np.asarray(image)).float() / 127.5 - 1.0
        x = tensor.from_torch(
            x[None].contiguous(),
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, None, None],
        )
        out = self._encoder.forward(x)
        ttnn.deallocate(x)
        mean = tensor.to_torch(out).float()
        ttnn.deallocate(out)
        b, h, w, c = mean.shape
        tokens = (
            mean.reshape(b, h // PATCH, PATCH, w // PATCH, PATCH, c).permute(0, 1, 3, 5, 2, 4).flatten(3).flatten(1, 2)
        )
        return ((tokens[0] - self._bn_mean) / self._bn_std).to(torch.bfloat16).contiguous()

    def deallocate_weights(self) -> None:
        self._decoder.deallocate_weights()
        self._encoder.deallocate_weights()
