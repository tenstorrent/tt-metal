# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein reference encoding on device: a prepared reference (``references.py``) -> its ``image_cond``
rows for the denoiser, through ``models/vae/encoder_flux2.py::Flux2VaeEncoder`` (design record ``docs/M8_DESIGN.md``
section 4.6). The reference is ``Pipeline.encode_references`` (``pipeline.py:84-95``): per reference ``index``,
``tokens, ids = autoencoder.encode(to_tensor(reference))``, ``ids = scaled_ids(ids)`` with ``l = -(1 + index)``,
tokens cast to bf16, everything concatenated in reference order. The ids are ``packing.reference_ids`` (host, exact);
the tokens come from the device encoder.

Parallel config on the (4, 8) ring parent: channels TP4 over mesh axis 0 (the decoder's channel axis), spatial axes
replicated over axis 1 (eight identical copies; the strided downsample is not spatially sharded in this version).
Weights: the encoder half of ``snapshot/vae`` through ``cache.load_model`` under the blob-keyed model name of the
decoder and the subfolder ``vae_encoder`` (one cache entry serves base and distilled, which link the same blob).

Eager, not traced: a reference's size varies per request and the encoder runs once per reference before the denoise;
the first call of a size compiles its kernels (~95 s on a cold ``TT_METAL_CACHE``). The host hop of the bf16 tokens
(1 MB per 1 MP reference) is the seam the denoiser already has for the text; a device handoff is a later zero-numerics
item.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn

from ...models.vae.encoder_flux2 import Flux2VaeEncoder
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from .packing import FLUX3_KLEIN_LATENT_CHANNELS, FLUX3_KLEIN_VAE_CHANNELS, latent_shape
from .references import PreparedReferences, reference_ids_for, to_tensor
from .vae_flux3_klein import (
    FLUX3_KLEIN_VAE_SCALE,
    load_vae_torch_state,
    register_klein_conv_slices,
    release_module_trace,
    set_conv_config_overrides,
    vae_model_name,
)

VAE_ENCODER_SUBFOLDER = "vae_encoder"


def build_flux3_klein_vae_encoder(
    mesh_device: ttnn.MeshDevice, *, tp_axis: int | None, ccl_manager: CCLManager | None
) -> tuple[Flux2VaeEncoder, Flux2VaeParallelConfig]:
    """``Flux2VaeEncoder`` with the Klein ``vae/config.json`` defaults, channels TP over ``tp_axis`` (``None``: no TP),
    spatial axes replicated."""
    parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=None, w_axis=None)
    encoder = Flux2VaeEncoder(
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        z_channels=FLUX3_KLEIN_VAE_CHANNELS,
        parallel_config=parallel_config,
        device=mesh_device,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    set_conv_config_overrides(encoder, config_tensors_in_dram=True)
    return encoder, parallel_config


def load_flux3_klein_vae_encoder_weights(
    encoder: Flux2VaeEncoder,
    snapshot: str | os.PathLike,
    *,
    parallel_config: Flux2VaeParallelConfig,
    mesh_device: ttnn.MeshDevice,
) -> str:
    """Upload the encoder half of ``snapshot/vae`` through ``cache.load_model`` (blob-keyed, subfolder
    ``vae_encoder``). Returns ``"HIT"`` / ``"MISS"`` / ``"NO CACHE DIR"`` for the log line."""
    model_name = vae_model_name(snapshot)
    cache_dir = (
        cache.model_cache_dir(
            model_name=model_name,
            subfolder=VAE_ENCODER_SUBFOLDER,
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            required=False,
        )
        if cache.cache_dir_is_set()
        else None
    )
    state = (
        "HIT"
        if cache_dir is not None and (Path(cache_dir) / cache.CACHE_DICT_FILE).is_file()
        else ("MISS" if cache_dir is not None else "NO CACHE DIR")
    )
    t0 = time.time()
    cache.load_model(
        encoder,
        model_name=model_name,
        subfolder=VAE_ENCODER_SUBFOLDER,
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        get_torch_state_dict=lambda: load_vae_torch_state(snapshot),
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"RECORD vae encoder weights: cache {state} ({cache_dir}) loaded in {time.time() - t0:.1f}s")
    return state


class Flux3KleinVaeEncode:
    """A prepared reference -> ``[h * w, 128]`` bf16 tokens on the host (module docstring)."""

    def __init__(self, encoder: Flux2VaeEncoder, *, device: ttnn.MeshDevice) -> None:
        self._encoder = encoder
        self._device = device
        self.last_wall_s: float = 0.0

    @property
    def encoder(self) -> Flux2VaeEncoder:
        return self._encoder

    def upload(self, image) -> ttnn.Tensor:
        """``to_tensor(image)`` -> ``[1, H, W, 3]`` bf16 ROW_MAJOR replicated on the mesh."""
        x = to_tensor(image).permute(1, 2, 0)[None].contiguous()
        return tensor.from_torch(
            x,
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, None, None],
        )

    def encode(self, image, *, traced: bool = False) -> torch.Tensor:
        """One prepared PIL reference -> bf16 ``[h * w, 128]`` host tokens in ``latent_ids`` order (eager; ``traced``
        captures / replays the encoder's conv pyramid for repeated same-size calls, tests only)."""
        width, height = image.size
        h, w = latent_shape(height, width)
        register_klein_conv_slices(tuple(self._device.shape), height, width)
        x = self.upload(image)
        ttnn.synchronize_device(self._device)
        t0 = time.time()
        out = self._encoder.encode(x, height=height, width=width, traced=traced)
        ttnn.synchronize_device(self._device)
        self.last_wall_s = time.time() - t0
        tokens = tensor.to_torch(out)[0]
        ttnn.deallocate(out)
        ttnn.deallocate(x)
        assert tuple(tokens.shape) == (h * w, FLUX3_KLEIN_LATENT_CHANNELS), tuple(tokens.shape)
        assert tokens.dtype == torch.bfloat16, tokens.dtype
        return tokens.contiguous()

    def encode_references(self, prepared: PreparedReferences) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``Pipeline.encode_references`` (``pipeline.py:84-95``): ``(cat(tokens_i) bf16 [N, 128], reference_ids_for
        fp32 [N, 4])``, ``None`` without references."""
        if not prepared.images:
            return None
        tokens = torch.cat([self.encode(image) for image in prepared.images])
        ids = reference_ids_for(prepared)
        assert tokens.shape[0] == ids.shape[0] == prepared.cond_tokens(), (tokens.shape, ids.shape)
        return tokens, ids

    def release_trace(self) -> None:
        release_module_trace(self._encoder)

    @staticmethod
    def latent_grid(image) -> tuple[int, int]:
        width, height = image.size
        return latent_shape(height, width)


__all__ = [
    "FLUX3_KLEIN_VAE_SCALE",
    "VAE_ENCODER_SUBFOLDER",
    "Flux3KleinVaeEncode",
    "build_flux3_klein_vae_encoder",
    "load_flux3_klein_vae_encoder_weights",
]
