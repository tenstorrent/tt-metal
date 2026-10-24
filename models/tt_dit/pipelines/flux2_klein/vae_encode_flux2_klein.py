# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein reference encoding on device: a prepared reference (``references.py``) -> its condition rows for
the denoiser, through ``models/vae/encoder_flux2.py::Flux2VaeEncoder``. The reference is ``Flux2KleinPipeline.
prepare_image_latents`` (``pipeline_flux2_klein.py:514-540``, cited as ``P``): per condition image
``_encode_vae_image`` (``P:462-477``: the posterior MEAN, ``sample_mode="argmax"``, ``_patchify_latents``, the
BatchNorm normalize), ``_prepare_image_ids`` (``P:320-367``, ``t = 10 + 10 i``), ``_pack_latents`` and one
``torch.cat`` in reference order (BFL ``sampling.py:52-90`` ``encode_image_refs`` is the same, casting the tokens to
bf16). The ids are ``packing.reference_ids`` (host, exact); the tokens come from the device encoder.

Parallel config on the (4, 8) ring parent: channels TP4 over mesh axis 0 (the decoder's channel axis), spatial axes
replicated over axis 1 (eight identical copies; the strided downsample is not spatially sharded in this version).
Weights: the encoder half of ``snapshot/vae`` through ``cache.load_model`` under the blob-keyed model name of the
decoder and the subfolder ``vae_encoder`` (one cache entry serves base and distilled, which link the same blob).

Eager, not traced: a reference's size varies per request and the encoder runs once per reference before the denoise;
the first call of a size compiles its kernels. The host hop of the bf16 tokens (1 MB per 1 MP reference) is the seam
the denoiser already has for the text; a device handoff is a later zero-numerics item.

Upload: the encoder wants the ``[1, H, W, 3]`` bf16 ROW_MAJOR image on EVERY chip. Writing it from the host to all 32
(6.3 MB x 32 PCIe writes per 1024x1024 reference) cost ~70 ms of host time per reference next to a 0.21 s encode, so
``Flux2KleinVaeEncode`` writes each mesh column ONE row band instead (``sharded_upload_view``: the image flattened to
``[1, 1, 8 R, C]`` fp32 on the host, no copy, sharded over the encoder's replicated axis -> 0.8 MB per chip, replicated
over the TP axis as before) and rebuilds the replicated image on the device: tilize, the encoder's CCL ``all_gather``
over that axis on dim 2, untilize, ``reshape`` to ``[1, H, W, 3]``: pure bf16 data movement, every chip ends with the
same bytes the replicated upload wrote (gate: tokens ``torch.equal``). The view needs tile-aligned bands (``R`` and
``C`` multiples of 32: 1024x1024, 1344x768, 1024x768, ...); a size without one (880x1168, 400x736, ...) takes the
replicated upload as before. ``replicate_on_device=False`` keeps the host replication for every size (A/B only).
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
from .packing import FLUX2_KLEIN_LATENT_CHANNELS, FLUX2_KLEIN_VAE_CHANNELS, latent_shape
from .references import PreparedReferences, reference_ids_for, to_pixels
from .vae_flux2_klein import (
    FLUX2_KLEIN_VAE_SCALE,
    _config_tensors_in_dram,
    _trace_captured,
    load_vae_torch_state,
    register_klein_conv_slices,
    release_module_trace,
    set_conv_config_overrides,
    vae_model_name,
)

VAE_ENCODER_SUBFOLDER = "vae_encoder"
UPLOAD_PAGE_MAX = 4096  # widest row (bf16 elements, 8 KB) of the sharded upload view
_TILE = 32


def sharded_upload_view(
    height: int, width: int, shards: int, *, page_max: int = UPLOAD_PAGE_MAX
) -> tuple[int, int] | None:
    """``(R, C)``: the ``[1, 1, R, C]`` row-major band each of ``shards`` mesh columns receives of a ``[1, height, width,
    3]`` image flattened to ``[1, 1, shards * R, C]`` (band ``k`` = image rows ``[k H / shards, (k + 1) H / shards)``,
    the flat order of the image), or ``None`` when no such view exists. ``C`` is the widest multiple of 32 up to
    ``page_max`` dividing a band's ``(H / shards) * W * 3`` elements with ``R`` a multiple of 32 (both tile-aligned, so
    the tilize before the all-gather pads nothing and the gathered tiles untilize to the image's bytes)."""
    if shards < 1 or height % shards:
        return None
    n = (height // shards) * width * 3
    for k in range(page_max // _TILE, 0, -1):
        c = _TILE * k
        if n % c == 0 and (n // c) % _TILE == 0:
            return n // c, c
    return None


def build_flux2_klein_vae_encoder(
    mesh_device: ttnn.MeshDevice, *, tp_axis: int | None, ccl_manager: CCLManager | None
) -> tuple[Flux2VaeEncoder, Flux2VaeParallelConfig]:
    """``Flux2VaeEncoder`` with the Klein ``vae/config.json`` defaults, channels TP over ``tp_axis`` (``None``: no TP),
    spatial axes replicated."""
    parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=None, w_axis=None)
    encoder = Flux2VaeEncoder(
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        z_channels=FLUX2_KLEIN_VAE_CHANNELS,
        parallel_config=parallel_config,
        device=mesh_device,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    set_conv_config_overrides(encoder, config_tensors_in_dram=_config_tensors_in_dram())
    return encoder, parallel_config


def load_flux2_klein_vae_encoder_weights(
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


class Flux2KleinVaeEncode:
    """A prepared reference -> ``[h * w, 128]`` bf16 tokens on the host (module docstring)."""

    def __init__(self, encoder: Flux2VaeEncoder, *, device: ttnn.MeshDevice, replicate_on_device: bool = True) -> None:
        self._encoder = encoder
        self._device = device
        # ``True``: one row band per mesh column + the on-device gather (module docstring); ``False``: every chip
        # written from the host. Sizes without a ``sharded_upload_view`` take the host path either way.
        self.replicate_on_device = replicate_on_device
        self.last_wall_s: float = 0.0
        self.last_upload_s: float = 0.0  # host wall of the image upload (+ the enqueue of the on-device gather)
        self.last_upload_path: str = ""  # "sharded" | "replicated" of the last encode
        # The image tensor the encoder trace was CAPTURED on: it is the trace's input buffer (``utils/tracing.py``
        # ``_update_input`` copies every later input into it), so it lives as long as the trace does (the decoder's
        # own rule for ``z`` in ``Flux2KleinVaeDecode.decode``). Freeing it after the capture call left the tracer
        # holding a deallocated tensor, whose ``device()`` no longer compares equal on the next replay.
        self._trace_input: ttnn.Tensor | None = None

    @property
    def encoder(self) -> Flux2VaeEncoder:
        return self._encoder

    @staticmethod
    def _pixels(image, pixels: torch.Tensor | None) -> torch.Tensor:
        width, height = image.size
        x = to_pixels(image) if pixels is None else pixels
        if tuple(x.shape) != (1, height, width, 3) or x.dtype != torch.float32:
            raise ValueError(f"pixels {x.dtype} {tuple(x.shape)} do not match the {width}x{height} prepared reference")
        return x

    def upload(self, image, pixels: torch.Tensor | None = None) -> ttnn.Tensor:
        """``to_pixels(image)`` (fp32 ``[1, H, W, 3]``; ``pixels`` when the caller holds it already, the pipeline's
        reference cache does) -> bf16 ROW_MAJOR replicated on the mesh from the host (``P:524``: the condition image
        enters the VAE in its dtype, bf16): one write per chip."""
        return tensor.from_torch(
            self._pixels(image, pixels),
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, None, None],
        )

    def replicate_axis(self) -> tuple[int, int] | None:
        """``(mesh axis, size)`` of the encoder's spatially replicated axis (the one its channel TP does not use), or
        ``None`` when there is no such axis of size > 1 or no CCL manager to gather over."""
        ctx = getattr(self._encoder, "_ctx", None)
        if ctx is None or ctx.ccl_manager is None or ctx.tp_axis is None:
            return None
        shape = tuple(self._device.shape)
        axes = [ax for ax in range(len(shape)) if ax != ctx.tp_axis and shape[ax] > 1]
        if len(axes) != 1:
            return None
        return axes[0], shape[axes[0]]

    def upload_sharded(self, image, pixels: torch.Tensor | None = None) -> ttnn.Tensor | None:
        """The image as ONE row band per mesh column (``sharded_upload_view``: ``[1, 1, R, C]`` bf16 ROW_MAJOR per chip,
        sharded over the replicated axis, replicated over the TP axis), gathered back to the replicated ``[1, H, W, 3]``
        on the device (tilize -> CCL all_gather on dim 2 -> untilize -> reshape; bf16 data movement only). ``None`` when
        the size has no view or the mesh no replicated axis: the caller takes :meth:`upload`."""
        width, height = image.size
        axis = self.replicate_axis()
        view = None if axis is None else sharded_upload_view(height, width, axis[1])
        if view is None:
            return None
        mesh_axis, shards = axis
        rows, cols = view
        x = self._pixels(image, pixels).reshape(1, 1, shards * rows, cols)  # a view: the image's flat order
        band = tensor.from_torch(
            x,
            device=self._device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, mesh_axis, None],
        )
        tiles = ttnn.to_layout(band, ttnn.TILE_LAYOUT)  # R, C multiples of 32: no padding
        ttnn.deallocate(band)
        gathered = self._encoder._ctx.ccl_manager.all_gather(tiles, dim=2, mesh_axis=mesh_axis, use_hyperparams=True)
        ttnn.deallocate(tiles)
        flat = ttnn.to_layout(gathered, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, shards * R, C] = the image's bytes
        ttnn.deallocate(gathered)
        replicated = ttnn.reshape(flat, [1, height, width, 3])
        if replicated.buffer_address() != flat.buffer_address():  # the row split moves the data; a view would alias
            ttnn.deallocate(flat)
        return replicated

    def upload_image(self, image, pixels: torch.Tensor | None = None) -> ttnn.Tensor:
        """The encoder's ``[1, H, W, 3]`` bf16 ROW_MAJOR input replicated on the mesh: :meth:`upload_sharded` when
        ``replicate_on_device`` and the size has a view (``last_upload_path`` ``"sharded"``), else :meth:`upload`
        (``"replicated"``)."""
        t0 = time.time()
        x = self.upload_sharded(image, pixels) if self.replicate_on_device else None
        self.last_upload_path = "sharded" if x is not None else "replicated"
        if x is None:
            x = self.upload(image, pixels)
        self.last_upload_s = time.time() - t0
        return x

    def encode(self, image, *, pixels: torch.Tensor | None = None, traced: bool = False) -> torch.Tensor:
        """One prepared PIL reference (``pixels``: its :func:`to_pixels`, when held) -> bf16 ``[h * w, 128]`` host
        tokens in ``latent_ids`` order (eager; ``traced`` captures / replays the encoder's conv pyramid for repeated
        same-size calls, tests only)."""
        width, height = image.size
        h, w = latent_shape(height, width)
        register_klein_conv_slices(tuple(self._device.shape), height, width)
        # No synchronize between the upload and the encode: both are enqueued on the same command queue in order.
        x = self.upload_image(image, pixels)
        capture = traced and not _trace_captured(self._encoder)
        t0 = time.time()
        out = self._encoder.encode(x, height=height, width=width, traced=traced)
        ttnn.synchronize_device(self._device)
        self.last_wall_s = time.time() - t0
        tokens = tensor.to_torch(out)[0]
        ttnn.deallocate(out)
        if capture:
            self._release_trace_input()
            self._trace_input = x
        else:
            # Eager: a plain input. Traced REPLAY: the tracer copied ``x`` into the captured input buffer (enqueued
            # before this deallocate on the same command queue), so this fresh upload is free to go.
            ttnn.deallocate(x)
        if tuple(tokens.shape) != (h * w, FLUX2_KLEIN_LATENT_CHANNELS) or tokens.dtype != torch.bfloat16:
            raise RuntimeError(f"encoder returned {tokens.dtype} {tuple(tokens.shape)} for a {h}x{w} reference")
        return tokens.contiguous()

    def encode_references(self, prepared: PreparedReferences) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``prepare_image_latents`` (``P:514-540``): ``(cat(tokens_i) bf16 [N, 128], reference_ids_for int64
        [N, 4])``, ``None`` without references."""
        if not prepared.images:
            return None
        tokens = torch.cat(
            [self.encode(image, pixels=prepared.pixel_tensor(i)) for i, image in enumerate(prepared.images)]
        )
        ids = reference_ids_for(prepared)
        if not tokens.shape[0] == ids.shape[0] == prepared.cond_tokens():
            raise RuntimeError(f"reference tokens {tuple(tokens.shape)} and ids {tuple(ids.shape)} disagree")
        return tokens, ids

    def release_trace(self) -> None:
        release_module_trace(self._encoder)
        self._release_trace_input()

    def _release_trace_input(self) -> None:
        if self._trace_input is not None:
            ttnn.deallocate(self._trace_input)
            self._trace_input = None

    @staticmethod
    def latent_grid(image) -> tuple[int, int]:
        width, height = image.size
        return latent_shape(height, width)


__all__ = [
    "FLUX2_KLEIN_VAE_SCALE",
    "UPLOAD_PAGE_MAX",
    "VAE_ENCODER_SUBFOLDER",
    "Flux2KleinVaeEncode",
    "build_flux2_klein_vae_encoder",
    "load_flux2_klein_vae_encoder_weights",
    "sharded_upload_view",
]
