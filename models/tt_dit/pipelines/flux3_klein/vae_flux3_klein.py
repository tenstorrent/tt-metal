# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein VAE decode on device: the DiT's fp32 image state -> the ``[1, H, W, 3]`` bf16 render, through
``models/vae/vae_flux2.py::Flux2VaeDecoder`` UNCHANGED (design record ``docs/M6_M7_DESIGN.md`` sections 2-3).

Facts the module rests on (measured 2026-10-17 on the Galaxy 4x8 ring, ``logs/m67_designer_vae_decode_4x8.log``):

* The Klein autoencoder is the FLUX.2 KL autoencoder: ``snap:vae/config.json`` is ``AutoencoderKLFlux2`` with
  ``latent_channels 32``, ``patch_size [2, 2]``, ``block_out_channels [128, 256, 512, 512]``, ``layers_per_block 2``,
  ``batch_norm_eps 1e-4`` = ``Flux2VaeDecoder``'s defaults. ``snap:ae.safetensors`` (BFL names, read by the vendored
  reference) equals ``snap:vae/diffusion_pytorch_model.safetensors`` (diffusers names) tensor by tensor for all 250
  float tensors; only the int64 ``bn.num_batches_tracked`` counter differs. Base and distilled snapshots link the same
  blobs, so the device weight cache is keyed by the ``vae/`` blob, not the snapshot (:func:`vae_blob`).
* Parallel config on the (4, 8) ring parent: channels TP4 over mesh axis 0, latent rows H8 over axis 1 (Flux2's own
  4x8 pipeline row). TP8 over channels is impossible: ``DistributedGroupNorm``'s fused path needs per-device
  ``C % 32 == 0`` and the 128-channel levels give 16. H over the DiT's SP axis alone (no TP) is 2.7x slower
  (replay 0.25-0.27 s vs 0.087 s) and was rejected.
* Golden ``final_tokens`` through this path vs the reference ``AutoEncoder.decode`` (fp32, CPU): pcc 0.999952 /
  0.999954, rel_rmse 0.0113 / 0.0110, PSNR 51.3 / 52.0 dB (base / distilled); traced == eager bit for bit; trace
  286,720 B per bank (2.3 MB); DRAM 28.8 MB per bank after capture; warm eager 0.095-0.108 s, replay 0.087 s.

Token layout -> decoder input (:meth:`Flux3KleinVaeDecode.decode`): the denoiser's state slot is fp32
``[1, L_img_pad, 128]`` ``mesh_axes=[None, sp_axis, None]`` in reference ``latent_ids(h, w)`` order (row-major over
``(row, col)``), ``L_img_pad = padded_length(h * w, SP)`` with the alignment rows exactly zero. The slot is typecast
to bf16 on device (the Flux2 pipeline feeds bf16 latents; measured equivalent to inverse-normalizing in fp32 first,
pcc vs the reference 0.999952 vs 0.999955), all-gathered over the SP axis with the DiT's CCL manager, the alignment
rows are sliced off (here and nowhere else), the real rows are re-partitioned over the VAE's H axis (contiguous
token chunks are whole latent rows because the tokens are row-major, hence ``h % h_factor == 0``), and
``preprocess_and_unpatchify`` applies the BatchNorm inverse and the 2x2 unpatchify (channel ``c * 4 + py * 2 + px``
on both sides: diffusers decode through this unpack equals the reference decode to 4.4e-6). The typecast precedes
the gather only to halve the gathered bytes; it is elementwise, so the order is numerically irrelevant.
Sizes with ``h % h_factor != 0`` (e.g. a 1344-px height, h 84, with H8) take a second, lazily built decoder with no
H partition (channels TP only, replicated rows); it is not exercised by the M7 rows.

The host finishes with ``tensor.to_torch(out)[0].permute(2, 0, 1).float()`` and the reference ``to_pil`` formula
``((x.clamp(-1, 1) + 1) * 127.5).to(uint8)`` (truncation toward zero), so the PSNR gate compares like with like.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from loguru import logger

import ttnn

from ...models.vae.vae_flux2 import Flux2VaeDecoder
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from .packing import FLUX3_KLEIN_IN_CHANNELS, FLUX3_KLEIN_PATCH, FLUX3_KLEIN_VAE_CHANNELS, latent_shape

VAE_SUBFOLDER = "vae"
VAE_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"
# autoencoder.py: 8x spatial compression per side before the 2x2 patchify.
FLUX3_KLEIN_VAE_SCALE = 8


def vae_blob(snapshot: str | os.PathLike) -> str:
    """The first 16 hex digits of the ``vae/`` weight blob (the HuggingFace cache symlink target); the cache key."""
    path = Path(snapshot) / VAE_SUBFOLDER / VAE_WEIGHTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: the snapshot has no diffusers vae/ partition")
    return Path(os.path.realpath(path)).name[:16]


def vae_model_name(snapshot: str | os.PathLike) -> str:
    return f"flux3_klein_vae_{vae_blob(snapshot)}"


def load_vae_torch_state(snapshot: str | os.PathLike) -> dict:
    """The diffusers ``AutoencoderKLFlux2`` state of ``snapshot/vae`` (the cache MISS path; ``Flux2VaeDecoder``'s
    ``_prepare_torch_state`` drops the encoder half and renames ``bn.*``)."""
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_pretrained(str(snapshot), subfolder=VAE_SUBFOLDER).state_dict()


def build_flux3_klein_vae_decoder(
    mesh_device: ttnn.MeshDevice,
    *,
    tp_axis: int | None,
    h_axis: int | None,
    ccl_manager: CCLManager | None,
) -> tuple[Flux2VaeDecoder, Flux2VaeParallelConfig]:
    """``Flux2VaeDecoder`` with its defaults (the Klein ``vae/config.json``) on ``mesh_device``."""
    parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=None)
    decoder = Flux2VaeDecoder(
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        z_channels=FLUX3_KLEIN_VAE_CHANNELS,
        parallel_config=parallel_config,
        device=mesh_device,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    return decoder, parallel_config


def load_flux3_klein_vae_weights(
    decoder: Flux2VaeDecoder,
    snapshot: str | os.PathLike,
    *,
    parallel_config: Flux2VaeParallelConfig,
    mesh_device: ttnn.MeshDevice,
) -> str:
    """Upload the decoder half of ``snapshot/vae`` through ``cache.load_model`` (blob-keyed). Returns ``"HIT"`` /
    ``"MISS"`` / ``"NO CACHE DIR"`` for the log line."""
    model_name = vae_model_name(snapshot)
    cache_dir = (
        cache.model_cache_dir(
            model_name=model_name,
            subfolder=VAE_SUBFOLDER,
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
        decoder,
        model_name=model_name,
        subfolder=VAE_SUBFOLDER,
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        get_torch_state_dict=lambda: load_vae_torch_state(snapshot),
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"RECORD vae weights: cache {state} ({cache_dir}) loaded in {time.time() - t0:.1f}s")
    return state


def release_module_trace(module: object | None) -> None:
    """Release the trace captured on ``module.forward``, if any (the Flux2 pipeline's helper)."""
    forward = getattr(type(module), "forward", None)
    tracers = getattr(forward, "_tracers", None)
    if tracers is None:
        return
    tracer = tracers.get(module)
    if tracer is not None:
        tracer.release_trace()


class Flux3KleinVaeDecode:
    """The denoiser state -> image stage (module docstring). ``decoder`` is H-partitioned over ``h_axis`` (``None``:
    replicated rows); ``sp_axis`` is the DiT's sequence-parallel axis the state arrives sharded on (``None`` on a
    single device); ``dit_ccl`` does the SP gather. ``snapshot`` is needed only to build the no-H fallback decoder."""

    def __init__(
        self,
        decoder: Flux2VaeDecoder,
        *,
        device: ttnn.MeshDevice,
        dit_ccl: CCLManager | None,
        sp_axis: int | None,
        h_axis: int | None,
        tp_axis: int | None = None,
        vae_ccl: CCLManager | None = None,
        snapshot: str | os.PathLike | None = None,
    ) -> None:
        self._decoder = decoder
        self._device = device
        self._dit_ccl = dit_ccl
        self._sp_axis = sp_axis
        self._h_axis = h_axis
        self._h_factor = int(device.shape[h_axis]) if h_axis is not None else 1
        self._tp_axis = tp_axis
        self._vae_ccl = vae_ccl
        self._snapshot = snapshot
        self._fallback: Flux2VaeDecoder | None = None
        self._traced_size: dict[int, tuple[int, int]] = {}
        if sp_axis is not None and dit_ccl is None:
            raise ValueError("an SP-sharded state needs the DiT's CCL manager for the gather")

    @property
    def decoder(self) -> Flux2VaeDecoder:
        return self._decoder

    @property
    def h_factor(self) -> int:
        return self._h_factor

    def partitioned(self, height: int, width: int) -> bool:
        """True when ``height x width`` decodes on the H-partitioned decoder (``h % h_factor == 0``)."""
        h, _ = latent_shape(height, width)
        return h % self._h_factor == 0

    def _fallback_decoder(self) -> Flux2VaeDecoder:
        if self._fallback is None:
            if self._snapshot is None:
                raise ValueError("a size with h % h_factor != 0 needs the snapshot to build the replicated decoder")
            logger.info("building the replicated-rows VAE decoder (h % h_factor != 0)")
            decoder, pc = build_flux3_klein_vae_decoder(
                self._device, tp_axis=self._tp_axis, h_axis=None, ccl_manager=self._vae_ccl
            )
            load_flux3_klein_vae_weights(decoder, self._snapshot, parallel_config=pc, mesh_device=self._device)
            self._fallback = decoder
        return self._fallback

    def gather_real_rows(self, state: ttnn.Tensor, *, num_image_rows: int) -> ttnn.Tensor:
        """bf16 ``[1, L_img, 128]`` replicated, ROW_MAJOR: typecast, SP gather, alignment rows sliced off."""
        z = ttnn.typecast(state, ttnn.bfloat16)
        if self._sp_axis is not None and int(self._device.shape[self._sp_axis]) > 1:
            gathered = self._dit_ccl.all_gather(z, dim=1, mesh_axis=self._sp_axis, use_hyperparams=True)
            ttnn.deallocate(z)
            z = gathered
        rows = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(z)
        if rows.shape[1] != num_image_rows:
            if rows.shape[1] < num_image_rows:
                raise ValueError(f"state holds {rows.shape[1]} rows, fewer than the {num_image_rows} real rows")
            sliced = ttnn.slice(rows, [0, 0, 0], [1, num_image_rows, FLUX3_KLEIN_IN_CHANNELS["image"]])
            ttnn.deallocate(rows)
            rows = sliced
        return rows

    def decode(self, state: ttnn.Tensor, *, num_image_rows: int, height: int, width: int, traced: bool) -> ttnn.Tensor:
        """``state`` fp32 ``[1, L_img_pad, 128]`` (``mesh_axes=[None, sp_axis, None]``) -> ``[1, H, W, 3]`` bf16
        replicated. ``traced=True`` captures the decoder trace on the first call of a size and replays it after; the
        returned tensor is then the trace's output buffer (read it, never deallocate it). In eager mode the caller
        owns the output."""
        h, w = latent_shape(height, width)
        if h * w != num_image_rows:
            raise ValueError(f"{height}x{width} has {h * w} latent rows, got num_image_rows {num_image_rows}")
        partitioned = self.partitioned(height, width)
        decoder = self._decoder if partitioned else self._fallback_decoder()
        rows = self.gather_real_rows(state, num_image_rows=num_image_rows)
        if partitioned and self._h_axis is not None and self._h_factor > 1:
            parted = ttnn.mesh_partition(rows, dim=1, cluster_axis=self._h_axis, memory_config=rows.memory_config())
            ttnn.deallocate(rows)
            rows = parted
        z = decoder.preprocess_and_unpatchify(
            rows, height=height // FLUX3_KLEIN_VAE_SCALE, width=width // FLUX3_KLEIN_VAE_SCALE
        )
        ttnn.deallocate(rows)
        if traced:
            key = id(decoder)
            if self._traced_size.get(key) not in (None, (height, width)):
                release_module_trace(decoder)  # one trace per decoder, keyed by size (Flux2 precedent)
            self._traced_size[key] = (height, width)
        replay = traced and _trace_captured(decoder)
        out = decoder.forward(z, traced=traced)
        if not traced or replay:
            # Eager: ``z`` is a plain intermediate. Traced REPLAY: the tracer copied ``z`` into the captured input
            # buffer (``utils/tracing.py`` ``_update_input``: ``ttnn.copy`` when the addresses differ), so this fresh
            # ``z`` would otherwise leak one ``[1, H/8, W/8, 32]`` bf16 buffer per generate (131 KB per device at
            # 1024x1024; found by the M6/M7 verifier). On the CAPTURE call ``z`` IS the trace's input buffer and
            # must stay alive for as long as the trace does. The deallocate is enqueued after the copy on the same
            # command queue, so the order is preserved.
            ttnn.deallocate(z)
        return out

    def release_trace(self) -> None:
        """Release the captured decoder trace(s); the weights stay."""
        for decoder in (self._decoder, self._fallback):
            if decoder is not None:
                release_module_trace(decoder)
        self._traced_size.clear()

    def trace_captured(self) -> bool:
        return _trace_captured(self._decoder)


def _trace_captured(decoder: Flux2VaeDecoder) -> bool:
    """Whether ``decoder.forward`` holds a captured trace (the next ``traced=True`` call is a replay)."""
    tracers = getattr(type(decoder).forward, "_tracers", None)
    tracer = None if tracers is None else tracers.get(decoder)
    return tracer is not None and tracer.trace_captured


__all__ = [
    "FLUX3_KLEIN_PATCH",
    "FLUX3_KLEIN_VAE_SCALE",
    "Flux3KleinVaeDecode",
    "build_flux3_klein_vae_decoder",
    "load_flux3_klein_vae_weights",
    "load_vae_torch_state",
    "release_module_trace",
    "vae_blob",
    "vae_model_name",
]
