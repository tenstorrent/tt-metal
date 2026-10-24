# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein VAE decode on device: the DiT's fp32 image state -> the ``[1, H, W, 3]`` bf16 render, through
``models/vae/vae_flux2.py::Flux2VaeDecoder`` UNCHANGED. The reference is ``Flux2KleinPipeline.__call__`` after the
denoising loop (``pipeline_flux2_klein.py:905-922``, cited as ``P``): ``_unpack_latents_with_ids`` -> inverse
BatchNorm ``z * sqrt(var + eps) + mean`` -> ``_unpatchify_latents`` -> ``AutoencoderKLFlux2.decode`` (BFL
``autoencoder.py:308-336`` ``inv_normalize`` + ``decode`` is the same arithmetic).

Facts the module rests on:

* The Klein autoencoder is the FLUX.2 KL autoencoder: ``snapshot/vae/config.json`` is ``AutoencoderKLFlux2`` with
  ``_name_or_path black-forest-labs/FLUX.2-dev``, ``latent_channels 32``, ``patch_size [2, 2]``, ``block_out_channels
  [128, 256, 512, 512]``, ``layers_per_block 2``, ``batch_norm_eps 1e-4`` = ``Flux2VaeDecoder``'s defaults. Base and
  distilled snapshots of a size link the same ``vae/`` blob, so the device weight cache is keyed by that blob, not
  the snapshot (:func:`vae_blob`).
* Parallel config on the (4, 8) ring parent: channels TP4 over mesh axis 0, latent rows H8 over axis 1 (Flux2's own
  4x8 pipeline row). TP8 over channels is impossible: ``DistributedGroupNorm``'s fused path needs per-device
  ``C % 32 == 0`` and the 128-channel levels give 16.

Token layout -> decoder input (:meth:`Flux2KleinVaeDecode.decode`): the denoiser's state slot is fp32
``[1, L_pad, 128]`` ``mesh_axes=[None, sp_axis, None]`` holding ``[image rows in latent_ids order | reference rows |
zero pad]``; only the first ``num_image_rows`` are decoded (``P:862`` slices the prediction to the latents, ``P:908``
unpacks ``latents`` alone). The slot holds bf16-valued numbers (the reference state is bf16, ``scheduling_flow_match_euler_discrete.py:526``), so the
typecast to bf16 on device is exact; the rows are all-gathered over the SP axis with the DiT's CCL manager, the
reference and alignment rows are sliced off (here and nowhere else), the real rows are re-partitioned over the VAE's
H axis (contiguous token chunks are whole latent rows because the tokens are row-major, hence ``h % h_factor == 0``),
and ``preprocess_and_unpatchify`` applies the BatchNorm inverse and the 2x2 unpatchify (channel ``c * 4 + py * 2 +
px`` on both sides, ``P:371-385``). Sizes with ``h % h_factor != 0`` (e.g. a 1344-px height, h 84, with H8) take a
second, lazily built decoder with no H partition (channels TP only, replicated rows).

The host finish (``image_processor.postprocess``: ``(x * 0.5 + 0.5).clamp(0, 1)`` then ``(x * 255).round()`` to
uint8) is the pipeline's; ``references.image_to_uint8`` pins the arithmetic.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from loguru import logger
from safetensors.torch import load_file

import ttnn

from ...layers.conv2d import Conv2d
from ...layers.normalization import plain_group_norm_grid
from ...models.vae.vae import VaeAttention
from ...models.vae.vae_flux2 import Flux2VaeDecoder
from ...parallel.config import Flux2VaeParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from .packing import FLUX2_KLEIN_LATENT_CHANNELS, FLUX2_KLEIN_PATCH, FLUX2_KLEIN_VAE_CHANNELS, latent_shape

VAE_SUBFOLDER = "vae"
VAE_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"
# vae/config.json: 8x spatial compression per side before the 2x2 patchify.
FLUX2_KLEIN_VAE_SCALE = 8

# DRAM slice counts of ``layers/conv2d.py::Conv2d`` for the Klein VAE's convolutions, per level of a full-resolution
# ``H x W`` image: level 0 = the image resolution (128 channels), level 1 = /2 (256), level 2 = /4 (512), level 3 = /8
# (512 and the 32 / 64 latent channels). Keyed ``(in_channels, out_channels) -> num_slices``; the counts are the
# shipped ``Conv2d.slice_default`` entries of the 1024x1024 levels (``(1024, 1024, 128, 128): 16`` ...) carried to the
# channel pairs the ENCODER adds (``3 -> 128``, ``128 -> 256``, ``256 -> 512``, ``512 -> 64``: 16 / 8 / 8 / 4 by the
# same bytes-per-slice rule) and to the non-1024 sizes the pipeline decodes / encodes on the replicated TP4 configs,
# whose ``(h, w)`` keys the shipped table does not list. ``register_klein_conv_slices`` installs them under the
# process's mesh shape (a (4, 8) mesh has no table of its own and falls back to ``slice_default``, which keeps every
# shipped entry: the registration is a superset, never a change, of the counts the 1024 decode already ran with).
#
# ttnn's limit (``op_slicing.cpp`` ``compute_max_num_slices`` + the TT_FATAL at the slice-config check): a DRAM
# WIDTH-sliced conv with a TILE output takes at most ``ceil(output_width / 32)`` slices. ``klein_conv_slices`` caps every
# level count at that bound for the level's OUTPUT width (= the input width for the stride-1 convs here); the stride-2
# ``VaeDownsample2d`` halves the width and caps its own instance entry (``encoder_flux2.py``), because the ``Conv2d``
# key ``(h, w, cin, cout)`` carries no stride and the shipped stride-1 count of the same key exceeds its bound
# (``TT_FATAL ... Number of slices (8) exceeds the maximum allowed (4)`` at the level-2 downsample of 1024x1024; the
# shipped 1024 decode entries all sit inside the bound, so the cap changes no count the 1024 decode ran with).
KLEIN_CONV_SLICES_PER_LEVEL = (
    # (256, 256) at level 0 and (512, 512) at level 1 are the two UPSAMPLER convs (``VaeUpsampler``: the 2x nearest
    # upsample lands at the next level and its conv keeps the channel count); they take the shipped 1024
    # ``slice_default`` counts ``(1024, 1024, 256, 256): 16`` / ``(512, 512, 512, 512): 16``.
    {(3, 128): 16, (128, 128): 16, (128, 3): 8, (256, 128): 16, (256, 256): 16},
    {(128, 256): 8, (256, 256): 4, (512, 256): 16, (512, 512): 16},
    {(256, 512): 8, (512, 512): 8},
    {(512, 512): 4, (512, 64): 4, (32, 512): 1},
)
# The DECODER's convolutions per level as ``(in_channels, out_channels, tensor_parallel)`` (``vae_flux2.py``:
# ``channel_counts = [512, 512, 512, 256, 128]``; level 3 = ``conv_in`` 32 -> 512 + the mid / up-block resnets, each
# upsampler conv runs at the level it upsamples INTO, ``conv_out`` 128 -> 3 has ``tensor_parallel=False``).
KLEIN_DECODER_CONVS_PER_LEVEL = (
    ((256, 256, True), (256, 128, True), (128, 128, True), (128, 3, False)),
    ((512, 512, True), (512, 256, True), (256, 256, True)),
    ((512, 512, True),),
    ((32, 512, True), (512, 512, True)),
)
# H-PARTITIONED decoder (``h_axis`` set): ``Conv2d`` keys its slice table by the PER-DEVICE input shape, which under
# an H split of factor F with the 3x3 pad-1 halo (``VaeConv2d``: ``neighbor_pad`` adds one row per side) is
# ``(H_level / F + 2, W_level, cin, cout)`` -- NOT the full-image key above. A lookup that misses runs ttnn's AUTO
# slicer (``op_slicing.cpp::determine_slice_config``: the smallest count whose modelled L1 use fits the L1 FREE AT
# THAT MOMENT; the model does not include the L1_SMALL config tensors), which is why large canvases decoded on a
# fresh mesh and failed after other allocations existed ("Not enough space to allocate ... L1_SMALL buffer", then
# "circular buffers clash with L1 buffers"). Canvases up to ``AUTO_SLICE_MAX_AREA`` keep the auto path (count 0 =
# ttnn auto); above it, in the non-auto modes, every decoder conv gets an EXPLICIT count from a bytes-per-slice
# budget: ``DECODER_SLICE_BUDGET_BYTES`` = 4 MiB of bf16 input per slice per device, the per-slice size of the shipped
# ``(1024, 1024, 128, 128): 16`` and ``(512, 512, 512, 512): 16`` entries on the (1, 4) TP4 mesh (1024 rows x 64
# columns x 32 channels x 2 B = 512 x 32 x 128 x 2 B = 4 MiB), capped by ``conv_slice_cap``.
AUTO_SLICE_MAX_AREA = 1536 * 1536
DECODER_SLICE_BUDGET_BYTES = 4 << 20
_registered_conv_slices: set[tuple] = set()


def conv_slice_cap(output_width: int) -> int:
    """ttnn's maximum DRAM width-slice count for a conv whose TILE output is ``output_width`` wide: ``ceil(w / 32)``
    (``op_slicing.cpp::compute_max_num_slices``, ``slice_rounding_value`` = TILE_HEIGHT for a TILE DRAM_WIDTH slice)."""
    return max(1, -(-int(output_width) // 32))


def downsample_extent(extent: int) -> int:
    """The spatial extent after the reference ``Downsample`` (``autoencoder.py:34-40``: ``conv(F.pad(x, (0, 1, 0, 1)))``,
    3x3 stride 2 pad 0): ``floor((H + 1 - 3) / 2) + 1 = floor(H / 2)`` (torch and the device agree: 35 -> 17, 17 -> 8).
    A 16-multiple side never produces an odd extent before a downsample."""
    return (int(extent) + 1 - 3) // 2 + 1


def klein_conv_slices(height: int, width: int) -> dict[tuple[int, int, int, int], int]:
    """``(h, w, in_channels, out_channels) -> num_slices`` for the four VAE levels of a ``height x width`` image, every
    stride-1 entry capped at ``conv_slice_cap(w_level)`` (the level's output width); the strided downsample's extents
    are ``downsample_extent`` (floor)."""
    out, h, w = {}, height, width
    for level in KLEIN_CONV_SLICES_PER_LEVEL:
        cap = conv_slice_cap(w)
        for (cin, cout), slices in level.items():
            out[(h, w, cin, cout)] = min(slices, cap)
        h, w = downsample_extent(h), downsample_extent(w)
    return out


def klein_downsample_slices(height: int, width: int) -> dict[tuple[int, int, int, int], int]:
    """The slice counts the encoder's three stride-2 downsamples (levels 0-2, ``channels -> channels``) take for a
    ``height x width`` image: the level's stride-1 count for the same ``(h, w, c, c)`` key capped at
    ``conv_slice_cap(downsample_extent(w))``, the downsample's OUTPUT width (the rule ``VaeDownsample2d`` applies per
    instance; here for the host test row)."""
    stride1 = klein_conv_slices(height, width)
    out, h, w = {}, height, width
    for channels in (128, 256, 512):
        key = (h, w, channels, channels)
        out[key] = min(stride1[key], conv_slice_cap(downsample_extent(w)))
        h, w = downsample_extent(h), downsample_extent(w)
    return out


def partitioned_conv_rows(level_height: int, h_factor: int, halo: int = 1) -> int:
    """Rows of a conv's per-device input at a level of ``level_height`` rows under an H split of ``h_factor``: the
    device's share plus the ``neighbor_pad`` halo (``halo`` rows per side); the full height when not split."""
    if h_factor <= 1:
        return int(level_height)
    return int(level_height) // int(h_factor) + 2 * halo


def decoder_slice_count(rows: int, width: int, cin: int, cout: int, tensor_parallel: bool, tp_factor: int) -> int:
    """Explicit DRAM width-slice count for one decoder conv on one device: the bf16 input bytes per slice stay
    under ``DECODER_SLICE_BUDGET_BYTES``. ``Conv2d`` shards the INPUT channels over ``tp_factor`` when the conv is
    tensor-parallel and ``cin >= cout`` (``VaeConv2d``: ``out_is_greater`` picks the output axis instead, then the
    input is replicated). Capped at ``conv_slice_cap(width)``, at least 1."""
    cin_device = cin // tp_factor if (tensor_parallel and tp_factor > 1 and cin >= cout) else cin
    cap = conv_slice_cap(width)
    row_bytes = int(rows) * int(cin_device) * 2
    count = max(1, -(-(row_bytes * int(width)) // DECODER_SLICE_BUDGET_BYTES))
    # ttnn spreads the output columns as evenly as it can (``op_slicing.cpp``: ``ceil(width / count)`` at most per
    # slice), so the widest slice decides; step up until it fits or the cap is reached.
    while count < cap and row_bytes * (-(-int(width) // count)) > DECODER_SLICE_BUDGET_BYTES:
        count += 1
    return aligned_slice_count(min(count, cap), width)


def aligned_slice_count(wanted: int, width: int) -> int:
    """The smallest slice count >= ``wanted`` that cuts ``width`` into TILE-aligned (32-column) slices: a divisor of
    ``width // 32``. Every shipped Flux2 entry has this property (16 / 32 / 64 / 128 slices of 1024 / 2048 / 4096);
    unaligned counts like 17 / 33 / 9 (slice widths of 120-121 px) render vertical striping and blown colours over
    the whole frame. A width that is not a multiple of 32 has no aligned cut, so the cap (ttnn's ``ceil(width / 32)``)
    is returned unchanged. ``0`` (auto) passes through."""
    wanted, width = int(wanted), int(width)
    if wanted <= 0 or width % 32:
        return wanted if wanted <= 0 else conv_slice_cap(width)
    tiles = width // 32
    for n in range(max(1, wanted), tiles + 1):
        if tiles % n == 0:
            return n
    return tiles


def _slice_mode() -> str:
    """``FLUX2_KLEIN_VAE_SLICE_MODE``: ``auto`` (default: ttnn's auto slicer at every size, with the conv config
    tensors in DRAM; the fastest decoder measured on the FLUX.2 autoencoder), ``aligned`` (explicit TILE-aligned
    counts above ``AUTO_SLICE_MAX_AREA``) or ``explicit`` (the unaligned budget counts, a regression probe only).
    The slice count changes the convs' rounding, so the choice is a PSNR-gate-class default."""
    mode = os.environ.get("FLUX2_KLEIN_VAE_SLICE_MODE", "auto").strip().lower()
    if mode not in ("aligned", "explicit", "auto"):
        raise ValueError(f"FLUX2_KLEIN_VAE_SLICE_MODE must be aligned, explicit or auto, got {mode!r}")
    return mode


def _config_tensors_in_dram() -> bool:
    """``FLUX2_KLEIN_VAE_CONFIG_TENSORS_DRAM`` (default 1): where the conv halo / sliding-window config tensors live."""
    return os.environ.get("FLUX2_KLEIN_VAE_CONFIG_TENSORS_DRAM", "1").strip() not in ("0", "false", "no")


def klein_partitioned_decoder_slices(
    height: int, width: int, *, h_factor: int, tp_factor: int
) -> dict[tuple[int, int, int, int], int]:
    """``(rows_device, w_level, cin, cout) -> num_slices`` for every decoder conv of a ``height x width`` image on
    the H-partitioned decoder (``h_factor > 1``; empty otherwise). ``0`` = ttnn's auto slicer (canvases up to
    ``AUTO_SLICE_MAX_AREA``, and every canvas in ``auto`` mode); larger canvases get ``decoder_slice_count``."""
    if h_factor <= 1:
        return {}
    mode = _slice_mode()
    explicit = mode != "auto" and int(height) * int(width) > AUTO_SLICE_MAX_AREA
    out, h, w = {}, int(height), int(width)
    for level in KLEIN_DECODER_CONVS_PER_LEVEL:
        rows = partitioned_conv_rows(h, h_factor)
        for cin, cout, tensor_parallel in level:
            count = decoder_slice_count(rows, w, cin, cout, tensor_parallel, tp_factor) if explicit else 0
            if explicit and mode == "explicit":  # the unaligned regression variant
                count = _unaligned_slice_count(rows, w, cin, cout, tensor_parallel, tp_factor)
            out[(rows, w, cin, cout)] = count
        h, w = downsample_extent(h), downsample_extent(w)
    return out


def _unaligned_slice_count(rows: int, width: int, cin: int, cout: int, tensor_parallel: bool, tp_factor: int) -> int:
    """The budget count WITHOUT the tile alignment (probe variant ``explicit``)."""
    cin_device = cin // tp_factor if (tensor_parallel and tp_factor > 1 and cin >= cout) else cin
    cap = conv_slice_cap(width)
    row_bytes = int(rows) * int(cin_device) * 2
    count = max(1, -(-(row_bytes * int(width)) // DECODER_SLICE_BUDGET_BYTES))
    while count < cap and row_bytes * (-(-int(width) // count)) > DECODER_SLICE_BUDGET_BYTES:
        count += 1
    return min(count, cap)


def register_klein_conv_slices(
    mesh_shape: tuple[int, int], height: int, width: int, *, h_factor: int = 1, tp_factor: int = 1
) -> None:
    """Make ``Conv2d.slice_params[mesh_shape]`` know the Klein VAE shapes of a ``height x width`` image: the
    full-image keys (encoder, replicated decoder) and, for ``h_factor > 1``, the per-device keys of the
    H-partitioned decoder (``klein_partitioned_decoder_slices``). Idempotent; existing entries are never
    overwritten, so the shipped counts win where both exist."""
    key = (tuple(mesh_shape), int(height), int(width), int(h_factor), int(tp_factor))
    if key in _registered_conv_slices:
        return
    table = Conv2d.slice_params.setdefault(tuple(mesh_shape), dict(Conv2d.slice_default))
    for shape, slices in klein_conv_slices(height, width).items():
        table.setdefault(shape, slices)
    for shape, slices in klein_partitioned_decoder_slices(
        height, width, h_factor=h_factor, tp_factor=tp_factor
    ).items():
        table.setdefault(shape, slices)
    _registered_conv_slices.add(key)


def set_conv_config_overrides(module, **overrides) -> int:
    """Merge ``overrides`` into ``Conv2d.conv_config_overrides`` of every conv under ``module`` (per instance; the
    class is untouched). Returns the number of convs changed. The Klein VAE sets ``config_tensors_in_dram=True``: the
    halo / sliding-window config tensors otherwise live in the 64 KB L1_SMALL region, which large-canvas decoder
    convs exhaust; Z-Image's VAE and the Qwen3 audio encoder run the same setting."""
    changed = 0
    for child in _walk_modules(module):
        if hasattr(child, "conv_config_overrides"):
            child.conv_config_overrides.update(overrides)
            changed += 1
    return changed


def _walk_modules(module):
    yield module
    for _, child in module.named_children():
        yield from _walk_modules(child)


def set_sdpa_chunk_size(module, q_chunk_size: int, k_chunk_size: int) -> int:
    """Rebuild the SDPA program config of every ``VaeAttention`` under ``module`` with the given chunk sizes (the
    class-level ``sdpa_chunk_size_map`` is NOT touched: this is per instance). Chunking changes the reduction
    blocking only (fp32 accumulation, ``exp_approx_mode=False`` kept). Returns the number of attentions changed."""
    changed = 0
    for child in _walk_modules(module):
        if isinstance(child, VaeAttention):
            old = child._sdpa_program_config
            child._sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=old.compute_with_storage_grid_size,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                exp_approx_mode=old.exp_approx_mode,
            )
            changed += 1
    return changed


def vae_checkpoint(snapshot: str | os.PathLike) -> Path:
    """``snapshot/vae/diffusion_pytorch_model.safetensors`` (the diffusers partition; the BFL-layout duplicate in the
    snapshot root is not read)."""
    path = Path(snapshot) / VAE_SUBFOLDER / VAE_WEIGHTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path}: the snapshot has no diffusers vae/ partition")
    return path


def vae_blob(snapshot: str | os.PathLike) -> str:
    """The first 16 hex digits of the ``vae/`` weight blob (the HuggingFace cache symlink target); the cache key."""
    return Path(os.path.realpath(vae_checkpoint(snapshot))).name[:16]


def vae_model_name(snapshot: str | os.PathLike) -> str:
    return f"flux2_klein_vae_{vae_blob(snapshot)}"


def load_vae_torch_state(snapshot: str | os.PathLike) -> dict:
    """The diffusers ``AutoencoderKLFlux2`` state of ``snapshot/vae`` as the safetensors file holds it (its keys ARE
    the module's ``state_dict`` keys; the cache MISS path). ``Flux2VaeDecoder._prepare_torch_state`` drops the
    encoder half and renames ``bn.*``; ``Flux2VaeEncoder`` does the converse."""
    return load_file(str(vae_checkpoint(snapshot)))


def build_flux2_klein_vae_decoder(
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
        z_channels=FLUX2_KLEIN_VAE_CHANNELS,
        parallel_config=parallel_config,
        device=mesh_device,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    set_conv_config_overrides(decoder, config_tensors_in_dram=_config_tensors_in_dram())
    return decoder, parallel_config


def load_flux2_klein_vae_weights(
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


class Flux2KleinVaeDecode:
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
            decoder, pc = build_flux2_klein_vae_decoder(
                self._device, tp_axis=self._tp_axis, h_axis=None, ccl_manager=self._vae_ccl
            )
            load_flux2_klein_vae_weights(decoder, self._snapshot, parallel_config=pc, mesh_device=self._device)
            self._fallback = decoder
        return self._fallback

    def gather_real_rows(self, state: ttnn.Tensor, *, num_image_rows: int) -> ttnn.Tensor:
        """bf16 ``[1, num_image_rows, 128]`` replicated, ROW_MAJOR: typecast, SP gather, the reference and
        alignment rows sliced off (``P:862`` / ``P:908``: only the latents are decoded)."""
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
            sliced = ttnn.slice(rows, [0, 0, 0], [1, num_image_rows, FLUX2_KLEIN_LATENT_CHANNELS])
            ttnn.deallocate(rows)
            rows = sliced
        return rows

    def decode(self, state: ttnn.Tensor, *, num_image_rows: int, height: int, width: int, traced: bool) -> ttnn.Tensor:
        """``state`` fp32 ``[1, L_pad, 128]`` (``mesh_axes=[None, sp_axis, None]``) -> ``[1, H, W, 3]`` bf16
        replicated. ``traced=True`` captures the decoder trace on the first call of a size and replays it after; the
        returned tensor is then the trace's output buffer (read it, never deallocate it). In eager mode the caller
        owns the output."""
        h, w = latent_shape(height, width)
        if h * w != num_image_rows:
            raise ValueError(f"{height}x{width} has {h * w} latent rows, got num_image_rows {num_image_rows}")
        partitioned = self.partitioned(height, width)
        tp_factor = int(self._device.shape[self._tp_axis]) if self._tp_axis is not None else 1
        register_klein_conv_slices(
            tuple(self._device.shape),
            height,
            width,
            h_factor=self._h_factor if partitioned else 1,
            tp_factor=tp_factor,
        )
        decoder = self._decoder if partitioned else self._fallback_decoder()
        rows = self.gather_real_rows(state, num_image_rows=num_image_rows)
        if partitioned and self._h_axis is not None and self._h_factor > 1:
            parted = ttnn.mesh_partition(rows, dim=1, cluster_axis=self._h_axis, memory_config=rows.memory_config())
            ttnn.deallocate(rows)
            rows = parted
        z = decoder.preprocess_and_unpatchify(
            rows, height=height // FLUX2_KLEIN_VAE_SCALE, width=width // FLUX2_KLEIN_VAE_SCALE
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
            # ``z`` would otherwise leak one ``[1, H/8, W/8, 32]`` bf16 buffer per generate. On the CAPTURE call
            # ``z`` IS the trace's input buffer and must stay alive for as long as the trace does. The deallocate is
            # enqueued after the copy on the same command queue, so the order is preserved.
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

    def trace_captured_for(self, height: int, width: int) -> bool:
        """True when the decoder that serves ``height x width`` holds a trace captured AT that size (the next traced
        decode of the size is a replay: nothing long-lived is allocated by it)."""
        partitioned = self.partitioned(height, width)
        decoder = self._decoder if partitioned else self._fallback
        if decoder is None:
            return False
        return _trace_captured(decoder) and self._traced_size.get(id(decoder)) == (height, width)


def _trace_captured(decoder: Flux2VaeDecoder) -> bool:
    """Whether ``decoder.forward`` holds a captured trace (the next ``traced=True`` call is a replay)."""
    tracers = getattr(type(decoder).forward, "_tracers", None)
    tracer = None if tracers is None else tracers.get(decoder)
    return tracer is not None and tracer.trace_captured


__all__ = [
    "FLUX2_KLEIN_PATCH",
    "FLUX2_KLEIN_VAE_SCALE",
    "KLEIN_CONV_SLICES_PER_LEVEL",
    "VAE_SUBFOLDER",
    "Flux2KleinVaeDecode",
    "build_flux2_klein_vae_decoder",
    "conv_slice_cap",
    "downsample_extent",
    "klein_conv_slices",
    "klein_downsample_slices",
    "load_flux2_klein_vae_weights",
    "load_vae_torch_state",
    "plain_group_norm_grid",
    "register_klein_conv_slices",
    "release_module_trace",
    "set_conv_config_overrides",
    "set_sdpa_chunk_size",
    "vae_blob",
    "vae_checkpoint",
    "vae_model_name",
]
