# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Top-level MiniMax-H3 visual VAE: spatial tiling, temporal chunking, encode.

MiniMax-H3 ships tiled (256 px tiles, overlap 64). Both encode and decode split the
frame into 256x256 **pixel** tiles, run the model independently per tile, and
linearly cross-fade the overlaps. That is not an optional memory optimisation --
reproducing the released model requires it -- and it is what bounds every work unit:

* the encoder always sees ``(1, 3, 17, 256, 256)`` and emits ``(1, 48, 5, 16, 16)``;
* the decoder always sees ``(1, 24, 7, 16, 16)``.

Because the tiles are independent, the mesh is used **data-parallel over (tile, chunk)
work units with the weights replicated**, rather than by sharding one tile's H/W. Two
consequences simplify this port considerably: reflect padding stays a local
slice-and-concat (``neighbor_pad_async`` has no reflect mode), and GroupNorm statistics
never need a cross-device reduction.

The tiling and blending stay on host. They are cheap, exactly specified, and porting
them to device buys nothing until a measurement says otherwise.

The encoder lives in ``encoder_minimax_h3.py`` on a ``LTXCausalConv3d``-shaped conv,
parameterised by ``temporal_taps`` so the keyframe path and the clip path are one
implementation. ``Conv2dViaConv3d`` cannot serve as ``conv_in``: it sizes the weight
from the *aligned* input-channel count but prepares an unpadded weight, so a 3-channel
conv fails its ``Parameter`` shape check.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import ttnn

from ....utils.tensor import fast_device_to_host, float_to_uint8, local_device_to_torch
from ....utils.yuv_d2h import fast_device_to_host_yuv
from .encoder_minimax_h3 import MiniMaxH3Encoder3d

DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_OVERLAP = 64


class MiniMaxH3VaeConfig:
    """The subset of ``vae/config.json`` this port needs, plus the derived geometry.

    ``clip_length`` (17) is not a multiple of ``temporal_compression_ratio`` (4), which
    is why the four derived fields exist rather than plain constants: the decoder has to
    re-derive the implicit leading pad and the overlap that ``token_drop`` leaves behind.
    """

    def __init__(self, **cfg) -> None:
        self.in_channels = cfg.get("in_channels", 3)
        self.out_channels = cfg.get("out_channels", 3)
        self.latent_channels = cfg.get("latent_channels", 24)
        self.block_out_channels = tuple(cfg.get("block_out_channels", (128, 256, 256, 512, 512, 1024)))
        self.layers_per_block = cfg.get("layers_per_block", 2)
        self.spatial_downsample_factors = tuple(cfg.get("spatial_downsample_factors", (2, 2, 2, 2, 1, 1)))
        self.temporal_downsample_factors = tuple(cfg.get("temporal_downsample_factors", (1, 2, 2, 1, 1, 1)))
        self.norm_num_groups = cfg.get("norm_num_groups", 32)
        self.norm_eps = cfg.get("norm_eps", 1e-6)
        self.decoder_num_layers = cfg.get("decoder_num_layers", 36)
        self.decoder_num_attention_heads = cfg.get("decoder_num_attention_heads", 32)
        self.decoder_attention_head_dim = cfg.get("decoder_attention_head_dim", 64)
        self.decoder_num_register_tokens = cfg.get("decoder_num_register_tokens", 4)
        self.decoder_ffn_mult = cfg.get("decoder_ffn_mult", 4)
        self.decoder_rope_theta = cfg.get("decoder_rope_theta", 100.0)
        self.decoder_rope_dim_ratio = cfg.get("decoder_rope_dim_ratio", 0.75)
        self.decoder_norm_eps = cfg.get("decoder_norm_eps", 1e-5)
        self.clip_length = cfg.get("clip_length", 17)
        self.token_drop = cfg.get("token_drop", 3)
        self.latents_mean = tuple(cfg.get("latents_mean", ()))
        self.latents_std = tuple(cfg.get("latents_std", ()))

        self.spatial_compression_ratio = math.prod(self.spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(self.temporal_downsample_factors)

        self.frame_pre_padding = (-self.clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(self.clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike) -> "MiniMaxH3VaeConfig":
        cfg = json.loads((Path(path) / "config.json").read_text())
        return cls(**{k: v for k, v in cfg.items() if not k.startswith("_")})


# Decode does substantial host work (stitching, unpatchify, readback), so it raises torch's thread
# count above the single thread a server worker pins for the denoise loop.
_DECODE_TORCH_THREADS = 8


def split_tiles(length: int, tile_size: int, min_overlap: int, ratio: int) -> tuple[list[int], list[int], list[int]]:
    """Lay ``tile_size``-wide tiles over ``length``, latent-aligned (reference ``_split_tiles``).

    The tile count is the smallest whose union covers ``length`` with every overlap at
    least ``min_overlap``; the slack is spread round-robin over the overlaps in whole
    ``ratio`` steps so each boundary lands on a latent grid point. Note every tile has
    length exactly ``tile_size`` unless one tile already covers the axis -- which is why
    the encoder only ever needs one activation shape.
    """
    if tile_size >= length:
        return [0], [length], []

    num_tiles = math.ceil(length / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1

    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (num_tiles - 1)] += ratio

    starts = [0]
    for i in range(num_tiles - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * num_tiles, overlaps


def blend(a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
    """Linear cross-fade of ``a``'s tail into ``b``'s head along ``dim``."""
    blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
    positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = blend_extent
    weight_a = (1 - positions / blend_extent).view(shape)
    weight_b = (positions / blend_extent).view(shape)

    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-blend_extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, blend_extent)
    blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

    if blend_extent == b.shape[dim]:
        return blended
    slice_rest = [slice(None)] * b.ndim
    slice_rest[dim] = slice(blend_extent, None)
    return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)


OUTPUT_TYPES = ("float", "yuv420")
"""A decoded clip is either a torch ``(1, 3, T, H, W)`` tensor or, on the YUV path, a planar
``(T, H*3//2, W)`` uint8 array. Only the temporal assembly in :meth:`MiniMaxH3Vae._decode` touches
both, and only along the frame axis -- these three keep that one difference in one place."""


def clip_num_frames(clip) -> int:
    return clip.shape[0] if isinstance(clip, np.ndarray) else clip.shape[2]


def clip_frames(clip, start: int, stop: int):
    return clip[start:stop] if isinstance(clip, np.ndarray) else clip[:, :, start:stop]


def concat_clip_frames(parts: list):
    if parts and isinstance(parts[0], np.ndarray):
        return np.concatenate(parts, axis=0)
    return torch.cat(parts, dim=2)


def blend_clip_frames(a, b, extent: int):
    """Cross-fade ``a``'s trailing frames into ``b``'s leading ones -- :func:`blend` along frames.

    The planar branch blends already-quantized uint8. BT.601 is affine and 4:2:0 decimation is
    linear, so a convex combination commutes with the conversion and the only cost is re-rounding:
    bounded at 1 LSB, which ``test_temporal_crossfade_survives_the_yuv_conversion`` pins.
    """
    if not isinstance(a, np.ndarray):
        return blend(a, b, extent, dim=-3)

    extent = min(a.shape[0], b.shape[0], extent)
    positions = np.arange(extent, dtype=np.float32).reshape(-1, *([1] * (a.ndim - 1)))
    head = a[-extent:].astype(np.float32) * (1.0 - positions / extent) + b[:extent].astype(np.float32) * (
        positions / extent
    )
    head = np.clip(np.rint(head), 0, 255).astype(np.uint8)
    return head if extent == b.shape[0] else np.concatenate([head, b[extent:]], axis=0)


def stitch_tiles(
    tiles: list[list[torch.Tensor]], height_overlaps: list[int], width_overlaps: list[int]
) -> torch.Tensor:
    """Blend a 2D grid of tiles back into one tensor (reference ``_stitch_tiles``)."""
    result_rows = []
    for i, row in enumerate(tiles):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
            if j > 0:
                tile = blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
            if i < len(tiles) - 1:
                tile = tile[..., : -height_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -width_overlaps[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def prepare_encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Encoder ``state_dict`` with ``quant_conv`` folded into ``conv_out``.

    The two are adjacent with no nonlinearity between them, so one 1024->48 k3 conv
    does both and the awkward 48-channel 1x1x1 conv disappears entirely.
    """
    encoder_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
    if not encoder_state:
        encoder_state = dict(state)
    quant_weight = state.get("quant_conv.weight")
    quant_bias = state.get("quant_conv.bias")
    encoder_state.pop("quant_conv.weight", None)
    encoder_state.pop("quant_conv.bias", None)
    if quant_weight is not None and "conv_out.weight" in encoder_state:
        # (48,48,1,1,1) composed with (48,1024,3,3,3) -> (48,1024,3,3,3)
        quant_2d = quant_weight.reshape(quant_weight.shape[0], quant_weight.shape[1])
        encoder_state["conv_out.weight"] = torch.einsum("oi,ijkmn->ojkmn", quant_2d, encoder_state["conv_out.weight"])
        encoder_state["conv_out.bias"] = quant_2d @ encoder_state["conv_out.bias"] + quant_bias
    return encoder_state


def _fold_pixel_denorm(
    decoder_state: dict[str, torch.Tensor],
    pixel_denorm: tuple[Sequence[float], Sequence[float]],
    out_channels: int,
) -> None:
    """Fold the ImageNet de-normalization into ``proj_out``, landing pixels in ``[-1, 1]``.

    ``proj_out`` emits a patch's pixels flattened as ``(C, pt, p, p)`` with ``C`` outermost --
    the order :func:`decoder_minimax_h3.unpatchify` reshapes to -- so a per-channel affine is a
    row scale plus a bias shift, the same move :func:`prepare_decoder_state` already makes for
    ``post_quant_conv``. The reference's ``x*std + mean`` lands in ``[0, 1]``; the extra
    ``2x - 1`` is the range ``rgb_to_yuv`` and ``float_to_uint8`` both take.

    Exact, not approximate: the tile cross-fade is a convex combination and this is affine, so
    it commutes with the blend. The reference's ``clamp`` is *not* affine and therefore stays
    after the stitch rather than being folded in here.
    """
    pixel_mean, pixel_std = pixel_denorm
    weight = decoder_state["proj_out.weight"]
    out_features = weight.shape[0]
    assert out_features % out_channels == 0, f"proj_out emits {out_features}, not a multiple of {out_channels}"
    per_channel = out_features // out_channels

    mean = torch.tensor(pixel_mean, dtype=weight.dtype)
    std = torch.tensor(pixel_std, dtype=weight.dtype)
    assert mean.numel() == out_channels and std.numel() == out_channels, "pixel_denorm must be per output channel"
    scale = (2.0 * std).repeat_interleave(per_channel)
    shift = (2.0 * mean - 1.0).repeat_interleave(per_channel)

    decoder_state["proj_out.weight"] = weight * scale.view(-1, 1)
    decoder_state["proj_out.bias"] = decoder_state["proj_out.bias"] * scale + shift


def prepare_decoder_state(
    state: dict[str, torch.Tensor],
    pixel_denorm: tuple[Sequence[float], Sequence[float]] | None = None,
    out_channels: int = 3,
) -> dict[str, torch.Tensor]:
    """Decoder ``state_dict`` with ``post_quant_conv`` folded into ``proj_in``.

    The two are adjacent with no nonlinearity, so ``proj_in(post_quant_conv(z))`` is one
    24->2048 linear and the awkward 24-channel 1x1x1 conv disappears. ``pixel_denorm``,
    when set, is then folded into ``proj_out``.
    """
    decoder_state = {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}
    if not decoder_state:
        decoder_state = dict(state)
    post_weight = state.get("post_quant_conv.weight")
    post_bias = state.get("post_quant_conv.bias")
    decoder_state.pop("post_quant_conv.weight", None)
    decoder_state.pop("post_quant_conv.bias", None)
    if post_weight is not None and "proj_in.weight" in decoder_state:
        post_2d = post_weight.reshape(post_weight.shape[0], post_weight.shape[1])
        decoder_state["proj_in.bias"] = decoder_state["proj_in.weight"] @ post_bias + decoder_state["proj_in.bias"]
        decoder_state["proj_in.weight"] = decoder_state["proj_in.weight"] @ post_2d
    if pixel_denorm is not None:
        _fold_pixel_denorm(decoder_state, pixel_denorm, out_channels)
    return decoder_state


class MiniMaxH3Vae:
    """Orchestrator for the H3 visual VAE: tiling, waves, stitch; owns the leaf modules.

    Constructs a fixed set of sub-models from config + task (decoder, image encoder,
    and a video encoder on ref2va). Each is a named ``Module`` loaded on first use.
    Production shapes are the tile: decoder ``(chunk+overlap, tile/ratio, tile/ratio)``,
    image encoder ``(1, tile, tile)`` taps=1, video encoder ``(clip_length, tile, tile)``
    taps=3.
    """

    def __init__(
        self,
        config: MiniMaxH3VaeConfig,
        *,
        task: str,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        weight_loader=None,
        device_stitch: bool = False,
        profile: bool = False,
        ccl_manager=None,
        pixel_denorm: tuple[Sequence[float], Sequence[float]] | None = None,
        pixel_norm: tuple[Sequence[float], Sequence[float]] | None = None,
        readback_uint8: bool = False,
        waves_per_device: int = 1,
    ) -> None:
        if task not in ("t2va", "ref2va"):
            raise ValueError(f"task must be 't2va' (also serves fl2va) or 'ref2va', got {task!r}")
        self.config = config
        self.task = task
        self.mesh_device = mesh_device
        # Used only to read a wave back (`_read_wave_units`); this VAE has no tensor parallelism and
        # runs no collectives in its forward. Optional because the standalone numerics tests build the
        # VAE on a 1x1 mesh with no fabric configured, where a CCL is neither needed nor openable.
        self.ccl_manager = ccl_manager
        self.dtype = dtype
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        # Hook for a caller that wants weights loaded through `utils.cache` rather than straight off
        # the host state dict. Called as `weight_loader(module, subfolder, state)` once per sub-model.
        # Defaults to a plain strict load, which is what every existing test does.
        self._weight_loader = weight_loader
        self._stitcher = None
        # Blend the tile grid on device and read back the assembled canvas, instead of reading
        # overlapping tiles and blending them on host.
        self.device_stitch = device_stitch
        # `(mean, std)` of the ImageNet normalization the decoder's pixels are still in. Set it and
        # the de-normalization is folded into `proj_out`, so `decode` emits `[-1, 1]` pixels and the
        # caller keeps no copy of the constants. Left unset the decoder emits reference-space values,
        # which is what the numerics tests compare against.
        self.pixel_denorm = pixel_denorm
        # The encode-side mirror: set it and `(x/255 - mean)/std` folds into each encoder's conv_in,
        # so `encode` / `encode_clip` take the decoder's raw **uint8** pixels (1 byte across PCIe
        # instead of 4) and the host never runs a normalize pass. Left unset they take
        # reference-space normalized fp32, which the numerics tests feed directly.
        self.pixel_norm = pixel_norm
        # Cast decoded tiles to uint8 before the DMA, halving what crosses PCIe. Applies to the
        # host path only: `device_stitch` reads back a canvas and never calls `_read_wave_units`,
        # so the two are alternatives, not a combination.
        self.readback_uint8 = readback_uint8
        # Synchronize after each decode forward so `device` and `readback` are separable in the
        # profile -- which also serializes them, so it is opt-in.
        self.profile = profile
        # Decode tiles per device per wave: each device runs a `waves_per_device`-sized batch, so a
        # full wave covers `num_devices * waves_per_device` tiles. >1 trades activation memory for
        # bigger matmuls and fewer waves; 1 is the original one-tile-per-device schedule.
        assert waves_per_device >= 1, f"waves_per_device must be >= 1, got {waves_per_device}"
        self.waves_per_device = waves_per_device
        # Pipeline warmup turns this off so the compile-pass decode does not dump a profile.
        self.log_profile = True
        self._encoder_state: dict[str, torch.Tensor] | None = None
        self._decoder_state: dict[str, torch.Tensor] | None = None
        # Per-decode breakdown, reset at the top of `decode`. Always collected: it is a handful of
        # `perf_counter` calls against a multi-second stage, and without it "VAE decode: 6.0 s" is a
        # number with nowhere to go.
        self._profile = self._empty_profile()
        self.last_decode_profile: dict[str, float] = {}

        # Geometry pinned to config; leaf ctors keep explicit shape args for tests.
        self.decoder = self._make_decoder()
        self.image_encoder = self._make_encoder(num_frames=1, temporal_taps=1)
        self.video_encoder = (
            self._make_encoder(num_frames=config.clip_length, temporal_taps=3) if task == "ref2va" else None
        )
        self.modules = tuple(m for m in (self.decoder, self.image_encoder, self.video_encoder) if m is not None)

    def decode_unit_shape(self) -> tuple[int, int, int]:
        """The ``(T, H, W)`` of one decoder work unit: one temporal chunk of one spatial tile."""
        ratio = self.config.spatial_compression_ratio
        return (
            self.config.tokens_chunk_size + self.config.token_overlap,
            self.tile_size // ratio,
            self.tile_size // ratio,
        )

    def _make_encoder(self, *, num_frames: int, temporal_taps: int) -> MiniMaxH3Encoder3d:
        config = self.config
        return MiniMaxH3Encoder3d(
            num_frames=num_frames,
            height=self.tile_size,
            width=self.tile_size,
            in_channels=config.in_channels,
            out_channels=2 * config.latent_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            spatial_downsample_factors=config.spatial_downsample_factors,
            temporal_downsample_factors=config.temporal_downsample_factors,
            temporal_taps=temporal_taps,
            mesh_device=self.mesh_device,
            dtype=self.dtype,
            pixel_norm=self.pixel_norm,
        )

    def _make_decoder(self):
        from .decoder_minimax_h3 import MiniMaxH3ViTDecoder3d

        num_frames, height, width = self.decode_unit_shape()
        return MiniMaxH3ViTDecoder3d(
            num_frames=num_frames,
            height=height,
            width=width,
            in_channels=self.config.latent_channels,
            out_channels=self.config.out_channels,
            patch_size=self.config.spatial_compression_ratio,
            patch_size_t=self.config.temporal_compression_ratio,
            num_layers=self.config.decoder_num_layers,
            num_heads=self.config.decoder_num_attention_heads,
            head_dim=self.config.decoder_attention_head_dim,
            num_register_tokens=self.config.decoder_num_register_tokens,
            ffn_mult=self.config.decoder_ffn_mult,
            rope_theta=self.config.decoder_rope_theta,
            rope_dim_ratio=self.config.decoder_rope_dim_ratio,
            eps=self.config.decoder_norm_eps,
            mesh_device=self.mesh_device,
        )

    def _encoder_subfolder(self, encoder: MiniMaxH3Encoder3d) -> str:
        num_frames, height, width = encoder.input_shape
        # `_pxnorm` keys the cache: the fold rewrites conv_in's weight and bias, so cached
        # bytes from a fold-less build must never load into a folded encoder or vice versa.
        # The dtype tag does the same for the prepared-weight bytes, which are dtype-specific.
        variant = "_pxnorm" if self.pixel_norm is not None else ""
        if self.dtype != ttnn.float32:
            variant += f"_{str(self.dtype).rsplit('.', 1)[-1].lower()}"
        return f"vae_encoder_t{num_frames}_h{height}_w{width}_taps{encoder.temporal_taps}{variant}"

    def _decoder_subfolder(self) -> str:
        num_frames, height, width = self.decoder.latent_shape
        return f"vae_decoder_t{num_frames}_h{height}_w{width}"

    def _state_for(self, module) -> dict[str, torch.Tensor]:
        state = self._decoder_state if module is self.decoder else self._encoder_state
        if state is None:
            which = "decoder" if module is self.decoder else "encoder"
            raise RuntimeError(f"call load_state() before {which} use")
        return state

    def _ensure_loaded(self, module, subfolder: str) -> None:
        if module.is_loaded():
            return
        state = dict(self._state_for(module))
        if self._weight_loader is not None:
            self._weight_loader(module, subfolder, state)
        else:
            module.load_torch_state_dict(state)

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """Prepare and retain both halves of the checkpoint for cache-miss reloads."""
        self._encoder_state = prepare_encoder_state(state)
        self._decoder_state = prepare_decoder_state(
            state, pixel_denorm=self.pixel_denorm, out_channels=self.config.out_channels
        )

    @staticmethod
    def _empty_profile() -> dict[str, float]:
        return {
            "host_prep": 0.0,
            "upload": 0.0,
            "device": 0.0,
            "readback": 0.0,
            "readback_mb": 0.0,
            "unpatchify": 0.0,
            "tiling": 0.0,
            "stitch": 0.0,
            "waves": 0,
            "units": 0,
            # Per-wave readback durations, not just their sum: a mean hides a slow first wave, and
            # comparing a mean against someone else's min-of-N is how a 2x phantom appears.
            "readback_each": [],
            "device_each": [],
        }

    def _read_wave_units(self, wave: ttnn.Tensor) -> torch.Tensor:
        """Read a ``wave_size``-way dim-0 fracture back to host in unit order.

        Both readers are correct; the choice is cost. ``ConcatMeshToTensor`` MPI-broadcasts every
        shard this host does not own, so each rank pulls the whole wave over MPI on top of its own
        DMA. ``fast_device_to_host`` moves the inter-host hop onto the fabric and DMAs only local
        shards. ``concat_dims=[0, 0]`` is one dim fractured over the whole mesh, which
        ``_reassemble_2d`` handles in its ``d0 == d1`` branch; the two are pinned bit-for-bit by
        ``tests/unit/test_fast_device_to_host.py::TestLinearisedShardReadback``.

        ``readback_uint8`` casts on device first, halving what crosses PCIe. The tiles it returns are
        quantized *before* the host cross-fade rather than after, which costs at most 1 LSB: the blend
        is a convex combination, so blending quantized values and quantizing a blend differ only by
        the rounding. It needs ``pixel_denorm``, since the cast maps ``[-1, 1]``.
        """
        if self.ccl_manager is None:
            return ttnn.to_torch(wave, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        pre_fn = float_to_uint8 if self.readback_uint8 else None
        return fast_device_to_host(wave, self.mesh_device, [0, 0], ccl_manager=self.ccl_manager, pre_transfer_fn=pre_fn)

    def _report_profile(self, total: float) -> None:
        """Log where a decode's wall time went, and stash it on `last_decode_profile`."""
        p = dict(self._profile)
        accounted = sum(
            p.get(k, 0.0)
            for k in (
                "host_prep",
                "upload",
                "device",
                "readback",
                "unpatchify",
                "tiling",
                "stitch",
                "blend",
                "concat",
                "dispatch",
                "compute",
            )
        )
        p["residual"] = max(0.0, total - accounted)
        p["total"] = total
        self.last_decode_profile = p
        if not self.log_profile or (ttnn.using_distributed_env() and int(ttnn.distributed_context_get_rank()) != 0):
            return
        each = p.get("readback_each") or []
        if each:
            logger.info(
                f"    readback per wave: min {min(each) * 1000:.0f} / median "
                f"{sorted(each)[len(each) // 2] * 1000:.0f} / max {max(each) * 1000:.0f} ms  "
                f"[{' '.join(f'{v * 1000:.0f}' for v in each)}]  on {p.get('shape')} {p.get('dtype')}"
            )
        each_d = p.get("device_each") or []
        if each_d:
            logger.info(
                f"    device   per wave: min {min(each_d) * 1000:.0f} / median "
                f"{sorted(each_d)[len(each_d) // 2] * 1000:.0f} / max {max(each_d) * 1000:.0f} ms  "
                f"[{' '.join(f'{v * 1000:.0f}' for v in each_d)}]"
            )
        waves, units = int(p["waves"]), int(p["units"])
        logger.info(
            f"VAE decode profile: {total:.2f} s over {waves} waves / {units} units "
            f"({self.mesh_device.get_num_devices()} devices, "
            f"{units / waves if waves else 0:.1f} units/wave)"
        )
        for name in ("device", "readback", "stitch", "unpatchify", "tiling", "upload", "host_prep", "residual"):
            share = 100 * p[name] / total if total else 0.0
            per_wave = f"  {p[name] / waves * 1000:6.0f} ms/wave" if waves and name in ("device", "readback") else ""
            logger.info(f"    {name:<12} {p[name]:6.2f} s  ({share:4.1f} %){per_wave}")
        logger.info(f"    readback volume {p['readback_mb'] / 1000:.2f} GB")

    def _run_encoder_units(self, units: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode independent ``(clip, tile)`` units, one per device, in mesh-sized waves.

        Every unit is the same shape and fully independent of the others -- that is what
        the reference's spatial tiling buys -- and the encoder contains no CCL, so handing
        each device a different unit and running one program is exact SPMD. Gated bit-exact
        against the replicated result in ``test_vae_data_parallel_minimax_h3.py``.

        The last wave is padded by repeating a unit rather than shrinking the program: the
        conv3d blockings and the GroupNorm core grid are chosen per shape at construction,
        so a short final wave would build a second set of them.
        """
        if not units:
            return []
        # A wave is one program, so a differently-shaped unit would be silently mis-stacked.
        odd = [tuple(u.shape) for u in units if u.shape != units[0].shape]
        assert not odd, f"units must share a shape; {units[0].shape} vs {odd[0]}"
        _, _, num_frames, height, width = units[0].shape
        encoder = self.image_encoder if num_frames == 1 else self.video_encoder
        if encoder is None:
            raise RuntimeError("video encode reached a non-ref2va VAE; construct with task='ref2va'")
        assert encoder.input_shape == (
            num_frames,
            height,
            width,
        ), f"unit shape {(num_frames, height, width)} != encoder {encoder.input_shape}"
        in_channels = encoder.conv_in.in_channels
        moments = 2 * self.config.latent_channels
        wave_size = self.mesh_device.get_num_devices()
        # Same counters `_stream_decoder_units` keeps, and same caveat: without the opt-in
        # `profile` sync, `device` times the enqueue and the wait lands in `readback`.
        profile = self._profile

        def prepare(unit: torch.Tensor) -> torch.Tensor:
            # Channel-last only -- the pad to `in_channels` happens on device, below. Padding here
            # would inflate the upload >10x (3 -> 32 channels of fp32) and the host_prep with it;
            # the device pad is bit-exact against the host pad and nearly free next to the DMA.
            return unit.permute(0, 2, 3, 4, 1).contiguous()

        def read_wave(encoded: ttnn.Tensor, count: int) -> list[torch.Tensor]:
            mark = time.perf_counter()
            out = fast_device_to_host(encoded, self.mesh_device, [0, 0], ccl_manager=self.ccl_manager).float()
            elapsed = time.perf_counter() - mark
            profile["readback"] += elapsed
            profile["readback_each"].append(elapsed)
            profile["readback_mb"] += out.numel() * out.element_size() / 1e6
            profile["shape"] = tuple(encoded.shape)
            profile["dtype"] = str(encoded.dtype)
            ttnn.deallocate(encoded)

            mark = time.perf_counter()
            tiles = [
                out[index : index + 1, ..., :moments].permute(0, 4, 1, 2, 3).contiguous() for index in range(count)
            ]
            profile["unpatchify"] += time.perf_counter() - mark
            return tiles

        # Same schedule as `_stream_decoder_units`: wave k's readback is deferred until wave
        # k + 1 is prepared, uploaded and enqueued, so the host work and the k - 1 transfer run
        # under wave k's compute instead of after it. Two waves' outputs are live at once, which
        # is one extra latent tile per device.
        results: list[torch.Tensor] = []
        pending: tuple[ttnn.Tensor, int] | None = None
        for start in range(0, len(units), wave_size):
            # Prepared per wave, not up front: `units` are cheap views into the source video,
            # but permute().contiguous() materialises 13.4 MB each, so preparing all of them
            # would cost 19 GB of host memory at 1440P/10s. Outputs are latents and tiny.
            mark = time.perf_counter()
            wave = [prepare(unit) for unit in units[start : start + wave_size]]
            count = len(wave)
            padded = wave + [wave[-1]] * (wave_size - count)
            batch = torch.cat(padded, dim=0)
            profile["host_prep"] += time.perf_counter() - mark

            raw = batch.dtype == torch.uint8
            assert raw == (self.pixel_norm is not None), (
                f"{batch.dtype} pixels against pixel_norm={'set' if self.pixel_norm else 'unset'}: a folded "
                "conv_in takes raw uint8 and an unfolded one takes normalized floats -- mixing them "
                "double- or un-normalizes with no error anywhere downstream"
            )
            mark = time.perf_counter()
            x_device = ttnn.from_torch(
                batch,
                dtype=ttnn.uint8 if raw else self.dtype,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            if raw:
                # ttnn.pad refuses uint8, so the cast comes first; the fp32 intermediate is
                # device-side and cheap next to the 4x it removes from the transfer.
                cast = ttnn.typecast(x_device, self.dtype)
                ttnn.deallocate(x_device)
                x_device = cast
            if batch.shape[-1] < in_channels:
                # Zero-pad the channel axis to conv_in's tile alignment on device (the padded
                # weight channels are zeros, so any fill works; zero matches the host pad
                # bit-for-bit). Same move as pipeline_wan_i2v's conditioning upload.
                padded_device = ttnn.pad(
                    x_device, [(0, 0), (0, 0), (0, 0), (0, 0), (0, in_channels - batch.shape[-1])], value=0.0
                )
                ttnn.deallocate(x_device)
                x_device = padded_device
            profile["upload"] += time.perf_counter() - mark
            profile["upload_mb"] = profile.get("upload_mb", 0.0) + batch.numel() * batch.element_size() / 1e6

            mark = time.perf_counter()
            encoded = encoder(x_device)
            ttnn.deallocate(x_device)
            # Opt-in sync, as in decode: it makes `device` and `readback` separable in the
            # profile but serializes the streaming, so leave it off to go fast.
            if self.profile:
                ttnn.synchronize_device(self.mesh_device)
            elapsed = time.perf_counter() - mark
            profile["device"] += elapsed
            profile["device_each"].append(elapsed)

            if pending is not None:
                results.extend(read_wave(*pending))
            pending = (encoded, count)
            profile["waves"] += 1
            profile["units"] += count

        if pending is not None:
            results.extend(read_wave(*pending))
        return results

    def encode_clip(self, x_BCTHW: torch.Tensor) -> torch.Tensor:
        """Encode one keyframe, spatially tiled -- the reference ``_encode_clip`` at T=1.

        A keyframe goes through here rather than :meth:`encode`, because a single frame
        must not be put through the temporal chunking.
        """
        assert (
            x_BCTHW.shape[2] == 1
        ), f"encode_clip is the keyframe path (T=1); got T={x_BCTHW.shape[2]}, use encode() for video"
        self._ensure_loaded(self.image_encoder, self._encoder_subfolder(self.image_encoder))

        mark = time.perf_counter()
        units = self._clip_tiles(x_BCTHW)
        self._profile["tiling"] += time.perf_counter() - mark
        encoded = self._run_encoder_units(units)
        mark = time.perf_counter()
        stitched = self._stitch_clip(encoded, x_BCTHW.shape[-2], x_BCTHW.shape[-1])
        self._profile["stitch"] += time.perf_counter() - mark
        return stitched

    def _tile_grid(self, height: int, width: int):
        ratio = self.config.spatial_compression_ratio
        return (
            split_tiles(height, self.tile_size, self.tile_overlap, ratio),
            split_tiles(width, self.tile_size, self.tile_overlap, ratio),
        )

    def _clip_tiles(self, x_BCTHW: torch.Tensor) -> list[torch.Tensor]:
        """One clip's spatial tile crops, row-major -- the reference ``_split_tiles`` order."""
        (y_starts, y_lengths, _), (x_starts, x_lengths, _) = self._tile_grid(x_BCTHW.shape[-2], x_BCTHW.shape[-1])
        return [
            x_BCTHW[..., y : y + y_length, x : x + x_length]
            for y, y_length in zip(y_starts, y_lengths)
            for x, x_length in zip(x_starts, x_lengths)
        ]

    def _stitch_clip(self, encoded: list[torch.Tensor], height: int, width: int) -> torch.Tensor:
        """Re-grid a row-major list of encoded tiles and cross-fade the overlaps."""
        (_, y_lengths, y_overlaps), (_, x_lengths, x_overlaps) = self._tile_grid(height, width)
        columns = len(x_lengths)
        assert len(encoded) == len(y_lengths) * columns, f"{len(encoded)} tiles for a {len(y_lengths)}x{columns} grid"
        rows = [encoded[i * columns : (i + 1) * columns] for i in range(len(y_lengths))]
        ratio = self.config.spatial_compression_ratio
        return stitch_tiles(rows, [o // ratio for o in y_overlaps], [o // ratio for o in x_overlaps])

    def encode(self, x_BCTHW: torch.Tensor) -> torch.Tensor:
        """Encode a video in ``clip_length``-frame chunks, dropping ``token_drop`` tails.

        Mirrors the reference ``_encode``: the final frame is repeated to reach a whole
        number of clips, and the trailing ``token_drop`` latent frames are then removed.

        Every ``(clip, tile)`` pair is an independent work unit, so they are collected
        across **all** clips and handed to the mesh together. Batching per clip instead
        would leave the wave ragged -- 768P is a 4x7 grid, so 28 units against 32 devices
        would waste an eighth of the mesh on every clip.
        """
        clip_length = self.config.clip_length
        num_frames = x_BCTHW.shape[2]
        if self.video_encoder is None:
            raise RuntimeError("video encode reached a non-ref2va VAE; construct with task='ref2va'")
        self._ensure_loaded(self.video_encoder, self._encoder_subfolder(self.video_encoder))
        if num_frames % clip_length != 0:
            pad = x_BCTHW[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
            x_BCTHW = torch.cat([x_BCTHW, pad], dim=2)

        height, width = x_BCTHW.shape[-2], x_BCTHW.shape[-1]
        clips = [x_BCTHW[:, :, i * clip_length : (i + 1) * clip_length] for i in range(x_BCTHW.shape[2] // clip_length)]
        mark = time.perf_counter()
        per_clip = [self._clip_tiles(clip) for clip in clips]
        tiles_per_clip = len(per_clip[0])
        flat = [unit for clip_units in per_clip for unit in clip_units]
        self._profile["tiling"] += time.perf_counter() - mark
        encoded = self._run_encoder_units(flat)
        mark = time.perf_counter()
        moments = torch.cat(
            [
                self._stitch_clip(encoded[i * tiles_per_clip : (i + 1) * tiles_per_clip], height, width)
                for i in range(len(clips))
            ],
            dim=2,
        )
        self._profile["stitch"] += time.perf_counter() - mark
        if self.config.token_drop > 0:
            moments = moments[:, :, : -self.config.token_drop]
        return moments

    # ---------------------------------------------------------------- decode

    def _stream_decoder_units(self, units: list[torch.Tensor]):
        """Yield decoded pixel tiles in ``units`` order, running the mesh a wave at a time.

        Same argument as :meth:`_run_encoder_units`: the reference decodes each spatial
        tile of each temporal chunk on its own and only cross-fades afterwards, and the ViT
        decoder holds no CCL, so one device per unit is exact SPMD. The temporal blend in
        :meth:`decode` still runs in order on the host -- it is the *decodes* that are
        independent, not the stitching.

        A generator, not a list, because the read of wave ``k`` is deferred until wave ``k+1``
        is enqueued: that only buys anything when the caller hands over every unit at once, and
        a caller given a list would have to hold the whole video to do so. Streaming lets it
        stitch and release each chunk while later waves are still in flight.
        """
        from .decoder_minimax_h3 import unpatchify

        if not units:
            return []
        odd = [tuple(u.shape) for u in units if u.shape != units[0].shape]
        assert not odd, f"units must share a shape; {units[0].shape} vs {odd[0]}"
        _, _, num_frames, height, width = units[0].shape
        decoder = self.decoder
        assert decoder.latent_shape == (
            num_frames,
            height,
            width,
        ), f"unit shape {(num_frames, height, width)} != decoder {decoder.latent_shape}"
        # A wave spans every device, each running a `waves_per_device`-sized batch (dim-0 shard).
        wave_size = self.mesh_device.get_num_devices() * self.waves_per_device

        profile = self._profile
        # Synchronizing after each forward is what makes `device` and `readback` separable in the
        # profile -- and it also serializes them, defeating the pipelining below. So it is opt-in
        # via the constructor's `profile` flag; leave it off to go fast.
        attribute = self.profile

        def read_wave(decoded, count: int) -> list[torch.Tensor]:
            mark = time.perf_counter()
            out = self._read_wave_units(decoded)
            elapsed = time.perf_counter() - mark
            profile["readback"] += elapsed
            profile["readback_each"].append(elapsed)
            profile["shape"] = tuple(decoded.shape)
            profile["dtype"] = str(decoded.dtype)
            profile["readback_mb"] += out.numel() * out.element_size() / 1e6

            mark = time.perf_counter()
            # `.float()` per tile rather than once over the whole batch: the batch is 32 x 22 MB and
            # upcasting it whole allocates a 5 GB fp32 intermediate only to slice it 32 ways. The blend
            # still runs in fp32, so this is numerically identical -- the device output is bf16 either
            # way, and upcasting earlier adds no information.
            tiles = [
                unpatchify(
                    out[index : index + 1].float(),
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    out_channels=self.config.out_channels,
                    patch_size=self.config.spatial_compression_ratio,
                    patch_size_t=self.config.temporal_compression_ratio,
                )
                for index in range(count)
            ]
            profile["unpatchify"] += time.perf_counter() - mark
            return tiles

        pending: tuple | None = None  # (decoded_device_tensor, count) for the wave in flight
        for start in range(0, len(units), wave_size):
            mark = time.perf_counter()
            wave = [
                unit.permute(0, 2, 3, 4, 1).reshape(1, num_frames * height * width, -1)
                for unit in units[start : start + wave_size]
            ]
            count = len(wave)
            padded = wave + [wave[-1]] * (wave_size - count)
            batch = torch.cat(padded, dim=0)
            profile["host_prep"] += time.perf_counter() - mark

            mark = time.perf_counter()
            tokens = ttnn.from_torch(
                batch,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            profile["upload"] += time.perf_counter() - mark

            mark = time.perf_counter()
            decoded = decoder(tokens)
            if attribute:
                ttnn.synchronize_device(self.mesh_device)
            elapsed = time.perf_counter() - mark
            profile["device"] += elapsed
            profile["device_each"].append(elapsed)

            # Read the previous wave only after this one is enqueued, so its transfer overlaps this
            # wave's compute instead of following it. Two waves' device output are live at once,
            # which is one extra tile per device.
            if pending is not None:
                yield from read_wave(*pending)
            pending = (decoded, count)
            profile["waves"] += 1
            profile["units"] += count

        if pending is not None:
            yield from read_wave(*pending)

    def _decode_clip_device_stitched(self, z_BCTHW: torch.Tensor, output_type: str = "float"):
        """One clip, decoded and stitched entirely on device, read back as the assembled canvas.

        The win is transfer volume, not compute. The host path reads back **overlapping** tiles --
        28 of 256x256 against a 768x1344 canvas, 2.51 GB over the whole video -- and then blends them
        on host. Blending first and reading back the canvas moves far less, and the two-axis
        all-gather that co-locates the neighbours costs little against the readback it removes, so
        the collective is nearly free.

        The tile -> gathered-position map comes from `gathered_tile_order`'s inverse and is **not**
        row-major: the two-axis gather transposes dim 0, so position
        `c * rows + r` holds shard `r * cols + c`. Assuming row-major here puts tiles in the wrong
        place, which the seam gate catches loudly -- but only because something finally reads them.
        """
        from .decoder_minimax_h3 import unpatchify  # noqa: F401  (host fallback parity)

        (y_starts, y_lengths, y_overlaps), (x_starts, x_lengths, x_overlaps) = self._decode_tile_grid(
            z_BCTHW.shape[-2], z_BCTHW.shape[-1]
        )
        grid_rows, grid_cols = len(y_lengths), len(x_lengths)

        mark = time.perf_counter()
        units = self._latent_tiles(z_BCTHW)
        self._profile["tiling"] += time.perf_counter() - mark
        assert len(units) == grid_rows * grid_cols, f"{len(units)} tiles for a {grid_rows}x{grid_cols} grid"

        mesh_rows, mesh_cols = tuple(self.mesh_device.shape)
        wave_size = self.mesh_device.get_num_devices()
        assert (
            len(units) <= wave_size
        ), f"a device-stitched clip must fit one wave; {len(units)} tiles against {wave_size} devices"
        # The co-location below all-gathers the *whole* of each cluster axis, so every device ends up
        # holding `wave_size` tiles regardless of how many the grid has. That is affordable when the
        # grid fills the mesh and ruinous when it does not: a 4x7 grid on a 4x32 quad would put 128
        # tiles on every chip to use 28 of them. Refuse rather than allocate it. Lifting this needs
        # the neighbour-exchange form, where a tile only ever pulls its two overlap strips.
        assert grid_rows == mesh_rows and grid_cols <= mesh_cols, (
            f"the gather-based device stitch needs a {mesh_rows}x<={mesh_cols} tile grid to match the "
            f"mesh, got {grid_rows}x{grid_cols} on {mesh_rows}x{mesh_cols}"
        )
        _, _, num_frames, height, width = units[0].shape
        decoder = self.decoder
        assert decoder.latent_shape == (
            num_frames,
            height,
            width,
        ), f"unit shape {(num_frames, height, width)} != decoder {decoder.latent_shape}"
        profile = self._profile

        mark = time.perf_counter()
        wave = [unit.permute(0, 2, 3, 4, 1).reshape(1, num_frames * height * width, -1) for unit in units]
        # Pad to the mesh with repeats of the last tile, as the host path does. The padding lands on
        # devices whose gathered positions map outside the grid and is never indexed.
        batch = torch.cat(wave + [wave[-1]] * (wave_size - len(wave)), dim=0)
        profile["host_prep"] += time.perf_counter() - mark

        mark = time.perf_counter()
        tokens = ttnn.from_torch(
            batch,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )
        profile["upload"] += time.perf_counter() - mark

        mark = time.perf_counter()
        from .stitch_device_minimax_h3 import DeviceTileStitcher, unpatchify_device

        decoded = decoder(tokens)
        # Row-major from here to the DMA. `unpatchify_device`'s rank-8 intermediate has trailing dims
        # of 16, which a tiled reshape pads to 32x32 -- a 4x blowup for a view -- and the stitch's
        # slices and concats land off tile boundaries on this grid's overlaps. One conversion here
        # keeps every step after it on a fast path, `mesh_partition` included.
        decoded = ttnn.to_layout(decoded, ttnn.ROW_MAJOR_LAYOUT)
        pixels = unpatchify_device(
            decoded,
            num_frames=num_frames,
            height=height,
            width=width,
            out_channels=self.config.out_channels,
            patch_size=self.config.spatial_compression_ratio,
            patch_size_t=self.config.temporal_compression_ratio,
        )
        # Co-locate every tile on every device. Two gathers, one per mesh axis.
        gathered = ttnn.all_gather(pixels, 0, cluster_axis=0, topology=ttnn.Topology.Ring)
        gathered = ttnn.all_gather(gathered, 0, cluster_axis=1, topology=ttnn.Topology.Ring)

        # Position of shard k in the gathered tensor: the inverse of the transpose.
        order = [r * mesh_cols + c for c in range(mesh_cols) for r in range(mesh_rows)]
        position = {shard: index for index, shard in enumerate(order)}
        if self._stitcher is None:
            self._stitcher = DeviceTileStitcher(self.mesh_device)
        # `ttnn.Shape` does not support slicing, so materialize it as a list once.
        gathered_shape = list(gathered.shape)

        def tile_at(row: int, col: int) -> ttnn.Tensor:
            index = position[row * grid_cols + col]
            return ttnn.slice(gathered, [index, 0, 0, 0, 0], [index + 1, *gathered_shape[1:]])

        rows = [[tile_at(i, j) for j in range(grid_cols)] for i in range(grid_rows)]
        canvas = self._stitcher.stitch(rows, y_overlaps, x_overlaps)
        elapsed = time.perf_counter() - mark
        profile["device"] += elapsed
        profile["device_each"].append(elapsed)
        profile["waves"] += 1
        profile["units"] += len(units)

        mark = time.perf_counter()
        canvas_shape = tuple(canvas.shape)
        canvas_dtype = str(canvas.dtype)
        if output_type == "yuv420":
            out = self._read_canvas_yuv(canvas)
            read_bytes = out.size
        else:
            out = local_device_to_torch(canvas).float()
            read_bytes = out.numel() * out.element_size()
        elapsed = time.perf_counter() - mark
        profile["readback"] += elapsed
        profile["readback_each"].append(elapsed)
        profile["shape"] = canvas_shape
        profile["dtype"] = canvas_dtype
        profile["readback_mb"] += read_bytes / 1e6
        return out

    def _read_canvas_yuv(self, canvas: ttnn.Tensor) -> np.ndarray:
        """Convert the replicated canvas to YUV 4:2:0 on device and read it back as planar uint8.

        Two things earn their keep here. The `clamp` is the reference's post-stitch `clamp(0, 1)`,
        moved to `[-1, 1]` by the `proj_out` fold and kept *after* the blend because clamping is the
        one step in the chain that is not affine. The `mesh_partition` pair splits a canvas every
        device already holds an identical copy of, so the DMA that follows runs across every PCIe
        link instead of one -- the same repeat/partition trick `fast_device_to_host` uses to spread
        a multi-host read.
        """
        canvas = ttnn.clamp(canvas, min=-1.0, max=1.0)
        canvas = ttnn.typecast(canvas, ttnn.bfloat16)
        canvas = ttnn.to_layout(canvas, ttnn.ROW_MAJOR_LAYOUT)

        mesh_rows, mesh_cols = tuple(self.mesh_device.shape)
        height, width = int(canvas.shape[-2]), int(canvas.shape[-1])
        # `fast_device_to_host_yuv` reconstructs H and W as `per_shard * mesh_extent`, so an uneven
        # split would silently assemble a canvas of the wrong size rather than fail.
        assert height % mesh_rows == 0, f"canvas height {height} does not split over {mesh_rows} mesh rows"
        assert width % mesh_cols == 0, f"canvas width {width} does not split over {mesh_cols} mesh columns"
        assert (height // mesh_rows) % 2 == 0 and (
            width // mesh_cols
        ) % 2 == 0, "4:2:0 needs an even per-shard height and width"

        canvas = ttnn.mesh_partition(canvas, dim=-2, cluster_axis=0)
        canvas = ttnn.mesh_partition(canvas, dim=-1, cluster_axis=1)

        planar = fast_device_to_host_yuv(canvas, self.mesh_device, ccl_manager=self.ccl_manager)
        return planar.reshape(planar.shape[0], height * 3 // 2, width)

    def decode_clip(self, z_BCTHW: torch.Tensor) -> torch.Tensor:
        """Decode one temporal clip, spatially tiled -- the reference ``_decode_clip``.

        Tiles are laid out in *pixel* space and mapped back onto the latent grid, so the
        blend extents are pixel-space too (unlike encode, where they are divided down).
        """
        self._ensure_loaded(self.decoder, self._decoder_subfolder())
        if self.device_stitch:
            return self._decode_clip_device_stitched(z_BCTHW)

        mark = time.perf_counter()
        units = self._latent_tiles(z_BCTHW)
        self._profile["tiling"] += time.perf_counter() - mark
        decoded = list(self._stream_decoder_units(units))
        mark = time.perf_counter()
        out = self._stitch_decoded(decoded, z_BCTHW.shape[-2], z_BCTHW.shape[-1])
        self._profile["stitch"] += time.perf_counter() - mark
        return out

    def decode_tile_grid(self, latent_height: int, latent_width: int):
        """Public alias for :meth:`_decode_tile_grid`, for callers that need the seam positions.

        Exists because re-deriving this with `split_tiles(...)` and hardcoded constants is easy to get
        wrong: at 1344x768 the real overlap of 64 gives a 4x7 grid, and an assumed 32 gives 4x6 with
        boundary columns that are not boundaries. Ask the object that owns the geometry.
        """
        return self._decode_tile_grid(latent_height, latent_width)

    def _decode_tile_grid(self, latent_height: int, latent_width: int):
        """Tiles are laid out in *pixel* space, then mapped back onto the latent grid."""
        ratio = self.config.spatial_compression_ratio
        return (
            split_tiles(latent_height * ratio, self.tile_size, self.tile_overlap, ratio),
            split_tiles(latent_width * ratio, self.tile_size, self.tile_overlap, ratio),
        )

    def _latent_tiles(self, z_BCTHW: torch.Tensor) -> list[torch.Tensor]:
        ratio = self.config.spatial_compression_ratio
        (y_starts, y_lengths, _), (x_starts, x_lengths, _) = self._decode_tile_grid(
            z_BCTHW.shape[-2], z_BCTHW.shape[-1]
        )
        return [
            z_BCTHW[..., y // ratio : y // ratio + y_length // ratio, x // ratio : x // ratio + x_length // ratio]
            for y, y_length in zip(y_starts, y_lengths)
            for x, x_length in zip(x_starts, x_lengths)
        ]

    def _stitch_decoded(self, decoded: list[torch.Tensor], latent_height: int, latent_width: int) -> torch.Tensor:
        """Re-grid decoded tiles. The blend extents are pixel-space here, not divided down."""
        (_, y_lengths, y_overlaps), (_, x_lengths, x_overlaps) = self._decode_tile_grid(latent_height, latent_width)
        columns = len(x_lengths)
        assert len(decoded) == len(y_lengths) * columns, f"{len(decoded)} tiles for a {len(y_lengths)}x{columns} grid"
        rows = [decoded[i * columns : (i + 1) * columns] for i in range(len(y_lengths))]
        return stitch_tiles(rows, y_overlaps, x_overlaps)

    def decode(self, z_BCTHW: torch.Tensor, *, output_type: str = "float"):
        """Decode a latent video, mirroring the chunking ``encode`` applied.

        ``output_type`` picks what crosses PCIe. ``"float"`` reads the decoder's own pixels back and
        returns ``(1, 3, T, H, W)``. ``"yuv420"`` converts on device and returns a planar
        ``(T, H*3//2, W)`` uint8 array ready for ``export_video_audio_yuv`` -- 1.5 bytes per pixel
        against 6, and no host blend, unpatchify or upcast. It needs the assembled canvas, so it
        requires ``device_stitch``, and it needs ``[-1, 1]`` pixels, so it requires ``pixel_denorm``.

        ``token_drop`` removes the tail of every encoded chunk, so consecutive decoded
        chunks overlap by ``frame_overlap`` pixel frames and are cross-faded. Ported from
        the reference ``_decode``; the trailing repeated latent frames produce pixel frames
        that were never asked for and are cut at the end.

        Runs with raised torch threads (``_DECODE_TORCH_THREADS``), because a server worker pins
        torch to one thread for the denoise loop's benefit and the host still does the tile
        stitching, unpatchify and temporal blend. The previous limit is restored on the way out.
        """
        if output_type not in OUTPUT_TYPES:
            raise ValueError(f"output_type must be one of {OUTPUT_TYPES}, got {output_type!r}")
        if self.readback_uint8 and self.pixel_denorm is None:
            raise ValueError("readback_uint8 needs pixel_denorm; the uint8 cast maps [-1, 1]")
        if output_type == "yuv420":
            if not self.device_stitch:
                raise ValueError("output_type='yuv420' needs device_stitch=True; it reads back the stitched canvas")
            if self.pixel_denorm is None:
                raise ValueError("output_type='yuv420' needs pixel_denorm; the YUV kernel takes [-1, 1] pixels")

        threads = _DECODE_TORCH_THREADS
        previous_threads = torch.get_num_threads()
        if threads > 0 and threads != previous_threads:
            torch.set_num_threads(threads)
        try:
            return self._decode(z_BCTHW, output_type)
        finally:
            torch.set_num_threads(previous_threads)

    def _decode(self, z_BCTHW: torch.Tensor, output_type: str = "float"):
        self._ensure_loaded(self.decoder, self._decoder_subfolder())
        self._profile = self._empty_profile()
        decode_started = time.perf_counter()
        config = self.config
        chunk_size = config.tokens_chunk_size
        temporal_ratio = config.temporal_compression_ratio
        chunk_num_frames = chunk_size * temporal_ratio

        num_tokens = z_BCTHW.shape[2] + config.token_drop
        pad_tokens = (-num_tokens) % chunk_size
        num_chunks = (num_tokens + pad_tokens) // chunk_size - int(config.token_drop > 0)
        if pad_tokens > 0:
            z_BCTHW = torch.cat([z_BCTHW, z_BCTHW[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        # Every (chunk, tile) decode is independent -- only the temporal cross-fade below is
        # ordered -- so all of them go to the mesh in one batch. Per-chunk batching would
        # offer 28 units against 32 devices at 768P and idle an eighth of the mesh.
        chunk_latents = [
            z_BCTHW[:, :, i * chunk_size : i * chunk_size + chunk_size + config.token_overlap]
            for i in range(num_chunks)
        ]
        if self.device_stitch and chunk_latents:
            # Each chunk's tile grid is decoded, unpatchified, all-gathered and blended on device,
            # and only the assembled canvas is read back. One chunk at a time, and serial on
            # purpose: deferring a chunk's readback until the next is enqueued does not help,
            # because the stage is device-bound and holding two canvases live only adds allocation.
            clips = [self._decode_clip_device_stitched(latents, output_type) for latents in chunk_latents]
        elif chunk_latents:
            latent_height, latent_width = chunk_latents[0].shape[-2], chunk_latents[0].shape[-1]
            mark = time.perf_counter()
            # Every (chunk, tile) unit in one stream. One chunk at a time would give the wave loop
            # a single wave per call, so it could never overlap a read with the next wave's compute,
            # and every wave but the last would be padded out to the mesh.
            all_units = [unit for latents in chunk_latents for unit in self._latent_tiles(latents)]
            self._profile["tiling"] += time.perf_counter() - mark
            tiles_per_chunk = len(all_units) // num_chunks

            clips = []
            # Holding every decoded tile would scale with the whole video. Stitching a chunk the
            # moment its last tile lands caps the pile at one chunk plus the wave in flight.
            buffered: list[torch.Tensor] = []
            for tile in self._stream_decoder_units(all_units):
                buffered.append(tile)
                while len(buffered) >= tiles_per_chunk:
                    mark = time.perf_counter()
                    clips.append(self._stitch_decoded(buffered[:tiles_per_chunk], latent_height, latent_width))
                    self._profile["stitch"] += time.perf_counter() - mark
                    del buffered[:tiles_per_chunk]
            assert not buffered, f"{len(buffered)} tiles left over for a {tiles_per_chunk}-tile chunk grid"
        else:
            clips = []

        assemble_mark = time.perf_counter()
        decoded, overlap = [], None
        for i in range(num_chunks):
            clip = clips[i]
            for j in range(int(config.token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip_frames(clip, frame_start, frame_start + chunk_num_frames)
                chunk = clip_frames(chunk, config.frame_pre_padding, clip_num_frames(chunk))
                if j == 0:
                    if overlap is not None:
                        chunk = blend_clip_frames(overlap, chunk, config.frame_overlap)
                    decoded.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded.append(overlap)

        self._profile["blend"] = time.perf_counter() - assemble_mark
        concat_mark = time.perf_counter()
        result = concat_clip_frames(decoded)
        self._profile["concat"] = time.perf_counter() - concat_mark
        if pad_tokens > 0:
            intra_tail = config.clip_length % temporal_ratio
            tokens_before_pad = z_BCTHW.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (tokens_before_pad + k) % chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            result = clip_frames(result, 0, clip_num_frames(result) - pad_frames)
        self._report_profile(time.perf_counter() - decode_started)
        return result

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the per-channel ``latents_mean`` / ``latents_std`` the pipeline expects."""
        mean = torch.tensor(self.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return (latents - mean) / std
