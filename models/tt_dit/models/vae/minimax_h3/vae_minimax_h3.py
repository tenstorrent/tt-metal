# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Top-level MiniMax-H3 visual VAE: spatial tiling, temporal chunking, encode.

MiniMax-H3 ships with **tiling enabled** (``use_tiling = True``,
``tile_sample_min_{height,width} = 256``, overlap 64). Both ``_encode_clip`` and
``_decode_clip`` split the frame into 256x256 **pixel** tiles, run the model
independently per tile, and linearly cross-fade the overlaps. That is not an optional
memory optimisation -- reproducing the released model requires it -- and it is what
bounds every work unit:

* the encoder always sees ``(1, 3, 17, 256, 256)`` and emits ``(1, 48, 5, 16, 16)``;
* the decoder always sees ``(1, 24, 7, 16, 16)``.

Because the tiles are independent, the mesh is used **data-parallel over (tile, chunk)
work units with the weights replicated**, rather than by sharding one tile's H/W. Two
consequences simplify this port considerably: reflect padding stays a local
slice-and-concat (``neighbor_pad_async`` has no reflect mode), and GroupNorm statistics
never need a cross-device reduction.

The tiling and blending stay on host. They are cheap, exactly specified, and porting
them to device buys nothing until a measurement says otherwise.

The earlier version of this file was a T=1-only keyframe encoder built on
``Conv2dViaConv3d``. That class cannot load ``conv_in``: it sizes the weight from the
*aligned* input-channel count but prepares an unpadded weight, so a 3-channel conv
fails its ``Parameter`` shape check. The encoder now lives in
``encoder_minimax_h3.py`` on a ``LTXCausalConv3d``-shaped conv, parameterised by
``temporal_taps`` so the keyframe path and the clip path are one implementation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

import ttnn

from ....layers.module import Module
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


# How many mesh-sized waves of decoded tiles to hold on the host before stitching. Keeps
# every wave full while bounding peak host memory: a decoded tile is 22 MB, so this caps
# the in-flight set at a few GB regardless of video length or resolution.
_DECODE_WAVES_IN_FLIGHT = 4


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


class MiniMaxH3Vae(Module):
    """Encode path for the H3 visual VAE, tiled and temporally chunked.

    Encoders are built lazily and cached per distinct tile shape. Each carries
    GroupNorms whose core grid is pinned at construction, so one encoder serves every
    tile of a given ``(T, H, W)`` -- which, with tiling on, is every tile of the frame.
    """

    def __init__(
        self,
        config: MiniMaxH3VaeConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        use_tiling: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.use_tiling = use_tiling
        self._encoders: dict[tuple[int, int, int, int], MiniMaxH3Encoder3d] = {}
        self._encoder_state: dict[str, torch.Tensor] | None = None
        self._decoders: dict[tuple[int, int, int], object] = {}
        self._decoder_state: dict[str, torch.Tensor] | None = None

    def forward(self, *args, **kwargs):
        """Unused: ``Module`` declares ``forward`` abstract, but this class has two entry points."""
        raise RuntimeError("use encode() or encode_clip(); MiniMaxH3Vae has no single forward")

    def load_encoder_state(self, state: dict[str, torch.Tensor]) -> None:
        """Hold the encoder ``state_dict`` so lazily-built per-shape encoders can load it.

        ``quant_conv`` is folded into ``conv_out`` here. The two are adjacent with no
        nonlinearity between them, so one 1024->48 k3 conv does both and the awkward
        48-channel 1x1x1 conv disappears entirely.
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
            encoder_state["conv_out.weight"] = torch.einsum(
                "oi,ijkmn->ojkmn", quant_2d, encoder_state["conv_out.weight"]
            )
            encoder_state["conv_out.bias"] = quant_2d @ encoder_state["conv_out.bias"] + quant_bias
        self._encoder_state = encoder_state
        self._is_loaded = True

    def _encoder_for(self, num_frames: int, height: int, width: int, temporal_taps: int) -> MiniMaxH3Encoder3d:
        key = (num_frames, height, width, temporal_taps)
        if key not in self._encoders:
            if self._encoder_state is None:
                raise RuntimeError("call load_encoder_state() before encoding")
            config = self.config
            encoder = MiniMaxH3Encoder3d(
                num_frames=num_frames,
                height=height,
                width=width,
                in_channels=config.in_channels,
                out_channels=2 * config.latent_channels,
                block_out_channels=config.block_out_channels,
                layers_per_block=config.layers_per_block,
                spatial_downsample_factors=config.spatial_downsample_factors,
                temporal_downsample_factors=config.temporal_downsample_factors,
                temporal_taps=temporal_taps,
                mesh_device=self.mesh_device,
                dtype=self.dtype,
            )
            encoder.load_torch_state_dict(dict(self._encoder_state))
            self._encoders[key] = encoder
        return self._encoders[key]

    def _run_encoder(self, tile_BCTHW: torch.Tensor, temporal_taps: int) -> torch.Tensor:
        return self._run_encoder_units([tile_BCTHW], temporal_taps)[0]

    def _run_encoder_units(self, units: list[torch.Tensor], temporal_taps: int) -> list[torch.Tensor]:
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
        encoder = self._encoder_for(num_frames, height, width, temporal_taps)
        in_channels = encoder.conv_in.in_channels
        moments = 2 * self.config.latent_channels
        wave_size = self.mesh_device.get_num_devices()

        def prepare(unit: torch.Tensor) -> torch.Tensor:
            x = unit.permute(0, 2, 3, 4, 1).contiguous()
            if x.shape[-1] < in_channels:
                x = torch.nn.functional.pad(x, (0, in_channels - x.shape[-1]))
            return x

        results: list[torch.Tensor] = []
        for start in range(0, len(units), wave_size):
            # Prepared per wave, not up front: `units` are cheap views into the source video,
            # but permute().contiguous() materialises 13.4 MB each, so preparing all of them
            # would cost 19 GB of host memory at 1440P/10s. Outputs are latents and tiny.
            wave = [prepare(unit) for unit in units[start : start + wave_size]]
            count = len(wave)
            padded = wave + [wave[-1]] * (wave_size - count)
            x_device = ttnn.from_torch(
                torch.cat(padded, dim=0),
                dtype=self.dtype,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            out = ttnn.to_torch(
                encoder(x_device),
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
            ).float()
            for index in range(count):
                results.append(out[index : index + 1, ..., :moments].permute(0, 4, 1, 2, 3).contiguous())
        return results

    def encode_clip(self, x_BCTHW: torch.Tensor, *, temporal_taps: int | None = None) -> torch.Tensor:
        """Encode one temporal clip, spatially tiled -- the reference ``_encode_clip``.

        A keyframe goes through here rather than :meth:`encode`, because a single frame
        must not be put through the temporal chunking.
        """
        if temporal_taps is None:
            temporal_taps = 1 if x_BCTHW.shape[2] == 1 else 3

        if not self.use_tiling:
            return self._run_encoder(x_BCTHW, temporal_taps)

        units = self._clip_tiles(x_BCTHW)
        encoded = self._run_encoder_units(units, temporal_taps)
        return self._stitch_clip(encoded, x_BCTHW.shape[-2], x_BCTHW.shape[-1])

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
        if num_frames % clip_length != 0:
            pad = x_BCTHW[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
            x_BCTHW = torch.cat([x_BCTHW, pad], dim=2)

        height, width = x_BCTHW.shape[-2], x_BCTHW.shape[-1]
        clips = [x_BCTHW[:, :, i * clip_length : (i + 1) * clip_length] for i in range(x_BCTHW.shape[2] // clip_length)]
        if not self.use_tiling:
            moments = torch.cat(self._run_encoder_units(clips, 3), dim=2)
        else:
            per_clip = [self._clip_tiles(clip) for clip in clips]
            tiles_per_clip = len(per_clip[0])
            flat = [unit for clip_units in per_clip for unit in clip_units]
            encoded = self._run_encoder_units(flat, 3)
            moments = torch.cat(
                [
                    self._stitch_clip(encoded[i * tiles_per_clip : (i + 1) * tiles_per_clip], height, width)
                    for i in range(len(clips))
                ],
                dim=2,
            )
        if self.config.token_drop > 0:
            moments = moments[:, :, : -self.config.token_drop]
        return moments

    # ---------------------------------------------------------------- decode

    def load_decoder_state(self, state: dict[str, torch.Tensor]) -> None:
        """Hold the decoder ``state_dict``, folding ``post_quant_conv`` into ``proj_in``.

        The two are adjacent with no nonlinearity, so ``proj_in(post_quant_conv(z))`` is one
        24->2048 linear and the awkward 24-channel 1x1x1 conv disappears.
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
        self._decoder_state = decoder_state

    def _decoder_for(self, num_frames: int, height: int, width: int):
        from .decoder_minimax_h3 import MiniMaxH3ViTDecoder3d

        key = (num_frames, height, width)
        if key not in self._decoders:
            if self._decoder_state is None:
                raise RuntimeError("call load_decoder_state() before decoding")
            decoder = MiniMaxH3ViTDecoder3d(
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
            decoder.load_torch_state_dict(dict(self._decoder_state))
            self._decoders[key] = decoder
        return self._decoders[key]

    def _run_decoder(self, latent_BCTHW: torch.Tensor) -> torch.Tensor:
        return self._run_decoder_units([latent_BCTHW])[0]

    def _run_decoder_units(self, units: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode independent ``(chunk, tile)`` units, one per device, in mesh-sized waves.

        Same argument as :meth:`_run_encoder_units`: the reference decodes each spatial
        tile of each temporal chunk on its own and only cross-fades afterwards, and the ViT
        decoder holds no CCL, so one device per unit is exact SPMD. The temporal blend in
        :meth:`decode` still runs in order on the host -- it is the *decodes* that are
        independent, not the stitching.
        """
        from .decoder_minimax_h3 import unpatchify

        if not units:
            return []
        odd = [tuple(u.shape) for u in units if u.shape != units[0].shape]
        assert not odd, f"units must share a shape; {units[0].shape} vs {odd[0]}"
        _, _, num_frames, height, width = units[0].shape
        decoder = self._decoder_for(num_frames, height, width)
        wave_size = self.mesh_device.get_num_devices()

        results: list[torch.Tensor] = []
        for start in range(0, len(units), wave_size):
            wave = [
                unit.permute(0, 2, 3, 4, 1).reshape(1, num_frames * height * width, -1)
                for unit in units[start : start + wave_size]
            ]
            count = len(wave)
            padded = wave + [wave[-1]] * (wave_size - count)
            tokens = ttnn.from_torch(
                torch.cat(padded, dim=0),
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )
            out = ttnn.to_torch(decoder(tokens), mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
            for index in range(count):
                results.append(
                    unpatchify(
                        out[index : index + 1],
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        out_channels=self.config.out_channels,
                        patch_size=self.config.spatial_compression_ratio,
                        patch_size_t=self.config.temporal_compression_ratio,
                    )
                )
        return results

    def decode_clip(self, z_BCTHW: torch.Tensor) -> torch.Tensor:
        """Decode one temporal clip, spatially tiled -- the reference ``_decode_clip``.

        Tiles are laid out in *pixel* space and mapped back onto the latent grid, so the
        blend extents are pixel-space too (unlike encode, where they are divided down).
        """
        if not self.use_tiling:
            return self._run_decoder(z_BCTHW)

        units = self._latent_tiles(z_BCTHW)
        return self._stitch_decoded(self._run_decoder_units(units), z_BCTHW.shape[-2], z_BCTHW.shape[-1])

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

    def decode(self, z_BCTHW: torch.Tensor) -> torch.Tensor:
        """Decode a latent video, mirroring the chunking ``encode`` applied.

        ``token_drop`` removed the tail of every encoded chunk, so consecutive decoded
        chunks overlap by ``frame_overlap`` pixel frames and are cross-faded. Ported from
        the reference ``_decode``; the trailing repeated latent frames produce pixel frames
        that were never asked for and are cut at the end.
        """
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
        if self.use_tiling and chunk_latents:
            latent_height, latent_width = chunk_latents[0].shape[-2], chunk_latents[0].shape[-1]
            tiles_per_chunk = len(self._latent_tiles(chunk_latents[0]))
            wave_size = self.mesh_device.get_num_devices()
            # Decoded tiles are 22 MB each, so decoding all 308 units of a 768P/5s video
            # before stitching any of them would hold 6.8 GB (and ~29 GB at 1440P/10s).
            # Group chunks into a few waves' worth, stitch, release. Groups are whole chunks
            # so a group never straddles a stitch boundary.
            chunks_per_group = max(1, -(-_DECODE_WAVES_IN_FLIGHT * wave_size // tiles_per_chunk))
            clips = []
            for group_start in range(0, num_chunks, chunks_per_group):
                group = chunk_latents[group_start : group_start + chunks_per_group]
                flat = self._run_decoder_units([unit for latents in group for unit in self._latent_tiles(latents)])
                clips.extend(
                    self._stitch_decoded(
                        flat[i * tiles_per_chunk : (i + 1) * tiles_per_chunk], latent_height, latent_width
                    )
                    for i in range(len(group))
                )
        else:
            clips = self._run_decoder_units(chunk_latents) if chunk_latents else []

        decoded, overlap = [], None
        for i in range(num_chunks):
            clip = clips[i]
            for j in range(int(config.token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames][:, :, config.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = blend(overlap, chunk, config.frame_overlap, dim=-3)
                    decoded.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded.append(overlap)

        result = torch.cat(decoded, dim=2)
        if pad_tokens > 0:
            intra_tail = config.clip_length % temporal_ratio
            tokens_before_pad = z_BCTHW.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (tokens_before_pad + k) % chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            result = result[:, :, :-pad_frames]
        return result

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the per-channel ``latents_mean`` / ``latents_std`` the pipeline expects."""
        mean = torch.tensor(self.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return (latents - mean) / std
