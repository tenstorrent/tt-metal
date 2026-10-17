# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein packed-sequence construction.

Host-side and pure torch: every value here is part of the checkpoint's numerical contract and is
compared bit-exactly against the vendored BFL reference (``models/tt_dit/reference/flux3_klein``),
so nothing in this module may be reassociated, re-typed or moved to device. The reference package
is never imported here; only the tests import it.

Two token orders exist and both are kept straight:

* ``pack()`` (reference ``pipeline.py:115-138``) takes one ``[text, image_cond, image]`` triple per
  sample and stores the tokens **mode-major**: ``data[mode]`` holds that mode's rows of every sample
  back to back, ``mode_indices[mode][s] = (start, end)`` locates sample ``s`` inside it.
* The DiT (reference ``dit.py:11`` and the single-stage concat loop ``dit.py:216-228``) lays one
  sample out as ``[image | image_cond | text]`` -- ``MODES`` order -- and concatenates the samples,
  so attention bounds are per sample over that sequence. This is the device order of the port.

An empty mode (text-to-image has no ``image_cond``) keeps ``(start, start)`` indices; the DiT gives it
one zero *phantom* token so the input projection and per-mode blocks run on a non-empty tensor
(``dit.py:192-198``). The phantom never enters the single-stage sequence (``b > a`` is false) and is
never attended (``packed_attention`` skips ``start == end``).

dtypes are part of the contract: text and image_cond tokens are bf16, the image state is fp32
(bf16-valued noise, ``pipeline.py:87-88``), image and image_cond ids are fp32 (``scaled_ids``), text
ids are int64 (``text_encoder.py:48-49``), and an empty mode gets fp32 ``(0, 4)`` ids and bf16
``(0, width)`` tokens. The DiT promotes the concatenated single-stage ids to fp32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# dit.py:11 -- sorted; the order tokens take inside a sample.
FLUX3_KLEIN_MODES = ("image", "image_cond", "text")
# dit.py:12
FLUX3_KLEIN_IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}
# autoencoder.py:13-14 and :EMBED_DIM: 8x VAE + 2x2 patchify = 16x; 32 VAE channels x 4 = 128 token channels.
FLUX3_KLEIN_DOWNSAMPLING_RATIO = 16
FLUX3_KLEIN_LATENT_CHANNELS = 128
FLUX3_KLEIN_VAE_CHANNELS = 32
FLUX3_KLEIN_PATCH = 2
# pipeline.py:24; scaled_ids (pipeline.py:141-144) scales the (h, w) axes of image and image_cond ids.
FLUX3_KLEIN_IMAGE_ROPE_SCALE = 0.5
# text_encoder.py:10-11: text length rounded up to a multiple of 80, capped at 4096. The padding rows are
# REAL tokens to the DiT (it attends to all of them; there is no text mask), so they are never masked here.
FLUX3_KLEIN_TEXT_PAD_MULTIPLE = 80
FLUX3_KLEIN_MAX_TEXT_LENGTH = 4096
# pipeline.py:30
FLUX3_KLEIN_MAX_REFERENCES = 10
# Tile height. Kept as a plain int so this module stays free of `ttnn`; it equals `ttnn.TILE_SIZE`.
FLUX3_KLEIN_TILE = 32

TokenIds = tuple[Tensor, Tensor]


# --- latent geometry ---------------------------------------------------------------------------------------------


def latent_shape(height: int, width: int) -> tuple[int, int]:
    """``(h, w)`` token grid of a ``height x width`` image (autoencoder.py:188-191)."""
    if height % FLUX3_KLEIN_DOWNSAMPLING_RATIO or width % FLUX3_KLEIN_DOWNSAMPLING_RATIO:
        raise ValueError(f"width and height must be multiples of {FLUX3_KLEIN_DOWNSAMPLING_RATIO}")
    return height // FLUX3_KLEIN_DOWNSAMPLING_RATIO, width // FLUX3_KLEIN_DOWNSAMPLING_RATIO


def latent_ids(h: int, w: int) -> Tensor:
    """Unscaled ``(h*w, 4)`` int64 ids ``(0, row, col, 0)`` in row-major token order (autoencoder.py:182-185).
    This is what the autoencoder's encode/decode take; the DiT takes :func:`scaled_ids` of it."""
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    zeros = torch.zeros_like(rows)
    return torch.stack([zeros, rows, cols, zeros], dim=-1).reshape(h * w, 4)


def scaled_ids(ids: Tensor) -> Tensor:
    """fp32 copy of ``ids`` with the ``(h, w)`` axes scaled by 0.5 (pipeline.py:141-144). Always a copy: the
    reference scales the fp32 view in place, which aliases when the input is already fp32."""
    ids = ids.to(torch.float32, copy=True)
    ids[:, 1:3] *= FLUX3_KLEIN_IMAGE_ROPE_SCALE
    return ids


def image_ids(h: int, w: int) -> Tensor:
    """DiT ids of the generated image: ``(0, 0.5*row, 0.5*col, 0)`` fp32 (pipeline.py:89-90)."""
    return scaled_ids(latent_ids(h, w))


def reference_ids(h: int, w: int, index: int) -> Tensor:
    """DiT ids of reference ``index`` (0-based): scaled like the image, ``l = -(1 + index)`` tells references
    apart (pipeline.py:106-109)."""
    ids = scaled_ids(latent_ids(h, w))
    ids[:, 3] = -(1 + index)
    return ids


def text_ids(length: int) -> Tensor:
    """Text ids ``(0, 0, 0, position)`` int64 (text_encoder.py:48-49)."""
    positions = torch.arange(length)
    return torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)


def padded_text_length(real_length: int) -> int:
    """The encoder's text length: ``real_length`` rounded up to a multiple of 80, capped at 4096
    (text_encoder.py:35). Every row up to this length is a real token to the DiT."""
    return min(
        math.ceil(real_length / FLUX3_KLEIN_TEXT_PAD_MULTIPLE) * FLUX3_KLEIN_TEXT_PAD_MULTIPLE,
        FLUX3_KLEIN_MAX_TEXT_LENGTH,
    )


def latent_grid_from_ids(ids: Tensor) -> tuple[int, int]:
    """``(h, w)`` recovered from unscaled ids the way ``decode`` does (autoencoder.py:175)."""
    return int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1


# --- 2x2 patchify / unpatchify (autoencoder.py:164-178) ----------------------------------------------------------


def patchify_latents(latents: Tensor) -> Tensor:
    """``(32, 2h, 2w)`` VAE latents -> ``(h*w, 128)`` tokens, channel-major within the patch (autoencoder.py:167-170):
    token channel ``c*4 + dy*2 + dx`` is VAE channel ``c`` at patch offset ``(dy, dx)``. The reference applies its
    BatchNorm between the two reshapes; it is per token channel, so it commutes with this function."""
    channels, height, width = latents.shape
    patch = FLUX3_KLEIN_PATCH
    if channels != FLUX3_KLEIN_VAE_CHANNELS or height % patch or width % patch:
        raise ValueError(f"latents of shape {tuple(latents.shape)} are not (32, 2h, 2w)")
    h, w = height // patch, width // patch
    z = latents.reshape(1, channels, h, patch, w, patch).permute(0, 1, 3, 5, 2, 4)
    z = z.reshape(1, FLUX3_KLEIN_LATENT_CHANNELS, h, w)
    return z[0].permute(1, 2, 0).reshape(h * w, FLUX3_KLEIN_LATENT_CHANNELS)


def unpatchify_tokens(tokens: Tensor, h: int, w: int) -> Tensor:
    """Inverse of :func:`patchify_latents`: ``(h*w, 128)`` tokens in id order -> ``(32, 2h, 2w)``
    (autoencoder.py:176-178)."""
    if tokens.shape != (h * w, FLUX3_KLEIN_LATENT_CHANNELS):
        raise ValueError(f"tokens of shape {tuple(tokens.shape)} do not fill a {h}x{w} grid of 128 channels")
    patch = FLUX3_KLEIN_PATCH
    z = tokens.reshape(h, w, FLUX3_KLEIN_LATENT_CHANNELS).permute(2, 0, 1).unsqueeze(0)
    z = z.reshape(1, FLUX3_KLEIN_VAE_CHANNELS, patch, patch, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(FLUX3_KLEIN_VAE_CHANNELS, patch * h, patch * w)


# --- packed layout -----------------------------------------------------------------------------------------------


@dataclass
class Flux3KleinSample:
    """One sample's inputs, each a ``(tokens, ids)`` pair. ``image_cond`` is ``None`` for text-to-image and
    otherwise the concatenation of every reference's tokens and ids (``pipeline.py:102-112``)."""

    text: TokenIds
    image: TokenIds
    image_cond: TokenIds | None = None

    @classmethod
    def from_reference_triple(cls, triple: list[TokenIds | None]) -> Flux3KleinSample:
        """From the reference ``pack()`` argument order ``[text, image_cond, image]``."""
        text, cond, image = triple
        return cls(text=text, image=image, image_cond=cond)

    def to_reference_triple(self) -> list[TokenIds | None]:
        return [self.text, self.image_cond, self.image]

    def item(self, mode: str) -> TokenIds | None:
        return {"image": self.image, "image_cond": self.image_cond, "text": self.text}[mode]


@dataclass(frozen=True)
class Flux3KleinSegment:
    """A run of one mode of one sample. ``(start, end)`` index the single-stage ``[image | image_cond | text]``
    sequence; ``(mode_start, mode_end)`` index the mode-major ``data[mode]``. Empty modes have no segment."""

    sample: int
    mode: str
    start: int
    end: int
    mode_start: int
    mode_end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Flux3KleinPacked:
    """The reference ``Packed`` (mode-major ``data``/``ids``/``timesteps``/``mode_indices``, ``dit.py:28-39``) plus
    the single-stage layout the DiT derives from it (``dit.py:216-228``): ``bounds[s]`` is sample ``s``'s
    ``(start, end)`` in the concatenated sequence and ``segments`` lists the runs in sequence order."""

    data: dict[str, Tensor]
    ids: dict[str, Tensor]
    timesteps: dict[str, Tensor]
    mode_indices: dict[str, list[tuple[int, int]]]
    bounds: list[tuple[int, int]]
    segments: list[Flux3KleinSegment]

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])

    @property
    def sequence_length(self) -> int:
        return self.bounds[-1][1] if self.bounds else 0

    def mode_rows(self, mode: str) -> int:
        return self.data[mode].shape[0]

    @property
    def phantom_modes(self) -> tuple[str, ...]:
        """Modes with no rows at all, which the DiT runs on one zero phantom token (``dit.py:192-198``)."""
        return tuple(m for m in FLUX3_KLEIN_MODES if self.mode_rows(m) == 0)

    def projection_rows(self, mode: str) -> int:
        """Rows the DiT's input projection and per-mode blocks see for ``mode``: the real rows, or the phantom."""
        return max(self.mode_rows(mode), 1)

    def segment(self, sample: int, mode: str) -> Flux3KleinSegment | None:
        for segment in self.segments:
            if segment.sample == sample and segment.mode == mode:
                return segment
        return None

    def sequence_ids(self) -> Tensor:
        """Ids in single-stage order, concatenated the way the DiT does it (``dit.py:230``, fp32 after promotion)."""
        return torch.cat([self.ids[s.mode][s.mode_start : s.mode_end] for s in self.segments])

    def sequence_modes(self) -> list[str]:
        """The mode of every row of the single-stage sequence."""
        return [s.mode for s in self.segments for _ in range(s.length)]


def pack(samples: list[Flux3KleinSample]) -> Flux3KleinPacked:
    """Bit-exact mirror of the reference ``pack()`` (``pipeline.py:115-138``) plus the single-stage layout.

    Mode-major concatenation keeps each input's dtype (``torch.cat`` of one dtype); an empty mode becomes
    bf16 ``(0, width)`` tokens and fp32 ``(0, 4)`` ids; timesteps are bf16 zeros per row.
    """
    if not samples:
        raise ValueError("pack needs at least one sample")
    data = {m: [] for m in FLUX3_KLEIN_MODES}
    ids = {m: [] for m in FLUX3_KLEIN_MODES}
    mode_indices = {m: [] for m in FLUX3_KLEIN_MODES}
    counts = dict.fromkeys(FLUX3_KLEIN_MODES, 0)
    for sample in samples:
        for mode in ("text", "image_cond", "image"):  # the reference's loop order; it does not affect the result
            item = sample.item(mode)
            start = counts[mode]
            if item is not None:
                tokens, item_ids = item
                if tokens.shape[0] != item_ids.shape[0] or item_ids.shape[-1] != 4:
                    raise ValueError(f"{mode}: tokens {tuple(tokens.shape)} and ids {tuple(item_ids.shape)} disagree")
                if tokens.shape[1] != FLUX3_KLEIN_IN_CHANNELS[mode]:
                    raise ValueError(
                        f"{mode}: expected {FLUX3_KLEIN_IN_CHANNELS[mode]} channels, got {tokens.shape[1]}"
                    )
                data[mode].append(tokens)
                ids[mode].append(item_ids)
                counts[mode] += tokens.shape[0]
            mode_indices[mode].append((start, counts[mode]))
    device = samples[0].text[0].device
    cat = lambda parts, width, dtype: torch.cat(parts) if parts else torch.zeros((0, width), device=device, dtype=dtype)
    data = {m: cat(data[m], FLUX3_KLEIN_IN_CHANNELS[m], torch.bfloat16) for m in FLUX3_KLEIN_MODES}
    ids = {m: cat(ids[m], 4, torch.float32) for m in FLUX3_KLEIN_MODES}
    timesteps = {m: torch.zeros(data[m].shape[0], device=device, dtype=torch.bfloat16) for m in FLUX3_KLEIN_MODES}
    bounds, segments = single_stage_layout(mode_indices)
    return Flux3KleinPacked(
        data=data, ids=ids, timesteps=timesteps, mode_indices=mode_indices, bounds=bounds, segments=segments
    )


def single_stage_layout(
    mode_indices: dict[str, list[tuple[int, int]]]
) -> tuple[list[tuple[int, int]], list[Flux3KleinSegment]]:
    """The DiT's concatenation order (``dit.py:216-228``): for each sample, its non-empty modes in ``MODES`` order
    ``[image | image_cond | text]``. Returns per-sample ``bounds`` and the ``segments`` in sequence order."""
    num_samples = len(mode_indices["image"])
    bounds, segments, offset = [], [], 0
    for s in range(num_samples):
        start = offset
        for m in FLUX3_KLEIN_MODES:
            a, b = mode_indices[m][s]
            if b > a:
                segments.append(
                    Flux3KleinSegment(sample=s, mode=m, start=offset, end=offset + b - a, mode_start=a, mode_end=b)
                )
                offset += b - a
        bounds.append((start, offset))
    return bounds, segments


def with_phantom_tokens(packed: Flux3KleinPacked) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """``(data, ids, timesteps)`` as the DiT sees them after the phantom rule (``dit.py:191-198``): an empty mode gets
    one zero row of its own dtype (bf16 tokens, fp32 ids, bf16 timestep). ``mode_indices`` are NOT changed, so the
    phantom is never attended and never enters the single-stage sequence."""
    data, ids, timesteps = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
    for m in FLUX3_KLEIN_MODES:
        if data[m].shape[0] == 0:
            data[m], ids[m], timesteps[m] = (
                data[m].new_zeros(1, data[m].shape[1]),
                ids[m].new_zeros(1, 4),
                timesteps[m].new_zeros(1),
            )
    return data, ids, timesteps


# --- tile padding (device layout only; nothing here changes what the model computes) ------------------------------


def padded_length(length: int, sp_factor: int = 1, tile: int = FLUX3_KLEIN_TILE) -> int:
    """``length`` rounded up to a multiple of ``sp_factor * tile`` so a segment fractures evenly over SP into
    tile-aligned rows per device. A segment of 0 rows stays 0 (the phantom is not a device row)."""
    if sp_factor < 1 or tile < 1:
        raise ValueError(f"sp_factor and tile must be positive, got {sp_factor}, {tile}")
    alignment = sp_factor * tile
    return ((length + alignment - 1) // alignment) * alignment


def pad_rows(rows: Tensor, length: int) -> tuple[Tensor, Tensor]:
    """Zero-pad ``rows`` along dim 0 to ``length``. Returns ``(padded, mask)`` with ``mask[i]`` True for every
    input row and False for the appended ones. The input rows are all real: text that the encoder already
    padded to a multiple of 80 arrives here as real rows and keeps ``True``, because the reference DiT attends
    to those tokens too."""
    if length < rows.shape[0]:
        raise ValueError(f"cannot pad {rows.shape[0]} rows down to {length}")
    pad = length - rows.shape[0]
    mask = torch.cat([torch.ones(rows.shape[0], dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
    if pad == 0:
        return rows, mask
    padding = rows.new_zeros((pad, *rows.shape[1:]))
    return torch.cat([rows, padding]), mask


def pad_segment(rows: Tensor, sp_factor: int = 1, tile: int = FLUX3_KLEIN_TILE) -> tuple[Tensor, Tensor]:
    """:func:`pad_rows` to :func:`padded_length` of the segment."""
    return pad_rows(rows, padded_length(rows.shape[0], sp_factor, tile))
