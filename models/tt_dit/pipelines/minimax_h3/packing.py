# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 packed-sequence construction.

Host-side and pure torch/numpy: every value here is part of the
checkpoint's numerical contract and is compared bit-exactly against the
reference, so nothing in this module may be reassociated or moved to device.

One H3 forward denoises a single sequence holding four modalities at once::

    [ text (L) | keyframe cond (C) | target audio (A) | target video (V) ]
      tag=1/0      tag=0              tag=2              tag=0

Text rows carry tag 1 except the rows of a keyframe's VLM vision block, which
H3 tags as video -- hence ``text_token_tags`` is an input rather than a
constant fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2

# AdaLN keeps one parameter set per (timestep, modality) pair, so a row's
# modulation index is `timestep_index * 3 + tag`.
MINIMAX_H3_MODALITY_NUM = 3

MINIMAX_H3_FPS = 24
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4
MINIMAX_H3_MIN_DURATION = 5.0
MINIMAX_H3_MAX_DURATION = 15.0

# The video VAE encodes 17 pixel frames per chunk and drops the 3 trailing
# latent frames of each, so `17n + 5` pixel frames give `5n + 2` latent frames.
MINIMAX_H3_FRAMES_PER_CHUNK = 17
MINIMAX_H3_LATENTS_PER_CHUNK = 5

MINIMAX_H3_AUDIO_LATENTS_PER_SECOND = 40
MINIMAX_H3_AUDIO_CHANNELS = 2

MINIMAX_H3_KEYFRAME_NOISE_AUG = 0.999
MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42

MINIMAX_H3_TEXT_ENCODER_LAYER = 50

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass
class MiniMaxH3PackedSequence:
    """Structural description of one packed H3 sequence.

    ``video_indices`` lists the conditioning rows first and then the target
    rows, so ``video_indices[:num_condition_video_rows]`` are exactly the rows
    the update mask suppresses.
    """

    sequence_length: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve a display aspect ratio into an H3 canvas ``(height, width)``.

    The short edge starts at 768 and the area is capped at ``768 * 1344``, but
    both axes are rounded to a multiple of 32 *after* the cap, so the result may
    sit slightly above that budget. Only the ratio of the arguments matters, so
    a keyframe's own dimensions can be passed directly.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_width}:{aspect_height}")

    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(f"H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height} ({ratio:g})")

    if ratio >= 1.0:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio

    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next ``17n + 5`` the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    """Latent frame count for an aligned ``17n + 5`` frame count, i.e. ``5n + 2``."""
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form 17n + 5, got {num_frames}")
    chunks = (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK
    return chunks * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """Audio latents covering ``num_frames`` video frames, on the 40 Hz grid."""
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def prepare_keyframe_image(image: Image.Image, height: int, width: int, stretch: bool) -> Image.Image:
    """Put a keyframe onto the target canvas.

    The first keyframe is the geometry anchor and is *stretched*; any later one
    follows that canvas and is cover-cropped. An image already at canvas size
    skips resampling entirely.
    """
    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)

    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    return image.resize(resized_size, Image.Resampling.LANCZOS).crop((left, top, left + width, top + height))


def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Pack ``(B, C, T, H, W)`` latents into frame-major transformer rows."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"latents of shape {tuple(latents.shape)} are not divisible by patch {patch_size}")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Inverse of :func:`patchify_video_latents`."""
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_tokens(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """Channel-major audio rows to ``(2, C, T)``, one batch item per stereo channel.

    H3's audio VAE is mono and shares one encoder/decoder across channels, so
    stereo reaches it as a batch of two.
    """
    rows = rows.reshape(MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return rows.permute(0, 2, 1).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis, scaled by 32, right endpoint excluded."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # numpy's endpoint=False linspace is `start + arange(num) * (stop - start) / num`,
    # which torch.linspace does not reproduce; the float64 grid must match exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of each latent frame. Spacing is non-uniform: ``5/3 * (1, 4, 4, 4, 4)``."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _temporal_position_span(num_latent_frames: int) -> float:
    """Rotary time spanned by ``num_latent_frames`` latent frames.

    Summed by numpy (pairwise) rather than sequentially: the reference computes
    the keyframe anchor this way and the two orders differ in the last ulp from
    16 latent frames onwards.
    """
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for index in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[index :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[index]
    return float(spans.sum())


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    """Build the ``t2va`` / ``fl2va`` layout.

    ``keyframe_anchors`` holds one ``"first"`` or ``"last"`` per conditioning
    block, in packed order.
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # Text rows occupy the time axis at their own row index and the media rows
    # continue from there, so prompt length shifts the whole media clock.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    frame_grid = torch.stack([g.reshape(-1) for g in torch.meshgrid(height_grid, width_grid, indexing="ij")], -1)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = float(num_text_tokens) + _temporal_position_span(num_latent_frames) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"a keyframe anchor must be 'first' or 'last', got {anchor!r}")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video rotary clock: one unit
    # per latent at 40 latents/s equals 24 fps * 5/3. They carry no height
    # coordinate and pin to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def build_rope_tables(
    position_ids: torch.Tensor, *, rope_freq_dim: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary cos/sin for every row of a packed sequence, from its ``(t, h, w)`` grid.

    One ``inv_freq`` of ``rope_freq_dim`` frequencies is shared by the three axes; each axis
    contributes its own block, the three are concatenated to ``3 * rope_freq_dim`` and then
    concatenated with themselves so ``rotate_half`` rotates ``2 * 3 * rope_freq_dim`` of each
    head's channels and passes the rest through.

    ``rope_freq_dim`` and ``rope_theta`` are the transformer's, read from its ``config.json``
    (16 and 10000.0 for the released checkpoint). ``rope_theta`` here is unrelated to the video
    VAE decoder's 100.0 and to the text encoder's 5e6.

    A mirror of the reference ``MiniMaxH3RotaryPosEmbed``, op for op and in its order, so the
    result is bit-exact against it -- pinned by ``test_rope_tables_match_reference``. The
    ``float64`` position grid is cast to ``float32`` first, as the reference does; casting later
    would change the last ulp of every angle. The head-width interleaved relayout the fused RoPE
    op wants is a separate step, ``attention_minimax_h3.prepare_rope_tables``.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)))
    position_ids = position_ids.to(torch.float32)
    freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
    freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row timesteps, reduced to the transformer's ``(timesteps, indices)`` pair.

    One forward serves rows at different noise levels: generated video and audio
    step their own schedules while conditioning rows stay pinned at their
    noise-augmentation level. Text rows never reach an output head and inherit
    the video timestep.
    """
    row_timesteps = torch.full((layout.sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = condition_video_timestep
    row_timesteps[layout.audio_indices[layout.num_condition_audio_rows :]] = audio_timestep
    row_timesteps[layout.audio_indices[: layout.num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


# Roles a resident-AdaLN row can take, in canonical slot order. Each maps to one noise level per
# step, and a row's role never changes across a request -- so, unlike the precomputed path's
# per-step `torch.unique`, the row->slot map is built once.
MINIMAX_H3_ADALN_ROLES = ("video", "audio", "condition_video", "condition_audio")


def build_slot_routing(layout: MiniMaxH3PackedSequence) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Fixed per-row AdaLN slot assignment for the resident (non-precomputed) path.

    A row's noise level is fixed by its role -- generated video and text at the video level,
    generated audio at the audio level, conditioning rows pinned at their augmentation level -- so
    the row->slot map is constant for the whole request. This is what replaces the precomputed
    path's per-step ``build_row_timesteps`` + ``torch.unique``, whose deduplicated level count
    varies step to step (the conditioning floor collides with the video level early in the schedule
    and separates later) and so cannot be a traced input. Roles with no rows are dropped, so a
    request carries the minimum fixed slot count: two for ``t2va``, three for ``fl2va`` and four for
    ``ref2va``.

    Returns ``(row_slot, roles)``: ``row_slot[r]`` is row ``r``'s slot index and ``roles`` names
    each slot in order, so :func:`slot_levels` builds the matching per-step level vector.
    """
    num_cond_video = layout.num_condition_video_rows
    num_cond_audio = layout.num_condition_audio_rows
    # Video and audio always carry generated rows; the two conditioning slots exist only when the
    # layout has rows to fill them.
    present = {
        "video": True,
        "audio": True,
        "condition_video": num_cond_video > 0,
        "condition_audio": num_cond_audio > 0,
    }
    roles = tuple(role for role in MINIMAX_H3_ADALN_ROLES if present[role])
    slot = {role: index for index, role in enumerate(roles)}

    # The same row->role assignment `build_row_timesteps` makes, recording the slot index rather
    # than the level value: default (text + generated video) at the video slot, then override the
    # conditioning and generated-audio spans.
    row_slot = torch.full((layout.sequence_length,), slot["video"], dtype=torch.long)
    if num_cond_video:
        row_slot[layout.video_indices[:num_cond_video]] = slot["condition_video"]
    row_slot[layout.audio_indices[num_cond_audio:]] = slot["audio"]
    if num_cond_audio:
        row_slot[layout.audio_indices[:num_cond_audio]] = slot["condition_audio"]
    return row_slot, roles


def slot_levels(
    roles: tuple[str, ...],
    *,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> torch.Tensor:
    """The per-step noise level of each slot, ordered to match :func:`build_slot_routing`'s roles.

    Fixed length (``len(roles)``) for the whole request -- no dedup -- so the modulation table the
    blocks project from has a constant shape and the step is traceable. Two slots may hold equal
    levels (the conditioning floor equals the video level early in the schedule); they are kept
    distinct rather than merged, which is precisely what fixes the shape.
    """
    values = {
        "video": video_timestep,
        "audio": audio_timestep,
        "condition_video": condition_video_timestep,
        "condition_audio": condition_audio_timestep,
    }
    return torch.tensor([values[role] for role in roles], dtype=torch.float32)


def adaln_indices(token_tags: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
    """Row to AdaLN table row.

    ``clamp(min=0)`` mirrors the reference, where padding rows carry tag -1 and
    their modulation is irrelevant because their output is discarded.
    """
    return timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)


def adaln_index_ranges(indices: torch.Tensor) -> list[tuple[int, int, int]]:
    """Compress a row-to-AdaLN-table map into ``(start, stop, table_row)`` runs.

    The packed layout is contiguous by modality and the timestep map is
    piecewise-constant over those same blocks, so this collapses to a handful of
    runs -- which is what lets the device apply modulation as slice plus
    broadcast instead of a gather. Returns runs over the rows given, so callers
    pass their own sequence-parallel shard and get shard-local ranges.
    """
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1-D, got shape {tuple(indices.shape)}")
    if indices.numel() == 0:
        return []

    values = indices.to(torch.long)
    boundaries = torch.nonzero(values[1:] != values[:-1], as_tuple=False).reshape(-1) + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long), boundaries])
    stops = torch.cat([boundaries, torch.tensor([values.numel()], dtype=torch.long)])
    return [(int(a), int(b), int(values[a])) for a, b in zip(starts, stops)]
