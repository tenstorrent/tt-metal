# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va``: the omni-reference half of the packed sequence.

Host-side and pure torch/numpy, for the same reason as
:mod:`packing`: every value here is part of the checkpoint's numerical contract
and is compared bit-exactly against the reference, so nothing may be
reassociated or moved to device.

A ``ref2va`` request carries an **ordered** list of references and packs one
block per reference ahead of the generated rows::

    [ text (L) | ref block 1 | ref block 2 | ... | target audio (A) | target video (V) ]

The order is semantic **twice**: it fixes the ``"<Picture i>"`` / ``"<Audio j>"``
/ ``"<Video k>"`` labels of the prompt presentation, *and* it advances the shared
audio/video rotary clock. Reordering the same references is a different request.

Two properties separate this from ``fl2va``:

* **A reference never binds the target geometry.** Every reference is prepared at its
  own resolution -- 2048 px short edge for an image with no area cap, the 768 px canvas
  of its own aspect ratio for a video -- with its own aspect-normalized spatial grid. One
  2048x2048 reference contributes 4096 vision tokens to the text stream *and* 4096 video
  condition rows, so a ref2va packed sequence runs 1.2x-3.0x t2va's.
* **A video reference packs its soundtrack rows immediately before its own video rows**,
  sharing one rotary origin, as the generated audio and video do.

Audio reference rows are clean: posterior mean, no float16 round trip, no noise
augmentation, and a literal ``t = 1.0`` at every step.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from .packing import (
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    _spatial_position_grid,
    _temporal_position_grid,
    resolve_canvas_size,
)

# A reference image is resized to a 2048 px short edge -- upscaling included -- and
# both axes rounded to a multiple of 32 independently. There is NO area cap, so a
# 4:1 reference is 8192x2048, i.e. 65536 patches to the conditioner.
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048

# The conditioner sees a reference video at 2 fps and Qwen3-VL merges every two of
# those frames into one vision block, labelled with the mean timestamp of the pair.
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2

# Documented per-request limits of the omni-reference task.
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_MAX_REFERENCES = 12


@dataclass
class MiniMaxH3Reference:
    """One omni-reference of a ``ref2va`` request: an image, a video, or audio.

    A reference carries exactly one medium -- plus, for a video, the ``audio`` of
    its own soundtrack, which is conditioned on as that reference's own. The list
    is passed **in the order the model should read them**.

    Unlike the reference implementation's dataclass this one does not decode
    paths: the pipeline is handed in-memory media, and keeping file I/O out of
    here is what lets the whole layout be gated without PyAV. Use
    :func:`decode_reference_video` / :func:`decode_reference_audio` to get there.

    ``fps`` defaults to H3's own 24 and ``sample_rate`` to the audio VAE's own, in
    both cases meaning "already at the model's rate, pass the samples through
    untouched".
    """

    image: Image.Image | np.ndarray | torch.Tensor | None = None
    video: list | np.ndarray | torch.Tensor | None = None
    fps: float | None = None
    audio: torch.Tensor | None = None
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        # A video reference conditions on its soundtrack too, so `audio` is a
        # second medium of a video reference rather than a conflicting one.
        media = [name for name in ("image", "video", "audio") if getattr(self, name) is not None]
        if media not in (["image"], ["video"], ["audio"], ["video", "audio"]):
            raise ValueError(
                "a MiniMaxH3Reference must carry exactly one of image, video or audio -- plus, for a "
                f"video, the audio of its soundtrack -- got {media if media else 'none of them'}"
            )
        for name in ("image", "video", "audio"):
            if isinstance(getattr(self, name), (str, os.PathLike)):
                raise ValueError(
                    f"{name} is a path; this module never opens media files. Decode it first with "
                    "decode_reference_video / decode_reference_audio, or pass in-memory media."
                )
        if self.fps is None:
            self.fps = float(MINIMAX_H3_FPS)

    @property
    def kind(self) -> str:
        """The modality this reference is packed as: ``image``, ``video`` or ``audio``."""
        if self.image is not None:
            return "image"
        return "video" if self.video is not None else "audio"

    @property
    def has_audio(self) -> bool:
        """Whether this reference contributes audio rows, i.e. carries a waveform."""
        return self.audio is not None


@dataclass
class MiniMaxH3PreparedReference:
    """One reference prepared for packing, in packed order.

    Resolved in three passes, which is why so many fields default: the request
    fixes ``kind`` / ``has_audio``, preparation fills ``image`` / ``frames`` /
    ``waveform``, and the VAE encode fixes the latent geometry the packed layout
    is built from.
    """

    kind: str
    has_audio: bool = False
    image: Any = None
    frames: Any = None
    waveform: torch.Tensor | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    @property
    def num_video_rows(self) -> int:
        """Packed video rows, for the ``(1, 2, 2)`` patch H3 packs video latents with."""
        return self.num_latent_frames * (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_audio_rows(self) -> int:
        """Packed audio rows: one per latent per stereo channel."""
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def reference_kind(index: int, entry: Any) -> str:
    """The modality of one ``references`` entry, validated at construction."""
    if not isinstance(entry, MiniMaxH3Reference):
        raise ValueError(
            f"references[{index}] must be a MiniMaxH3Reference, got {type(entry)}. Build one with "
            "MiniMaxH3Reference(image=...), (video=...) or (audio=...)."
        )
    return entry.kind


def _temporal_position_span(num_latent_frames: int) -> float:
    """Rotary time a video reference advances the shared clock by.

    Summed sequentially in float64, which is *not* how
    :func:`packing._temporal_position_span` sums the same series -- that one reproduces a
    numpy pairwise sum. The two differ in the last ulp from 16 latent frames on, and by
    ~2 ulp in the opposite direction at the production 37:

        n=16  pairwise 86.66666666666667   sequential 86.66666666666669
        n=37  pairwise 206.66666666666663  sequential 206.66666666666657

    The reference keeps both, one per call site. A single shared implementation passes
    below 16 frames and is wrong at the working point, which a video reference truncated
    to the target frame count always reaches.
    """
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(h, w)`` rotary coordinates of one latent frame, and the width axis they came from.

    The width axis is returned because audio rows pin to *its* two extremes, and
    for a soundtrack that is the **video's own** grid rather than the target's.
    """
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    """Place one channel-major audio block.

    Audio rows carry no height coordinate and pin to the two extremes of the width
    grid of *their own* block -- the target grid for a standalone audio reference,
    the video's grid for a soundtrack.
    """
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    """Build the ``[text | reference blocks | target audio | target video]`` layout.

    ``references`` are in packed order with their latent geometry already
    resolved. ``text_token_tags`` tags text 1 except the rows of a reference's
    vision block, which H3 tags 0 (video).
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_reference_video_rows = sum(reference.num_video_rows for reference in references if reference.kind != "audio")
    num_reference_audio_rows = sum(reference.num_audio_rows for reference in references)
    sequence_length = (
        num_text_tokens
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Reference blocks, in request order. `rotary_time` is the shared audio/video
    # clock: it starts where the text rows end and every block pushes it forward by
    # the time that block occupies.
    video_indices, audio_indices = [], []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            rows = slice(cursor, cursor + reference.num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(reference.latent_height, reference.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            # An image takes a single INTEGER rotary slot, not a latent frame's 5/3 units.
            rotary_time += 1.0
        elif reference.kind == "audio":
            rows = slice(cursor, cursor + reference.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            # A standalone audio reference has no spatial grid of its own, so it
            # pins to the TARGET's width extremes.
            _fill_audio_positions(position_ids, rows, reference.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(reference.num_audio_latents)
        elif reference.kind == "video":
            # The soundtrack rows are packed immediately BEFORE the video rows and
            # share their origin, so the two are rotary-aligned exactly as the
            # generated audio and video are.
            audio_rows = slice(cursor, cursor + reference.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + reference.num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(
                reference.latent_height, reference.latent_width, patch_h, patch_w
            )
            _fill_audio_positions(position_ids, audio_rows, reference.num_audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(reference.num_latent_frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(reference.num_latent_frames, 1)
            # The longer of its two spans. `_temporal_position_span` here is the
            # SEQUENTIAL one; see its docstring.
            rotary_time += max(float(reference.num_audio_latents), _temporal_position_span(reference.num_latent_frames))
        else:
            raise ValueError(f"a reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}")

    # The generated rows. Target audio and target video share the origin the
    # reference blocks left behind.
    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
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
        num_condition_video_rows=num_reference_video_rows,
        num_condition_audio_rows=num_reference_audio_rows,
    )


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    """``(height, width)`` a reference image is encoded at: 2048 px short edge, axes to a multiple of 32.

    Upscaling is intended and, unlike the target canvas, there is **no area cap** --
    so a 4:1 reference is 8192x2048.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"a reference image must have a positive size, got {width}x{height}")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"a reference image must be within 1:4 and 4:1, got {width}x{height}")

    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def reference_media_to_uint8(media) -> np.ndarray:
    """An in-memory reference image or video as channels-last ``uint8`` RGB.

    A ``torch.Tensor`` is channels-first, as everywhere else in diffusers, and a
    ``np.ndarray`` channels-last; floating point is read over ``[0, 1]``.
    """
    if isinstance(media, list):
        return np.stack([reference_media_to_uint8(item) for item in media])
    if isinstance(media, Image.Image):
        return np.asarray(media.convert("RGB"))
    if isinstance(media, torch.Tensor):
        media = media.movedim(-3, -1).cpu().numpy()
    media = np.asarray(media)
    if media.dtype != np.uint8:
        media = (media * 255.0).round().clip(0, 255).astype(np.uint8)
    return media


def prepare_reference_image(image: Image.Image, height: int, width: int) -> Image.Image:
    """LANCZOS-resize a reference image onto its own resolution; an exact match skips resampling."""
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    """Resample a reference video onto H3's own 24 fps, dropping and duplicating whole frames.

    The reference implementation decoded every video reference through ``ffmpeg``'s
    ``fps`` filter, a constant-frame-rate resampler: every source frame lands on
    the output slot its timestamp rounds to, ``round(index * 24 / fps)``, and a slot
    holds the last frame that landed on it -- so a frame whose successor lands on
    the same slot is dropped and one whose successor skips slots is repeated. The
    end of the stream rounds onto the grid the same way, which is what fixes the
    length at ``round(num_frames * 24 / fps)`` slots.

    An exact identity -- the same array, no copy -- for frames already at 24 fps,
    which is the parity-exact route.
    """
    if fps <= 0:
        raise ValueError(f"a reference video must have a positive frame rate, got {fps}")
    if fps == MINIMAX_H3_FPS:
        return frames

    scale = MINIMAX_H3_FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    # Every frame is held until the slot of the next one, the last until the slot
    # the stream's end rounds to.
    return np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)


def prepare_reference_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Put a reference video on the canvas its OWN aspect ratio resolves to, capped at ``num_frames``.

    Frames already at that canvas flow through untouched, with no resampling pass
    and no copy, and that is the parity-exact route: the reference rescaled with
    ``ffmpeg``'s own LANCZOS scaler while decoding, so only frames decoded at the
    canvas reproduce its pixels bit for bit. Any other size is resized frame by
    frame with PIL.
    """
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"a reference video must be (num_frames, height, width, 3) RGB, got {tuple(frames.shape)}")
    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """The frames the conditioner sees, and one timestamp per vision block.

    The conditioner reads a reference at 2 fps: every 12th of the 24 fps frames,
    deduplicated. Qwen3-VL then merges the sampled frames in pairs -- repeating the
    last when there is an odd number -- and a merged pair is labelled with the
    **mean** of its two timestamps, which ``"{:.1f}"`` renders with round-half-to-even,
    so the first block of a 2 fps pair is ``"<0.2 seconds>"`` and not ``"<0.3 seconds>"``.
    """
    stride = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices, cursor = [], 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride

    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    ]
    return [frames[index] for index in indices], block_timestamps


def prepare_reference_waveform(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int, max_duration: float
) -> torch.Tensor:
    """Put a reference soundtrack on the audio VAE's rate, as a stereo ``(2, samples)`` waveform.

    The reference truncates at the **native** rate and resamples once, in torch,
    which this mirrors -- truncating after resampling would land on a different
    sample. A mono waveform is upmixed by repeating its channel.
    """
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, MINIMAX_H3_AUDIO_CHANNELS):
        raise ValueError(
            f"a reference soundtrack must be a (channels, num_samples) mono or stereo waveform, got "
            f"{tuple(waveform.shape)}"
        )
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != MINIMAX_H3_AUDIO_CHANNELS:
        waveform = waveform.expand(MINIMAX_H3_AUDIO_CHANNELS, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform

    import torchaudio

    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def build_ref2va_presentation(
    tokenizer,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    """Tokenize H3's presentation of a ``ref2va`` request.

    Every reference prepends a label, in packed order and numbered **per
    modality**: ``"<Picture i>: "`` plus a vision block for an image,
    ``"<Audio j>: "`` *alone* for audio -- a waveform never reaches the conditioner
    -- and ``"<Video k>: "`` plus one timestamped vision block per merged frame
    pair for a video. A video that carries sound is labelled ``"<Audio j>: "``
    **before** ``"<Video k>: "``, mirroring the order its rows are packed in. The
    prompt follows verbatim, with no chat template and no special tokens.

    Returns ``(token_ids, token_tags)``. A vision block is tagged 0 (video) *in
    full*, including its ``<|vision_start|>`` / ``<|vision_end|>`` sentinels, and
    everything else 1 (text) -- that tag is what the DiT's AdaLN keys off, and it
    is a different tagging from Qwen3-VL's own ``mm_token_type_ids``, which marks
    only the pad run.
    """

    def text(value: str) -> tuple[list[int], list[int]]:
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [MINIMAX_H3_TEXT_TAG] * len(token_ids)

    def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
        token_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return token_ids, [MINIMAX_H3_VIDEO_TAG] * len(token_ids)

    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in reference.block_timestamps:
                # "{:.1f}" rounds half to even, so the mean of a 2 fps pair renders "<0.2 seconds>".
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
    emit(text(prompt))
    return token_ids, token_tags


def trim_reference_num_frames(num_frames: int) -> int:
    """Snap a reference video's frame count **down** to a ``17n + 5`` the video VAE encodes unpadded.

    A reference is truncated to the target's own frame count, which already is of
    that form, so this only bites when the reference is *shorter* than the video
    being generated.
    """
    if num_frames < 1:
        raise ValueError(f"a reference video must have at least one frame, got {num_frames}")
    return (
        max(1, (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK) * MINIMAX_H3_FRAMES_PER_CHUNK
        + MINIMAX_H3_LATENTS_PER_CHUNK
    )


# ------------------------------------------------------------------ media decode

# PyAV-backed decode, kept at the bottom because nothing above it needs PyAV and
# the whole layout is gated without touching this half.


def _import_av():
    try:
        import av
    except ImportError as error:
        raise ImportError(
            "decoding a MiniMax-H3 reference from a file needs PyAV (`pip install av`), or pass the "
            "decoded media itself: frames plus their fps for a video, a (channels, num_samples) "
            "waveform plus its sample rate for audio"
        ) from error
    return av


def _decode_soundtrack(av, container, stream) -> tuple[torch.Tensor, int]:
    """An audio stream as a ``(channels, num_samples)`` float32 waveform at the container's own rate."""
    sample_rate = int(stream.codec_context.sample_rate)
    # Planar float is a format conversion only: the rate and the channel layout
    # stay the container's own, and a mono soundtrack is upmixed later by
    # `prepare_reference_waveform`.
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(frame)]
    # Whatever the resampler is still holding.
    chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(None)]
    return torch.cat(chunks, dim=-1).to(torch.float32), sample_rate


def decode_reference_video(path: str | os.PathLike) -> tuple[np.ndarray, float, tuple[torch.Tensor, int] | None]:
    """Decode a video file into ``uint8`` RGB frames, its frame rate, and its soundtrack.

    The display-matrix rotation is undone, snapped to the nearest quarter turn,
    which is what ``ffmpeg`` does to display a frame upright. A non-square pixel
    aspect ratio is left alone: the reference resolved a canvas from *display*
    geometry, so a stream carrying a sample aspect ratio conditions at the wrong
    shape, and correcting it here would be untested guesswork.
    """
    av = _import_av()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames, rotation = [], 0.0
        for frame in container.decode(stream):
            # `VideoFrame.rotation` arrived in PyAV 14.1; older releases decode
            # fine, they just can't report a display matrix, so stay upright.
            rotation = getattr(frame, "rotation", 0.0)
            frames.append(frame.to_ndarray(format="rgb24"))
        reported_rate = stream.average_rate or stream.guessed_rate
        if reported_rate is None:
            raise ValueError(
                f"the video stream in {path} reports no frame rate; decode it yourself and pass the frames plus their fps"
            )
        frame_rate = float(reported_rate)
        soundtrack = None
        if container.streams.audio:
            # Decoding the frames drained the container, so the soundtrack needs a second pass.
            container.seek(0)
            soundtrack = _decode_soundtrack(av, container, container.streams.audio[0])

    if not frames:
        raise ValueError(f"no video frames to decode in {path}")
    stacked = np.stack(frames)
    turns = round(rotation / 90.0) % 4
    if turns:
        stacked = np.ascontiguousarray(np.rot90(stacked, k=-turns, axes=(1, 2)))
    return stacked, frame_rate, soundtrack


def decode_reference_audio(path: str | os.PathLike) -> tuple[torch.Tensor, int]:
    """Decode an audio file (or a video's soundtrack) into a waveform at its own sample rate."""
    av = _import_av()
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream to decode in {path}")
        return _decode_soundtrack(av, container, container.streams.audio[0])


def reference_from_video_file(path: str | os.PathLike, *, with_audio: bool = True) -> MiniMaxH3Reference:
    """A video :class:`MiniMaxH3Reference` from a file, soundtrack included unless declined.

    Conditioning on a file's motion **alone** means ``with_audio=False``; the
    reference implementation's dataclass brings the soundtrack along and there is
    no way to decline it short of decoding the frames separately.
    """
    frames, frame_rate, soundtrack = decode_reference_video(path)
    if soundtrack is None or not with_audio:
        return MiniMaxH3Reference(video=frames, fps=frame_rate)
    waveform, sample_rate = soundtrack
    return MiniMaxH3Reference(video=frames, fps=frame_rate, audio=waveform, sample_rate=sample_rate)
