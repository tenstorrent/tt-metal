# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` reference conditioning: request to packed condition rows.

The two passes between a request and the packed layout, mirroring the reference's
``MiniMaxH3Ref2VASetupStep.prepare_references`` and
``MiniMaxH3Ref2VAReferenceEncoderStep.encode_references``:

1. :func:`prepare_references` -- validate the request and put every reference on
   **its own** resolution. An image goes to a 2048 px short edge, a video onto the
   768 px canvas of its own aspect ratio at 24 fps truncated to the generated frame
   count, a soundtrack onto the audio VAE's sample rate truncated to the generated
   duration. None of this touches the target canvas.
2. :func:`encode_references` -- encode each one and, in doing so, **resolve the
   latent geometry the packed layout is built from**. This is why the encode has to
   run before the layout, unlike ``fl2va`` where a keyframe's geometry is the
   target's by construction.

Three recipes, one per modality:

============  ==========================  ====================================
modality      posterior                   then
============  ==========================  ====================================
image         **sampled**, seed 42        fp16 round trip, video normalization
video         **sampled**, seed 42        fp16 round trip, video normalization
soundtrack    **mean** (``mode()``)       audio normalization, channel-major
============  ==========================  ====================================

The visual rows are then noise-augmented to ``t = 0.999``; the audio rows are **not**,
and run at a literal ``t = 1.0`` for every denoising step. Getting that
backwards produces a plausible soundtrack that ignores its reference, and no downstream
check would catch it.

The encoders are injected as callables, as in :mod:`conditioning`, so the whole
module is host-testable and the device VAEs plug straight in.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from .conditioning import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD, keyframe_condition_rows, sample_posterior
from .packing import MINIMAX_H3_FPS, MINIMAX_H3_MAX_DURATION, MINIMAX_H3_MIN_DURATION, align_num_frames
from .packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    reference_kind,
    reference_media_to_uint8,
    resample_reference_frames,
    resolve_reference_image_size,
    trim_reference_num_frames,
)

# The audio VAE hops 800 samples at 32 kHz. Its `encode` right-pads the waveform up
# to a whole hop with ZEROS, which our device encoder does not do -- it asserts
# divisibility instead -- so the padding happens here, on host.
MINIMAX_H3_AUDIO_HOP = 800


def check_references(references: Sequence[MiniMaxH3Reference]) -> list[str]:
    """Validate a ``references`` list against the documented per-request limits.

    Returns the modality of each entry, in request order. The "audio cannot be
    alone" rule is the reference's: an audio reference has to be paired with at
    least one image or video, because a ref2va request with no visual reference has
    nothing to condition the picture on.
    """
    if not references:
        raise ValueError("ref2va needs at least one reference; use a t2va call for text-only requests")

    kinds = [reference_kind(index, entry) for index, entry in enumerate(references)]
    for kind, limit in (
        ("image", MINIMAX_H3_MAX_REFERENCE_IMAGES),
        ("video", MINIMAX_H3_MAX_REFERENCE_VIDEOS),
        ("audio", MINIMAX_H3_MAX_REFERENCE_AUDIOS),
    ):
        if kinds.count(kind) > limit:
            raise ValueError(f"H3 accepts at most {limit} {kind} references, got {kinds.count(kind)}")
    if len(kinds) > MINIMAX_H3_MAX_REFERENCES:
        raise ValueError(f"H3 accepts at most {MINIMAX_H3_MAX_REFERENCES} references in total, got {len(kinds)}")
    if set(kinds) == {"audio"}:
        raise ValueError("an audio reference must be paired with at least one image or video reference")
    return kinds


def resolve_num_frames(
    references: Sequence[MiniMaxH3Reference], num_frames: int | None, audio_sampling_rate: int
) -> int:
    """The generated frame count, derived from the references when it was left open.

    Only derivable when **exactly one** reference carries audio -- with two, the
    request is ambiguous about which duration to generate. The duration ceiling is
    checked against the *aligned* count, because a 14.99 s soundtrack rounds up to
    362 frames, i.e. 15.083 s, and it is the aligned count that gets generated.
    """
    if num_frames is not None:
        return align_num_frames(num_frames)

    audio_bearing = [index for index, entry in enumerate(references) if entry.has_audio]
    if len(audio_bearing) != 1:
        raise ValueError(
            "num_frames may only be left to the references when exactly one of them carries audio, got "
            f"{len(audio_bearing)}"
        )
    index = audio_bearing[0]
    sample_rate = references[index].sample_rate or audio_sampling_rate
    duration = references[index].audio.shape[-1] / sample_rate
    if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
        raise ValueError(
            f"references[{index}] is {duration:g} s long, outside the {MINIMAX_H3_MIN_DURATION} to "
            f"{MINIMAX_H3_MAX_DURATION} seconds H3 generates"
        )
    aligned = align_num_frames(round(duration * MINIMAX_H3_FPS))
    if aligned / MINIMAX_H3_FPS > MINIMAX_H3_MAX_DURATION:
        raise ValueError(
            f"references[{index}] is {duration:g} s, which rounds up to {aligned} frames (17n + 5), i.e. "
            f"{aligned / MINIMAX_H3_FPS:g} s -- past the {MINIMAX_H3_MAX_DURATION} s H3 generates. Pass "
            "num_frames to generate a shorter video from this soundtrack."
        )
    return aligned


def prepare_references(
    references: Sequence[MiniMaxH3Reference],
    num_frames: int | None,
    audio_sampling_rate: int,
) -> tuple[list[MiniMaxH3PreparedReference], int]:
    """Prepare every reference at its own resolution, and resolve the frame count.

    A video goes through the two passes the reference's ``ffmpeg`` decode applied,
    **in this order**: the constant-frame-rate resample onto 24 fps, then the
    LANCZOS rescale onto its own canvas. Frames handed over at 24 fps and already at
    that canvas therefore reach the VAE untouched, which is the parity-exact route.
    """
    check_references(references)
    num_frames = resolve_num_frames(references, num_frames, audio_sampling_rate)

    prepared = [
        MiniMaxH3PreparedReference(kind=reference_kind(index, entry), has_audio=entry.has_audio)
        for index, entry in enumerate(references)
    ]
    for reference, entry in zip(prepared, references):
        if reference.kind == "image":
            image = entry.image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(reference_media_to_uint8(image))
            # EXIF-transpose and RGB before anything else: a phone photo carries its
            # rotation in EXIF and would condition sideways, and a palette or RGBA
            # PNG would reach the channel permute with the wrong channel count.
            image = ImageOps.exif_transpose(image).convert("RGB")
            height, width = resolve_reference_image_size(*image.size)
            reference.image = prepare_reference_image(image, height, width)
        elif reference.kind == "video":
            frames = resample_reference_frames(reference_media_to_uint8(entry.video), float(entry.fps))
            reference.frames = prepare_reference_frames(frames, num_frames)
        if reference.has_audio:
            reference.waveform = prepare_reference_waveform(
                entry.audio,
                entry.sample_rate or audio_sampling_rate,
                audio_sampling_rate,
                max_duration=num_frames / MINIMAX_H3_FPS,
            )
    return prepared, num_frames


def normalize_reference_pixels(frames: np.ndarray, device: torch.device | str | None = None) -> torch.Tensor:
    """``(T, H, W, 3)`` uint8 frames to ``(1, 3, T, H, W)`` ImageNet-normalized fp32.

    H3 conditions on VLM-style normalized pixels, not on the ``[-1, 1]`` range most
    video VAEs take. Shared by the image path (``T == 1``) and the video path so the
    two cannot drift.
    """
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(3, 0, 1, 2)[None]
    return (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std


def pad_waveform_to_hop(waveform: torch.Tensor) -> torch.Tensor:
    """Right-pad a waveform with zeros up to a whole 800-sample hop.

    The reference audio VAE's ``encode`` does this internally
    (``autoencoder_kl_minimax_h3_audio.py:607``); the device encoder asserts the length
    is already a multiple of the hop instead. Load-bearing, not defensive: a 5.1667 s
    soundtrack is 165333 samples, which is not a multiple of 800.
    """
    samples = waveform.shape[-1]
    padded = math.ceil(samples / MINIMAX_H3_AUDIO_HOP) * MINIMAX_H3_AUDIO_HOP
    if padded == samples:
        return waveform
    return torch.nn.functional.pad(waveform, (0, padded - samples))


def encode_references(
    references: Sequence[MiniMaxH3PreparedReference],
    *,
    encode_clip: Callable[[torch.Tensor], torch.Tensor],
    encode_video: Callable[[torch.Tensor], torch.Tensor],
    encode_audio: Callable[[torch.Tensor], torch.Tensor],
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    audio_latents_mean: Sequence[float],
    audio_latents_std: Sequence[float],
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_latent_channels: int = 32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Encode the references into packed condition rows, resolving their geometry.

    **Mutates** ``references``, filling in ``num_latent_frames``,
    ``latent_height``, ``latent_width`` and ``num_audio_latents`` -- the layout is
    built from those, so this must run before it.

    Args:
        encode_clip: ``(1, 3, 1, H, W)`` normalized pixels to ``[mean, logvar]``
            moments, spatially tiled and with no temporal chunking. The image path.
        encode_video: ``(1, 3, T, H, W)`` to moments, **with** the VAE's
            17-frames-per-5-latents temporal chunking. The video path -- a different
            entry point, not the same one with a bigger T.
        encode_audio: ``(2, 1, samples)`` to the posterior **mean** ``(2, C, T)``.
        audio_latent_channels: width of an audio row, 32 for the released checkpoint.

    Returns ``(video_rows, audio_rows)``, both float32 on CPU and both ``None`` when
    the references carry no rows of that modality. Video rows are still **clean** --
    the caller noise-augments them with ``scheduler.scale_noise``; audio rows are
    never augmented at all.
    """
    video_rows: list[torch.Tensor] = []
    audio_rows: list[torch.Tensor] = []
    audio_mean = torch.tensor(tuple(audio_latents_mean), dtype=torch.float32).view(1, 1, -1)
    audio_std = torch.tensor(tuple(audio_latents_std), dtype=torch.float32).view(1, 1, -1)

    for reference in references:
        if reference.kind != "audio":
            if reference.kind == "image":
                frames = np.asarray(reference.image)[None]
            else:
                # Snapped DOWN to a 17n + 5 the VAE encodes without padding.
                frames = reference.frames[: trim_reference_num_frames(reference.frames.shape[0])]
            pixels = normalize_reference_pixels(frames, device=device)
            # A single frame takes the spatial encoder alone; a video takes the temporal
            # chunking that turns 17n + 5 frames into 5n + 2 latent frames.
            moments = encode_clip(pixels) if reference.kind == "image" else encode_video(pixels)
            latents = sample_posterior(moments)
            reference.num_latent_frames = latents.shape[2]
            reference.latent_height, reference.latent_width = latents.shape[3], latents.shape[4]
            # fp16 round trip and normalization, shared with the fl2va keyframe path.
            video_rows.append(keyframe_condition_rows(latents, latents_mean, latents_std, patch_size))

        if reference.has_audio:
            waveform = pad_waveform_to_hop(reference.waveform.to(device) if device else reference.waveform)
            # The audio VAE is mono; the two stereo channels are two batch items.
            latents = encode_audio(waveform[:, None]).float().cpu().transpose(1, 2)  # (2, T, C)
            reference.num_audio_latents = latents.shape[1]
            normalized = (latents - audio_mean) / audio_std
            # Channel-major rows: the whole left channel, then the whole right one.
            audio_rows.append(normalized.reshape(-1, audio_latent_channels))

    return (
        torch.cat(video_rows) if video_rows else None,
        torch.cat(audio_rows) if audio_rows else None,
    )


def reference_condition_shapes(
    references: Sequence[MiniMaxH3PreparedReference],
) -> tuple[tuple[int, int, int], ...]:
    """The ``(frames, height, width)`` of every **visual** reference, in packed order.

    What the conditioning noise is drawn at: one draw per visual reference, none for
    audio. Only valid after :func:`encode_references` has resolved the geometry.
    """
    return tuple(
        (reference.num_latent_frames, reference.latent_height, reference.latent_width)
        for reference in references
        if reference.kind != "audio"
    )


def split_condition_blocks(
    references: Sequence[MiniMaxH3PreparedReference],
    video_rows: torch.Tensor | None,
    audio_rows: torch.Tensor | None,
) -> list[tuple[torch.Tensor, str]]:
    """The reference region as typed blocks, **in packed order**.

    ``video_rows`` and ``audio_rows`` arrive as two per-modality concatenations, which
    is how the reference carries them and how the denoising loop's write mask slices
    them. The packed sequence interleaves the two -- a video reference's soundtrack rows
    sit immediately before its own video rows -- so this walks the references once,
    cutting both tensors as it goes. The layout's row indices and the presentation come
    from the same walk, so there is no second ordering to keep in step.
    """
    blocks: list[tuple[torch.Tensor, str]] = []
    video_cursor = audio_cursor = 0
    for reference in references:
        if reference.has_audio:
            rows = reference.num_audio_rows
            if audio_rows is None:
                raise ValueError(f"a {reference.kind} reference carries audio but no audio rows were encoded")
            blocks.append((audio_rows[audio_cursor : audio_cursor + rows], "audio"))
            audio_cursor += rows
        if reference.kind != "audio":
            rows = reference.num_video_rows
            if video_rows is None:
                raise ValueError(f"a {reference.kind} reference carries pixels but no video rows were encoded")
            blocks.append((video_rows[video_cursor : video_cursor + rows], "video"))
            video_cursor += rows

    # A leftover row means a reference was skipped or double-counted, which shifts
    # every row after it.
    if video_rows is not None and video_cursor != video_rows.shape[0]:
        raise ValueError(f"consumed {video_cursor} of {video_rows.shape[0]} reference video rows")
    if audio_rows is not None and audio_cursor != audio_rows.shape[0]:
        raise ValueError(f"consumed {audio_cursor} of {audio_rows.shape[0]} reference audio rows")
    return blocks
