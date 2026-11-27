# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-side temporal assembly of decoded MiniMax-H3 clips: the cross-fade between consecutive
chunks, in one batch at the end of a decode or streamed clip by clip as the device hands them out.

Host only (numpy and torch, no ttnn), so it is tested without a device.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from ....utils.host_buffers import empty_uint8


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


def split_clip(clip, chunk_num_frames: int, frame_pre_padding: int, token_drop: int) -> tuple:
    """One decoded clip's ``(kept_part, overlap_tail_or_None)``, each without its pre-padding frames.

    With ``token_drop`` a clip covers two chunks: the first is its own output, the second only
    cross-fades into the next clip's head (or ends the video after the last clip).
    """
    pieces = []
    for j in range(int(token_drop > 0) + 1):
        frame_start = j * chunk_num_frames
        chunk = clip_frames(clip, frame_start, frame_start + chunk_num_frames)
        pieces.append(clip_frames(chunk, frame_pre_padding, clip_num_frames(chunk)))
    return pieces[0], (pieces[1] if len(pieces) > 1 else None)


def write_clip_part(out, start: int, segment, previous, frame_overlap: int) -> None:
    """Write ``segment`` into ``out`` at frame ``start``, its head cross-faded with ``previous``'s tail."""
    frames = clip_num_frames(segment)
    slab = clip_frames(out, start, start + frames)
    extent = 0 if previous is None else min(clip_num_frames(previous), frames, frame_overlap)
    if extent:
        head = blend_clip_frames(previous, clip_frames(segment, 0, extent), extent)
        if isinstance(out, np.ndarray):
            slab[:extent] = head
            np.copyto(slab[extent:], segment[extent:])
        else:
            slab[:, :, :extent] = head
            slab[:, :, extent:].copy_(segment[:, :, extent:])
    elif isinstance(out, np.ndarray):
        np.copyto(slab, segment)
    else:
        slab.copy_(segment)


def assemble_clip_parts(parts: list[tuple], frame_overlap: int):
    """Temporal assembly into one preallocated buffer: seam math on overlap frames, memcpy for the rest.

    ``parts`` is ``[(segment, previous_overlap_or_None), ...]`` in output order; each fills its own slab.
    """
    total = sum(clip_num_frames(segment) for segment, _ in parts)
    first = parts[0][0]
    if isinstance(first, np.ndarray):
        shape = (total, *first.shape[1:])
        # A fresh buffer this size faults in page by page; a released one is already mapped.
        out = empty_uint8(shape) if first.dtype == np.uint8 else np.empty(shape, dtype=first.dtype)
    else:
        shape = list(first.shape)
        shape[2] = total
        out = torch.empty(shape, dtype=first.dtype)

    jobs, start = [], 0
    for segment, previous in parts:
        jobs.append((out, start, segment, previous, frame_overlap))
        start += clip_num_frames(segment)
    if len(jobs) == 1:
        write_clip_part(*jobs[0])
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
            for done in [pool.submit(write_clip_part, *job) for job in jobs]:
                done.result()
    return out


class StreamingClipAssembly:
    """The planar assembly of :func:`assemble_clip_parts`, done clip by clip as the decode hands them out.

    A part depends only on its own clip and the previous clip's tail, so it is final the moment its
    clip lands. Each finished run of frames goes to ``on_frames`` as a read-only view of the one
    output buffer, in order, with the last ``trim`` frames (the latent padding) never handed out.
    Writes and callbacks run on one background thread, so :meth:`add` never waits on the host.
    Every decoded clip of one decode has the same length, which sizes the buffer from the first.
    """

    def __init__(
        self,
        num_clips: int,
        *,
        chunk_num_frames: int,
        frame_pre_padding: int,
        token_drop: int,
        frame_overlap: int,
        trim: int,
        on_frames: Callable[[np.ndarray], None],
    ) -> None:
        self._num_clips = num_clips
        self._chunk_num_frames = chunk_num_frames
        self._frame_pre_padding = frame_pre_padding
        self._token_drop = token_drop
        self._frame_overlap = frame_overlap
        self._trim = trim
        self._on_frames = on_frames
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-clip-stream")
        self._futures: list[Future] = []
        self._lock = threading.Lock()
        self._out = None
        self._added = 0
        self._written = 0
        self._handed = 0
        self._part_frames = 0
        self._keep = 0
        self._tail = None
        self._failed = False

    def add(self, clip) -> None:
        """Queue one decoded clip; raises the frame consumer's error once it has failed, so the decode stops."""
        if self._failed:
            self.result()
        with self._lock:
            index = self._added
            self._added += 1
            if index >= self._num_clips:
                raise ValueError(f"clip {index + 1} for a {self._num_clips}-clip decode")
            self._futures.append(self._pool.submit(self._add, index, clip))

    def _add(self, index: int, clip) -> None:
        if self._failed:
            return
        try:
            part, tail = split_clip(clip, self._chunk_num_frames, self._frame_pre_padding, self._token_drop)
            if not isinstance(part, np.ndarray):
                raise TypeError("streamed assembly takes the planar uint8 clips of the yuv420 path")
            if index == 0:
                self._part_frames = clip_num_frames(part)
                total = self._num_clips * self._part_frames + (0 if tail is None else clip_num_frames(tail))
                self._keep = total - self._trim
                self._out = empty_uint8((total, *part.shape[1:]))
            elif clip_num_frames(part) != self._part_frames:
                raise ValueError(f"clip {index} has {clip_num_frames(part)} frames, the first had {self._part_frames}")
            start = index * self._part_frames
            write_clip_part(self._out, start, part, self._tail, self._frame_overlap)
            self._written = start + self._part_frames
            self._tail = tail
            if index == self._num_clips - 1 and tail is not None:
                write_clip_part(self._out, self._written, tail, None, self._frame_overlap)
                self._written += clip_num_frames(tail)
            self._hand_out()
        except BaseException:
            self._failed = True
            raise

    def _hand_out(self) -> None:
        stop = min(self._written, self._keep)
        if stop > self._handed:
            frames = self._out[self._handed : stop]
            frames.flags.writeable = False
            self._handed = stop
            self._on_frames(frames)

    def result(self) -> np.ndarray:
        """Wait for every clip, re-raise the first error, and return the whole untrimmed buffer."""
        for future in list(self._futures):
            future.result()
        if self._added != self._num_clips:
            raise ValueError(f"{self._added} clips for a {self._num_clips}-clip decode")
        return self._out

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
