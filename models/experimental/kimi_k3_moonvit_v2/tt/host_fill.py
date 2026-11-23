# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The per-image device inputs of `host_prep.prepare_host_inputs`, built fast, rows in window order, unpadded.

`host_prep` is the readable form, checked against the reference module; these functions produce the same values
(`tests/test_host_prep.py` compares them bit for bit) for `DeviceInput.write_rows` / `fill_rows`, which pad them in
place. What makes them fast:

  * Only the image's `h*w` rows are built: the padding to `max_patches` rows is already zero in the staging buffers.
  * Patches go from uint8 pixels to normalized, window-ordered bf16 rows through a 256-entry table of the
    normalized values (`normalized_patches_into`), with no float copy of the image.
  * The 64x64 position table is interpolated as a batch of channel chunks (`PosEmbedTable`): torch runs the
    reference's single interpolation on one thread, and spreads a batch over its threads. Each chunk is
    channels-last, so it takes the same kernel and the same per-element arithmetic as the whole table.
  * RoPE cos and sin are sums of per-axis tables (`rope_window_rows`) instead of cos and sin evaluated per row.

The patch fill, the one done for every image, is split across a small thread pool and runs in numpy, which releases
the GIL. It avoids torch's own threads: from a pool thread, each torch op starts a thread team of its own, and after
the service has been idle, waking torch's threads costs milliseconds per op.
"""

from __future__ import annotations

import functools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from .host_prep import pos_embeds
from .image_prep import IMAGE_MEAN, IMAGE_STD
from .model_config import MoonViTConfig

# Channels per interpolation chunk: a multiple of every CPU vector width, so a chunk's channels-last kernel computes
# each value as the whole table's does.
POS_EMB_CHUNK = 64


# Smallest share of a fill worth a pool thread, in elements: a smaller fill runs inline.
MIN_TASK_ELEMENTS = 1 << 20
POOL_THREADS = min(16, os.cpu_count() or 1)


@functools.cache
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=POOL_THREADS, thread_name_prefix="moonvit-host")


def _run_split(count: int, elements_each: int, work) -> None:
    """Run `work(start, stop)` over range(count) in pieces of at least MIN_TASK_ELEMENTS elements: inline when there
    is one piece, else on the pool."""
    tasks = max(1, min(count, POOL_THREADS, count * elements_each // MIN_TASK_ELEMENTS))
    bounds = [count * i // tasks for i in range(tasks + 1)]
    if tasks == 1:
        work(0, count)
        return
    for future in [_pool().submit(work, a, b) for a, b in zip(bounds, bounds[1:])]:
        future.result()


def _window_view(x: torch.Tensor, h: int, w: int, merge_kernel_size: tuple[int, int]) -> torch.Tensor:
    """x [h*w, ...] in raster order -> view [h/kh, w/kw, kh, kw, ...]: indexed (window row, window col, row in window,
    col in window), i.e. window order when flattened (`host_prep.window_order`)."""
    kh, kw = merge_kernel_size
    return x.view(h // kh, kh, w // kw, kw, *x.shape[1:]).transpose(1, 2)


@functools.cache
def _normalized_table(dtype: torch.dtype) -> np.ndarray:
    """The 256 normalized values of `image_prep.pixels_to_patches` ((v / 255.0 in float64, to float32) - mean) /
    std), cast to dtype, as numpy integers of dtype's width: one lookup per pixel gives the cast value, bit for
    bit."""
    values = (np.arange(256, dtype=np.uint8) / 255.0).astype(np.float32)
    values -= IMAGE_MEAN
    values *= 1.0 / IMAGE_STD
    return _as_numpy_bits(torch.from_numpy(values).to(dtype))


def _as_numpy_bits(x: torch.Tensor) -> np.ndarray:
    """A numpy view of x's memory as integers of its element width (numpy has no bfloat16)."""
    return x.view({2: torch.int16, 4: torch.int32}[x.element_size()]).numpy()


def normalized_patches_into(dst: torch.Tensor, pixels: np.ndarray, config: MoonViTConfig) -> None:
    """uint8 pixels [h*p, w*p, 3] (`image_prep.resize_image`) -> dst[:, :patch_dim] (dst [h*w, >= patch_dim] of a
    2- or 4-byte float dtype): the patches of `image_prep.pixels_to_patches`, flattened as `host_prep.patch_rows`,
    in window order and cast to dst's dtype, bit for bit."""
    p = config.patch_size
    kh, kw = config.merge_kernel_size
    h, w = pixels.shape[0] // p, pixels.shape[1] // p
    table = _normalized_table(dst.dtype)
    out = _as_numpy_bits(dst)[:, : config.patch_dim]
    rows_per_window_row = (w // kw) * kh * kw

    def work(i0: int, i1: int) -> None:
        # Window rows i0..i1-1: (window row, kh, py, window col, kw, px, c) -> (window row, window col, kh, kw, c, py, px)
        src = pixels[i0 * kh * p : i1 * kh * p].reshape(i1 - i0, kh, p, w // kw, kw, p, 3)
        dst_rows = out[i0 * rows_per_window_row : i1 * rows_per_window_row]
        np.take(
            table,
            src.transpose(0, 3, 1, 4, 6, 2, 5),
            out=dst_rows.reshape(i1 - i0, w // kw, kh, kw, 3, p, p),
            mode="clip",  # uint8 indices are always in range; "raise" would buffer the output
        )

    _run_split(h // kh, rows_per_window_row * config.patch_dim, work)


def patch_values_into(dst: torch.Tensor, pixel_values: torch.Tensor, h: int, w: int, config: MoonViTConfig) -> None:
    """`pixel_values` (h*w, 3, p, p) in raster order -> dst[:, :patch_dim] in window order, cast to dst's dtype."""
    rows = pixel_values.reshape(h * w, config.patch_dim)
    kh, kw = config.merge_kernel_size
    dst[:, : config.patch_dim].view(h // kh, w // kw, kh, kw, config.patch_dim).copy_(
        _window_view(rows, h, w, config.merge_kernel_size)
    )


class PosEmbedTable:
    """`host_prep.pos_embeds` in window order, from the weight's channel chunks interpolated as one batch."""

    def __init__(self, weight: torch.Tensor, config: MoonViTConfig) -> None:
        self.weight = weight
        self.config = config
        height, width, hidden = weight.shape
        # The reference permutes the contiguous (H, W, C) weight to (1, C, H, W), which is channels-last, so torch
        # takes its channels-last kernel, which runs on one thread. Here the channels are cut into chunks that form
        # a batch, (C / chunk, chunk, H, W), channels-last: the same kernel and the same per-element arithmetic, but
        # torch spreads the batch over its threads.
        self.batched = weight.is_contiguous() and weight.dtype == torch.float32 and hidden % POS_EMB_CHUNK == 0
        self.batch = (
            weight.view(height, width, hidden // POS_EMB_CHUNK, POS_EMB_CHUNK)
            .permute(2, 3, 0, 1)
            .contiguous(memory_format=torch.channels_last)
            if self.batched
            else None
        )

    def window_rows(self, h: int, w: int) -> torch.Tensor:
        """(h*w, hidden) float32, rows in window order: `to_window_order(pos_embeds(weight, h, w, mode))`."""
        config = self.config
        hidden = self.weight.shape[-1]
        kh, kw = config.merge_kernel_size
        out = torch.empty(h * w, hidden, dtype=torch.float32)
        windows = out.view(h // kh, w // kw, kh, kw, hidden)
        mode = config.pos_emb_interpolation_mode
        if not self.batched or (h, w) == tuple(self.weight.shape[:-1]):
            windows.copy_(_window_view(pos_embeds(self.weight, h, w, mode), h, w, config.merge_kernel_size))
            return out
        chunks = F.interpolate(self.batch, size=(h, w), mode=mode)  # (C / chunk, chunk, h, w), channels-last
        k = hidden // POS_EMB_CHUNK
        # Memory (chunk index, y, x, channel) -> (window row, window col, row in window, col in window, chunk, channel)
        by_pixel = chunks.permute(0, 2, 3, 1).view(k, h // kh, kh, w // kw, kw, POS_EMB_CHUNK)
        windows.view(h // kh, w // kw, kh, kw, k, POS_EMB_CHUNK).copy_(by_pixel.permute(1, 3, 2, 4, 0, 5))
        return out


@functools.cache
def _rope_axis_tables(config: MoonViTConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """cos_x, sin_x, cos_y, sin_y [max_side_patches, head_dim] float32: `host_prep.rope_cos_sin`'s lanes for the column
    (x) or the row (y) at each position, zero in the other axis' lanes, so a row's table is cos_x[x] + cos_y[y]."""
    dim = config.head_dim
    # As rope_cos_sin computes them, so the angles are the same float32 products.
    freqs = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    angles = torch.outer(torch.arange(config.max_side_patches).float(), freqs)  # (side, head_dim/4)
    lanes = torch.arange(dim)
    freq_of_lane, axis_of_lane = lanes // 4, (lanes // 2) % 2  # lane l: pair l // 2, which is (freq, axis)
    tables = []
    for axis in (0, 1):
        for fn in (torch.cos, torch.sin):
            t = fn(angles)[:, freq_of_lane]
            t[:, axis_of_lane != axis] = 0.0
            tables.append(t)
    cos_x, sin_x, cos_y, sin_y = tables
    return cos_x, sin_x, cos_y, sin_y


def rope_window_rows(config: MoonViTConfig, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin (h*w, head_dim) float32 in window order: `to_window_order` of `host_prep.rope_cos_sin`.

    Window order (window row i, window col j, row a, col b in the window) puts patch (y, x) = (kh*i + a, kw*j + b)
    there, so each table is a broadcast sum over (i, a) and (j, b): one pass, exact, as one of the two terms of
    every lane is zero."""
    kh, kw = config.merge_kernel_size
    d = config.head_dim

    def table(x_part: torch.Tensor, y_part: torch.Tensor) -> torch.Tensor:
        out = y_part[:h].view(h // kh, 1, kh, 1, d) + x_part[:w].view(1, w // kw, 1, kw, d)
        return out.view(h * w, d)

    cos_x, sin_x, cos_y, sin_y = _rope_axis_tables(config)
    return table(cos_x, cos_y), table(sin_x, sin_y)
