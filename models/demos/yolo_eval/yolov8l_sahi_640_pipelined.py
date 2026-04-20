#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SAHI-640 pipelined YOLOv8L 4K inference on a sub-mesh.

Slices 3840x2160 frames into 24 tiles of 640x640 (6 cols x 4 rows with
overlap on the last row) and runs all 24 tiles across a 4x6 = 24-device
sub-mesh of the Galaxy.  No padding devices.

Architecture (3-stage pipeline):
  - Prep process:   read frame, slice into 24 x 640^2 tiles, preprocess -> shared memory
  - Main process:   host_prep + h2d + device compute + d2h
  - BG process:     per-tile NMS + cross-tile merge + draw + encode

Usage:
    python models/demos/yolo_eval/yolov8l_sahi_640_pipelined.py \\
        --input path/to/4k_video.mp4 --serve --port 9090 --display-width 1920
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms as _tv_batched_nms
from torchvision.ops import nms as _tv_nms

from models.demos.yolo_eval.yolov8l_native_vs_sahi_demo import (
    _EMA_ALPHA,
    FrameSource,
    _coco_names_dict,
    _load_coco_names,
    draw_hud,
)
from models.demos.yolo_eval.yolov8l_sahi_pipelined import TileGrid, TileSpec

_TILE_SIZE_640 = 640
_CONF_FLOOR_640 = 0.50
_PAD_VALUE = 114  # YOLOv8 letterbox grey


_INV_255 = torch.tensor(1.0 / 255.0, dtype=torch.bfloat16)


def _load_fused_tile_ext():
    """JIT-compile the C++ fused tile conversion extension (cached after first build)."""
    from torch.utils.cpp_extension import load

    _dir = os.path.dirname(os.path.abspath(__file__))
    return load(
        name="fused_tile_convert",
        sources=[os.path.join(_dir, "fused_tile_convert.cpp")],
        extra_cflags=["-O3", "-march=native", "-fopenmp", "-ffast-math"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )


def _build_tile_specs(grid: TileGrid) -> torch.Tensor:
    """Build (n_tiles, 6) int32 tile_specs tensor from TileGrid for the C++ kernel.

    Only includes actual tiles — the C++ kernel zero-fills device slots beyond
    this count (when N > n_tiles).
    """
    n = len(grid.tiles)
    specs = torch.zeros((n, 6), dtype=torch.int32)
    for i, ts in enumerate(grid.tiles):
        specs[i, 0] = ts.row_start
        specs[i, 1] = ts.col_start
        specs[i, 2] = ts.src_h
        specs[i, 3] = ts.src_w
        specs[i, 4] = int(ts.needs_pad)
        specs[i, 5] = _PAD_VALUE
    return specs


def _fused_slice_and_preprocess(
    frame_bgr: np.ndarray,
    grid: TileGrid,
    n_devices: int,
    n_threads: int = 8,
    bf16_buf: torch.Tensor | None = None,
    uint8_buf: np.ndarray | None = None,
    pool: ThreadPoolExecutor | None = None,
    return_uint8: bool = False,
) -> tuple[torch.Tensor, list[tuple[int, int]], dict]:
    """Slice tiles directly into contiguous NCHW-RGB buffer, avoiding ascontiguousarray.

    Writes each tile's channels directly from the frame into a pre-allocated
    (N, 3, H, W) uint8 buffer.  This replaces the old two-step approach
    (slice into NHWC, then ``ascontiguousarray`` on the non-contiguous
    flip+transpose view) with per-tile channel copies that have simple
    stride patterns (contiguous writes, stride-3 reads).

    Threading: each tile is independent; numpy releases the GIL during the
    array slice assignments, so ``ThreadPoolExecutor`` gives real parallelism.

    When ``bf16_buf`` is provided (and not ``return_uint8``), the bf16
    conversion is **fused** into each tile thread: immediately after slicing,
    the thread converts its tile to bf16/255 in-place.  This keeps tile data
    hot in L2/L3 (~2.4 MB per tile) instead of streaming the full 24-tile
    tensor through main memory in a serial second pass.

    When ``uint8_buf`` is provided, reuses the pre-allocated numpy array
    instead of allocating a new one each frame (~2ms savings).

    When ``pool`` is provided, reuses that thread pool instead of creating
    and destroying one each frame (~1-2ms savings).
    """
    N = max(grid.n_tiles, n_devices)
    H, W = grid.tile_h, grid.tile_w
    n_tiles = len(grid.tiles)

    # Fuse bf16 conversion into per-tile threads when possible.
    # Each thread converts its own tile right after slicing, keeping
    # data in L2/L3.  Eliminates the serial bf16 pass (~5ms savings).
    _fuse_bf16 = bf16_buf is not None and not return_uint8

    # --- Threaded fused slice (BGR->RGB + HWC->CHW + optional bf16) ---
    t0 = time.perf_counter()
    out = uint8_buf if uint8_buf is not None else np.empty((N, 3, H, W), dtype=np.uint8)
    shifts: list[tuple[int, int]] = [(0, 0)] * N

    def _slice_tile(i: int) -> tuple[int, int]:
        if i >= n_tiles:
            out[i] = 0
            if _fuse_bf16:
                bf16_buf[i].zero_()
            return (0, 0)
        ts = grid.tiles[i]
        if ts.needs_pad:
            out[i] = _PAD_VALUE
        src = frame_bgr[
            ts.row_start : ts.row_start + ts.src_h,
            ts.col_start : ts.col_start + ts.src_w,
        ]
        # Channel-wise copy: BGR->RGB + HWC->CHW fused
        out[i, 0, : ts.src_h, : ts.src_w] = src[:, :, 2]  # R
        out[i, 1, : ts.src_h, : ts.src_w] = src[:, :, 1]  # G
        out[i, 2, : ts.src_h, : ts.src_w] = src[:, :, 0]  # B
        # Fused bf16 conversion — tile data is hot in L2/L3
        if _fuse_bf16:
            bf16_buf[i].copy_(torch.from_numpy(out[i]))
            bf16_buf[i].mul_(_INV_255)
        return (ts.col_start, ts.row_start)

    if pool is not None:
        shifts = list(pool.map(_slice_tile, range(N)))
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as _pool:
            shifts = list(_pool.map(_slice_tile, range(N)))
    t_slice = (time.perf_counter() - t0) * 1000

    # --- bf16 conversion (skipped when fused above) ---
    t0 = time.perf_counter()
    if return_uint8:
        tensor = torch.from_numpy(out)
        t_preprocess = 0.0
    elif _fuse_bf16:
        tensor = bf16_buf
        t_preprocess = 0.0  # included in t_slice
    elif bf16_buf is not None:
        bf16_buf.copy_(torch.from_numpy(out))
        bf16_buf.mul_(_INV_255)
        tensor = bf16_buf
        t_preprocess = (time.perf_counter() - t0) * 1000
    else:
        tensor = torch.from_numpy(out).to(torch.bfloat16).div_(255.0)
        t_preprocess = (time.perf_counter() - t0) * 1000

    return tensor, shifts, {"slice_ms": t_slice, "preprocess_ms": t_preprocess}


# ---------------------------------------------------------------------------
# Overlap grid builder for 640x640 tiles
# ---------------------------------------------------------------------------


def build_overlap_grid(
    frame_h: int,
    frame_w: int,
    tile_h: int = _TILE_SIZE_640,
    tile_w: int = _TILE_SIZE_640,
) -> TileGrid:
    """Build a tile grid that shifts the last row/col to avoid partial tiles.

    Instead of padding partial edge tiles with grey, shifts the last row and/or
    column inward so every tile is a full ``tile_h x tile_w`` crop.  The overlap
    between the last two rows/columns is handled by cross-tile NMS.

    For 3840x2160 with 640x640:
        x: [0, 640, 1280, 1920, 2560, 3200]  -- 6 cols, exact fit
        y: [0, 640, 1280, 1520]               -- 4 rows, last overlaps by 400 px
        Total: 24 tiles, all 640x640, no padding.
    """
    x_starts = list(range(0, frame_w, tile_w))
    if x_starts[-1] + tile_w > frame_w:
        x_starts[-1] = frame_w - tile_w

    y_starts = list(range(0, frame_h, tile_h))
    if y_starts[-1] + tile_h > frame_h:
        y_starts[-1] = frame_h - tile_h

    specs: list[TileSpec] = []
    for r in y_starts:
        for c in x_starts:
            sh = min(tile_h, frame_h - r)
            sw = min(tile_w, frame_w - c)
            specs.append(TileSpec(r, c, sh, sw, needs_pad=(sh < tile_h or sw < tile_w)))

    return TileGrid(
        tile_h=tile_h,
        tile_w=tile_w,
        frame_h=frame_h,
        frame_w=frame_w,
        n_rows=len(y_starts),
        n_cols=len(x_starts),
        n_tiles=len(specs),
        tiles=tuple(specs),
    )


# ---------------------------------------------------------------------------
# Compute the smallest sub-mesh that fits all tiles
# ---------------------------------------------------------------------------


def compute_mesh_shape(sys_rows: int, sys_cols: int, n_tiles: int) -> tuple[int, int]:
    """Find the smallest sub-rectangle of (sys_rows, sys_cols) with area >= n_tiles.

    Prefers shapes that use more rows (closer to square, better topology).
    """
    best = (sys_rows, sys_cols)
    best_excess = best[0] * best[1] - n_tiles
    for r in range(sys_rows, 0, -1):
        c = (n_tiles + r - 1) // r  # ceil division
        if c <= sys_cols:
            excess = r * c - n_tiles
            if excess < best_excess:
                best = (r, c)
                best_excess = excess
                if excess == 0:
                    break
    return best


# ---------------------------------------------------------------------------
# Prep process: read frame, slice + preprocess, write to shared memory
# ---------------------------------------------------------------------------


def _prep_process_worker(
    video_path: str,
    grid: TileGrid,
    tiles_per_frame: int,
    total_devices: int,
    shm_tensor_bufs: list,
    shm_ring: torch.Tensor,
    ring_size: int,
    shm_shifts: torch.Tensor,
    shm_timings: torch.Tensor,
    go_event,
    ready_event,
    stop_event,
    frame_h: int,
    frame_w: int,
    display_width: int,
):
    """Separate process: read, fused slice+preprocess, scale frame to ring buffer.

    Deferred-scale optimisation: the ready event is signalled as soon as the
    tensor is written to shared memory.  Frame scaling + ring write happen
    AFTER signalling, overlapping with the main process's host_prep + h2d
    (which do NOT use host-memory bandwidth as heavily as PCIe d2h transfers).
    This shaves ~7 ms off the prep critical path without causing d2h contention.

    Pre-allocated buffers: uint8 numpy array and thread pool are created
    once and reused across frames, saving ~3-4 ms per frame of allocation
    and pool creation overhead.

    Timings layout (shm_timings):
        [0] n_frames_valid  [1] read_ms  [2] slice_ms  [3] preprocess_ms
        [4] sp_wall_ms (slice+preprocess)  [5] total_ms  [6] sentinel
        [7] ring_slot
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    torch.set_num_threads(1)

    src = FrameSource(video_path)
    ring_idx = 0
    prep_frame_idx = 0  # for double-buffer alternation

    # C++ fused per-range kernel: 8 Python threads, each processes 3 tiles
    # with a single C++ call (avoids 24x pybind11 overhead).
    _fused_ext = _load_fused_tile_ext()
    tile_specs = _build_tile_specs(grid)
    slice_pool = ThreadPoolExecutor(max_workers=8)
    read_pool = ThreadPoolExecutor(max_workers=1)

    # Read-ahead: pre-read the first frame, then overlap subsequent reads
    # with scale+sp processing.  cv2.VideoCapture.read() releases the GIL
    # during H.264 decode, so real parallelism with sp (also GIL-free).
    def _read_next():
        ok, f = src.read()
        if not ok:
            src.reset()
            ok, f = src.read()
        return ok, f

    pending_read = read_pool.submit(_read_next)  # pre-read frame 0

    try:
        while not stop_event.is_set():
            while not go_event.wait(timeout=0.5):
                if stop_event.is_set():
                    return
            go_event.clear()
            if stop_event.is_set():
                return

            t_prep_start = time.perf_counter()

            # Collect pre-read frame (usually instant — was decoded during
            # previous frame's processing + go_event wait).
            t0 = time.perf_counter()
            ok, frame = pending_read.result()
            t_read = (time.perf_counter() - t0) * 1000

            if not ok:
                shm_timings[6] = -1.0
                ready_event.set()
                pending_read = read_pool.submit(_read_next)
                continue

            # Start pre-reading NEXT frame — overlaps with sp + post-ready work.
            # cv2.VideoCapture.read() releases GIL during H.264 decode.
            pending_read = read_pool.submit(_read_next)

            # C++ fused tile ranges via ThreadPoolExecutor.
            shm_tensor = shm_tensor_bufs[prep_frame_idx % 2]
            frame_tensor = torch.from_numpy(frame)
            N = max(grid.n_tiles, total_devices)
            chunk = (N + 7) // 8

            def _cpp_range(thread_id):
                start = thread_id * chunk
                end = min(start + chunk, N)
                if start < end:
                    _fused_ext.fused_convert_tile_range(
                        frame_tensor,
                        shm_tensor,
                        tile_specs,
                        start,
                        end,
                        True,
                    )

            t0 = time.perf_counter()
            list(slice_pool.map(_cpp_range, range(8)))
            t_sp = (time.perf_counter() - t0) * 1000
            sp_timings = {"slice_ms": t_sp, "preprocess_ms": 0.0}
            prep_frame_idx += 1
            for i, ts in enumerate(grid.tiles[:tiles_per_frame]):
                shm_shifts[i, 0] = ts.col_start
                shm_shifts[i, 1] = ts.row_start

            # Signal ready as soon as bf16 tensor is written.
            # Scale + ring write happen AFTER ready, overlapping with main's
            # host_prep + h2d (not the bandwidth-sensitive d2h window).
            slot = ring_idx % ring_size
            ring_idx += 1

            shm_timings[0] = 1.0
            shm_timings[1] = t_read
            shm_timings[2] = sp_timings["slice_ms"]
            shm_timings[3] = sp_timings["preprocess_ms"]
            shm_timings[4] = t_sp
            shm_timings[5] = (time.perf_counter() - t_prep_start) * 1000
            shm_timings[6] = 0.0
            shm_timings[7] = float(slot)
            ready_event.set()

            # Deferred scale + ring write: happen AFTER ready, overlapping with
            # main's host_prep + h2d.  Read-ahead ensures read doesn't block.
            if display_width > 0 and frame.shape[1] != display_width:
                target_h = int(round(frame.shape[0] * display_width / frame.shape[1]))
                scaled = cv2.resize(frame, (display_width, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled = frame.copy() if display_width > 0 else frame
            shm_ring[slot].copy_(torch.from_numpy(scaled))

    except Exception:
        import traceback

        traceback.print_exc()
        shm_timings[6] = -1.0
        ready_event.set()


# ---------------------------------------------------------------------------
# Parallel per-tile NMS (threaded)
# ---------------------------------------------------------------------------


def _fused_nms_merge(
    preds_batch: torch.Tensor,
    conf: float,
    iou: float,
    shifts: list[tuple[int, int]],
    n_valid: int,
    merge_iou: float = 0.5,
    class_agnostic: bool = False,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    max_det: int = 300,
    pool: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused per-tile NMS + cross-tile merge. Returns numpy arrays directly.

    Combines _parallel_tile_nms + torch_merge_tiles + scaling into one pass.
    Eliminates per-tile dict allocations, clone+shift loops, and .item() overhead.

    Returns (boxes_np [K,4], scores_np [K], cls_ids_np [K]) in display coordinates.
    """
    bs = preds_batch.shape[0]
    _empty = (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32))

    # --- Fully vectorized prep across ALL tiles at once ---
    preds = preds_batch.transpose(-1, -2)  # [B, N, 84]
    xy = preds[..., :2]
    half_wh = preds[..., 2:4] / 2
    box_xyxy = torch.cat([xy - half_wh, xy + half_wh], dim=-1)  # [B, N, 4]
    cls_scores = preds[..., 4:]  # [B, N, 80]
    max_cls = cls_scores.amax(dim=-1)  # [B, N]

    # Batch-wide conf mask + gather (single nonzero call for all tiles)
    mask = max_cls > conf  # [B, N]
    tile_idx, anchor_idx = torch.nonzero(mask, as_tuple=True)

    if tile_idx.numel() == 0:
        return _empty

    # Gather all passing detections at once
    all_boxes = box_xyxy[tile_idx, anchor_idx]  # [M, 4]
    all_cls_scores = cls_scores[tile_idx, anchor_idx]  # [M, 80]
    all_conf, all_cls_id = all_cls_scores.max(dim=1)  # [M], [M]

    # Second conf filter (on max class score)
    pass2 = all_conf > conf
    tile_idx = tile_idx[pass2]
    all_boxes = all_boxes[pass2].float()
    all_conf = all_conf[pass2].float()
    all_cls_id = all_cls_id[pass2]

    # Pre-compute NMS input: offset boxes by class for multi-class NMS
    nms_boxes = all_boxes + all_cls_id.float().unsqueeze(1) * 7680

    # Sort by tile_idx for contiguous per-tile slicing
    sort_idx = tile_idx.argsort()
    tile_idx = tile_idx[sort_idx]
    all_boxes = all_boxes[sort_idx]
    all_conf = all_conf[sort_idx]
    all_cls_id = all_cls_id[sort_idx]
    nms_boxes = nms_boxes[sort_idx]

    # Find start/end for each tile
    tile_counts = torch.bincount(tile_idx, minlength=bs)
    tile_ends = tile_counts.cumsum(0)
    tile_starts = tile_ends - tile_counts

    # Per-tile NMS — returns global index offsets, not dicts
    def _nms_tile_idx(xi: int) -> torch.Tensor:
        s, e = int(tile_starts[xi]), int(tile_ends[xi])
        if s == e:
            return torch.empty(0, dtype=torch.long)
        keep = _tv_nms(nms_boxes[s:e], all_conf[s:e], iou)[:max_det]
        return keep + s  # global indices into sorted arrays

    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    if pool is not None:
        kept_lists = list(pool.map(_nms_tile_idx, range(n_valid)))
    else:
        kept_lists = [_nms_tile_idx(i) for i in range(n_valid)]
    torch.set_num_threads(_prev_threads)

    # Gather all kept detections (single cat, not 24 separate cats)
    non_empty = [k for k in kept_lists if k.numel() > 0]
    if not non_empty:
        return _empty

    kept = torch.cat(non_empty)
    boxes = all_boxes[kept]  # [K, 4]
    confs = all_conf[kept]  # [K]
    cls_ids = all_cls_id[kept]  # [K]
    kept_tiles = tile_idx[kept]  # [K]

    # Vectorized shift: apply tile offsets to all boxes at once
    shifts_t = torch.tensor(shifts[:n_valid], dtype=torch.float32)  # [n_valid, 2]
    shift_xy = shifts_t[kept_tiles]  # [K, 2]
    boxes[:, 0] += shift_xy[:, 0]
    boxes[:, 2] += shift_xy[:, 0]
    boxes[:, 1] += shift_xy[:, 1]
    boxes[:, 3] += shift_xy[:, 1]

    # Cross-tile merge NMS
    if class_agnostic:
        cross_keep = _tv_nms(boxes, confs, merge_iou)
    else:
        cross_keep = _tv_batched_nms(boxes, confs, cls_ids.int(), merge_iou)

    # Apply scale and convert to numpy in one shot (no .item() per detection)
    boxes = boxes[cross_keep]
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y

    return boxes.numpy(), confs[cross_keep].numpy(), cls_ids[cross_keep].int().numpy()


def _draw_boxes_np(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    names_dict: dict,
) -> None:
    """Draw detection boxes directly from numpy arrays. No Python object overhead."""
    n = len(boxes)
    if n == 0:
        return
    h, w = img.shape[:2]
    max_w = w * 0.25
    max_h = h * 0.25
    # Vectorized nan/inf filter (single numpy call vs per-element math.isfinite)
    valid = np.isfinite(boxes).all(axis=1) & (np.abs(boxes).max(axis=1) < 10000)
    for i in range(n):
        if not valid[i]:
            continue
        x1 = max(0, min(int(boxes[i, 0]), w - 1))
        y1 = max(0, min(int(boxes[i, 1]), h - 1))
        x2 = max(0, min(int(boxes[i, 2]), w - 1))
        y2 = max(0, min(int(boxes[i, 3]), h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) > max_w or (y2 - y1) > max_h:
            continue
        cid = int(cls_ids[i])
        label = f"{names_dict.get(cid, str(cid))} {scores[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# ---------------------------------------------------------------------------
# BG postprocess: NMS + merge + draw + encode
# ---------------------------------------------------------------------------


def _postprocess_worker_shm(
    q_in: mp.Queue,
    tiles_per_frame: int,
    names_dict: dict,
    conf: float,
    iou: float,
    merge_iou: float,
    merge_class_agnostic: bool,
    jpeg_quality: int,
    frame_file: str,
    hud_label: str,
    output_path: str | None,
    is_image: bool,
    shm_ring: torch.Tensor,
    ring_size: int,
    shm_preds: torch.Tensor | None = None,
):
    """BG process: per-tile NMS + cross-tile merge + draw + encode."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Pin BG to physical cores 24-31 + SMT siblings 56-63 (8 physical cores,
    # separate CCD from main inference on cores 0-23 / 32-55).
    _bg_cores = set(range(24, 32)) | set(range(56, 64))
    try:
        os.sched_setaffinity(0, _bg_cores)
        os.nice(19)  # Lowest priority so D2H DMA gets scheduler preference
    except OSError:
        pass
    torch.set_num_threads(4)  # 4 physical cores for BG; per-tile uses OMP=1
    cv2.setNumThreads(2)  # Limit OpenCV threads to avoid oversubscription

    TAG = "[bg-640]"
    ema_fps = 0.0
    fc = 0
    t_post_sum = t_nms_sum = t_merge_sum = t_draw_sum = t_encode_sum = 0.0
    t_wall_start = 0.0
    LOG_INTERVAL = 10
    nms_pool = ThreadPoolExecutor(max_workers=4)
    # 3 encode workers: each cv2.imencode takes ~28ms, NMS+merge+draw takes ~13ms.
    # With 3 workers, encode(N) completes before we need its slot (3*13=39 > 28ms),
    # so the encode wait drops to ~0ms.
    encode_pool = ThreadPoolExecutor(max_workers=3)
    encode_futures: list = []

    def _write_frame_ts(path: str, img: np.ndarray, quality: int):
        """Thread-safe write_frame: unique tmp name per thread."""
        import threading

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        tmp = path + f".tmp.{threading.get_ident()}"
        try:
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, path)
        except OSError:
            pass

    while True:
        try:
            item = q_in.get()
            if item is None:
                for ef in encode_futures:
                    ef.result()
                return

            (pred_slot, ring_slot, shifts, tpf, scale_x, scale_y) = item
            # Slice physical→logical (free view, no copy)
            preds_torch = shm_preds[pred_slot, :, :84, :8400]

            t_post_start = time.perf_counter()
            if fc == 0:
                t_wall_start = t_post_start

            # Direct view — ring_size=4 ensures prep won't reuse this slot
            # for 4 more frames (~133ms at 30 FPS), well after BG finishes.
            canvas = shm_ring[ring_slot].numpy()

            # Debug raw model output stats (~95ms — only at startup)
            if fc == 0:
                raw = preds_torch  # [24, 84, 8400]
                bbox_raw = raw[:, :4, :]
                cls_raw = raw[:, 4:, :]
                print(
                    f"{TAG} raw output frame {fc}: "
                    f"bbox min={bbox_raw.min():.2f} max={bbox_raw.max():.2f} "
                    f"cls min={cls_raw.min():.2f} max={cls_raw.max():.2f} "
                    f"nan={raw.isnan().sum()} inf={raw.isinf().sum()}",
                    flush=True,
                )

            # Fused NMS + cross-tile merge + scale (returns numpy arrays)
            t0 = time.perf_counter()
            boxes_np, scores_np, cls_np = _fused_nms_merge(
                preds_torch,
                conf,
                iou,
                shifts=shifts,
                n_valid=tpf,
                merge_iou=merge_iou,
                class_agnostic=merge_class_agnostic,
                scale_x=scale_x,
                scale_y=scale_y,
                pool=nms_pool,
            )
            t_nms = time.perf_counter() - t0
            t_merge = 0.0  # merged into nms

            # Debug: log detection stats every 5 frames
            n_dets = len(boxes_np)
            if fc % 5 == 0 and n_dets > 0:
                widths = boxes_np[:, 2] - boxes_np[:, 0]
                heights = boxes_np[:, 3] - boxes_np[:, 1]
                print(
                    f"{TAG} frame {fc}: {n_dets} dets, "
                    f"scores=[{scores_np.min():.2f},{scores_np.max():.2f}], "
                    f"widths=[{widths.min():.0f},{widths.max():.0f}], "
                    f"heights=[{heights.min():.0f},{heights.max():.0f}]",
                    flush=True,
                )
            elif fc % 5 == 0:
                print(f"{TAG} frame {fc}: 0 detections", flush=True)

            t0 = time.perf_counter()
            _draw_boxes_np(canvas, boxes_np, scores_np, cls_np, names_dict)
            t_draw = time.perf_counter() - t0

            # Multi-worker encode: 3 threads overlap cv2.imencode (~28ms each)
            # with NMS+merge+draw (~13ms).  Only block if all 3 workers busy.
            # cv2.imencode releases the GIL so threads give real parallelism.
            t0 = time.perf_counter()
            canvas = draw_hud(canvas, hud_label, ema_fps)
            # Drain completed futures
            encode_futures = [ef for ef in encode_futures if not ef.done()]
            # Block only if all workers are saturated
            if len(encode_futures) >= 3:
                encode_futures[0].result()
                encode_futures.pop(0)
            encode_futures.append(encode_pool.submit(_write_frame_ts, frame_file, canvas, jpeg_quality))
            t_encode = time.perf_counter() - t0

            dt_post = time.perf_counter() - t_post_start
            fc += 1
            t_post_sum += dt_post
            t_nms_sum += t_nms
            t_merge_sum += t_merge
            t_draw_sum += t_draw
            t_encode_sum += t_encode

            wall_elapsed = time.perf_counter() - t_wall_start
            if fc > 1:
                wall_fps = (fc - 1) / wall_elapsed
                ema_fps = _EMA_ALPHA * wall_fps + (1 - _EMA_ALPHA) * ema_fps
            else:
                ema_fps = 1.0 / max(dt_post, 1e-9)
                print(
                    f"{TAG} First frame: {dt_post * 1000:.1f}ms "
                    f"(nms={t_nms * 1000:.1f} merge={t_merge * 1000:.1f} "
                    f"draw={t_draw * 1000:.1f} encode={t_encode * 1000:.1f})",
                    flush=True,
                )

            if fc % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                throughput = fc / wall_elapsed
                print(
                    f"{TAG} BG avg {n}f: post={t_post_sum / n * 1000:.1f}ms "
                    f"(nms={t_nms_sum / n * 1000:.1f} "
                    f"merge={t_merge_sum / n * 1000:.1f} "
                    f"draw={t_draw_sum / n * 1000:.1f} "
                    f"encode={t_encode_sum / n * 1000:.1f})  "
                    f"Throughput: {throughput:.1f} FPS",
                    flush=True,
                )
                t_post_sum = t_nms_sum = t_merge_sum = 0.0
                t_draw_sum = t_encode_sum = 0.0

            if is_image:
                if output_path:
                    cv2.imwrite(output_path, canvas)
                    print(f"{TAG} Saved to {output_path}", flush=True)
                print(f"{TAG} Image done.", flush=True)

        except Exception:
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_sahi_640_pipelined(args):
    """Single-frame pipelined YOLOv8L SAHI-640 on a 24-device sub-mesh."""
    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    # Pin main inference process to cores 0-31 (physical cores)
    try:
        os.sched_setaffinity(0, set(range(0, 24)) | set(range(32, 56)))
    except OSError:
        pass

    TAG = "[sahi-640]"
    l1_small = yolov8l_l1_small_size_for_res(_TILE_SIZE_640, _TILE_SIZE_640)
    trace_region = 6_434_816  # tuned for 640x640

    # --- Frame source (peek for dimensions) --------------------------------
    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"{TAG} ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    sample = src.peek()
    frame_h, frame_w = sample.shape[:2]
    is_image = src.is_image
    src.release()

    # --- Tile grid (overlap to keep all tiles full-size) -------------------
    grid = build_overlap_grid(frame_h, frame_w, _TILE_SIZE_640, _TILE_SIZE_640)
    tiles_per_frame = grid.n_tiles

    print(
        f"{TAG} Frame: {frame_w}x{frame_h} -> "
        f"{grid.n_cols}x{grid.n_rows} = {tiles_per_frame} tiles "
        f"of {_TILE_SIZE_640}x{_TILE_SIZE_640}",
        flush=True,
    )
    for i, ts in enumerate(grid.tiles):
        print(
            f"{TAG}   tile {i:2d}: start=({ts.col_start:4d},{ts.row_start:4d}) "
            f"src={ts.src_w}x{ts.src_h} pad={ts.needs_pad}",
            flush=True,
        )

    # --- Compute sub-mesh shape (exact fit, no padding) --------------------
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    sys_rows, sys_cols = sys_shape
    mesh_rows, mesh_cols = compute_mesh_shape(sys_rows, sys_cols, tiles_per_frame)
    total_devices = mesh_rows * mesh_cols

    print(
        f"{TAG} System: {sys_rows}x{sys_cols}={sys_rows * sys_cols} devices  "
        f"-> sub-mesh: {mesh_rows}x{mesh_cols}={total_devices} devices "
        f"for {tiles_per_frame} tiles",
        flush=True,
    )

    if total_devices < tiles_per_frame:
        print(
            f"{TAG} ERROR: cannot fit {tiles_per_frame} tiles on " f"{sys_rows}x{sys_cols} system",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    n_padding = total_devices - tiles_per_frame

    # --- Open sub-mesh -----------------------------------------------------
    mesh_shape = ttnn.MeshShape(mesh_rows, mesh_cols)
    print(
        f"{TAG} Opening mesh {mesh_rows}x{mesh_cols}={total_devices} "
        f"(l1_small={l1_small}, trace_region={trace_region})",
        flush=True,
    )
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        l1_small_size=l1_small,
        trace_region_size=trace_region,
        num_command_queues=2,
    )
    mesh_device.enable_program_cache()

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    # --- Build runner (batch = total_devices) ------------------------------
    print(
        f"{TAG} Building YOLOv8lPerformantRunner " f"({_TILE_SIZE_640}x{_TILE_SIZE_640}, batch={total_devices})...",
        flush=True,
    )
    runner = YOLOv8lPerformantRunner(
        mesh_device,
        device_batch_size=total_devices,
        inp_h=_TILE_SIZE_640,
        inp_w=_TILE_SIZE_640,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    print(f"{TAG} Runner ready.", flush=True)

    # Staging buffer is pre-allocated during runner construction (before trace capture).

    # --- NMS / merge config ------------------------------------------------
    coco_names = _load_coco_names()
    names_dict = _coco_names_dict(coco_names)
    conf = max(args.conf, _CONF_FLOOR_640)
    merge_iou = args.sahi_merge_threshold
    merge_class_agnostic = args.sahi_class_agnostic
    print(
        f"{TAG} NMS conf={conf} iou={args.iou} " f"merge_iou={merge_iou} class_agnostic={merge_class_agnostic}",
        flush=True,
    )

    # --- Display dimensions ------------------------------------------------
    display_width = args.display_width
    if display_width > 0:
        disp_h = int(round(frame_h * display_width / frame_w))
        disp_w = display_width
    else:
        disp_h, disp_w = frame_h, frame_w
    scale_x = disp_w / frame_w
    scale_y = disp_h / frame_h

    # --- Shared memory: ring buffer for bg frames --------------------------
    _ctx = mp.get_context("spawn")
    RING_SIZE = 4  # 1 frame/batch, small ring is enough
    shm_ring = torch.zeros(RING_SIZE, disp_h, disp_w, 3, dtype=torch.uint8).share_memory_()

    # --- Shared memory: prediction ring (avoids pickling 67 MB per frame) --
    PRED_RING = 2
    # Physical shape: tile-aligned padding of [84, 8400] → [96, 8416].
    # batch_to_torch(physical=True) copies full physical buffers contiguously
    # (24 large memcpy calls vs 2016 strided = 4.4ms vs 7.8ms).
    # BG process slices to [:, :84, :8400] for logical data (free view).
    _phys_pred_h = ((84 + 31) // 32) * 32  # 96
    _phys_pred_w = ((8400 + 31) // 32) * 32  # 8416
    shm_preds = torch.zeros(
        PRED_RING, tiles_per_frame, _phys_pred_h, _phys_pred_w, dtype=torch.bfloat16
    ).share_memory_()
    print(
        f"{TAG} shm_preds: [{PRED_RING}, {tiles_per_frame}, {_phys_pred_h}, {_phys_pred_w}] "
        f"logical=[:, :, :84, :8400] "
        f"({shm_preds.nelement() * 2 / 1e6:.1f} MB)",
        flush=True,
    )
    pred_write_idx = 0

    # --- BG process --------------------------------------------------------
    q_post: mp.Queue = _ctx.Queue(maxsize=4)
    hud_label = f"YOLOv8L SAHI-640 4K " f"({mesh_rows}x{mesh_cols}={total_devices} chips, " f"{tiles_per_frame} tiles)"

    bg_proc = _ctx.Process(
        target=_postprocess_worker_shm,
        args=(
            q_post,
            tiles_per_frame,
            names_dict,
            conf,
            args.iou,
            merge_iou,
            merge_class_agnostic,
            args.jpeg_quality,
            args._frame_file,
            hud_label,
            getattr(args, "output", None),
            is_image,
            shm_ring,
            RING_SIZE,
            shm_preds,
        ),
        daemon=True,
        name="sahi640-bg",
    )
    bg_proc.start()

    # --- Shared memory: prep process buffers --------------------------------
    # Double-buffered bfloat16 shared memory — prep writes normalized bf16
    # into buffer[prep_frame_idx % 2], main reads buffer[main_frame_idx % 2].
    # bf16 conversion runs in the prep process to keep the main process's
    # L3 cache clean for PCIe DMA (h2d reads from_torch's host buffer,
    # which is hot in L3 only when no other large writes have polluted it).
    _shm_buf0 = torch.zeros(total_devices, 3, _TILE_SIZE_640, _TILE_SIZE_640, dtype=torch.bfloat16).share_memory_()
    _shm_buf1 = torch.zeros(total_devices, 3, _TILE_SIZE_640, _TILE_SIZE_640, dtype=torch.bfloat16).share_memory_()
    shm_tensor_bufs = [_shm_buf0, _shm_buf1]  # indexed by frame parity
    shm_shifts = torch.zeros(total_devices, 2, dtype=torch.int32).share_memory_()
    shm_timings = torch.zeros(10, dtype=torch.float32).share_memory_()
    go_event = _ctx.Event()
    ready_event = _ctx.Event()
    stop_event = _ctx.Event()

    shm_mb = (
        (_shm_buf0.nelement() + _shm_buf1.nelement()) * 2
        + shm_ring.nelement()
        + shm_preds.nelement() * 4
        + shm_shifts.nelement() * 4
        + shm_timings.nelement() * 4
    ) / 1e6
    print(f"{TAG} Shared memory: {shm_mb:.0f} MB (double-buffered bf16)", flush=True)

    prep_proc = _ctx.Process(
        target=_prep_process_worker,
        args=(
            args.input,
            grid,
            tiles_per_frame,
            total_devices,
            shm_tensor_bufs,
            shm_ring,
            RING_SIZE,
            shm_shifts,
            shm_timings,
            go_event,
            ready_event,
            stop_event,
            frame_h,
            frame_w,
            display_width,
        ),
        daemon=True,
        name="sahi640-prep",
    )
    prep_proc.start()

    # --- Signal handling ---------------------------------------------------
    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"{TAG} 3-stage pipelined loop (double-buffered D2H/compute overlap, " f"{tiles_per_frame} tiles)...",
        flush=True,
    )

    # ===================================================================
    # Main loop — pipelined: submit(N) + get_result(N-1) overlap D2H
    # with compute.  submit() is fast (~6 ms host_prep + queue); the
    # heavy D2H blocks inside get_result() while CQ0 runs the trace.
    #
    # Because get_result() returns the PREVIOUS frame's predictions,
    # we carry per-frame metadata (shifts, ring_slot) in prev_* vars.
    # ===================================================================
    # main_frame_idx tracks which double-buffer to read (same parity as
    # prep_frame_idx in the prep process).
    main_frame_idx = 0

    # Thread pool for overlapping compose with prep_wait (1 worker)
    _submit_executor = ThreadPoolExecutor(max_workers=1)

    # Metadata tracking for pipelined frames
    prev_meta = None
    batch_count = 0
    drop_count = 0
    LOG_INTERVAL = 10
    t_prep_wait_sum = t_convert_sum = 0.0
    t_host_prep_sum = t_queue_sum = t_d2h_sum = 0.0
    t_enqueue_sum = t_batch_total_sum = 0.0
    t_prep_read_sum = t_prep_sp_sum = t_prep_total_sum = 0.0
    t_staging_wait_sum = t_pcie_d2h_sum = t_compose_sum = 0.0
    t_wait_op_sum = t_h2d_sum = t_wait_h2d_sum = t_stg_copy_sum = t_reshard_sum = 0.0
    try:
        # --- Request first frame from prep ---------------------------------
        go_event.set()
        while not ready_event.wait(timeout=0.5):
            if stop:
                return
        ready_event.clear()
        if shm_timings[6].item() < 0:
            print(f"{TAG} No frames available.", flush=True)
            return

        cur_shifts = [(int(shm_shifts[i, 0].item()), int(shm_shifts[i, 1].item())) for i in range(tiles_per_frame)]
        cur_ring_slot = int(shm_timings[7].item())

        print(
            f"{TAG} First frame: prep={shm_timings[5].item():.1f}ms "
            f"[read={shm_timings[1].item():.1f} "
            f"sp={shm_timings[4].item():.1f} "
            f"(slice={shm_timings[2].item():.1f} "
            f"pp={shm_timings[3].item():.1f})]",
            flush=True,
        )

        # --- Pipelined execution (D2H/compute overlap via pre-allocated staging) ---
        print(f"{TAG} Using pipelined execution (D2H/compute overlap)", flush=True)

        # Prime: submit first frame (shm buf[0] already has bf16 from prep)
        runner.submit(shm_tensor_bufs[main_frame_idx % 2])
        main_frame_idx += 1
        prev_meta = (cur_shifts[:tiles_per_frame], cur_ring_slot)

        # Signal prep for next frame
        if not is_image:
            go_event.set()
            _prep_pending = True
        else:
            _prep_pending = False

        # Deferred compose: compose(N-1) runs in a thread and overlaps with
        # the next iteration's prep_wait.  We join before submit to ensure
        # shm_preds is written before BG reads it.
        _compose_future = None
        _compose_enqueue_item = None

        while not stop:
            t_batch_start = time.perf_counter()

            # --- Wait for next frame from prep ---
            # Compose(N-1) thread runs during this wait (GIL released by
            # Event.wait and batch_to_torch's C++ code).
            t0 = time.perf_counter()
            if _prep_pending:
                while not ready_event.wait(timeout=0.5):
                    if stop:
                        break
                ready_event.clear()
                _next_valid = (not stop) and shm_timings[6].item() >= 0
            else:
                _next_valid = False
            t_prep_wait = (time.perf_counter() - t0) * 1000

            # --- Read metadata from shm FIRST, then release prep early ---
            if _next_valid:
                next_shifts = [
                    (int(shm_shifts[i, 0].item()), int(shm_shifts[i, 1].item())) for i in range(tiles_per_frame)
                ]
                next_ring_slot = int(shm_timings[7].item())
                _prep_timings = {
                    "read_ms": shm_timings[1].item(),
                    "sp_ms": shm_timings[4].item(),
                    "total_ms": shm_timings[5].item(),
                }
            else:
                _prep_timings = None

            if not _next_valid and not is_image:
                # Drain compose before exiting
                if _compose_future is not None:
                    _compose_future.result()
                    try:
                        q_post.put_nowait(_compose_enqueue_item)
                    except Exception:
                        pass
                    _compose_future = None
                break

            result_meta = prev_meta  # metadata for get_result's return value
            t_convert = 0.0

            # --- Submit next frame (bf16 already in shm from prep) ---
            # compose(N-1) thread runs concurrently during h2d (GIL released
            # by batch_to_torch C++ code).  Must join before d2h writes to
            # host_staging.
            t0_submit = time.perf_counter()
            if _next_valid:
                runner.submit(shm_tensor_bufs[main_frame_idx % 2])
                main_frame_idx += 1
                _lt = runner.last_timing
                t_host_prep = _lt.get("host_prep_ms", 0)
                t_wait_op = _lt.get("wait_op_ms", 0)
                t_h2d = _lt.get("h2d_ms", 0)
                t_wait_h2d = _lt.get("wait_h2d_ms", 0)
                t_stg_copy = _lt.get("staging_ms", 0)
                t_reshard = _lt.get("reshard_ms", 0)
                prev_meta = (next_shifts[:tiles_per_frame], next_ring_slot)
            else:
                t_host_prep = t_wait_op = t_h2d = t_wait_h2d = t_stg_copy = t_reshard = 0
            t_submit = (time.perf_counter() - t0_submit) * 1000

            # Signal prep AFTER submit — avoids prep's C++ kernel memory
            # traffic overlapping with h2d PCIe DMA.  Prep gets the
            # compose_join + d2h + compose + next_prep_wait window (~10ms).
            # NOTE: go_after_compose_join was tested (v13) and is 2.6 FPS worse
            # because prep_wait increases by more than pcie_d2h decreases.
            if not is_image:
                go_event.set()
                _prep_pending = True

            # --- Join compose AFTER submit (ran during h2d, should be done) ---
            # Must complete before d2h writes to host_staging.
            _dropped = False
            if _compose_future is not None:
                _compose_future.result()
                t_compose = runner._compose_timing
                try:
                    q_post.put_nowait(_compose_enqueue_item)
                except Exception:
                    _dropped = True
                _compose_future = None
            else:
                t_compose = 0.0

            # --- PCIe D2H ---
            t0_d2h = time.perf_counter()
            has_result = runner.pcie_d2h()
            t_staging_wait = runner.last_timing.get("staging_wait_ms", 0)
            t_pcie_d2h = runner.last_timing.get("pcie_d2h_ms", 0)

            # --- Launch compose(N-1): reads host_staging written by d2h ---
            if has_result:
                pred_slot = pred_write_idx % PRED_RING
                _compose_future = _submit_executor.submit(runner.compose, dest=shm_preds[pred_slot])
                r_shifts, r_ring_slot = result_meta
                pred_write_idx += 1
                _compose_enqueue_item = (
                    pred_slot,
                    r_ring_slot,
                    r_shifts,
                    tiles_per_frame,
                    scale_x,
                    scale_y,
                )
            t_enqueue = 0.0
            t_d2h = t_staging_wait + t_pcie_d2h + t_compose

            dt_batch = time.perf_counter() - t_batch_start

            # --- Logging ---------------------------------------------------
            batch_count += 1
            if _dropped:
                drop_count += 1
            t_prep_wait_sum += t_prep_wait
            t_convert_sum += t_convert
            t_host_prep_sum += t_host_prep
            t_queue_sum += t_submit
            t_d2h_sum += t_d2h
            t_staging_wait_sum += t_staging_wait
            t_pcie_d2h_sum += t_pcie_d2h
            t_compose_sum += t_compose
            t_enqueue_sum += t_enqueue
            t_wait_op_sum += t_wait_op
            t_h2d_sum += t_h2d
            t_wait_h2d_sum += t_wait_h2d
            t_stg_copy_sum += t_stg_copy
            t_reshard_sum += t_reshard
            t_batch_total_sum += dt_batch
            if _prep_timings:
                t_prep_read_sum += _prep_timings["read_ms"]
                t_prep_sp_sum += _prep_timings["sp_ms"]
                t_prep_total_sum += _prep_timings["total_ms"]

            if batch_count == 1:
                _pt = _prep_timings or {}
                fps = 1.0 / max(dt_batch, 1e-9)
                print(
                    f"{TAG} Batch 1: {dt_batch * 1000:.1f}ms ({fps:.1f} FPS)  |  "
                    f"bf16={t_convert:.1f}  host_prep={t_host_prep:.1f}  "
                    f"submit={t_submit:.1f}(wait_op={t_wait_op:.1f}+h2d={t_h2d:.1f}+wait_h2d={t_wait_h2d:.1f}+stg={t_stg_copy:.1f}+reshard={t_reshard:.1f})  "
                    f"d2h={t_d2h:.1f}(wait={t_staging_wait:.1f}+pcie={t_pcie_d2h:.1f}+compose={t_compose:.1f})  "
                    f"prep_wait={t_prep_wait:.1f}  "
                    f"prep(proc)={_pt.get('total_ms', 0):.1f}"
                    f"[read={_pt.get('read_ms', 0):.1f} "
                    f"sp={_pt.get('sp_ms', 0):.1f}]",
                    flush=True,
                )

            if batch_count % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                avg_ms = t_batch_total_sum / n * 1000
                fps = n / t_batch_total_sum
                print(
                    f"{TAG} avg {n}f: {avg_ms:.1f}ms/f ({fps:.1f} FPS)  |  "
                    f"prep(proc)={t_prep_total_sum / n:.1f}"
                    f"[read={t_prep_read_sum / n:.1f} "
                    f"sp={t_prep_sp_sum / n:.1f}]  "
                    f"prep_wait={t_prep_wait_sum / n:.1f}  "
                    f"bf16={t_convert_sum / n:.1f}  host_prep={t_host_prep_sum / n:.1f}  "
                    f"submit={t_queue_sum / n:.1f}(wait_op={t_wait_op_sum / n:.1f}+h2d={t_h2d_sum / n:.1f}+wait_h2d={t_wait_h2d_sum / n:.1f}+stg={t_stg_copy_sum / n:.1f}+reshard={t_reshard_sum / n:.1f})  "
                    f"d2h={t_d2h_sum / n:.1f}(wait={t_staging_wait_sum / n:.1f}+pcie={t_pcie_d2h_sum / n:.1f}+compose={t_compose_sum / n:.1f})  "
                    f"drops={drop_count}  (ms)",
                    flush=True,
                )
                t_prep_wait_sum = t_convert_sum = 0.0
                t_host_prep_sum = t_queue_sum = t_d2h_sum = 0.0
                t_enqueue_sum = t_batch_total_sum = 0.0
                t_prep_read_sum = t_prep_sp_sum = t_prep_total_sum = 0.0
                t_staging_wait_sum = t_pcie_d2h_sum = t_compose_sum = 0.0
                t_wait_op_sum = t_h2d_sum = t_wait_h2d_sum = t_stg_copy_sum = t_reshard_sum = 0.0

        # Drain any pending compose before flush
        if _compose_future is not None:
            try:
                _compose_future.result()
                q_post.put(_compose_enqueue_item, timeout=2)
            except Exception:
                pass

        # Flush: read last frame's output from device
        if batch_count > 0 and not stop:
            try:
                last_preds = runner.flush_pipeline(mesh_composer=output_mesh_composer)
                if last_preds is not None and prev_meta is not None:
                    r_shifts, r_ring_slot = prev_meta
                    pred_slot = pred_write_idx % PRED_RING
                    shm_preds[pred_slot, :tiles_per_frame, :84, :8400].copy_(last_preds[:tiles_per_frame])
                    pred_write_idx += 1
                    q_post.put(
                        (pred_slot, r_ring_slot, r_shifts, tiles_per_frame, scale_x, scale_y),
                        timeout=2,
                    )
            except Exception:
                pass

        # --- Shutdown compose executor ---
        _submit_executor.shutdown(wait=False)

        # Image mode: wait for bg to finish
        if is_image:
            while not stop and bg_proc.is_alive():
                time.sleep(0.5)
            while not stop:
                time.sleep(1)

    finally:
        print(
            f"\n{TAG} Shutting down... ({batch_count} frames processed)",
            flush=True,
        )

        # --- Shutdown prep ---
        stop_event.set()
        go_event.set()
        prep_proc.join(timeout=5)
        if prep_proc.is_alive():
            prep_proc.kill()
            prep_proc.join(timeout=2)

        # --- Drain queue so bg process isn't blocked on put ---
        try:
            while not q_post.empty():
                q_post.get_nowait()
        except Exception:
            pass

        # --- Shutdown bg ---
        try:
            q_post.put_nowait(None)
        except Exception:
            pass
        bg_proc.join(timeout=5)
        if bg_proc.is_alive():
            bg_proc.kill()
            bg_proc.join(timeout=2)

        # --- Release device resources ---
        try:
            runner.release()
        except Exception:
            pass
        try:
            ttnn.synchronize_device(mesh_device)
        except Exception:
            pass
        try:
            ttnn.close_mesh_device(mesh_device)
        except Exception:
            pass
        print(f"{TAG} Done.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv8L SAHI-640 4K inference (24 tiles of 640x640).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to 4K video or image.")
    p.add_argument("--conf", type=float, default=0.25, help="NMS confidence.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--serve", action="store_true", help="Stream MJPEG.")
    p.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    p.add_argument("--port", type=int, default=9090, help="HTTP port.")
    p.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality.")
    p.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="Scale output width. 0 = native.",
    )
    p.add_argument(
        "--sahi-merge-threshold",
        type=float,
        default=0.4,
        help="Cross-tile merge NMS IoU.",
    )
    p.add_argument(
        "--sahi-class-agnostic",
        action="store_true",
        help="Merge across classes.",
    )
    p.add_argument("--output", type=str, default=None, help="Save result (image mode).")
    return p.parse_args()


def main():
    args = parse_args()

    tmpdir = tempfile.gettempdir()
    frame_file = os.path.join(tmpdir, "yolov8l_sahi640.jpg")
    args._frame_file = frame_file

    http_proc = None
    if args.serve:
        server_script = str(Path(__file__).resolve().parent / "_mjpeg_server.py")
        print(
            f"[main] Starting MJPEG server on " f"http://{args.host}:{args.port}/",
            flush=True,
        )
        http_proc = subprocess.Popen(
            [
                sys.executable,
                server_script,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--frame-file",
                frame_file,
            ]
        )
        time.sleep(1)

    try:
        run_sahi_640_pipelined(args)
    except KeyboardInterrupt:
        pass
    finally:
        if http_proc and http_proc.poll() is None:
            http_proc.terminate()
            try:
                http_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                http_proc.kill()
        for p in (frame_file, frame_file + ".tmp"):
            try:
                os.unlink(p)
            except OSError:
                pass
        print("[main] Done.", flush=True)


if __name__ == "__main__":
    main()
