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
from torchvision.ops import nms as _tv_nms

from models.demos.yolo_eval.yolov8l_native_vs_sahi_demo import (
    _EMA_ALPHA,
    FrameSource,
    _coco_names_dict,
    _load_coco_names,
    draw_hud,
    scale_to_width,
)
from models.demos.yolo_eval.yolov8l_sahi_pipelined import (
    TileGrid,
    TileSpec,
    _draw_scaled_preds,
    _ScaledPred,
    torch_merge_tiles,
)

_TILE_SIZE_640 = 640
_CONF_FLOOR_640 = 0.50
_PAD_VALUE = 114  # YOLOv8 letterbox grey


_INV_255 = torch.tensor(1.0 / 255.0, dtype=torch.bfloat16)


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

    When ``uint8_buf`` is provided, reuses the pre-allocated numpy array
    instead of allocating a new one each frame (~2ms savings).

    When ``pool`` is provided, reuses that thread pool instead of creating
    and destroying one each frame (~1-2ms savings).
    """
    N = max(grid.n_tiles, n_devices)
    H, W = grid.tile_h, grid.tile_w
    n_tiles = len(grid.tiles)

    # --- Step 1: Threaded fused slice (BGR->RGB + HWC->CHW in one step) ---
    t0 = time.perf_counter()
    out = uint8_buf if uint8_buf is not None else np.empty((N, 3, H, W), dtype=np.uint8)
    shifts: list[tuple[int, int]] = [(0, 0)] * N

    def _slice_tile(i: int) -> tuple[int, int]:
        if i >= n_tiles:
            out[i] = 0
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
        return (ts.col_start, ts.row_start)

    if pool is not None:
        shifts = list(pool.map(_slice_tile, range(N)))
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as _pool:
            shifts = list(_pool.map(_slice_tile, range(N)))
    t_slice = (time.perf_counter() - t0) * 1000

    # --- Step 2: Contiguous uint8 -> bfloat16 / 255 ---
    t0 = time.perf_counter()
    if return_uint8:
        # Skip bf16 conversion — caller handles it (e.g., overlap with D2H)
        tensor = torch.from_numpy(out)
        t_preprocess = 0.0
    elif bf16_buf is not None:
        bf16_buf.copy_(torch.from_numpy(out))
        bf16_buf.mul_(_INV_255)
        tensor = bf16_buf
    else:
        tensor = torch.from_numpy(out).to(torch.bfloat16).div_(255.0)
    if not return_uint8:
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

    # Limit OpenMP threads for torch ops (copy_, mul_).  The default on a
    # 64-core system would spin up 64 threads for the 30 MB bf16 conversion —
    # massive overhead.  4 threads keeps the data in L2/L3 and avoids
    # cross-CCD traffic that inflates compose in the main process.
    torch.set_num_threads(4)

    src = FrameSource(video_path)
    ring_idx = 0
    prep_frame_idx = 0  # for double-buffer alternation

    # Pre-allocate: reuse across frames to avoid per-frame allocation overhead
    N = max(grid.n_tiles, total_devices)
    uint8_buf = np.empty((N, 3, grid.tile_h, grid.tile_w), dtype=np.uint8)
    # 4 threads instead of 8 — reduces memory bandwidth contention with
    # compose in the main process (8 threads inflates compose from 5ms to 12ms).
    slice_pool = ThreadPoolExecutor(max_workers=8)

    try:
        while not stop_event.is_set():
            while not go_event.wait(timeout=0.5):
                if stop_event.is_set():
                    return
            go_event.clear()
            if stop_event.is_set():
                return

            t_prep_start = time.perf_counter()

            # Read one frame
            t0 = time.perf_counter()
            ok, frame = src.read()
            if not ok:
                src.reset()
                ok, frame = src.read()
            t_read = (time.perf_counter() - t0) * 1000

            if not ok:
                shm_timings[6] = -1.0
                ready_event.set()
                continue

            # Fused slice + BGR→RGB + HWC→CHW + bf16/255 conversion.
            # Writes normalized bf16 directly into double-buffered shm.
            shm_tensor = shm_tensor_bufs[prep_frame_idx % 2]
            t0 = time.perf_counter()
            tensor, shifts, sp_timings = _fused_slice_and_preprocess(
                frame,
                grid,
                total_devices,
                uint8_buf=uint8_buf,
                bf16_buf=shm_tensor,
                pool=slice_pool,
            )
            t_sp = (time.perf_counter() - t0) * 1000
            prep_frame_idx += 1
            for i, (sx, sy) in enumerate(shifts[:tiles_per_frame]):
                shm_shifts[i, 0] = sx
                shm_shifts[i, 1] = sy

            # Pre-compute ring slot (actual write deferred after ready signal)
            slot = ring_idx % ring_size

            # Signal ready NOW — tensor is in shm, main can proceed.
            # Scale + ring write happen below, overlapping with main's
            # host_prep + h2d (safe: no PCIe transfers during that phase).
            shm_timings[0] = 1.0
            shm_timings[1] = t_read
            shm_timings[2] = sp_timings["slice_ms"]
            shm_timings[3] = sp_timings["preprocess_ms"]
            shm_timings[4] = t_sp
            shm_timings[5] = (time.perf_counter() - t_prep_start) * 1000
            shm_timings[6] = 0.0
            shm_timings[7] = float(slot)
            ready_event.set()

            # Deferred: scale frame and write to ring buffer.  Runs during
            # the next iteration's host_prep + h2d (~7 ms), well before the
            # postprocess worker reads ring[slot] (~50 ms later).
            scaled = scale_to_width(frame, display_width) if display_width > 0 else frame.copy()
            shm_ring[slot].copy_(torch.from_numpy(scaled))
            ring_idx += 1

    except Exception:
        import traceback

        traceback.print_exc()
        shm_timings[6] = -1.0
        ready_event.set()


# ---------------------------------------------------------------------------
# Parallel per-tile NMS (threaded)
# ---------------------------------------------------------------------------


def _parallel_tile_nms(
    preds_batch: torch.Tensor,
    conf: float,
    iou: float,
    max_det: int = 300,
    pool: ThreadPoolExecutor | None = None,
) -> list[dict]:
    """Same contract as ``_tile_nms`` but runs per-tile NMS on threads.

    The vectorized prep (transpose, xywh→xyxy, amax) is done once on the
    full batch.  Per-tile masking + torchvision.nms runs in a thread pool.
    PyTorch ops release the GIL, so threads give real parallelism.
    """
    bs = preds_batch.shape[0]

    # --- Vectorized prep (shared across threads) ---
    preds = preds_batch.transpose(-1, -2)  # [B, N, 84]
    xy = preds[..., :2]
    half_wh = preds[..., 2:4] / 2
    box_xyxy = torch.cat([xy - half_wh, xy + half_wh], dim=-1)
    cls_scores = preds[..., 4:]  # [B, N, 80]
    max_cls = cls_scores.amax(dim=-1)  # [B, N]

    def _process_tile(xi: int) -> dict:
        mask = max_cls[xi] > conf
        boxes = box_xyxy[xi][mask]
        scores = cls_scores[xi][mask]

        if boxes.shape[0] == 0:
            e = boxes[:, :4].float()
            return {"boxes": {"xyxy": e, "conf": e[:, 0], "cls": e[:, 0]}}

        conf_val, cls_id = scores.max(1, keepdim=True)
        x = torch.cat([boxes, conf_val, cls_id.float()], dim=1)
        x = x[conf_val.view(-1) > conf]

        if x.shape[0] == 0:
            x = x.float()
            return {"boxes": {"xyxy": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]}}

        # Convert to float32 only for the few remaining detections —
        # torchvision NMS requires float32.
        x = x.float()
        c = x[:, 5:6] * 7680
        i = _tv_nms(x[:, :4] + c, x[:, 4], iou)
        i = i[:max_det]
        det = x[i]
        return {"boxes": {"xyxy": det[:, :4], "conf": det[:, 4], "cls": det[:, 5]}}

    if pool is not None:
        results = list(pool.map(_process_tile, range(bs)))
    else:
        results = [_process_tile(i) for i in range(bs)]

    return results


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
    torch.set_num_threads(2)
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

            # Debug raw model output stats (very infrequent — bf16 min/max
            # takes ~95ms with 2 threads on 24×84×8400 elements).
            if fc % 300 == 0:
                raw = preds_torch  # [24, 84, 8400]
                bbox_raw = raw[:, :4, :]  # cx, cy, w, h
                cls_raw = raw[:, 4:, :]  # class logits
                print(
                    f"{TAG} raw output frame {fc}: "
                    f"bbox min={bbox_raw.min():.2f} max={bbox_raw.max():.2f} "
                    f"cls min={cls_raw.min():.2f} max={cls_raw.max():.2f} "
                    f"nan={raw.isnan().sum()} inf={raw.isinf().sum()}",
                    flush=True,
                )

            # Per-tile NMS (24 tiles, parallel)
            t0 = time.perf_counter()
            results_list = _parallel_tile_nms(preds_torch, conf, iou, pool=nms_pool)
            t_nms = time.perf_counter() - t0

            # Cross-tile merge (handles overlap deduplication)
            t0 = time.perf_counter()
            merged_preds = torch_merge_tiles(
                results_list,
                shifts,
                tpf,
                names_dict,
                conf=conf,
                merge_iou=merge_iou,
                class_agnostic=merge_class_agnostic,
            )
            t_merge = time.perf_counter() - t0

            # Scale box coords to display resolution
            scaled = [
                _ScaledPred(
                    minx=float(p.minx) * scale_x,
                    miny=float(p.miny) * scale_y,
                    maxx=float(p.maxx) * scale_x,
                    maxy=float(p.maxy) * scale_y,
                    score=float(p.score),
                    cat_name=p.cat_name,
                )
                for p in merged_preds
            ]

            # Debug: log detection stats every 5 frames
            if fc % 5 == 0 and scaled:
                scores = [p.score for p in scaled]
                widths = [p.maxx - p.minx for p in scaled]
                heights = [p.maxy - p.miny for p in scaled]
                print(
                    f"{TAG} frame {fc}: {len(scaled)} dets, "
                    f"scores=[{min(scores):.2f},{max(scores):.2f}], "
                    f"widths=[{min(widths):.0f},{max(widths):.0f}], "
                    f"heights=[{min(heights):.0f},{max(heights):.0f}]",
                    flush=True,
                )
            elif fc % 5 == 0:
                print(f"{TAG} frame {fc}: 0 detections", flush=True)

            t0 = time.perf_counter()
            _draw_scaled_preds(canvas, scaled)
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
    # batch_to_torch(physical=True) copies full physical buffers contiguously.
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
    # The bf16 shm backing produces cache-friendly host tensors that keep
    # compose at ~5ms (vs ~17ms with heap-allocated bf16 tensors).
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

    # Thread pool for overlapping compose with submit (1 worker)
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

        while not stop:
            t_batch_start = time.perf_counter()

            # --- Wait for next frame from prep ---
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

            if not _next_valid and not is_image:
                break

            # Read next frame metadata from shared memory
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

            # --- Signal prep BEFORE submit — gives prep the full submit+pcie+compose
            # window to finish.
            if not is_image:
                go_event.set()
                _prep_pending = True
            else:
                _prep_pending = False

            result_meta = prev_meta  # metadata for get_result's return value
            t_convert = 0.0

            # --- Submit next frame ---
            t0_submit = time.perf_counter()
            if _next_valid:
                runner.submit(shm_tensor_bufs[main_frame_idx % 2])
                main_frame_idx += 1
                t_host_prep = runner.last_timing.get("host_prep_ms", 0)
                prev_meta = (next_shifts[:tiles_per_frame], next_ring_slot)
            else:
                t_host_prep = 0
            t_submit = (time.perf_counter() - t0_submit) * 1000

            # --- PCIe D2H (GIL released during C++ DMA) ---
            t0_d2h = time.perf_counter()
            has_result = runner.pcie_d2h()
            t_staging_wait = runner.last_timing.get("staging_wait_ms", 0)
            t_pcie_d2h = runner.last_timing.get("pcie_d2h_ms", 0)

            # --- Compose: physical batch_to_torch (24 contiguous memcpy, ~2.7ms) ---
            if has_result:
                pred_slot = pred_write_idx % PRED_RING
                runner.compose(dest=shm_preds[pred_slot])
            t_compose = runner._compose_timing
            t_d2h_wall = (time.perf_counter() - t0_d2h) * 1000

            # --- Enqueue result to BG ---
            t0 = time.perf_counter()
            _dropped = False
            if has_result:
                r_shifts, r_ring_slot = result_meta
                pred_write_idx += 1
                item = (
                    pred_slot,
                    r_ring_slot,
                    r_shifts,
                    tiles_per_frame,
                    scale_x,
                    scale_y,
                )
                try:
                    q_post.put_nowait(item)
                except Exception:
                    _dropped = True
            t_enqueue = (time.perf_counter() - t0) * 1000
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
                    f"convert={t_convert:.1f}  "
                    f"host_prep={t_host_prep:.1f}  "
                    f"submit={t_submit:.1f}  "
                    f"d2h={t_d2h:.1f}(wait={t_staging_wait:.1f}+pcie={t_pcie_d2h:.1f}+compose={t_compose:.1f})  "
                    f"enqueue={t_enqueue:.1f}  "
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
                    f"convert={t_convert_sum / n:.1f}  "
                    f"host_prep={t_host_prep_sum / n:.1f}  "
                    f"submit={t_queue_sum / n:.1f}  "
                    f"d2h={t_d2h_sum / n:.1f}(wait={t_staging_wait_sum / n:.1f}+pcie={t_pcie_d2h_sum / n:.1f}+compose={t_compose_sum / n:.1f})  "
                    f"enqueue={t_enqueue_sum / n:.1f}  "
                    f"drops={drop_count}  (ms)",
                    flush=True,
                )
                t_prep_wait_sum = t_convert_sum = 0.0
                t_host_prep_sum = t_queue_sum = t_d2h_sum = 0.0
                t_enqueue_sum = t_batch_total_sum = 0.0
                t_prep_read_sum = t_prep_sp_sum = t_prep_total_sum = 0.0
                t_staging_wait_sum = t_pcie_d2h_sum = t_compose_sum = 0.0

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

        # --- Shutdown compose executor ---
        _submit_executor.shutdown(wait=False)

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
