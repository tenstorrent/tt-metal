#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Multi-frame parallel SAHI worker for YOLOv8L 4K inference on full Galaxy mesh.

Processes 5 frames simultaneously by tiling each 3840×2160 frame into
6 tiles of 1280×1280 (3 cols × 2 rows with overlap on the bottom row)
and runs all tiles across the full 8×4 = 32-device Galaxy mesh.

  5 frames × 6 tiles + 2 padding = 32 tiles → 32 devices

Architecture (3-stage pipeline, all optimizations from yolov8l_sahi_640_pipelined):
  - Prep process:  read-ahead + C++ fused tile conversion → shared memory
  - Main process:  pipelined submit(N) + pcie_d2h(N-1) + compose(N-1)
  - BG process:    fused NMS+merge + draw + multi-worker encode

Optimizations:
  - C++ LUT uint8→bf16/255 with AVX2 NT stores (GIL-free)
  - Pre-read-ahead threading for video decode overlap
  - Deferred cv2.resize after ready_event (overlaps with main's h2d)
  - Pipelined submit/d2h/compose (device compute overlaps with host I/O)
  - Go signal after submit (avoids PCIe bandwidth contention)
  - Double-buffered shm_tensor (prep writes one while main reads other)
  - Shared-memory prediction ring (eliminates pickle overhead)
  - Physical-shape compose (contiguous memcpys)
  - Fused NMS+merge returning numpy arrays
  - Multi-worker JPEG encode with canvas.copy()
  - CPU affinity (separate CCDs for main vs BG)

Usage:
    python models/demos/yolo_eval/yolov8l_sahi_5frame_pipelined.py \\
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

from models.demos.yolo_eval.yolov8l_native_vs_sahi_demo import (
    _EMA_ALPHA,
    FrameSource,
    _coco_names_dict,
    _load_coco_names,
    draw_hud,
)
from models.demos.yolo_eval.yolov8l_sahi_640_pipelined import (
    _build_tile_specs,
    _draw_boxes_np,
    _fused_nms_merge,
    _load_fused_tile_ext,
    build_overlap_grid,
)

_TILE_SIZE = 1280
_CONF_FLOOR = 0.50
_PAD_VALUE = 114


# ---------------------------------------------------------------------------
# Prep process: read N frames, C++ fused tile conversion, deferred scale
# ---------------------------------------------------------------------------


def _prep_process_worker(
    video_path: str,
    grid,  # TileGrid (pickle-safe dataclass)
    tiles_per_frame: int,
    frames_per_batch: int,
    total_devices: int,
    shm_tensor_bufs: list[torch.Tensor],  # double-buffered [devices, 3, H, W] bf16
    shm_ring: torch.Tensor,
    ring_size: int,
    shm_shifts: torch.Tensor,  # [tiles_per_frame, 2] int32 (constant grid)
    shm_ring_slots: torch.Tensor,  # [frames_per_batch] int32
    shm_timings: torch.Tensor,  # [10] float32
    go_event,
    ready_event,
    stop_event,
    frame_h: int,
    frame_w: int,
    display_width: int,
):
    """Prep process: read N frames, C++ fused bf16 conversion, deferred scale.

    Timings layout in shm_timings:
        [0] = n_frames_valid (float)
        [1] = read_ms
        [2] = convert_ms (C++ kernel wall time)
        [3] = total_prep_ms
        [4] = unused
        [5] = unused
        [6] = sentinel: -1.0 = error/no-frames, >= 0 = valid
        [7] = prep_frame_idx % 2 (which double-buffer was written)
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)

    src = FrameSource(video_path)

    # Load C++ fused tile extension + build tile specs (constant for all frames)
    _fused_ext = _load_fused_tile_ext()
    tile_specs = _build_tile_specs(grid)

    # Thread pools (persistent across batches)
    # 10 convert workers: 2 per frame, each handles 3 tiles.
    # More parallelism than 5 workers since each C++ call is GIL-free.
    convert_pool = ThreadPoolExecutor(max_workers=frames_per_batch * 2)
    read_pool = ThreadPoolExecutor(max_workers=1)

    ring_idx = 0
    prep_frame_idx = 0

    # Write constant shifts once (all frames have the same tile grid)
    for i, ts in enumerate(grid.tiles[:tiles_per_frame]):
        shm_shifts[i, 0] = ts.col_start
        shm_shifts[i, 1] = ts.row_start

    # Pre-read helper
    def _read_next():
        ok, frame = src.read()
        if not ok:
            src.reset()
            ok, frame = src.read()
        return ok, frame

    # Read-ahead: read N frames in background thread.
    # cv2.VideoCapture.read() releases GIL during H.264 decode,
    # so this gets true parallelism with deferred scale / idle wait.
    def _read_batch():
        batch = []
        for _ in range(frames_per_batch):
            ok, frame = _read_next()
            if not ok:
                break
            batch.append(frame)
        return batch

    # Kick off the very first read-ahead before entering the loop
    _read_future = read_pool.submit(_read_batch)

    try:
        while not stop_event.is_set():
            go_event.wait()
            go_event.clear()
            if stop_event.is_set():
                return

            t_prep_start = time.perf_counter()
            buf_idx = prep_frame_idx % 2
            shm_tensor = shm_tensor_bufs[buf_idx]

            # --- Phase 1: Collect pre-read frames (should already be done) ----
            t0 = time.perf_counter()
            frames = _read_future.result()
            t_read = (time.perf_counter() - t0) * 1000
            n_valid = len(frames)

            if n_valid == 0:
                shm_timings[6] = -1.0
                ready_event.set()
                # Still kick off read-ahead (video may have looped)
                _read_future = read_pool.submit(_read_batch)
                continue

            # --- Phase 2: C++ fused tile conversion (parallel per-tile-group) -
            # Split each frame's 6 tiles into 2 groups of 3 for more parallelism.
            # Each C++ call is GIL-free, so 10 threads run truly in parallel.
            t0 = time.perf_counter()
            _half = tiles_per_frame // 2

            def _convert_tile_group(args):
                fi, start, end = args
                tile_offset = fi * tiles_per_frame
                bf16_view = shm_tensor[tile_offset : tile_offset + tiles_per_frame]
                frame_tensor = torch.from_numpy(frames[fi])
                _fused_ext.fused_convert_tile_range(frame_tensor, bf16_view, tile_specs, start, end, True)

            _tasks = []
            for fi in range(n_valid):
                _tasks.append((fi, 0, _half))
                _tasks.append((fi, _half, tiles_per_frame))
            list(convert_pool.map(_convert_tile_group, _tasks))
            t_convert = (time.perf_counter() - t0) * 1000

            # Zero-pad remaining device slots (batch=32, valid=30)
            tile_offset = n_valid * tiles_per_frame
            if tile_offset < total_devices:
                shm_tensor[tile_offset:total_devices].zero_()

            prep_frame_idx += 1

            # Compute ring slots BEFORE signaling ready
            for fi in range(n_valid):
                shm_ring_slots[fi] = ring_idx % ring_size
                ring_idx += 1

            # --- Signal ready (bf16 tensor is in shm) -------------------------
            t_prep_total = (time.perf_counter() - t_prep_start) * 1000
            shm_timings[0] = float(n_valid)
            shm_timings[1] = t_read
            shm_timings[2] = t_convert
            shm_timings[3] = t_prep_total
            shm_timings[6] = 0.0  # valid
            shm_timings[7] = float(buf_idx)
            ready_event.set()

            # --- Post-ready work (overlaps with main's submit+d2h+compose) ----

            # Start read-ahead for NEXT batch (CPU-bound, no bandwidth contention)
            _read_future = read_pool.submit(_read_batch)

            # Deferred scale + ring write: overlaps with main's submit + d2h.
            for fi in range(n_valid):
                slot = int(shm_ring_slots[fi].item())
                frame = frames[fi]
                if display_width > 0 and frame.shape[1] != display_width:
                    target_h = int(round(frame.shape[0] * display_width / frame.shape[1]))
                    scaled = cv2.resize(frame, (display_width, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    scaled = frame
                shm_ring[slot].copy_(torch.from_numpy(scaled))

    except Exception:
        import traceback

        traceback.print_exc()
        shm_timings[6] = -1.0
        ready_event.set()


# ---------------------------------------------------------------------------
# BG postprocess: fused NMS + merge + draw + multi-worker encode
# ---------------------------------------------------------------------------


def _postprocess_worker(
    q_in: mp.Queue,
    tiles_per_frame: int,
    shifts: list[tuple[int, int]],  # constant tile shifts (6 tuples)
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
    shm_preds: torch.Tensor,
    log_pred_h: int,
    log_pred_w: int,
    video_output_path: str | None = None,
    video_fps: float = 30.0,
):
    """BG process: fused NMS + merge + draw + multi-worker JPEG encode.

    Receives (pred_slot, ring_slots, n_frames, scale_x, scale_y) per batch.
    Reads predictions from shm_preds and display frames from shm_ring.
    """
    TAG = "[bg-5f]"
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # CPU affinity: pin to separate CCD from main process
    try:
        os.sched_setaffinity(0, set(range(24, 32)) | set(range(56, 64)))
        os.nice(5)  # mild deprioritization (was 19, starved encode threads)
    except (OSError, AttributeError):
        pass

    torch.set_num_threads(4)
    cv2.setNumThreads(4)

    ema_fps = 0.0
    fc = 0
    t_post_sum = t_nms_sum = t_draw_sum = t_encode_sum = 0.0
    t_wall_start = 0.0
    LOG_INTERVAL = 10

    nms_pool = ThreadPoolExecutor(max_workers=4)
    # 3 JPEG encode workers — each imencode ~5-8ms, overlapped with NMS
    encode_pool = ThreadPoolExecutor(max_workers=3)
    encode_futures: list = []

    # Optional MP4 video writer (runs in background thread to avoid blocking)
    video_writer = None
    _video_pool = None
    _video_future = None
    if video_output_path and not is_image:
        disp_h, disp_w = shm_ring.shape[1], shm_ring.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_output_path, fourcc, video_fps, (disp_w, disp_h))
        _video_pool = ThreadPoolExecutor(max_workers=1)
        print(f"{TAG} Video output: {video_output_path} ({disp_w}x{disp_h} @ {video_fps:.1f} FPS)", flush=True)

    def _write_frame_ts(path: str, img: np.ndarray, quality: int):
        """Thread-safe: encode JPEG and atomically write to disk."""
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

    try:
        while True:
            item = q_in.get()
            if item is None:
                for ef in encode_futures:
                    ef.result()
                if video_writer is not None:
                    if _video_future is not None:
                        _video_future.result()
                    _video_pool.shutdown(wait=True)
                    video_writer.release()
                    video_writer = None
                    print(f"{TAG} Video saved: {video_output_path} ({fc} frames)", flush=True)
                return

            (pred_slot, ring_slots, n_frames, scale_x, scale_y) = item

            for fi in range(n_frames):
                t_post_start = time.perf_counter()
                if fc == 0:
                    t_wall_start = t_post_start

                # Slice this frame's predictions from shm_preds
                start = fi * tiles_per_frame
                end = start + tiles_per_frame
                preds_torch = shm_preds[pred_slot, start:end, :log_pred_h, :log_pred_w]
                ring_slot = ring_slots[fi]

                # Canvas from ring (already at display resolution)
                canvas = shm_ring[ring_slot].numpy()

                # Debug raw output on first frame
                if fc == 0:
                    raw = preds_torch
                    bbox_raw = raw[:, :4, :]
                    cls_raw = raw[:, 4:, :]
                    print(
                        f"{TAG} raw output frame {fc}: "
                        f"bbox min={bbox_raw.min():.2f} max={bbox_raw.max():.2f} "
                        f"cls min={cls_raw.min():.2f} max={cls_raw.max():.2f} "
                        f"nan={raw.isnan().sum()} inf={raw.isinf().sum()}",
                        flush=True,
                    )

                # Fused NMS + cross-tile merge + scale
                t0 = time.perf_counter()
                boxes_np, scores_np, cls_np = _fused_nms_merge(
                    preds_torch,
                    conf,
                    iou,
                    shifts=shifts,
                    n_valid=tiles_per_frame,
                    merge_iou=merge_iou,
                    class_agnostic=merge_class_agnostic,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    pool=nms_pool,
                )
                t_nms = time.perf_counter() - t0

                # Debug detection stats
                n_dets = len(boxes_np)
                if fc == 0 or (fc % 25 == 0 and n_dets > 0):
                    widths = boxes_np[:, 2] - boxes_np[:, 0] if n_dets else []
                    heights = boxes_np[:, 3] - boxes_np[:, 1] if n_dets else []
                    print(
                        f"{TAG} frame {fc}: {n_dets} dets"
                        + (
                            f", scores=[{scores_np.min():.2f},{scores_np.max():.2f}], "
                            f"widths=[{widths.min():.0f},{widths.max():.0f}], "
                            f"heights=[{heights.min():.0f},{heights.max():.0f}]"
                            if n_dets
                            else ""
                        ),
                        flush=True,
                    )

                t0 = time.perf_counter()
                _draw_boxes_np(canvas, boxes_np, scores_np, cls_np, names_dict)
                t_draw = time.perf_counter() - t0

                # Multi-worker JPEG encode + optional video writer (both threaded)
                t0 = time.perf_counter()
                canvas = draw_hud(canvas, hud_label, ema_fps)
                frame_copy = canvas.copy()  # single copy shared by encode + video
                if video_writer is not None:
                    if _video_future is None or _video_future.done():
                        _video_future = _video_pool.submit(video_writer.write, frame_copy)
                    # else: video writer busy, skip this frame for mp4
                encode_futures = [ef for ef in encode_futures if not ef.done()]
                if len(encode_futures) >= 3:
                    encode_futures[0].result()
                    encode_futures.pop(0)
                encode_futures.append(encode_pool.submit(_write_frame_ts, frame_file, frame_copy, jpeg_quality))
                t_encode = time.perf_counter() - t0

                dt_post = time.perf_counter() - t_post_start
                fc += 1
                t_post_sum += dt_post
                t_nms_sum += t_nms
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
                        f"(nms={t_nms * 1000:.1f} draw={t_draw * 1000:.1f} "
                        f"encode={t_encode * 1000:.1f})",
                        flush=True,
                    )

                if fc % LOG_INTERVAL == 0:
                    n = LOG_INTERVAL
                    throughput = fc / wall_elapsed
                    print(
                        f"{TAG} BG avg {n}f: post={t_post_sum / n * 1000:.1f}ms "
                        f"(nms={t_nms_sum / n * 1000:.1f} "
                        f"draw={t_draw_sum / n * 1000:.1f} "
                        f"encode={t_encode_sum / n * 1000:.1f})  "
                        f"Throughput: {throughput:.1f} FPS",
                        flush=True,
                    )
                    t_post_sum = t_nms_sum = t_draw_sum = t_encode_sum = 0.0

            if is_image:
                if output_path:
                    cv2.imwrite(output_path, canvas)
                    print(f"{TAG} Saved to {output_path}", flush=True)
                print(f"{TAG} Image done.", flush=True)

    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        if video_writer is not None:
            if _video_future is not None:
                try:
                    _video_future.result()
                except Exception:
                    pass
            if _video_pool is not None:
                _video_pool.shutdown(wait=False)
            video_writer.release()
            print(f"{TAG} Video saved (on exit): {video_output_path} ({fc} frames)", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_sahi_5frame_pipelined(args):
    """5-frame pipelined YOLOv8L SAHI on the full Galaxy mesh.

    Uses the same pipelined submit/d2h/compose pattern as the 640 pipeline
    but processes 5 frames × 6 tiles = 30 tiles (+2 padding) per batch.
    """
    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res, yolov8l_trace_region_size_e2e_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    TAG = "[5f-1280]"
    l1_small = yolov8l_l1_small_size_for_res(1280, 1280)
    trace_region = yolov8l_trace_region_size_e2e_for_res(1280, 1280)

    # --- Frame source (peek for dimensions) --------------------------------
    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"{TAG} ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    sample = src.peek()
    frame_h, frame_w = sample.shape[:2]
    is_image = src.is_image
    # Get source FPS for video output
    if not is_image and hasattr(src, "_cap"):
        source_fps = src._cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        source_fps = 30.0
    src.release()

    # --- Tile grid (overlap on bottom row) ---------------------------------
    grid = build_overlap_grid(frame_h, frame_w, _TILE_SIZE, _TILE_SIZE)
    tiles_per_frame = grid.n_tiles

    print(
        f"{TAG} Frame: {frame_w}x{frame_h} -> "
        f"{grid.n_cols}x{grid.n_rows} = {tiles_per_frame} tiles/frame of {_TILE_SIZE}x{_TILE_SIZE}",
        flush=True,
    )
    for i, ts in enumerate(grid.tiles):
        print(
            f"{TAG}   tile {i}: start=({ts.col_start:>4},{ts.row_start:>4}) "
            f"src={ts.src_w}x{ts.src_h} pad={ts.needs_pad}",
            flush=True,
        )

    # --- System mesh -------------------------------------------------------
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    sys_rows, sys_cols = sys_shape
    total_devices = sys_rows * sys_cols

    frames_per_batch = total_devices // tiles_per_frame
    n_padding = total_devices - (frames_per_batch * tiles_per_frame)

    if frames_per_batch < 1:
        print(
            f"{TAG} ERROR: tiles_per_frame={tiles_per_frame} > " f"total_devices={total_devices}.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(
        f"{TAG} System: {sys_rows}x{sys_cols}={total_devices} devices  ->  "
        f"{frames_per_batch} frames x {tiles_per_frame} tiles + {n_padding} padding = {total_devices}",
        flush=True,
    )

    # --- Open full mesh ----------------------------------------------------
    mesh_shape = ttnn.MeshShape(sys_rows, sys_cols)
    print(
        f"{TAG} Opening full mesh {sys_rows}x{sys_cols}={total_devices} "
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
    print(f"{TAG} Building YOLOv8lPerformantRunner ({_TILE_SIZE}x{_TILE_SIZE}, batch={total_devices})...", flush=True)
    runner = YOLOv8lPerformantRunner(
        mesh_device,
        device_batch_size=total_devices,
        inp_h=_TILE_SIZE,
        inp_w=_TILE_SIZE,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    print(f"{TAG} Runner ready.", flush=True)

    # --- Physical/logical shape for prediction ring ------------------------
    _phys_pred_h, _phys_pred_w = runner._phys_per_shard
    _log_pred_h, _log_pred_w = runner._log_per_shard
    print(
        f"{TAG} Pred shard: logical=[{_log_pred_h},{_log_pred_w}] "
        f"physical=[{_phys_pred_h},{_phys_pred_w}] "
        f"compose_physical={runner._compose_physical}",
        flush=True,
    )

    # --- NMS config --------------------------------------------------------
    coco_names = _load_coco_names()
    names_dict = _coco_names_dict(coco_names)
    conf = max(args.conf, _CONF_FLOOR)
    merge_iou = args.sahi_merge_threshold
    merge_class_agnostic = args.sahi_class_agnostic
    # Constant tile shifts (same grid for every frame)
    tile_shifts = [(ts.col_start, ts.row_start) for ts in grid.tiles[:tiles_per_frame]]

    # --- Display dimensions ------------------------------------------------
    display_width = args.display_width
    if display_width > 0:
        disp_h = int(round(frame_h * display_width / frame_w))
        disp_w = display_width
    else:
        disp_h, disp_w = frame_h, frame_w
    scale_x = disp_w / frame_w
    scale_y = disp_h / frame_h

    # --- Shared memory -----------------------------------------------------
    _ctx = mp.get_context("spawn")

    # Display frame ring buffer for BG process (display resolution)
    RING_SIZE = frames_per_batch * 3  # 3 batches of headroom
    shm_ring = torch.zeros(RING_SIZE, disp_h, disp_w, 3, dtype=torch.uint8).share_memory_()

    # Double-buffered bf16 tile tensors for prep→main handoff
    shm_buf0 = torch.zeros(total_devices, 3, _TILE_SIZE, _TILE_SIZE, dtype=torch.bfloat16).share_memory_()
    shm_buf1 = torch.zeros(total_devices, 3, _TILE_SIZE, _TILE_SIZE, dtype=torch.bfloat16).share_memory_()
    shm_tensor_bufs = [shm_buf0, shm_buf1]

    # Prediction ring for main→BG handoff (physical shape for fast compose)
    PRED_RING = 2
    shm_preds = torch.zeros(PRED_RING, total_devices, _phys_pred_h, _phys_pred_w, dtype=torch.bfloat16).share_memory_()

    # Metadata shared memory
    shm_shifts = torch.zeros(tiles_per_frame, 2, dtype=torch.int32).share_memory_()
    shm_ring_slots = torch.zeros(frames_per_batch, dtype=torch.int32).share_memory_()
    shm_timings = torch.zeros(10, dtype=torch.float32).share_memory_()
    go_event = _ctx.Event()
    ready_event = _ctx.Event()
    stop_event = _ctx.Event()

    shm_mb = (
        shm_buf0.nelement() * 2 * 2  # two buffers
        + shm_ring.nelement()
        + shm_preds.nelement() * 2
        + shm_shifts.nelement() * 4
        + shm_ring_slots.nelement() * 4
        + shm_timings.nelement() * 4
    ) / 1e6
    print(
        f"{TAG} Shared memory: {shm_mb:.0f} MB "
        f"(ring={RING_SIZE}x{disp_w}x{disp_h}, "
        f"pred_ring={PRED_RING}x{total_devices}x{_phys_pred_h}x{_phys_pred_w})",
        flush=True,
    )

    # --- BG process --------------------------------------------------------
    frame_file = args._frame_file
    hud_label = f"YOLOv8L SAHI 1280 x{frames_per_batch} ({sys_rows}x{sys_cols}={total_devices} chips)"
    q_post: mp.Queue = _ctx.Queue(maxsize=8)

    bg_proc = _ctx.Process(
        target=_postprocess_worker,
        args=(
            q_post,
            tiles_per_frame,
            tile_shifts,
            names_dict,
            conf,
            args.iou,
            merge_iou,
            merge_class_agnostic,
            args.jpeg_quality,
            frame_file,
            hud_label,
            getattr(args, "output", None),
            is_image,
            shm_ring,
            RING_SIZE,
            shm_preds,
            _log_pred_h,
            _log_pred_w,
            getattr(args, "video_output", None),
            source_fps,
        ),
        daemon=True,
        name="5f-bg",
    )
    bg_proc.start()

    # --- Prep process ------------------------------------------------------
    prep_proc = _ctx.Process(
        target=_prep_process_worker,
        args=(
            args.input,
            grid,
            tiles_per_frame,
            frames_per_batch,
            total_devices,
            shm_tensor_bufs,
            shm_ring,
            RING_SIZE,
            shm_shifts,
            shm_ring_slots,
            shm_timings,
            go_event,
            ready_event,
            stop_event,
            frame_h,
            frame_w,
            display_width,
        ),
        daemon=True,
        name="5f-prep",
    )
    prep_proc.start()

    # --- Signal handling ---------------------------------------------------
    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"{TAG} 3-stage pipeline ({frames_per_batch}-frame batch) running...", flush=True)

    # ===================================================================
    # Main loop — pipelined submit/d2h/compose
    # ===================================================================
    batch_count = 0
    frame_count_total = 0
    LOG_INTERVAL = 5
    main_frame_idx = 0
    pred_write_idx = 0

    # Timing accumulators
    t_prep_wait_sum = t_submit_sum = t_d2h_sum = t_compose_sum = 0.0
    t_enqueue_sum = t_batch_total_sum = 0.0
    t_prep_read_sum = t_prep_convert_sum = t_prep_total_sum = 0.0
    t_host_prep_sum = t_h2d_sum = t_wait_op_sum = 0.0
    t_pcie_d2h_sum = t_staging_wait_sum = 0.0
    drop_count = 0

    # Compose runs in a background thread (overlaps with next submit's h2d)
    _submit_executor = ThreadPoolExecutor(max_workers=1)
    _compose_future = None
    _compose_enqueue_item = None

    try:
        # --- Prime first batch from prep -----------------------------------
        go_event.set()
        while not ready_event.wait(timeout=0.5):
            if stop:
                return
        ready_event.clear()

        if shm_timings[6].item() < 0:
            print(f"{TAG} No frames available.", flush=True)
            return

        n_frames_valid = int(shm_timings[0].item())
        buf_idx = int(shm_timings[7].item())
        cur_ring_slots = [int(shm_ring_slots[fi].item()) for fi in range(n_frames_valid)]

        print(
            f"{TAG} First prep: {n_frames_valid} frames, "
            f"read={shm_timings[1].item():.1f}ms, "
            f"convert={shm_timings[2].item():.1f}ms, "
            f"total={shm_timings[3].item():.1f}ms",
            flush=True,
        )

        # Submit first batch
        runner.submit(shm_tensor_bufs[buf_idx])
        main_frame_idx += 1
        prev_meta = (n_frames_valid, cur_ring_slots)

        # Signal prep for next batch (after submit copied the data)
        if not is_image:
            go_event.set()
            _prep_pending = True
        else:
            _prep_pending = False

        # --- Steady-state pipeline loop ------------------------------------
        while not stop:
            t_batch_start = time.perf_counter()

            # --- Wait for next prep result ---------------------------------
            t0 = time.perf_counter()
            _next_valid = False
            if _prep_pending:
                while not ready_event.wait(timeout=0.5):
                    if stop:
                        break
                ready_event.clear()
                if not stop:
                    _next_valid = shm_timings[6].item() >= 0
            t_prep_wait = (time.perf_counter() - t0) * 1000

            if stop:
                break

            # Read prep metadata
            if _next_valid:
                n_frames_valid_next = int(shm_timings[0].item())
                buf_idx_next = int(shm_timings[7].item())
                next_ring_slots = [int(shm_ring_slots[fi].item()) for fi in range(n_frames_valid_next)]
                _prep_timings = {
                    "read_ms": shm_timings[1].item(),
                    "convert_ms": shm_timings[2].item(),
                    "total_ms": shm_timings[3].item(),
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

            # --- Submit next batch -----------------------------------------
            result_meta = prev_meta
            t0_submit = time.perf_counter()
            if _next_valid:
                runner.submit(shm_tensor_bufs[buf_idx_next])
                main_frame_idx += 1
                _lt = runner.last_timing
                t_host_prep = _lt.get("host_prep_ms", 0)
                t_wait_op = _lt.get("wait_op_ms", 0)
                t_h2d = _lt.get("h2d_ms", 0)
                prev_meta = (n_frames_valid_next, next_ring_slots)
            else:
                t_host_prep = t_wait_op = t_h2d = 0
            t_submit = (time.perf_counter() - t0_submit) * 1000

            # Signal prep AFTER submit (avoids h2d PCIe contention with convert)
            if not is_image and _next_valid:
                go_event.set()
                _prep_pending = True
            else:
                _prep_pending = False

            # --- Join compose from previous iteration ----------------------
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

            # --- PCIe D2H -------------------------------------------------
            t0_d2h = time.perf_counter()
            has_result = runner.pcie_d2h()
            t_staging_wait = runner.last_timing.get("staging_wait_ms", 0)
            t_pcie_d2h = runner.last_timing.get("pcie_d2h_ms", 0)

            # --- Launch compose (reads host_staging) -----------------------
            if has_result:
                pred_slot = pred_write_idx % PRED_RING
                _compose_future = _submit_executor.submit(runner.compose, dest=shm_preds[pred_slot])
                r_n_frames, r_ring_slots = result_meta
                pred_write_idx += 1
                _compose_enqueue_item = (
                    pred_slot,
                    r_ring_slots,
                    r_n_frames,
                    scale_x,
                    scale_y,
                )
            t_d2h = (time.perf_counter() - t0_d2h) * 1000

            dt_batch = time.perf_counter() - t_batch_start

            # --- Logging ---------------------------------------------------
            batch_count += 1
            n_f = result_meta[0] if result_meta else frames_per_batch
            frame_count_total += n_f
            if _dropped:
                drop_count += 1
            t_prep_wait_sum += t_prep_wait
            t_submit_sum += t_submit
            t_d2h_sum += t_d2h
            t_compose_sum += t_compose
            t_batch_total_sum += dt_batch * 1000
            t_host_prep_sum += t_host_prep
            t_h2d_sum += t_h2d
            t_wait_op_sum += t_wait_op
            t_pcie_d2h_sum += t_pcie_d2h
            t_staging_wait_sum += t_staging_wait
            if _prep_timings:
                t_prep_read_sum += _prep_timings["read_ms"]
                t_prep_convert_sum += _prep_timings["convert_ms"]
                t_prep_total_sum += _prep_timings["total_ms"]

            if batch_count == 1:
                _pt = _prep_timings or {}
                eff_fps = n_f / max(dt_batch, 1e-9)
                print(
                    f"{TAG} Batch 1: {dt_batch*1000:.1f}ms "
                    f"({n_f}f, {eff_fps:.1f} eff FPS)  |  "
                    f"prep_wait={t_prep_wait:.1f}  "
                    f"submit={t_submit:.1f}(host={t_host_prep:.1f} h2d={t_h2d:.1f})  "
                    f"d2h={t_d2h:.1f}(wait={t_staging_wait:.1f} pcie={t_pcie_d2h:.1f})  "
                    f"compose={t_compose:.1f}  "
                    f"prep(proc)={_pt.get('total_ms',0):.1f}"
                    f"[read={_pt.get('read_ms',0):.1f} "
                    f"convert={_pt.get('convert_ms',0):.1f}]",
                    flush=True,
                )

            if batch_count % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                avg_ms = t_batch_total_sum / n
                total_frames = sum(frames_per_batch for _ in range(n))  # approximate
                eff_fps = total_frames / (t_batch_total_sum / 1000)
                print(
                    f"{TAG} avg {n}b: {avg_ms:.1f}ms/b ({eff_fps:.1f} eff FPS)  |  "
                    f"prep(proc)={t_prep_total_sum/n:.1f}"
                    f"[read={t_prep_read_sum/n:.1f} "
                    f"convert={t_prep_convert_sum/n:.1f}]  "
                    f"prep_wait={t_prep_wait_sum/n:.1f}  "
                    f"submit={t_submit_sum/n:.1f}(host={t_host_prep_sum/n:.1f} h2d={t_h2d_sum/n:.1f})  "
                    f"d2h={t_d2h_sum/n:.1f}(wait={t_staging_wait_sum/n:.1f} pcie={t_pcie_d2h_sum/n:.1f})  "
                    f"compose={t_compose_sum/n:.1f}  "
                    f"drops={drop_count}  (ms)",
                    flush=True,
                )
                t_prep_wait_sum = t_submit_sum = t_d2h_sum = t_compose_sum = 0.0
                t_batch_total_sum = 0.0
                t_prep_read_sum = t_prep_convert_sum = t_prep_total_sum = 0.0
                t_host_prep_sum = t_h2d_sum = t_wait_op_sum = 0.0
                t_pcie_d2h_sum = t_staging_wait_sum = 0.0

        # --- Drain pending compose -----------------------------------------
        if _compose_future is not None:
            try:
                _compose_future.result()
                q_post.put(_compose_enqueue_item, timeout=2)
            except Exception:
                pass

        # --- Flush last batch ----------------------------------------------
        if batch_count > 0 and not stop:
            try:
                last_preds = runner.flush_pipeline(mesh_composer=output_mesh_composer)
                if last_preds is not None and prev_meta is not None:
                    r_n_frames, r_ring_slots = prev_meta
                    pred_slot = pred_write_idx % PRED_RING
                    # Copy logical predictions to shm_preds
                    n_tiles = r_n_frames * tiles_per_frame
                    shm_preds[pred_slot, :n_tiles, :_log_pred_h, :_log_pred_w].copy_(last_preds[:n_tiles])
                    pred_write_idx += 1
                    q_post.put(
                        (pred_slot, r_ring_slots, r_n_frames, scale_x, scale_y),
                        timeout=2,
                    )
            except Exception:
                pass

        # Image mode: wait for BG to finish
        if is_image:
            while not stop and bg_proc.is_alive():
                time.sleep(0.5)
            while not stop:
                time.sleep(1)

    finally:
        _submit_executor.shutdown(wait=False)

        # --- Shutdown BG ---
        print(f"{TAG} Shutting down... ({frame_count_total} frames in {batch_count} batches)", flush=True)
        try:
            q_post.put_nowait(None)
        except Exception:
            pass
        bg_proc.join(timeout=10)
        if bg_proc.is_alive():
            bg_proc.kill()

        # --- Shutdown prep ---
        stop_event.set()
        go_event.set()
        prep_proc.join(timeout=5)
        if prep_proc.is_alive():
            prep_proc.kill()

        # --- Release devices ---
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
        description="YOLOv8L 5-frame SAHI 4K inference on full Galaxy mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to 4K video or image.")
    p.add_argument("--conf", type=float, default=0.25, help="NMS confidence.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--serve", action="store_true", help="Stream MJPEG over HTTP.")
    p.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind address.")
    p.add_argument("--port", type=int, default=9090, help="HTTP port.")
    p.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality.")
    p.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="Width to scale output. 0 = native resolution.",
    )
    p.add_argument(
        "--sahi-merge-threshold",
        type=float,
        default=0.4,
        help="Cross-tile merge NMS IoU threshold.",
    )
    p.add_argument(
        "--sahi-class-agnostic",
        action="store_true",
        help="Merge boxes across classes during cross-tile NMS.",
    )
    p.add_argument("--output", type=str, default=None, help="Save result (image mode).")
    p.add_argument(
        "--video-output",
        type=str,
        default=None,
        help="Path to write MP4 output video (e.g. output.mp4). "
        "Writes every BG-processed frame at the source framerate.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Setup frame file for MJPEG streaming
    tmpdir = tempfile.gettempdir()
    frame_file = os.path.join(tmpdir, "yolov8l_5frame_sahi.jpg")
    args._frame_file = frame_file

    # Launch MJPEG server if requested
    http_proc = None
    if args.serve:
        server_script = str(Path(__file__).resolve().parent / "_mjpeg_server.py")
        print(f"[main] Starting MJPEG server on http://{args.host}:{args.port}/", flush=True)
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
        run_sahi_5frame_pipelined(args)
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
