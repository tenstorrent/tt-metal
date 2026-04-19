#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Multi-frame parallel SAHI worker for YOLOv8L 4K inference on full device mesh.

Processes multiple frames simultaneously by tiling each frame into a grid
(e.g. 3x2 = 6 tiles for 3840x2160) and running ALL tiles from ALL frames
as a single batch across the full device mesh.

For 3840x2160 on Galaxy (8x4 = 32 devices):
  5 frames x 6 tiles + 2 padding = 32 tiles -> 32 devices

Architecture (3-stage pipeline):
  - Prep process (own GIL):  read N frames, slice + preprocess -> shared memory
  - Main process:            host_prep + h2d + device compute + d2h
  - BG process (own GIL):    per-frame NMS + merge + draw + encode

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
import torch

from models.demos.yolo_eval.yolov8l_native_vs_sahi_demo import (
    _CONF_FLOOR_1280,
    _EMA_ALPHA,
    _TILE_SIZE,
    FrameSource,
    _coco_names_dict,
    _load_coco_names,
    draw_hud,
    scale_to_width,
    write_frame,
)
from models.demos.yolo_eval.yolov8l_sahi_pipelined import (
    TileGrid,
    _draw_scaled_preds,
    _ScaledPred,
    _tile_nms,
    slice_and_preprocess,
    torch_merge_tiles,
)

# ---------------------------------------------------------------------------
# Prep process: read N frames, slice + preprocess, write to shared memory
# ---------------------------------------------------------------------------


def _prep_process_worker_multiframe(
    video_path: str,
    grid: TileGrid,
    tiles_per_frame: int,
    frames_per_batch: int,
    total_devices: int,
    shm_tensor: torch.Tensor,
    shm_frames: torch.Tensor,
    shm_shifts: torch.Tensor,
    shm_timings: torch.Tensor,
    go_event,
    ready_event,
    stop_event,
    frame_h: int,
    frame_w: int,
):
    """Separate PROCESS for reading N frames + slice + preprocess.

    Writes directly into pre-allocated shared memory tensors.

    Timings layout in shm_timings:
        [0] = n_frames_valid (float)
        [1] = read_ms (sequential frame reads)
        [2] = slice_sum_ms (sum of per-thread slice times)
        [3] = pp_sum_ms (sum of per-thread preprocess times)
        [4] = parallel_wall_ms (wall time for ThreadPoolExecutor phase)
        [5] = total_prep_ms (end-to-end)
        [6] = sentinel: -1.0 = error/no-frames, >= 0 = valid
    """

    def _process_one_frame(fi, frame):
        """Slice+preprocess one frame, write to shared memory (GIL-free)."""
        tensor_f, shifts_f, sp_timings = slice_and_preprocess(frame, grid, tiles_per_frame)
        tile_offset = fi * tiles_per_frame
        end = tile_offset + tiles_per_frame
        shm_tensor[tile_offset:end].copy_(tensor_f[:tiles_per_frame])
        for i, (sx, sy) in enumerate(shifts_f[:tiles_per_frame]):
            shm_shifts[tile_offset + i, 0] = sx
            shm_shifts[tile_offset + i, 1] = sy
        shm_frames[fi, :frame_h, :frame_w, :].copy_(torch.from_numpy(frame))
        return sp_timings

    src = FrameSource(video_path)

    try:
        while not stop_event.is_set():
            go_event.wait()
            go_event.clear()
            if stop_event.is_set():
                return

            t_prep_start = time.perf_counter()

            # Phase 1: Read all frames sequentially (video decode is serial)
            t0 = time.perf_counter()
            frames = []
            for fi in range(frames_per_batch):
                ok, frame = src.read()
                if not ok:
                    src.reset()
                    ok, frame = src.read()
                if not ok:
                    break
                frames.append(frame)
            t_read_total = (time.perf_counter() - t0) * 1000
            n_valid = len(frames)

            if n_valid == 0:
                shm_timings[6] = -1.0
                ready_event.set()
                continue

            # Phase 2: Parallel slice + preprocess + shm write
            # numpy/torch ops release GIL -> threads run truly in parallel
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_valid) as pool:
                futures = [pool.submit(_process_one_frame, fi, frames[fi]) for fi in range(n_valid)]
                results = [f.result() for f in futures]
            t_parallel = (time.perf_counter() - t0) * 1000

            # Aggregate per-thread timings (sums, for diagnostics)
            t_slice_total = sum(r["slice_ms"] for r in results)
            t_pp_total = sum(r["preprocess_ms"] for r in results)

            # Zero-pad remaining tile slots
            tile_offset = n_valid * tiles_per_frame
            if tile_offset < total_devices:
                shm_tensor[tile_offset:total_devices].zero_()
                for i in range(tile_offset, total_devices):
                    shm_shifts[i, 0] = 0
                    shm_shifts[i, 1] = 0

            t_prep_total = (time.perf_counter() - t_prep_start) * 1000

            shm_timings[0] = float(n_valid)
            shm_timings[1] = t_read_total
            shm_timings[2] = t_slice_total  # sum of per-thread slice (diagnostic)
            shm_timings[3] = t_pp_total  # sum of per-thread pp (diagnostic)
            shm_timings[4] = t_parallel  # wall time for parallel phase
            shm_timings[5] = t_prep_total
            shm_timings[6] = 0.0  # valid

            ready_event.set()

    except Exception:
        import traceback

        traceback.print_exc()
        shm_timings[6] = -1.0
        ready_event.set()


# ---------------------------------------------------------------------------
# BG postprocess: reads pre-scaled frames from shared memory ring buffer
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
):
    """BG process: NMS + merge + draw + encode.

    Receives batched predictions (all frames per batch in one queue item)
    and reads pre-scaled frames from ``shm_ring`` (shared memory ring
    buffer).  Each queue item carries predictions for N frames + ring
    slot indices + scale factors.
    """
    TAG = "[bg-shm]"
    ema_fps = 0.0
    fc = 0
    t_post_sum = t_nms_sum = t_merge_sum = t_draw_sum = t_encode_sum = 0.0
    t_wall_start = 0.0
    LOG_INTERVAL = 10

    try:
        while True:
            item = q_in.get()
            if item is None:
                return

            (all_preds, ring_slots, all_shifts, n_frames, tpf, scale_x, scale_y) = item

            for fi in range(n_frames):
                t_post_start = time.perf_counter()
                if fc == 0:
                    t_wall_start = t_post_start

                start = fi * tpf
                end = start + tpf
                preds_torch = all_preds[start:end]
                ring_slot = ring_slots[fi]
                shifts = all_shifts[start:end]

                # Read pre-scaled frame from shared memory (copy so draw is safe)
                canvas = shm_ring[ring_slot].numpy().copy()

                # Per-tile NMS
                t0 = time.perf_counter()
                results_list = _tile_nms(preds_torch, conf, iou)
                t_nms = time.perf_counter() - t0

                # Cross-tile merge (boxes are in full-image coords)
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

                # Scale box coords from full-image to display
                scaled = [
                    _ScaledPred(
                        minx=p.minx * scale_x,
                        miny=p.miny * scale_y,
                        maxx=p.maxx * scale_x,
                        maxy=p.maxy * scale_y,
                        score=p.score,
                        cat_name=p.cat_name,
                    )
                    for p in merged_preds
                ]

                # Draw + encode + write
                t0 = time.perf_counter()
                _draw_scaled_preds(canvas, scaled)
                t_draw = time.perf_counter() - t0

                t0 = time.perf_counter()
                canvas = draw_hud(canvas, hud_label, ema_fps)
                write_frame(frame_file, canvas, jpeg_quality)
                t_encode = time.perf_counter() - t0

                # Bookkeeping
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
                    t_post_sum = t_nms_sum = t_merge_sum = t_draw_sum = t_encode_sum = 0.0

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


def run_sahi_multiframe_pipelined(args):
    """Multi-frame pipelined YOLOv8L SAHI on the full device mesh.

    Prep process reads N frames, slices into tiles, writes to shared memory.
    Main process runs a single inference pass on ALL tiles (batch=total_devices).
    BG process handles per-frame NMS + merge + draw + encode.
    """
    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    TAG = "[multi-frame]"
    l1_small = yolov8l_l1_small_size_for_res(1280, 1280)
    trace_region = 35_000_000

    # --- Frame source (peek for dimensions, then release) ------------------
    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"{TAG} ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    sample = src.peek()
    frame_h, frame_w = sample.shape[:2]
    is_image = src.is_image
    src.release()

    # --- Tile grid ---------------------------------------------------------
    grid = TileGrid.build(frame_h, frame_w, _TILE_SIZE, _TILE_SIZE)
    tiles_per_frame = grid.n_tiles

    # --- System mesh -------------------------------------------------------
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    sys_rows, sys_cols = sys_shape
    total_devices = sys_rows * sys_cols

    frames_per_batch = total_devices // tiles_per_frame
    n_padding = total_devices - (frames_per_batch * tiles_per_frame)

    if frames_per_batch < 1:
        print(
            f"{TAG} ERROR: tiles_per_frame={tiles_per_frame} > "
            f"total_devices={total_devices}. Cannot fit even 1 frame.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(
        f"{TAG} Frame: {frame_w}x{frame_h} -> " f"{grid.n_cols}x{grid.n_rows} = {tiles_per_frame} tiles/frame",
        flush=True,
    )
    print(
        f"{TAG} System mesh: {sys_rows}x{sys_cols} = {total_devices} devices",
        flush=True,
    )
    print(
        f"{TAG} Batch: {frames_per_batch} frames x {tiles_per_frame} tiles"
        f" + {n_padding} padding = {total_devices} devices",
        flush=True,
    )
    for i, ts in enumerate(grid.tiles):
        print(
            f"{TAG}   tile {i}: start=({ts.col_start},{ts.row_start}) " f"src={ts.src_w}x{ts.src_h} pad={ts.needs_pad}",
            flush=True,
        )

    # --- Open full mesh ----------------------------------------------------
    mesh_shape = ttnn.MeshShape(sys_rows, sys_cols)
    print(
        f"{TAG} Opening full mesh {sys_rows}x{sys_cols} = " f"{total_devices} devices",
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
        f"{TAG} Building YOLOv8lPerformantRunner " f"(1280x1280, batch={total_devices})...",
        flush=True,
    )
    runner = YOLOv8lPerformantRunner(
        mesh_device,
        device_batch_size=total_devices,
        inp_h=1280,
        inp_w=1280,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    print(f"{TAG} ready.", flush=True)

    # --- Merge config ------------------------------------------------------
    coco_names = _load_coco_names()
    names_dict = _coco_names_dict(coco_names)
    conf = max(args.conf, _CONF_FLOOR_1280)
    merge_iou = args.sahi_merge_threshold
    merge_class_agnostic = args.sahi_class_agnostic

    print(
        f"{TAG} Merge: torchvision NMS (iou={merge_iou}, " f"class_agnostic={merge_class_agnostic})  conf={conf}",
        flush=True,
    )

    # --- Display dimensions (for pre-scaling frames) -------------------------
    display_width = args.display_width
    if display_width > 0:
        disp_h = int(round(frame_h * display_width / frame_w))
        disp_w = display_width
    else:
        disp_h, disp_w = frame_h, frame_w
    scale_x = disp_w / frame_w
    scale_y = disp_h / frame_h

    # --- Shared memory: ring buffer for bg process frames ------------------
    _ctx = mp.get_context("spawn")
    RING_SIZE = frames_per_batch * 3  # headroom so bg doesn't stall main
    shm_ring = torch.zeros(RING_SIZE, disp_h, disp_w, 3, dtype=torch.uint8).share_memory_()
    ring_write_idx = 0

    # --- BG process (shared-memory postprocess) ----------------------------
    q_post: mp.Queue = _ctx.Queue(maxsize=frames_per_batch * 2)
    hud_label = f"YOLOv8L SAHI 4K x{frames_per_batch} " f"({sys_rows}x{sys_cols}={total_devices} chips)"

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
        ),
        daemon=True,
        name="mf-postprocess",
    )
    bg_proc.start()

    # --- Shared memory: prep process buffers --------------------------------
    shm_tensor = torch.zeros(total_devices, 3, _TILE_SIZE, _TILE_SIZE, dtype=torch.bfloat16).share_memory_()
    shm_frames = torch.zeros(frames_per_batch, frame_h, frame_w, 3, dtype=torch.uint8).share_memory_()
    shm_shifts = torch.zeros(total_devices, 2, dtype=torch.int32).share_memory_()
    shm_timings = torch.zeros(8, dtype=torch.float32).share_memory_()
    go_event = _ctx.Event()
    ready_event = _ctx.Event()
    stop_event = _ctx.Event()

    shm_mb = (
        shm_tensor.nelement() * 2
        + shm_frames.nelement()
        + shm_ring.nelement()
        + shm_shifts.nelement() * 4
        + shm_timings.nelement() * 4
    ) / 1e6
    print(f"{TAG} Shared memory: {shm_mb:.0f} MB " f"(ring: {RING_SIZE}x{disp_w}x{disp_h})", flush=True)

    prep_proc = _ctx.Process(
        target=_prep_process_worker_multiframe,
        args=(
            args.input,
            grid,
            tiles_per_frame,
            frames_per_batch,
            total_devices,
            shm_tensor,
            shm_frames,
            shm_shifts,
            shm_timings,
            go_event,
            ready_event,
            stop_event,
            frame_h,
            frame_w,
        ),
        daemon=True,
        name="mf-prep",
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
        f"{TAG} 3-stage pipeline ({frames_per_batch}-frame batch) running...",
        flush=True,
    )

    # ===================================================================
    # Main loop
    # ===================================================================
    batch_count = 0
    frame_count_total = 0
    LOG_INTERVAL = 5  # log every N batches
    # Accumulators (all in seconds, converted to ms at log time)
    t_prep_wait_sum = t_device_sum = t_dev_compute_sum = 0.0
    t_host_prep_sum = t_h2d_trace_sum = 0.0
    t_d2h_sum = t_enqueue_sum = t_copy_frames_sum = t_batch_total_sum = 0.0
    # Prep sub-timings (already in ms from prep process)
    t_prep_read_sum = t_prep_slice_sum = t_prep_pp_sum = 0.0
    t_prep_parallel_sum = t_prep_total_sum = 0.0

    try:
        # --- Request first batch from prep ---------------------------------
        go_event.set()
        ready_event.wait()
        ready_event.clear()
        if shm_timings[6].item() < 0:
            print(f"{TAG} No frames available.", flush=True)
            return

        n_frames_valid = int(shm_timings[0].item())
        cur_tensor = shm_tensor
        # Pre-scale frames and write to bg ring buffer (no queue pickle)
        for fi in range(n_frames_valid):
            slot = ring_write_idx % RING_SIZE
            frame_np = shm_frames[fi].numpy()
            if display_width > 0:
                scaled_frame = scale_to_width(frame_np, display_width)
            else:
                scaled_frame = frame_np.copy()
            shm_ring[slot].copy_(torch.from_numpy(scaled_frame))
            ring_write_idx += 1
        cur_shifts = [(int(shm_shifts[i, 0].item()), int(shm_shifts[i, 1].item())) for i in range(total_devices)]

        print(
            f"{TAG} First batch: {n_frames_valid} frames  "
            f"prep={shm_timings[5].item():.1f}ms "
            f"[read={shm_timings[1].item():.1f} "
            f"parallel={shm_timings[4].item():.1f} "
            f"(slice_sum={shm_timings[2].item():.1f} "
            f"pp_sum={shm_timings[3].item():.1f})]",
            flush=True,
        )

        while not stop:
            t_batch_start = time.perf_counter()

            # --- Device inference (all tiles in one pass) ------------------
            t0 = time.perf_counter()
            preds = runner.run(torch_input_tensor=cur_tensor)
            t_submit = time.perf_counter() - t0

            # Signal prep AFTER runner.run — host_prep has already copied
            # shm_tensor to device DRAM, so it can safely be overwritten.
            if not is_image:
                go_event.set()
                _prep_pending = True
            else:
                _prep_pending = False

            # --- Sync (device compute, GIL released -> prep runs) ----------
            t0_sync = time.perf_counter()
            ttnn.synchronize_device(mesh_device)
            t_dev_compute = time.perf_counter() - t0_sync
            t_device = t_submit + t_dev_compute
            t_host_prep = runner.last_timing["host_prep_ms"]
            t_h2d_trace = runner.last_timing["h2d_and_trace_ms"]

            # --- D2H -------------------------------------------------------
            t0 = time.perf_counter()
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds_torch = ttnn.to_torch(
                preds,
                dtype=torch.float32,
                mesh_composer=output_mesh_composer,
            )
            t_d2h = time.perf_counter() - t0

            # --- Send batched results to bg process --------------------------
            # Single put() with all frames' preds to avoid per-item queue
            # backpressure.  .clone() detaches from the full d2h storage so
            # pickle serialises only the valid slice.
            t0 = time.perf_counter()
            n_valid_tiles = n_frames_valid * tiles_per_frame
            base_ring = (ring_write_idx - n_frames_valid) % RING_SIZE
            ring_slots = [(base_ring + fi) % RING_SIZE for fi in range(n_frames_valid)]
            q_post.put(
                (
                    preds_torch[:n_valid_tiles].clone(),
                    ring_slots,
                    cur_shifts[:n_valid_tiles],
                    n_frames_valid,
                    tiles_per_frame,
                    scale_x,
                    scale_y,
                )
            )
            t_enqueue = time.perf_counter() - t0

            # --- Wait for prep result (next batch) -------------------------
            t0 = time.perf_counter()
            if _prep_pending:
                ready_event.wait()
                ready_event.clear()
                _next_valid = shm_timings[6].item() >= 0
            else:
                _next_valid = False
            t_prep_wait = time.perf_counter() - t0

            # --- Copy next batch from shared memory ------------------------
            if _prep_pending and _next_valid:
                n_frames_valid = int(shm_timings[0].item())
                cur_tensor = shm_tensor

                t0 = time.perf_counter()
                # Pre-scale frames and write to bg ring buffer
                for fi in range(n_frames_valid):
                    slot = ring_write_idx % RING_SIZE
                    frame_np = shm_frames[fi].numpy()
                    if display_width > 0:
                        scaled_frame = scale_to_width(frame_np, display_width)
                    else:
                        scaled_frame = frame_np.copy()
                    shm_ring[slot].copy_(torch.from_numpy(scaled_frame))
                    ring_write_idx += 1
                cur_shifts = [
                    (
                        int(shm_shifts[i, 0].item()),
                        int(shm_shifts[i, 1].item()),
                    )
                    for i in range(total_devices)
                ]
                t_copy_frames = time.perf_counter() - t0

                _prep_timings = {
                    "read_ms": shm_timings[1].item(),
                    "slice_sum_ms": shm_timings[2].item(),
                    "pp_sum_ms": shm_timings[3].item(),
                    "parallel_ms": shm_timings[4].item(),
                    "total_ms": shm_timings[5].item(),
                }
            else:
                t_copy_frames = 0.0
                _prep_timings = None
                if not is_image:
                    break  # video ended

            dt_batch = time.perf_counter() - t_batch_start

            # --- Logging ---------------------------------------------------
            batch_count += 1
            frame_count_total += n_frames_valid
            t_prep_wait_sum += t_prep_wait
            t_device_sum += t_device
            t_dev_compute_sum += t_dev_compute
            t_host_prep_sum += t_host_prep
            t_h2d_trace_sum += t_h2d_trace
            t_d2h_sum += t_d2h
            t_enqueue_sum += t_enqueue
            t_copy_frames_sum += t_copy_frames
            t_batch_total_sum += dt_batch
            if _prep_timings:
                t_prep_read_sum += _prep_timings["read_ms"]
                t_prep_slice_sum += _prep_timings["slice_sum_ms"]
                t_prep_pp_sum += _prep_timings["pp_sum_ms"]
                t_prep_parallel_sum += _prep_timings["parallel_ms"]
                t_prep_total_sum += _prep_timings["total_ms"]

            if batch_count == 1:
                _pt = _prep_timings or {}
                eff_fps = n_frames_valid / max(dt_batch, 1e-9)
                print(
                    f"{TAG} First batch (main): {dt_batch*1000:.1f}ms "
                    f"({n_frames_valid}f, {eff_fps:.1f} eff FPS)  |  "
                    f"device={t_device*1000:.1f}"
                    f"[host_prep={t_host_prep:.1f} "
                    f"h2d={t_h2d_trace:.1f} "
                    f"compute={t_dev_compute*1000:.1f}]  "
                    f"d2h={t_d2h*1000:.1f}  "
                    f"enqueue={t_enqueue*1000:.1f}  "
                    f"prep_wait={t_prep_wait*1000:.1f}  "
                    f"copy_frames={t_copy_frames*1000:.1f}  "
                    f"prep(proc)={_pt.get('total_ms',0):.1f}"
                    f"[read={_pt.get('read_ms',0):.1f} "
                    f"parallel={_pt.get('parallel_ms',0):.1f} "
                    f"(slice_sum={_pt.get('slice_sum_ms',0):.1f} "
                    f"pp_sum={_pt.get('pp_sum_ms',0):.1f})]",
                    flush=True,
                )

            if batch_count % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                avg_batch_s = t_batch_total_sum / n
                total_frames = n * frames_per_batch
                eff_fps = total_frames / t_batch_total_sum
                print(
                    f"{TAG} MAIN avg {n}b ({total_frames}f): "
                    f"batch={avg_batch_s*1000:.1f}ms "
                    f"({eff_fps:.1f} eff FPS)  |  "
                    f"prep(proc)={t_prep_total_sum/n:.1f}ms"
                    f"[read={t_prep_read_sum/n:.1f} "
                    f"parallel={t_prep_parallel_sum/n:.1f} "
                    f"(slice_sum={t_prep_slice_sum/n:.1f} "
                    f"pp_sum={t_prep_pp_sum/n:.1f})]  "
                    f"prep_wait={t_prep_wait_sum/n*1000:.1f}  "
                    f"host_prep={t_host_prep_sum/n:.1f}  "
                    f"h2d+trace={t_h2d_trace_sum/n:.1f}  "
                    f"device_compute={t_dev_compute_sum/n*1000:.1f}  "
                    f"d2h={t_d2h_sum/n*1000:.1f}  "
                    f"enqueue={t_enqueue_sum/n*1000:.1f}  "
                    f"copy_frames={t_copy_frames_sum/n*1000:.1f}  (ms)",
                    flush=True,
                )
                t_prep_wait_sum = t_device_sum = t_dev_compute_sum = 0.0
                t_host_prep_sum = t_h2d_trace_sum = 0.0
                t_d2h_sum = t_enqueue_sum = 0.0
                t_copy_frames_sum = t_batch_total_sum = 0.0
                t_prep_read_sum = t_prep_slice_sum = t_prep_pp_sum = 0.0
                t_prep_parallel_sum = t_prep_total_sum = 0.0

        # Image mode: wait for bg to finish
        if is_image:
            while not stop and bg_proc.is_alive():
                time.sleep(0.5)
            while not stop:
                time.sleep(1)

    finally:
        # --- Shutdown bg process ---
        try:
            q_post.put(None)
        except Exception:
            pass
        bg_proc.join(timeout=10)
        if bg_proc.is_alive():
            bg_proc.kill()

        # --- Shutdown prep process ---
        stop_event.set()
        go_event.set()
        prep_proc.join(timeout=5)
        if prep_proc.is_alive():
            prep_proc.kill()

        print(
            f"{TAG} Shutting down... " f"({frame_count_total} frames in {batch_count} batches)",
            flush=True,
        )
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
        description="YOLOv8L multi-frame SAHI 4K inference on full device mesh.",
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
    return p.parse_args()


def main():
    args = parse_args()

    # Setup frame file for MJPEG streaming
    tmpdir = tempfile.gettempdir()
    frame_file = os.path.join(tmpdir, "yolov8l_multiframe_sahi.jpg")
    args._frame_file = frame_file

    # Launch MJPEG server if requested
    http_proc = None
    if args.serve:
        server_script = str(Path(__file__).resolve().parent / "_mjpeg_server.py")
        print(
            f"[main] Starting MJPEG server on http://{args.host}:{args.port}/",
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
        run_sahi_multiframe_pipelined(args)
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
