#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8L SAHI-640 multi-stream pipelined inference on the full 8x4 Galaxy mesh.

Takes a 1280x1280 video source, tiles it into a 2x2 grid of 640x640 tiles
(TILES_PER_STREAM=4) and replicates that batch N_STREAMS=8 times across
the full 8x4 = 32-device mesh -- one logical stream per group of 4 devices.

Architecture mirrors yolov8l_sahi_640_pipelined.py's 3-stage pipeline:
    - Prep process:  read 1 frame, slice into 4 tiles, broadcast 8x -> shm
    - Main process:  host_prep + h2d + compute + d2h of the 32-tile batch
    - BG process:    split 32-tile preds into 8 groups of 4, NMS+merge per
                     stream, push 8 tagged dets messages (stream_id=0..7)

V1 note: all 8 streams read the same frame, so produced boxes are identical
across streams. The browser still plays 8 independent <video> elements and
overlays dets tagged with the matching stream_id, so the display shows 8
concurrent stream viewports -- a clean visualisation of the 8-way parallel
compute on the full 32-device mesh.

Delivery: split-delivery only -- serves the raw 1280x1280 MP4 via
byte-range, and pushes per-stream dets over /dets SSE. No H.264 re-encode,
no WebRTC.

Usage:
    python models/demos/yolo_eval/yolov8l_sahi_5frame_pipelined.py \\
        --input sample_images/14052767_1280x1280_30fps.mp4 --serve --port 9090
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from models.demos.yolo_eval.yolov8l_native_vs_sahi_demo import FrameSource, _coco_names_dict, _load_coco_names
from models.demos.yolo_eval.yolov8l_sahi_640_pipelined import (
    _CONF_FLOOR_640,
    _TILE_SIZE_640,
    _build_tile_specs,
    _fused_nms_merge,
    _load_fused_tile_ext,
    build_overlap_grid,
)

# ---------------------------------------------------------------------------
# Multi-stream topology constants
# ---------------------------------------------------------------------------
# Hardcoded to the full 8x4 Galaxy mesh. 8 logical streams, 4 tiles each.
# 1280x1280 source tiles into 2x2 = 4 tiles of 640x640 (exact fit, no pad).
N_STREAMS = 8
STREAM_W = 1280
STREAM_H = 1280
TILES_PER_STREAM = 4  # 2x2 grid of 640x640 over 1280x1280
TOTAL_TILES = N_STREAMS * TILES_PER_STREAM  # 32
MESH_ROWS = 8
MESH_COLS = 4
TAG = "[sahi-640-multi]"


# ---------------------------------------------------------------------------
# Prep worker: read 1 frame, slice into 4 tiles, broadcast 8x across batch.
# ---------------------------------------------------------------------------


def _prep_process_worker_multi(
    video_path: str,
    stream_grid,
    shm_tensor_bufs: list,
    shm_timings: torch.Tensor,
    go_event,
    ready_event,
    stop_event,
    pace: bool = False,
):
    """Read a 1280x1280 frame, convert its 4 tiles to bf16 NCHW, then copy
    those 4 tiles into each of the 8 stream slots of the 32-tile shm batch.

    Uses the same C++ fused tile conversion as the 640 pipeline but on a
    single-stream grid of 4 tiles; the result is then broadcast 8x into the
    full 32-slot buffer.

    Timings layout (same keys as the 640 pipeline's shm_timings):
        [0] n_frames_valid  [1] read_ms  [2] slice_ms  [3] preprocess_ms
        [4] sp_wall_ms  [5] total_ms  [6] sentinel
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # More threads help the 8-way broadcast memcpy (80MB/frame).  set_num_threads(1)
    # was leaving ~12ms/frame of memory bandwidth on the table.
    torch.set_num_threads(4)

    src = FrameSource(video_path)

    _pace_interval = 0.0
    if pace:
        _cap = cv2.VideoCapture(video_path)
        _src_fps = _cap.get(cv2.CAP_PROP_FPS)
        _cap.release()
        if _src_fps > 0:
            _pace_interval = 1.0 / _src_fps
    _pace_t0 = 0.0

    _fused_ext = _load_fused_tile_ext()
    tile_specs = _build_tile_specs(stream_grid)  # 4 specs for 2x2 stream grid
    slice_pool = ThreadPoolExecutor(max_workers=4)
    bcast_pool = ThreadPoolExecutor(max_workers=8)
    read_pool = ThreadPoolExecutor(max_workers=1)

    # Scratch NCHW bf16 buffer for a single stream's 4 tiles -- C++ kernel
    # writes here, then we broadcast into each of the 8 stream slots.
    stream_bf16 = torch.zeros(TILES_PER_STREAM, 3, _TILE_SIZE_640, _TILE_SIZE_640, dtype=torch.bfloat16)

    def _read_next():
        ok, f = src.read()
        if not ok:
            src.reset()
            ok, f = src.read()
        return ok, f

    pending_read = read_pool.submit(_read_next)
    prep_frame_idx = 0
    _PREP_LOG_INTERVAL = 30
    _prep_log_sum_read = 0.0
    _prep_log_sum_slice = 0.0
    _prep_log_sum_bcast = 0.0
    _prep_log_sum_total = 0.0

    try:
        while not stop_event.is_set():
            while not go_event.wait(timeout=0.5):
                if stop_event.is_set():
                    return
            go_event.clear()
            if stop_event.is_set():
                return

            t_prep_start = time.perf_counter()

            t0 = time.perf_counter()
            ok, frame = pending_read.result()
            t_read = (time.perf_counter() - t0) * 1000

            if not ok:
                shm_timings[6] = -1.0
                ready_event.set()
                pending_read = read_pool.submit(_read_next)
                continue

            pending_read = read_pool.submit(_read_next)

            # Defensive resize: source is expected to be exactly STREAM_W x
            # STREAM_H but tolerate mismatches.
            if frame.shape[0] != STREAM_H or frame.shape[1] != STREAM_W:
                frame = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_LINEAR)

            frame_tensor = torch.from_numpy(frame)

            # 4-way C++ fused slice for the 4 tiles of a single stream.
            chunk = (TILES_PER_STREAM + 3) // 4

            def _cpp_range(thread_id):
                start = thread_id * chunk
                end = min(start + chunk, TILES_PER_STREAM)
                if start < end:
                    _fused_ext.fused_convert_tile_range(
                        frame_tensor,
                        stream_bf16,
                        tile_specs,
                        start,
                        end,
                        True,
                    )

            t0 = time.perf_counter()
            list(slice_pool.map(_cpp_range, range(4)))
            t_sp = (time.perf_counter() - t0) * 1000

            # Broadcast the 4-tile stream batch into all 8 stream slots.
            # 8-way parallel memcpy — each thread writes one stream slot, so
            # writes are disjoint (no contention).  Memory-bandwidth-bound,
            # but parallel dispatch recovers the per-thread bandwidth we lost
            # with a single-threaded copy.
            t_bcast_start = time.perf_counter()
            shm_tensor = shm_tensor_bufs[prep_frame_idx % 2]

            def _bcast_one(s):
                shm_tensor[s * TILES_PER_STREAM : (s + 1) * TILES_PER_STREAM].copy_(stream_bf16)

            list(bcast_pool.map(_bcast_one, range(N_STREAMS)))
            t_bcast = (time.perf_counter() - t_bcast_start) * 1000
            prep_frame_idx += 1

            if _pace_interval > 0:
                if _pace_t0 == 0.0:
                    _pace_t0 = time.perf_counter()
                else:
                    target_t = _pace_t0 + prep_frame_idx * _pace_interval
                    now = time.perf_counter()
                    if target_t > now:
                        time.sleep(target_t - now)

            t_total = (time.perf_counter() - t_prep_start) * 1000
            shm_timings[0] = 1.0
            shm_timings[1] = t_read
            shm_timings[2] = t_sp
            shm_timings[3] = 0.0
            shm_timings[4] = t_sp
            shm_timings[5] = t_total
            shm_timings[6] = 0.0
            ready_event.set()

            _prep_log_sum_read += t_read
            _prep_log_sum_slice += t_sp
            _prep_log_sum_bcast += t_bcast
            _prep_log_sum_total += t_total
            if prep_frame_idx % _PREP_LOG_INTERVAL == 0:
                n = _PREP_LOG_INTERVAL
                print(
                    f"[prep] avg {n}f: total={_prep_log_sum_total / n:.2f}ms "
                    f"read={_prep_log_sum_read / n:.2f} "
                    f"slice={_prep_log_sum_slice / n:.2f} "
                    f"bcast={_prep_log_sum_bcast / n:.2f}",
                    flush=True,
                )
                _prep_log_sum_read = 0.0
                _prep_log_sum_slice = 0.0
                _prep_log_sum_bcast = 0.0
                _prep_log_sum_total = 0.0

    except Exception:
        import traceback

        traceback.print_exc()
        shm_timings[6] = -1.0
        ready_event.set()


# ---------------------------------------------------------------------------
# BG postprocess: split 32-tile preds into 8 streams, NMS+merge each.
# ---------------------------------------------------------------------------


def _postprocess_worker_shm_multi(
    q_in: mp.Queue,
    stream_shifts: list,
    names_dict: dict,
    conf: float,
    iou: float,
    merge_iou: float,
    merge_class_agnostic: bool,
    merge_mode: str,
    merge_match: str,
    shm_preds: torch.Tensor,
    dets_q: mp.Queue,
):
    """Split the 32-tile prediction batch into 8 groups of 4 tiles each and
    run _fused_nms_merge per stream. Each stream pushes its own dets message
    tagged with stream_id so the browser can route it to the matching
    overlay canvas.

    scale_x/scale_y are 1.0 -- boxes are shipped in source (1280x1280)
    coords and the browser performs letterbox scaling per overlay canvas.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    _bg_cores = set(range(24, 32)) | set(range(56, 64))
    try:
        os.sched_setaffinity(0, _bg_cores)
        os.nice(19)
    except OSError:
        pass
    torch.set_num_threads(4)
    cv2.setNumThreads(2)

    ema_fps = 0.0
    fc = 0
    t_post_sum = 0.0
    t_wall_start = 0.0
    t_last_post = 0.0
    LOG_INTERVAL = 300
    # Alpha for the short-term FPS EMA sent to the browser HUD. Higher than the
    # lifetime `_EMA_ALPHA` because the HUD should react within a second or two
    # to pipeline load changes rather than creep toward a cumulative average.
    _HUD_FPS_ALPHA = 0.25
    nms_pool = ThreadPoolExecutor(max_workers=4)

    while True:
        try:
            item = q_in.get()
            if item is None:
                return

            pred_slot = item[0]
            preds_torch = shm_preds[pred_slot, :, :84, :8400]  # [32, 84, 8400]

            t_post_start = time.perf_counter()
            if fc == 0:
                t_wall_start = t_post_start

            # Per-stream NMS + merge. Slice 4 tiles out of the 32-tile batch
            # per stream and run _fused_nms_merge on that 4-tile sub-batch.
            for s in range(N_STREAMS):
                start = s * TILES_PER_STREAM
                end = start + TILES_PER_STREAM
                sub_preds = preds_torch[start:end]
                boxes_np, scores_np, cls_np = _fused_nms_merge(
                    sub_preds,
                    conf,
                    iou,
                    shifts=stream_shifts,
                    n_valid=TILES_PER_STREAM,
                    merge_iou=merge_iou,
                    class_agnostic=merge_class_agnostic,
                    scale_x=1.0,
                    scale_y=1.0,
                    pool=nms_pool,
                    merge_mode=merge_mode,
                    merge_match=merge_match,
                )

                _n = int(len(boxes_np))
                msg = {
                    "k": "dets",
                    "stream_id": s,
                    "frame_id": int(fc),
                    "n": _n,
                    "fps": float(ema_fps),
                    "boxes": boxes_np.astype(np.float32).tolist() if _n else [],
                    "scores": scores_np.astype(np.float32).tolist() if _n else [],
                    "cls": cls_np.astype(np.int32).tolist() if _n else [],
                }
                try:
                    dets_q.put_nowait(msg)
                except Exception:
                    try:
                        dets_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        dets_q.put_nowait(msg)
                    except Exception:
                        pass

            dt_post = time.perf_counter() - t_post_start
            fc += 1
            t_post_sum += dt_post

            # Instantaneous inter-frame FPS -> responsive EMA for the browser
            # HUD. Using dt between successive post-worker arrivals captures
            # end-to-end pipeline throughput (not just NMS time), and the EMA
            # keeps jitter visible without letting one slow frame dominate.
            wall_elapsed = time.perf_counter() - t_wall_start
            if fc > 1:
                dt_inter = time.perf_counter() - t_last_post
                inst_fps = 1.0 / max(dt_inter, 1e-9)
                ema_fps = _HUD_FPS_ALPHA * inst_fps + (1 - _HUD_FPS_ALPHA) * ema_fps
            else:
                ema_fps = 1.0 / max(dt_post, 1e-9)
                print(
                    f"{TAG} First frame post: {dt_post * 1000:.1f}ms " f"(8 streams x NMS+merge)",
                    flush=True,
                )
            t_last_post = time.perf_counter()

            if fc % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                throughput = fc / wall_elapsed
                print(
                    f"{TAG} BG avg {n}f: post={t_post_sum / n * 1000:.1f}ms  " f"Throughput: {throughput:.1f} FPS",
                    flush=True,
                )
                t_post_sum = 0.0

        except Exception:
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_sahi_5frame_pipelined(args):
    """Multi-stream pipelined YOLOv8L SAHI-640 on the full 8x4 = 32-device mesh."""
    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    try:
        os.sched_setaffinity(0, set(range(0, 24)) | set(range(32, 56)))
    except OSError:
        pass

    l1_small = yolov8l_l1_small_size_for_res(_TILE_SIZE_640, _TILE_SIZE_640)
    trace_region = 6_434_816

    if not args.input:
        print(f"{TAG} ERROR: --input is required", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"{TAG} ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    sample = src.peek()
    src_h, src_w = sample.shape[:2]
    src.release()

    if src_w != STREAM_W or src_h != STREAM_H:
        print(
            f"{TAG} WARN: source is {src_w}x{src_h}, expected {STREAM_W}x{STREAM_H}; " f"frames will be resized.",
            flush=True,
        )

    # Per-stream tile grid: 1280x1280 -> 2x2 = 4 tiles of 640x640.
    stream_grid = build_overlap_grid(STREAM_H, STREAM_W, _TILE_SIZE_640, _TILE_SIZE_640, add_whole_frame=False)
    assert (
        stream_grid.n_tiles == TILES_PER_STREAM
    ), f"expected {TILES_PER_STREAM} tiles per stream, got {stream_grid.n_tiles}"
    print(
        f"{TAG} Per-stream grid: {stream_grid.n_cols}x{stream_grid.n_rows} "
        f"= {stream_grid.n_tiles} tiles of {_TILE_SIZE_640}x{_TILE_SIZE_640}",
        flush=True,
    )
    print(
        f"{TAG} Total batch: {N_STREAMS} streams x {TILES_PER_STREAM} tiles "
        f"= {TOTAL_TILES} tiles on {MESH_ROWS}x{MESH_COLS} mesh",
        flush=True,
    )

    # Per-tile affine shifts for ONE stream -- reused identically for all 8
    # streams because each stream lives in its own 1280x1280 coordinate
    # frame. box_frame = box_tile * (1, 1) + (col_start, row_start).
    stream_shifts = [(1.0, 1.0, float(ts.col_start), float(ts.row_start)) for ts in stream_grid.tiles]

    # --- Open full 8x4 mesh --------------------------------------------------
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    sys_rows, sys_cols = sys_shape
    if sys_rows < MESH_ROWS or sys_cols < MESH_COLS:
        print(
            f"{TAG} ERROR: system mesh {sys_rows}x{sys_cols} cannot host required " f"{MESH_ROWS}x{MESH_COLS} sub-mesh",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    mesh_shape = ttnn.MeshShape(MESH_ROWS, MESH_COLS)
    print(
        f"{TAG} Opening mesh {MESH_ROWS}x{MESH_COLS}={TOTAL_TILES} "
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

    # --- Build runner (batch = TOTAL_TILES = 32) -----------------------------
    print(
        f"{TAG} Building YOLOv8lPerformantRunner " f"({_TILE_SIZE_640}x{_TILE_SIZE_640}, batch={TOTAL_TILES})...",
        flush=True,
    )
    runner = YOLOv8lPerformantRunner(
        mesh_device,
        device_batch_size=TOTAL_TILES,
        inp_h=_TILE_SIZE_640,
        inp_w=_TILE_SIZE_640,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    print(f"{TAG} Runner ready.", flush=True)

    # --- NMS / merge config --------------------------------------------------
    coco_names = _load_coco_names()
    names_dict = _coco_names_dict(coco_names)
    conf = max(args.conf, _CONF_FLOOR_640)
    print(
        f"{TAG} NMS conf={conf} iou={args.iou} "
        f"merge_iou={args.merge_threshold} class_agnostic={args.class_agnostic} "
        f"merge_mode={args.merge_mode} merge_match={args.merge_match}",
        flush=True,
    )

    # --- Pre-warm the JIT C++ tile extension --------------------------------
    # torch.utils.cpp_extension.load() uses a `lock` baton file in
    # ~/.cache/torch_extensions/.../<ext>/lock.  When a previous process is
    # killed mid-load, that file is orphaned (no kernel-level lock left) and
    # a future load() spins forever in FileBaton.wait().  Two-part defense:
    #   1) Defensively unlink any pre-existing baton at parent startup.
    #   2) Build the extension once HERE in the parent, so the spawned prep
    #      worker only has to find the cached .so (no baton dance at all).
    try:
        _ext_dir = os.path.expanduser("~/.cache/torch_extensions/py310_cpu/fused_tile_convert")
        _baton = os.path.join(_ext_dir, "lock")
        if os.path.exists(_baton):
            os.unlink(_baton)
            print(f"{TAG} cleared stale JIT baton {_baton}", flush=True)
    except Exception as _e:
        print(f"{TAG} baton cleanup warning: {_e}", flush=True)
    print(f"{TAG} pre-warming fused_tile_convert C++ ext...", flush=True)
    _load_fused_tile_ext()
    print(f"{TAG} C++ ext ready.", flush=True)

    # --- Shared memory -------------------------------------------------------
    _ctx = mp.get_context("spawn")
    PRED_RING = 2
    _phys_pred_h, _phys_pred_w = runner._phys_per_shard
    _log_pred_h, _log_pred_w = runner._log_per_shard
    shm_preds = torch.zeros(PRED_RING, TOTAL_TILES, _phys_pred_h, _phys_pred_w, dtype=torch.bfloat16).share_memory_()
    print(
        f"{TAG} shm_preds: [{PRED_RING}, {TOTAL_TILES}, {_phys_pred_h}, {_phys_pred_w}] "
        f"logical=[:, :, :{_log_pred_h}, :{_log_pred_w}] "
        f"({shm_preds.nelement() * 2 / 1e6:.1f} MB)",
        flush=True,
    )
    pred_write_idx = 0

    q_post: mp.Queue = _ctx.Queue(maxsize=4)

    # --- Split server (raw MP4 + per-stream SSE dets) ------------------------
    from models.demos.yolo_eval import _split_server as _split

    dets_q = _ctx.Queue(maxsize=256)
    server_proc = _ctx.Process(
        target=_split.run_server,
        args=(args.host, int(args.port), dets_q, args.input, STREAM_W, STREAM_H),
        daemon=True,
        name="sahi640-multi-split",
    )
    server_proc.start()
    print(
        f"{TAG} Split server: http://{args.host}:{args.port}/  " f"source={args.input} ({STREAM_W}x{STREAM_H})",
        flush=True,
    )

    # --- BG process ----------------------------------------------------------
    bg_proc = _ctx.Process(
        target=_postprocess_worker_shm_multi,
        args=(
            q_post,
            stream_shifts,
            names_dict,
            conf,
            args.iou,
            args.merge_threshold,
            args.class_agnostic,
            args.merge_mode,
            args.merge_match,
            shm_preds,
            dets_q,
        ),
        daemon=True,
        name="sahi640-multi-bg",
    )
    bg_proc.start()

    # --- Prep process --------------------------------------------------------
    _shm_buf0 = torch.zeros(TOTAL_TILES, 3, _TILE_SIZE_640, _TILE_SIZE_640, dtype=torch.bfloat16).share_memory_()
    _shm_buf1 = torch.zeros(TOTAL_TILES, 3, _TILE_SIZE_640, _TILE_SIZE_640, dtype=torch.bfloat16).share_memory_()
    shm_tensor_bufs = [_shm_buf0, _shm_buf1]
    shm_timings = torch.zeros(10, dtype=torch.float32).share_memory_()
    go_event = _ctx.Event()
    ready_event = _ctx.Event()
    stop_event = _ctx.Event()

    shm_mb = (_shm_buf0.nelement() + _shm_buf1.nelement()) * 2 / 1e6
    print(f"{TAG} Shared memory (tiles): {shm_mb:.0f} MB (double-buffered bf16)", flush=True)

    prep_proc = _ctx.Process(
        target=_prep_process_worker_multi,
        args=(
            args.input,
            stream_grid,
            shm_tensor_bufs,
            shm_timings,
            go_event,
            ready_event,
            stop_event,
            getattr(args, "pace", False),
        ),
        daemon=True,
        name="sahi640-multi-prep",
    )
    prep_proc.start()

    # --- Signal handling -----------------------------------------------------
    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"{TAG} 3-stage pipelined multi-stream loop "
        f"({N_STREAMS} streams x {TILES_PER_STREAM} tiles = {TOTAL_TILES})...",
        flush=True,
    )

    # ===================================================================
    # Main loop -- same submit / d2h / compose pipelining as the 640
    # pipeline, but without display canvas plumbing since the browser
    # plays the source file directly.
    # ===================================================================
    main_frame_idx = 0
    _submit_executor = ThreadPoolExecutor(max_workers=1)
    _prep_executor = ThreadPoolExecutor(max_workers=1)
    batch_count = 0
    drop_count = 0
    LOG_INTERVAL = 10
    t_batch_total_sum = 0.0
    t_wait_sum = 0.0
    t_submit_sum = 0.0
    t_compose_wait_sum = 0.0
    t_qput_sum = 0.0
    t_d2h_sum = 0.0
    t_compose_start_sum = 0.0

    try:
        go_event.set()
        while not ready_event.wait(timeout=0.5):
            if stop:
                return
        ready_event.clear()
        if shm_timings[6].item() < 0:
            print(f"{TAG} No frames available.", flush=True)
            return

        runner.submit(shm_tensor_bufs[main_frame_idx % 2])
        main_frame_idx += 1
        go_event.set()
        _prep_pending = True

        _compose_future = None
        _compose_enqueue_item = None
        _prepare_future = None

        print(
            f"{TAG} First frame: prep={shm_timings[5].item():.1f}ms "
            f"(read={shm_timings[1].item():.1f} sp={shm_timings[4].item():.1f})",
            flush=True,
        )

        while not stop:
            t_batch_start = time.perf_counter()

            if _prep_pending:
                while not ready_event.wait(timeout=0.5):
                    if stop:
                        break
                ready_event.clear()
                _next_valid = (not stop) and shm_timings[6].item() >= 0
            else:
                _next_valid = False
            t_after_wait = time.perf_counter()

            if not _next_valid:
                if _compose_future is not None:
                    _compose_future.result()
                    try:
                        q_post.put_nowait(_compose_enqueue_item)
                    except Exception:
                        pass
                    _compose_future = None
                break

            # Kick off prepare_input (host from_torch loop, ~2-3ms) on a
            # thread so it overlaps with d2h (~9ms) below.  enqueue_frame,
            # which queues device commands on CQ0/CQ1, must still run on
            # this thread to serialize with pcie_d2h() on CQ1.
            if _next_valid:
                _prepare_future = _prep_executor.submit(runner.prepare_input, shm_tensor_bufs[main_frame_idx % 2])
                main_frame_idx += 1
            t_after_submit = time.perf_counter()

            go_event.set()
            _prep_pending = True

            _dropped = False
            t_compose_wait_start = time.perf_counter()
            if _compose_future is not None:
                _compose_future.result()
                t_after_compose_wait = time.perf_counter()
                try:
                    q_post.put_nowait(_compose_enqueue_item)
                except Exception:
                    _dropped = True
                t_after_qput = time.perf_counter()
                _compose_future = None
            else:
                t_after_compose_wait = t_compose_wait_start
                t_after_qput = t_compose_wait_start

            has_result = runner.pcie_d2h()
            t_after_d2h = time.perf_counter()

            if has_result:
                pred_slot = pred_write_idx % PRED_RING
                _compose_future = _submit_executor.submit(runner.compose, dest=shm_preds[pred_slot])
                pred_write_idx += 1
                _compose_enqueue_item = (pred_slot,)

            # Finish the submit pipeline: collect the prepared host tensor
            # (which ran in parallel with d2h) and queue the device commands.
            if _prepare_future is not None:
                tt_inputs_host = _prepare_future.result()
                runner.enqueue_frame(tt_inputs_host)
                _prepare_future = None
            t_after_compose_start = time.perf_counter()

            dt_batch = t_after_compose_start - t_batch_start
            batch_count += 1
            if _dropped:
                drop_count += 1
            t_batch_total_sum += dt_batch
            t_wait_sum += t_after_wait - t_batch_start
            t_submit_sum += t_after_submit - t_after_wait
            t_compose_wait_sum += t_after_compose_wait - t_compose_wait_start
            t_qput_sum += t_after_qput - t_after_compose_wait
            t_d2h_sum += t_after_d2h - t_after_qput
            t_compose_start_sum += t_after_compose_start - t_after_d2h

            if batch_count == 1:
                fps = 1.0 / max(dt_batch, 1e-9)
                print(
                    f"{TAG} Batch 1: {dt_batch * 1000:.1f}ms ({fps:.1f} FPS)",
                    flush=True,
                )

            if batch_count % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                avg_ms = t_batch_total_sum / n * 1000
                fps = n / t_batch_total_sum
                print(
                    f"{TAG} avg {n}f: {avg_ms:.1f}ms/f ({fps:.1f} FPS)  drops={drop_count} | "
                    f"wait={t_wait_sum / n * 1000:.2f} "
                    f"submit={t_submit_sum / n * 1000:.2f} "
                    f"cwait={t_compose_wait_sum / n * 1000:.2f} "
                    f"qput={t_qput_sum / n * 1000:.2f} "
                    f"d2h={t_d2h_sum / n * 1000:.2f} "
                    f"cstart={t_compose_start_sum / n * 1000:.2f}",
                    flush=True,
                )
                t_batch_total_sum = 0.0
                t_wait_sum = 0.0
                t_submit_sum = 0.0
                t_compose_wait_sum = 0.0
                t_qput_sum = 0.0
                t_d2h_sum = 0.0
                t_compose_start_sum = 0.0

        if _compose_future is not None:
            try:
                _compose_future.result()
                q_post.put(_compose_enqueue_item, timeout=2)
            except Exception:
                pass

        if batch_count > 0 and not stop:
            try:
                last_preds = runner.flush_pipeline(mesh_composer=output_mesh_composer)
                if last_preds is not None:
                    pred_slot = pred_write_idx % PRED_RING
                    shm_preds[pred_slot, :TOTAL_TILES, :_log_pred_h, :_log_pred_w].copy_(last_preds[:TOTAL_TILES])
                    pred_write_idx += 1
                    q_post.put((pred_slot,), timeout=2)
            except Exception:
                pass

        _submit_executor.shutdown(wait=False)

    finally:
        print(
            f"\n{TAG} Shutting down... ({batch_count} frames processed)",
            flush=True,
        )

        stop_event.set()
        go_event.set()
        prep_proc.join(timeout=5)
        if prep_proc.is_alive():
            prep_proc.kill()
            prep_proc.join(timeout=2)

        try:
            while not q_post.empty():
                q_post.get_nowait()
        except Exception:
            pass

        try:
            q_post.put_nowait(None)
        except Exception:
            pass
        bg_proc.join(timeout=5)
        if bg_proc.is_alive():
            bg_proc.kill()
            bg_proc.join(timeout=2)

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
        description=f"YOLOv8L SAHI-640 multi-stream inference "
        f"({N_STREAMS} streams x {TILES_PER_STREAM} tiles = {TOTAL_TILES} on "
        f"{MESH_ROWS}x{MESH_COLS} mesh).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help=f"Path to a {STREAM_W}x{STREAM_H} MP4 source.")
    p.add_argument("--conf", type=float, default=0.25, help="NMS confidence.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--serve", action="store_true", help="Serve split-delivery HTTP.")
    p.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    p.add_argument("--port", type=int, default=9090, help="HTTP port.")
    p.add_argument("--merge-threshold", type=float, default=0.4, help="Cross-tile merge IoU.")
    p.add_argument("--class-agnostic", action="store_true", help="Merge across classes.")
    p.add_argument(
        "--merge-mode",
        choices=["nms", "greedy-nmm", "nmm", "wbf"],
        default="greedy-nmm",
    )
    p.add_argument("--merge-match", choices=["iou", "ios"], default="ios")
    p.add_argument("--pace", action="store_true", help="Pace prep to source FPS.")
    # Supervisor-compat flags (accepted, ignored in split-only mode).
    p.add_argument("--frame-width", type=int, default=STREAM_W)
    p.add_argument("--frame-height", type=int, default=STREAM_H)
    p.add_argument("--serve-split", action="store_true", help="Split delivery (default on).")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        run_sahi_5frame_pipelined(args)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"{TAG} main done.", flush=True)


if __name__ == "__main__":
    main()
