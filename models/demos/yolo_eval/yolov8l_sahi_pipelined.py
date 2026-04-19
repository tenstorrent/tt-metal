#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Pipelined SAHI worker for YOLOv8L 4K inference on a multi-chip mesh.

Optimisations over the sequential ``_run_sahi_worker``:

1. **Fused slice + preprocess** — ``slice_and_preprocess()`` copies tiles directly
   into a pre-allocated NHWC buffer, then fuses BGR->RGB + HWC->CHW + bfloat16
   conversion.  Eliminates N intermediate tile arrays and the ``np.stack`` copy.

2. **Overlapped preprocess** — the next frame's read + slice + preprocess runs in
   a background thread during ``synchronize_device()`` (C-level wait releases GIL).
   Hides ~20 ms of host preprocessing behind ~65 ms of device compute.

3. **2-stage pipeline** — main process handles device inference + d2h;
   child process handles NMS + merge + draw + encode.  Separate GILs eliminate
   contention between host torch/numpy work and drawing/encoding.

Usage (called from ``yolov8l_native_vs_sahi_demo.py`` when ``--pipeline`` is set):

    python models/demos/yolo_eval/yolov8l_native_vs_sahi_demo.py \\
        --input path/to/4k_video.mp4 --sahi-only --pipeline --serve --port 9090
"""
from __future__ import annotations

import math
import multiprocessing as mp
import signal
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms, nms

# Re-use helpers from the main demo script (FrameSource, draw_hud, etc.).
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

# ---------------------------------------------------------------------------
# TileGrid -- pre-computed slice geometry (built once per resolution)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    """One tile's crop region inside the source frame."""

    row_start: int
    col_start: int
    src_h: int  # actual height in the source (may be < tile_h at bottom edge)
    src_w: int  # actual width in the source  (may be < tile_w at right edge)
    needs_pad: bool


@dataclass(frozen=True)
class TileGrid:
    """Pre-computed tiling geometry for a fixed frame size."""

    tile_h: int
    tile_w: int
    frame_h: int
    frame_w: int
    n_rows: int
    n_cols: int
    n_tiles: int
    tiles: tuple[TileSpec, ...]

    @staticmethod
    def build(frame_h: int, frame_w: int, tile_h: int = _TILE_SIZE, tile_w: int = _TILE_SIZE) -> "TileGrid":
        specs: list[TileSpec] = []
        for r_start in range(0, frame_h, tile_h):
            for c_start in range(0, frame_w, tile_w):
                sh = min(tile_h, frame_h - r_start)
                sw = min(tile_w, frame_w - c_start)
                specs.append(TileSpec(r_start, c_start, sh, sw, needs_pad=(sh < tile_h or sw < tile_w)))
        n_rows = max(1, (frame_h + tile_h - 1) // tile_h)
        n_cols = max(1, (frame_w + tile_w - 1) // tile_w)
        return TileGrid(
            tile_h=tile_h,
            tile_w=tile_w,
            frame_h=frame_h,
            frame_w=frame_w,
            n_rows=n_rows,
            n_cols=n_cols,
            n_tiles=len(specs),
            tiles=tuple(specs),
        )


# ---------------------------------------------------------------------------
# Lightweight draw helper for scaled-down canvas (avoids SAHI object overhead)
# ---------------------------------------------------------------------------


@dataclass
class _ScaledPred:
    minx: float
    miny: float
    maxx: float
    maxy: float
    score: float
    cat_name: str


def _draw_scaled_preds(img: np.ndarray, preds: list[_ScaledPred]) -> np.ndarray:
    h, w = img.shape[:2]
    for p in preds:
        fx1, fy1 = float(p.minx), float(p.miny)
        fx2, fy2 = float(p.maxx), float(p.maxy)
        # Skip garbage coordinates (model quantization artifacts)
        if not (math.isfinite(fx1) and math.isfinite(fy1) and math.isfinite(fx2) and math.isfinite(fy2)):
            continue
        if max(abs(fx1), abs(fy1), abs(fx2), abs(fy2)) > 10000:
            continue
        x1 = max(0, min(int(fx1), w - 1))
        y1 = max(0, min(int(fy1), h - 1))
        x2 = max(0, min(int(fx2), w - 1))
        y2 = max(0, min(int(fy2), h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        # Skip oversized boxes — quantization artifacts produce tile-sized detections
        if (x2 - x1) > w * 0.25 or (y2 - y1) > h * 0.25:
            continue
        label = f"{p.cat_name} {p.score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Fused slice + preprocess (eliminates intermediate tile arrays + np.stack)
# ---------------------------------------------------------------------------

_PAD_VALUE = 114  # matches letterbox grey


def slice_and_preprocess(
    frame_bgr: np.ndarray,
    grid: TileGrid,
    n_devices: int,
) -> tuple[torch.Tensor, list[tuple[int, int]], dict]:
    """Slice a BGR frame into tiles and convert directly to a bfloat16 NCHW tensor.

    Fuses the previous ``slice_frame_direct`` + ``preprocess_tiles_batch`` into a
    single pass: tile crops are written directly into a pre-allocated NHWC buffer,
    eliminating N intermediate tile arrays and the ``np.stack`` copy.

    Returns ``(tensor, shifts, timings)`` -- ``tensor`` is ``[N, 3, H, W]`` bfloat16,
    ``shifts`` is ``[(col_start, row_start), ...]`` per tile for SAHI merge.
    ``timings`` has ``slice_ms`` and ``preprocess_ms``.
    """
    N = max(grid.n_tiles, n_devices)
    H, W = grid.tile_h, grid.tile_w

    # --- Slice: crop tiles from frame (+ letterbox pad) ---
    t0 = time.perf_counter()
    buf = np.empty((N, H, W, 3), dtype=np.uint8)
    shifts: list[tuple[int, int]] = []

    for i, ts in enumerate(grid.tiles):
        if ts.needs_pad:
            buf[i] = _PAD_VALUE
        buf[i, : ts.src_h, : ts.src_w] = frame_bgr[
            ts.row_start : ts.row_start + ts.src_h,
            ts.col_start : ts.col_start + ts.src_w,
        ]
        shifts.append((ts.col_start, ts.row_start))

    for i in range(grid.n_tiles, N):
        buf[i] = 0
        shifts.append((0, 0))
    t_slice = (time.perf_counter() - t0) * 1000

    # --- Preprocess: BGR->RGB + HWC->CHW + bfloat16 + /255 ---
    t0 = time.perf_counter()
    arr = np.ascontiguousarray(buf[:, :, :, ::-1].transpose(0, 3, 1, 2))
    t_bgr_rgb_transpose = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    t = torch.from_numpy(arr).to(torch.bfloat16)
    t_to_bf16 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    t = t.div_(255.0)
    t_normalize = (time.perf_counter() - t0) * 1000

    t_preprocess = t_bgr_rgb_transpose + t_to_bf16 + t_normalize

    timings = {
        "slice_ms": t_slice,
        "preprocess_ms": t_preprocess,
        "bgr_rgb_transpose_ms": t_bgr_rgb_transpose,
        "to_bf16_ms": t_to_bf16,
        "normalize_ms": t_normalize,
    }
    return t, shifts, timings


# ---------------------------------------------------------------------------
# Torch-based cross-tile merge (replaces SAHI ObjectPrediction + GreedyNMMPostprocess)
# ---------------------------------------------------------------------------


def torch_merge_tiles(
    results_list: list[dict],
    shifts: list[tuple[int, int]],
    n_valid: int,
    names_dict: dict,
    conf: float,
    merge_iou: float = 0.5,
    class_agnostic: bool = False,
) -> list[_ScaledPred]:
    """Merge per-tile NMS results into full-image detections using torchvision NMS.

    Replaces the SAHI pipeline (result_to_object_predictions -> shift -> GreedyNMMPostprocess)
    with pure tensor ops + C++ NMS.  ~1-2 ms vs ~100 ms under GIL contention.

    Parameters
    ----------
    results_list : list of dicts from ``postprocess()``
        Each dict has ``boxes`` -> ``{xyxy, conf, cls}``.
    shifts : list of (shift_x, shift_y) per tile.
    n_valid : number of real tiles (rest are padding).
    names_dict : {cls_id: name}.
    conf : confidence threshold (already applied by NMS but we double-check).
    merge_iou : IoU threshold for cross-tile merge NMS.
    class_agnostic : if True, merge across classes.

    Returns
    -------
    list[_ScaledPred] in full-image coordinates.
    """
    all_boxes = []
    all_scores = []
    all_cls = []

    for j in range(n_valid):
        r = results_list[j]
        boxes_xyxy = r["boxes"]["xyxy"]  # (K, 4) float32 tensor -- tile coords
        scores = r["boxes"]["conf"]  # (K,)
        cls_ids = r["boxes"]["cls"]  # (K,)

        if boxes_xyxy.numel() == 0:
            continue

        # Shift tile coords -> full-image coords (tensor arithmetic, no Python loop)
        sx, sy = shifts[j]
        shifted = boxes_xyxy.clone()
        shifted[:, 0] += sx
        shifted[:, 2] += sx
        shifted[:, 1] += sy
        shifted[:, 3] += sy

        all_boxes.append(shifted)
        all_scores.append(scores)
        all_cls.append(cls_ids)

    if not all_boxes:
        return []

    boxes_cat = torch.cat(all_boxes, dim=0)  # (N_total, 4)
    scores_cat = torch.cat(all_scores, dim=0)  # (N_total,)
    cls_cat = torch.cat(all_cls, dim=0).int()  # (N_total,)

    # Cross-tile merge NMS (C++ backend, releases GIL)
    if class_agnostic:
        keep = nms(boxes_cat, scores_cat, merge_iou)
    else:
        keep = batched_nms(boxes_cat, scores_cat, cls_cat, merge_iou)

    boxes_kept = boxes_cat[keep]
    scores_kept = scores_cat[keep]
    cls_kept = cls_cat[keep]

    # Build lightweight prediction list (no SAHI objects)
    preds = []
    for i in range(boxes_kept.shape[0]):
        s = scores_kept[i].item()
        if s < conf:
            continue
        c = cls_kept[i].item()
        preds.append(
            _ScaledPred(
                minx=boxes_kept[i, 0].item(),
                miny=boxes_kept[i, 1].item(),
                maxx=boxes_kept[i, 2].item(),
                maxy=boxes_kept[i, 3].item(),
                score=s,
                cat_name=names_dict.get(c, str(c)),
            )
        )
    return preds


# ---------------------------------------------------------------------------
# Standalone per-tile NMS (no ttnn / common_demo_utils dependency)
# ---------------------------------------------------------------------------


def _tile_nms(
    preds_batch: torch.Tensor,
    conf: float,
    iou: float,
    max_det: int = 300,
) -> list[dict]:
    """Per-tile NMS on raw ``[B, 84, N]`` model output.

    Standalone replacement for ``common_demo_utils.postprocess`` — uses only
    torch + torchvision so the bg process never imports ttnn.  Also skips the
    ``scale_boxes`` call (no-op for 1280x1280 tiles) and the dummy-tensor
    bookkeeping.

    Returns ``list[dict]`` with ``boxes.xyxy / conf / cls`` — same contract
    as ``postprocess`` results, consumed by ``torch_merge_tiles``.
    """
    bs = preds_batch.shape[0]
    nc = preds_batch.shape[1] - 4  # 80 for COCO

    # [B, 84, N] -> [B, N, 84]
    preds = preds_batch.transpose(-1, -2)

    # xywh -> xyxy (new tensor, no in-place aliasing issue)
    xy = preds[..., :2]
    half_wh = preds[..., 2:4] / 2
    box_xyxy = torch.cat([xy - half_wh, xy + half_wh], dim=-1)

    cls_scores = preds[..., 4:]  # [B, N, nc]
    max_cls = cls_scores.amax(dim=-1)  # [B, N]

    results: list[dict] = []
    for xi in range(bs):
        mask = max_cls[xi] > conf
        boxes = box_xyxy[xi][mask]
        scores = cls_scores[xi][mask]

        if boxes.shape[0] == 0:
            results.append({"boxes": {"xyxy": boxes[:, :4], "conf": boxes[:, 0], "cls": boxes[:, 0]}})
            continue

        conf_val, cls_id = scores.max(1, keepdim=True)
        x = torch.cat([boxes, conf_val, cls_id.float()], dim=1)  # [K, 6]
        x = x[conf_val.view(-1) > conf]

        if x.shape[0] == 0:
            results.append({"boxes": {"xyxy": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]}})
            continue

        # Class-aware NMS (offset boxes by class * max_wh)
        c = x[:, 5:6] * 7680
        i = nms(x[:, :4] + c, x[:, 4], iou)
        i = i[:max_det]
        det = x[i]
        results.append({"boxes": {"xyxy": det[:, :4], "conf": det[:, 4], "cls": det[:, 5]}})

    return results


# ---------------------------------------------------------------------------
# Pipelined SAHI worker
# ---------------------------------------------------------------------------


def _postprocess_worker(
    q_in: mp.Queue,
    n_devices: int,
    names_dict: dict,
    conf: float,
    iou: float,
    merge_iou: float,
    merge_class_agnostic: bool,
    display_width: int,
    jpeg_quality: int,
    frame_file: str,
    hud_label: str,
    output_path: str | None,
    is_image: bool,
):
    """Background PROCESS for NMS + merge + draw + encode.

    Runs in a separate process with its own GIL -- no contention with the
    main process's tensor / host_prep / device work.  Receives preds_torch
    + frame_4k through ``mp.Queue`` (pickle, ~3 ms for 25 MB frame memcpy).
    Uses standalone ``_tile_nms`` for NMS (no ttnn dependency).
    """
    TAG = "[sahi-bg]"

    ema_fps = 0.0
    fc = 0
    t_post_sum = t_nms_sum = t_merge_sum = t_scale_sum = t_draw_sum = t_encode_sum = 0.0
    t_wall_start = 0.0
    LOG_INTERVAL = 10

    try:
        while True:
            item = q_in.get()
            if item is None:
                return

            (preds_torch, frame_4k, shifts, n_valid, n_batch) = item

            t_post_start = time.perf_counter()
            if fc == 0:
                t_wall_start = t_post_start

            # --- Per-tile NMS: transpose + xywh->xyxy + conf filter + NMS ---
            t0 = time.perf_counter()
            results_list = _tile_nms(preds_torch, conf, iou)
            t_nms = time.perf_counter() - t0

            # --- Cross-tile merge: shift coords + batched_nms ---
            t0 = time.perf_counter()
            merged_preds = torch_merge_tiles(
                results_list,
                shifts,
                n_valid,
                names_dict,
                conf=conf,
                merge_iou=merge_iou,
                class_agnostic=merge_class_agnostic,
            )
            t_merge = time.perf_counter() - t0

            # --- Scale 4K -> display width ---
            t0 = time.perf_counter()
            if display_width > 0:
                canvas = scale_to_width(frame_4k, display_width)
                sx = canvas.shape[1] / frame_4k.shape[1]
                sy = canvas.shape[0] / frame_4k.shape[0]
                scaled = [
                    _ScaledPred(
                        minx=p.minx * sx,
                        miny=p.miny * sy,
                        maxx=p.maxx * sx,
                        maxy=p.maxy * sy,
                        score=p.score,
                        cat_name=p.cat_name,
                    )
                    for p in merged_preds
                ]
            else:
                canvas = frame_4k.copy()
                sx = sy = 1.0
                scaled = merged_preds
            t_scale = time.perf_counter() - t0

            # --- Draw bounding boxes + labels ---
            t0 = time.perf_counter()
            _draw_scaled_preds(canvas, scaled)
            t_draw = time.perf_counter() - t0

            # --- HUD overlay + JPEG encode + write to frame file ---
            t0 = time.perf_counter()
            canvas = draw_hud(canvas, hud_label, ema_fps)
            write_frame(frame_file, canvas, jpeg_quality)
            t_encode = time.perf_counter() - t0

            # --- Bookkeeping ---
            dt_post = time.perf_counter() - t_post_start
            fc += 1
            t_post_sum += dt_post
            t_nms_sum += t_nms
            t_merge_sum += t_merge
            t_scale_sum += t_scale
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
                    f"scale={t_scale * 1000:.1f} draw={t_draw * 1000:.1f} "
                    f"encode={t_encode * 1000:.1f})",
                    flush=True,
                )

            if fc % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                throughput = fc / wall_elapsed
                print(
                    f"{TAG} BG avg {n}f: post={t_post_sum / n * 1000:.1f}ms "
                    f"(nms={t_nms_sum / n * 1000:.1f} "
                    f"merge={t_merge_sum / n * 1000:.1f} "
                    f"scale={t_scale_sum / n * 1000:.1f} "
                    f"draw={t_draw_sum / n * 1000:.1f} "
                    f"encode={t_encode_sum / n * 1000:.1f})  "
                    f"Throughput: {throughput:.1f} FPS",
                    flush=True,
                )
                t_post_sum = t_nms_sum = t_merge_sum = t_scale_sum = t_draw_sum = t_encode_sum = 0.0

            if is_image:
                if output_path:
                    cv2.imwrite(output_path, canvas)
                    print(f"{TAG} Saved to {output_path}", flush=True)
                print(f"{TAG} Image done.", flush=True)

    except Exception:
        import traceback

        traceback.print_exc()


def _prep_process_worker(
    video_path: str,
    grid: TileGrid,
    n_devices: int,
    shm_tensor: torch.Tensor,
    shm_frame: torch.Tensor,
    shm_shifts: torch.Tensor,
    shm_timings: torch.Tensor,
    go_event,
    ready_event,
    stop_event,
    frame_h: int,
    frame_w: int,
):
    """Separate PROCESS for frame read + slice + preprocess.

    Runs with its own GIL so numpy/torch work never contends with the main
    process's ttnn operations.  Writes directly into pre-allocated shared
    memory tensors — zero pickle, zero copy on the consumer side.

    Protocol: main sets ``go_event``; prep writes into shared buffers then
    sets ``ready_event``.  Main sets ``stop_event`` to shut down.
    """
    src = FrameSource(video_path)
    n_tiles = grid.n_tiles

    try:
        while not stop_event.is_set():
            go_event.wait()
            go_event.clear()
            if stop_event.is_set():
                return

            t0 = time.perf_counter()
            ok_r, frame_r = src.read()
            if not ok_r:
                src.reset()
                ok_r, frame_r = src.read()
            if not ok_r:
                # Signal ready with n_valid=0 to indicate end
                shm_timings[6] = -1.0  # sentinel for "no frame"
                ready_event.set()
                continue
            t_read = (time.perf_counter() - t0) * 1000

            tensor_r, shifts_r, sp_timings = slice_and_preprocess(frame_r, grid, n_devices)
            sp_timings["read_ms"] = t_read

            # Write into shared buffers (single memcpy each, no pickle)
            t0_ipc = time.perf_counter()
            shm_tensor.copy_(tensor_r)
            # Copy frame — use view to handle the exact frame dimensions
            shm_frame[:frame_h, :frame_w, :].copy_(torch.from_numpy(frame_r))
            # Pack shifts into shared tensor: [(col, row), ...] for each tile
            for i, (sx, sy) in enumerate(shifts_r):
                shm_shifts[i, 0] = sx
                shm_shifts[i, 1] = sy
            # Pack timings: [read, slice, preprocess, bgr, bf16, norm, ipc]
            shm_timings[0] = sp_timings["read_ms"]
            shm_timings[1] = sp_timings["slice_ms"]
            shm_timings[2] = sp_timings["preprocess_ms"]
            shm_timings[3] = sp_timings["bgr_rgb_transpose_ms"]
            shm_timings[4] = sp_timings["to_bf16_ms"]
            shm_timings[5] = sp_timings["normalize_ms"]
            shm_timings[6] = (time.perf_counter() - t0_ipc) * 1000  # ipc_ms

            ready_event.set()

    except Exception:
        import traceback

        traceback.print_exc()
        shm_timings[6] = -1.0  # signal error to main
        ready_event.set()  # unblock main on error


def run_sahi_worker_pipelined(args):
    """3-stage pipelined YOLOv8L SAHI:
    - prep process:  read + slice + preprocess (own GIL, no contention)
    - main process:  host_prep + h2d + device compute + d2h
    - bg process:    NMS + merge + draw + encode (own GIL)

    Key: prep runs in a PROCESS (not thread) so numpy/torch work never competes
    for the main process's GIL.  Tensor is shared via /dev/shm (zero-copy).
    """
    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers, preprocess
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    TAG = "[sahi-pipe]"
    l1_small = yolov8l_l1_small_size_for_res(1280, 1280)
    trace_region = 35_000_000
    in_res = (_TILE_SIZE, _TILE_SIZE)

    # --- Frame source --------------------------------------------------
    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"{TAG} ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    sample = src.peek()
    frame_h, frame_w = sample.shape[:2]

    # --- Pre-compute tile grid -----------------------------------------
    grid = TileGrid.build(frame_h, frame_w, _TILE_SIZE, _TILE_SIZE)
    n_devices = grid.n_tiles  # e.g. 6 for 3840x2160

    print(
        f"{TAG} TileGrid: {frame_w}x{frame_h} -> {grid.n_cols}x{grid.n_rows} "
        f"= {grid.n_tiles} tiles of {_TILE_SIZE}x{_TILE_SIZE}",
        flush=True,
    )
    for i, ts in enumerate(grid.tiles):
        print(
            f"{TAG}   tile {i}: start=({ts.col_start},{ts.row_start}) "
            f"src_size={ts.src_w}x{ts.src_h} pad={ts.needs_pad}",
            flush=True,
        )

    # --- Mesh shape ----------------------------------------------------
    if args.mesh_shape == "auto":
        mesh_rows, mesh_cols = grid.n_rows, grid.n_cols
    else:
        mesh_rows, mesh_cols = (int(x) for x in args.mesh_shape.split("x"))
        assert mesh_rows * mesh_cols == n_devices, (
            f"--mesh-shape {args.mesh_shape} = {mesh_rows * mesh_cols} devices " f"but tile grid has {n_devices} tiles"
        )

    mesh_shape = ttnn.MeshShape(mesh_rows, mesh_cols)
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    print(
        f"{TAG} System mesh: {sys_shape[0]}x{sys_shape[1]} = {sys_shape[0] * sys_shape[1]} devices",
        flush=True,
    )
    print(
        f"{TAG} Opening sub-mesh {mesh_rows}x{mesh_cols} = {n_devices} devices",
        flush=True,
    )

    # --- Open mesh device + runner -------------------------------------
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        l1_small_size=l1_small,
        trace_region_size=trace_region,
        num_command_queues=2,
    )
    mesh_device.enable_program_cache()

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    print(f"{TAG} Building YOLOv8lPerformantRunner (1280x1280, batch={n_devices})...", flush=True)
    runner = YOLOv8lPerformantRunner(
        mesh_device,
        device_batch_size=n_devices,
        inp_h=1280,
        inp_w=1280,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    print(f"{TAG} ready.", flush=True)

    # --- Merge config ----------------------------------------------------
    coco_names = _load_coco_names()
    names_dict = _coco_names_dict(coco_names)
    conf = max(args.conf, _CONF_FLOOR_1280)
    merge_iou = args.sahi_merge_threshold
    merge_class_agnostic = args.sahi_class_agnostic

    do_full_pred = args.perform_standard_pred
    print(
        f"{TAG} Merge: torchvision NMS (iou={merge_iou}, class_agnostic={merge_class_agnostic})  "
        f"conf={conf}  perform_standard_pred={do_full_pred}",
        flush=True,
    )

    # --- Background PROCESS (separate GIL) -----------------------------
    # Use 'spawn' context so the child process starts clean without
    # inheriting ttnn device state from the parent (fork would duplicate
    # device handles / internal threads and hang).
    _ctx = mp.get_context("spawn")
    q_post: mp.Queue = _ctx.Queue(maxsize=4)
    hud_label = f"YOLOv8L SAHI 4K pipe ({mesh_rows}x{mesh_cols}={n_devices} chips)"

    bg_proc = _ctx.Process(
        target=_postprocess_worker,
        args=(
            q_post,
            n_devices,
            names_dict,
            conf,
            args.iou,
            merge_iou,
            merge_class_agnostic,
            args.display_width,
            args.jpeg_quality,
            args._frame_file,
            hud_label,
            args.output if hasattr(args, "output") else None,
            src.is_image,
        ),
        daemon=True,
        name="sahi-postprocess",
    )
    bg_proc.start()

    # --- Prep PROCESS (separate GIL) ------------------------------------
    # Pre-allocated shared-memory tensors (backed by /dev/shm).  Prep process
    # writes directly into them; main reads with zero-copy for the bf16 tensor
    # and a numpy copy for frame_4k (needed because bg post-process pickles it).
    _N_tiles = max(grid.n_tiles, n_devices)
    shm_tensor = torch.zeros(_N_tiles, 3, _TILE_SIZE, _TILE_SIZE, dtype=torch.bfloat16).share_memory_()
    shm_frame = torch.zeros(frame_h, frame_w, 3, dtype=torch.uint8).share_memory_()
    shm_shifts = torch.zeros(_N_tiles, 2, dtype=torch.int32).share_memory_()
    shm_timings = torch.zeros(7, dtype=torch.float32).share_memory_()
    go_event = _ctx.Event()
    ready_event = _ctx.Event()
    stop_event = _ctx.Event()

    prep_proc = _ctx.Process(
        target=_prep_process_worker,
        args=(
            src.path,
            grid,
            n_devices,
            shm_tensor,
            shm_frame,
            shm_shifts,
            shm_timings,
            go_event,
            ready_event,
            stop_event,
            frame_h,
            frame_w,
        ),
        daemon=True,
        name="sahi-prep",
    )
    prep_proc.start()

    # ===================================================================
    # Signal handling
    # ===================================================================
    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"{TAG} 3-stage pipeline (prep-proc + main + bg-proc) running...", flush=True)

    # ===================================================================
    # Main process: device inference + d2h  (preprocess overlapped)
    # ===================================================================
    frame_count_main = 0
    LOG_INTERVAL_MAIN = 10
    t_prep_wait_sum = t_device_sum = 0.0
    t_host_prep_sum = t_h2d_trace_sum = t_dev_compute_sum = 0.0
    t_d2h_sum = t_enqueue_sum = t_main_total_sum = 0.0
    t_prep_read_sum = t_prep_slice_sum = t_prep_preprocess_sum = 0.0
    t_prep_bgr_sum = t_prep_bf16_sum = t_prep_norm_sum = t_prep_ipc_sum = 0.0

    try:
        # --- Request first frame from prep process --------------------
        go_event.set()
        ready_event.wait()
        ready_event.clear()
        if shm_timings[6].item() < 0:
            print(f"{TAG} No frames available.", flush=True)
            return
        # Tensor: zero-copy view (safe — runner.run reads before next go signal)
        cur_tensor = shm_tensor
        # Frame: must copy (bg process pickles it after next go overwrites shm)
        cur_frame_4k = shm_frame.numpy().copy()
        cur_shifts = [(int(shm_shifts[i, 0].item()), int(shm_shifts[i, 1].item())) for i in range(_N_tiles)]
        _first_sp = {
            "read_ms": shm_timings[0].item(),
            "slice_ms": shm_timings[1].item(),
            "preprocess_ms": shm_timings[2].item(),
            "bgr_rgb_transpose_ms": shm_timings[3].item(),
            "to_bf16_ms": shm_timings[4].item(),
            "normalize_ms": shm_timings[5].item(),
        }
        print(
            f"{TAG} First preprocess (prep-proc): "
            f"read={_first_sp['read_ms']:.1f} "
            f"slice={_first_sp['slice_ms']:.1f} "
            f"preprocess={_first_sp['preprocess_ms']:.1f}"
            f"[bgr_rgb_chw={_first_sp['bgr_rgb_transpose_ms']:.1f} "
            f"to_bf16={_first_sp['to_bf16_ms']:.1f} "
            f"normalize={_first_sp['normalize_ms']:.1f}]  (ms)",
            flush=True,
        )

        while not stop:
            t_frame_start = time.perf_counter()

            # --- Device inference (tensor was pre-computed) ------------
            t0 = time.perf_counter()
            preds = runner.run(torch_input_tensor=cur_tensor)
            t_submit = time.perf_counter() - t0

            # Signal prep AFTER runner.run — host_prep has already copied
            # the tensor to device DRAM, so shm_tensor can safely be
            # overwritten.  Prep runs truly in parallel with device compute.
            if not src.is_image:
                go_event.set()
                _prep_pending = True
            else:
                _prep_pending = False

            t0_dev = time.perf_counter()
            ttnn.synchronize_device(mesh_device)  # GIL released -> prep runs
            t_dev_compute = time.perf_counter() - t0_dev
            t_device = t_submit + t_dev_compute
            t_host_prep = runner.last_timing["host_prep_ms"]
            t_h2d_trace = runner.last_timing["h2d_and_trace_ms"]

            # --- Collect prep result (from shared memory, no pickle) ---
            t0 = time.perf_counter()
            if _prep_pending:
                ready_event.wait()
                ready_event.clear()
                _next_valid = shm_timings[6].item() >= 0
            else:
                _next_valid = False
            t_prep_wait = time.perf_counter() - t0
            # Extract prep sub-timings from shared buffer
            if _prep_pending and _next_valid:
                _prep_timings = {
                    "read_ms": shm_timings[0].item(),
                    "slice_ms": shm_timings[1].item(),
                    "preprocess_ms": shm_timings[2].item(),
                    "bgr_rgb_transpose_ms": shm_timings[3].item(),
                    "to_bf16_ms": shm_timings[4].item(),
                    "normalize_ms": shm_timings[5].item(),
                    "ipc_ms": shm_timings[6].item(),
                }
            else:
                _prep_timings = None

            # --- D2H (GIL-contention-free) ----------------------------
            t0 = time.perf_counter()
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds_torch = ttnn.to_torch(
                preds,
                dtype=torch.float32,
                mesh_composer=output_mesh_composer,
            )
            t_d2h = time.perf_counter() - t0

            n_valid = grid.n_tiles
            n_batch = n_devices

            # --- Optional: full-image prediction for large objects -----
            if do_full_pred:
                full_im_tensor = preprocess([cur_frame_4k], res=in_res)
                full_im_tensor = full_im_tensor.repeat(n_devices, 1, 1, 1)
                full_preds = runner.run(torch_input_tensor=full_im_tensor)
                ttnn.synchronize_device(mesh_device)
                if isinstance(full_preds, (list, tuple)):
                    full_preds = full_preds[0]
                full_preds_torch_extra = ttnn.to_torch(
                    full_preds,
                    dtype=torch.float32,
                    mesh_composer=output_mesh_composer,
                )
                preds_torch = torch.cat([preds_torch, full_preds_torch_extra[:1]], dim=0)
                cur_shifts.append((0, 0))
                n_valid += 1
                n_batch += 1

            # --- Send to bg process ---
            t0 = time.perf_counter()
            q_post.put((preds_torch, cur_frame_4k, cur_shifts, n_valid, n_batch))
            t_enqueue = time.perf_counter() - t0

            # --- Swap to next frame (or exit) -------------------------
            if _prep_pending and _next_valid:
                # Tensor: zero-copy view of shm (safe — runner.run reads it
                # before the next go_event triggers an overwrite).
                cur_tensor = shm_tensor
                # Frame: must copy — bg process will pickle it after the
                # next go_event, when prep may be overwriting shm_frame.
                cur_frame_4k = shm_frame.numpy().copy()
                cur_shifts = [(int(shm_shifts[i, 0].item()), int(shm_shifts[i, 1].item())) for i in range(_N_tiles)]
            else:
                break  # video ended or image mode

            dt_main = time.perf_counter() - t_frame_start

            # --- Main-thread timing logs -------------------------------
            frame_count_main += 1
            t_prep_wait_sum += t_prep_wait
            t_device_sum += t_device
            t_host_prep_sum += t_host_prep
            t_h2d_trace_sum += t_h2d_trace
            t_dev_compute_sum += t_dev_compute
            t_d2h_sum += t_d2h
            t_enqueue_sum += t_enqueue
            t_main_total_sum += dt_main
            if _prep_timings is not None:
                t_prep_read_sum += _prep_timings["read_ms"]
                t_prep_slice_sum += _prep_timings["slice_ms"]
                t_prep_preprocess_sum += _prep_timings["preprocess_ms"]
                t_prep_bgr_sum += _prep_timings["bgr_rgb_transpose_ms"]
                t_prep_bf16_sum += _prep_timings["to_bf16_ms"]
                t_prep_norm_sum += _prep_timings["normalize_ms"]
                t_prep_ipc_sum += _prep_timings.get("ipc_ms", 0)

            if frame_count_main == 1:
                _pt = _prep_timings or {}
                print(
                    f"{TAG} First frame (main): {dt_main * 1000:.1f}ms "
                    f"(device={t_device * 1000:.1f}"
                    f"[host_prep={t_host_prep:.1f} h2d+trace={t_h2d_trace:.1f} "
                    f"device_compute={t_dev_compute * 1000:.1f}] "
                    f"d2h={t_d2h * 1000:.1f} "
                    f"prep_wait={t_prep_wait * 1000:.1f}"
                    f"[read={_pt.get('read_ms',0):.1f} slice={_pt.get('slice_ms',0):.1f} "
                    f"preprocess={_pt.get('preprocess_ms',0):.1f}])",
                    flush=True,
                )

            if frame_count_main % LOG_INTERVAL_MAIN == 0:
                n = LOG_INTERVAL_MAIN
                print(
                    f"{TAG} MAIN avg {n}f: "
                    f"total={t_main_total_sum / n * 1000:.1f}ms "
                    f"({1.0 / (t_main_total_sum / n):.1f} FPS)  |  "
                    f"prep(proc)={t_prep_read_sum / n:.1f}+"
                    f"{t_prep_slice_sum / n:.1f}+"
                    f"{t_prep_preprocess_sum / n:.1f}"
                    f"[bgr_rgb_chw={t_prep_bgr_sum / n:.1f} "
                    f"to_bf16={t_prep_bf16_sum / n:.1f} "
                    f"norm={t_prep_norm_sum / n:.1f}] "
                    f"ipc={t_prep_ipc_sum / n:.1f}  "
                    f"prep_wait={t_prep_wait_sum / n * 1000:.1f}  "
                    f"host_prep={t_host_prep_sum / n:.1f}  "
                    f"h2d+trace={t_h2d_trace_sum / n:.1f}  "
                    f"device_compute={t_dev_compute_sum / n * 1000:.1f}  "
                    f"d2h={t_d2h_sum / n * 1000:.1f}  "
                    f"enqueue={t_enqueue_sum / n * 1000:.1f}  (ms)",
                    flush=True,
                )
                t_prep_wait_sum = t_device_sum = 0.0
                t_host_prep_sum = t_h2d_trace_sum = t_dev_compute_sum = 0.0
                t_d2h_sum = t_enqueue_sum = t_main_total_sum = 0.0
                t_prep_read_sum = t_prep_slice_sum = t_prep_preprocess_sum = 0.0
                t_prep_bgr_sum = t_prep_bf16_sum = t_prep_norm_sum = t_prep_ipc_sum = 0.0

        # Image mode: wait for bg process to finish the single image
        if src.is_image:
            while not stop and bg_proc.is_alive():
                time.sleep(0.5)
            while not stop:
                time.sleep(1)

    finally:
        # Signal bg process to exit and wait
        try:
            q_post.put(None)
        except Exception:
            pass
        bg_proc.join(timeout=10)
        if bg_proc.is_alive():
            bg_proc.kill()
        # Signal prep process to exit and wait
        stop_event.set()
        go_event.set()  # unblock if waiting on go_event
        prep_proc.join(timeout=5)
        if prep_proc.is_alive():
            prep_proc.kill()

        print(f"{TAG} Shutting down...", flush=True)
        src.release()
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
