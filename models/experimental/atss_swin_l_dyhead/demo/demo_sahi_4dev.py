#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ATSS Swin-L DyHead SAHI-style 4-device demo (T3K, 1x4 mesh).

Slices a 1280x1280 image into 4 tiles of 640x640 (2x2, no overlap — exact fit),
runs all 4 tiles in parallel across a 1x4 sub-mesh of Wormhole devices with
2CQ + trace, then merges per-tile detections via greedy NMM into the original
1280x1280 frame.

Pipeline transferred from
  models/demos/yolo_eval/yolov8l_sahi_640_pipelined.py  (sdawle/yolo_bh_demos)
adapted for the ATSS detector (per-tile postprocess instead of yolo decode).

Usage:
  cd $TT_METAL_HOME
  source python_env/bin/activate
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd)

  python3 models/experimental/atss_swin_l_dyhead/demo/demo_sahi_4dev.py \
      --image path/to/your/1280x1280_image.jpg \
      --output-dir path/to/output
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

import ttnn
from loguru import logger

from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

from models.experimental.atss_swin_l_dyhead.common import (
    ATSS_CHECKPOINT,
    ATSS_PAD_SIZE_DIVISOR,
    ATSS_PIXEL_MEAN,
    ATSS_PIXEL_STD,
    ATSS_SCORE_THR,
    ATSS_NMS_IOU_THR,
    ATSS_MAX_PER_IMG,
)
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel
from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess
from models.experimental.atss_swin_l_dyhead.demo.demo_inference import COCO_CLASSES, draw_detections


TILE_SIZE = 640
FRAME_SIZE = 1280


@dataclass(frozen=True)
class TileSpec:
    row_start: int
    col_start: int
    src_h: int
    src_w: int


def build_overlap_grid(frame_h: int, frame_w: int, tile_h: int = TILE_SIZE, tile_w: int = TILE_SIZE) -> List[TileSpec]:
    """Tile a frame into full tile_h x tile_w crops, shifting the last row/col
    inward if they would otherwise exceed the frame. For 1280x1280 with 640x640
    this returns 4 exact tiles with no overlap.
    """
    x_starts = list(range(0, frame_w, tile_w))
    if x_starts[-1] + tile_w > frame_w:
        x_starts[-1] = frame_w - tile_w
    y_starts = list(range(0, frame_h, tile_h))
    if y_starts[-1] + tile_h > frame_h:
        y_starts[-1] = frame_h - tile_h

    specs = []
    for r in y_starts:
        for c in x_starts:
            sh = min(tile_h, frame_h - r)
            sw = min(tile_w, frame_w - c)
            specs.append(TileSpec(r, c, sh, sw))
    return specs


def slice_image_to_tiles(img_bgr: np.ndarray, tiles: List[TileSpec], tile_h: int, tile_w: int) -> np.ndarray:
    """Slice a BGR HxWx3 image into [N, 3, tile_h, tile_w] float tensor (BGR, [0,255])."""
    n = len(tiles)
    out = np.empty((n, 3, tile_h, tile_w), dtype=np.float32)
    for i, ts in enumerate(tiles):
        crop = img_bgr[ts.row_start : ts.row_start + ts.src_h, ts.col_start : ts.col_start + ts.src_w, :]
        # Pad if last-row/col shift was not enough (shouldn't happen at 1280x1280 -> 640x640)
        if crop.shape[0] != tile_h or crop.shape[1] != tile_w:
            padded = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            padded[: crop.shape[0], : crop.shape[1], :] = crop
            crop = padded
        out[i] = crop.transpose(2, 0, 1).astype(np.float32)
    return out


def _pairwise_overlap(boxes: torch.Tensor, match_metric: str) -> torch.Tensor:
    n = boxes.shape[0]
    if n == 0:
        return torch.zeros(0, 0)
    x1 = torch.max(boxes[:, None, 0], boxes[None, :, 0])
    y1 = torch.max(boxes[:, None, 1], boxes[None, :, 1])
    x2 = torch.min(boxes[:, None, 2], boxes[None, :, 2])
    y2 = torch.min(boxes[:, None, 3], boxes[None, :, 3])
    inter = (x2 - x1).clamp_(min=0) * (y2 - y1).clamp_(min=0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if match_metric == "iou":
        denom = area[:, None] + area[None, :] - inter
    else:
        denom = torch.min(area[:, None], area[None, :])
    return inter / denom.clamp_(min=1e-9)


def cross_tile_greedy_nmm(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    threshold: float = 0.5,
    match_metric: str = "ios",
    class_agnostic: bool = False,
    max_det: int = ATSS_MAX_PER_IMG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy non-maximum-merging: for each top-scoring box, union-of-extents
    merge with overlapping boxes (above ``threshold`` on ``match_metric``).
    Direct port of yolov8l_sahi_640_pipelined._cross_tile_greedy_nmm.
    """
    n = boxes.shape[0]
    if n == 0:
        return boxes, scores, cls_ids
    order = scores.argsort(descending=True)
    b = boxes[order]
    s = scores[order]
    c = cls_ids[order]
    ov = _pairwise_overlap(b, match_metric)
    consumed = torch.zeros(n, dtype=torch.bool)
    out_b, out_s, out_c = [], [], []
    for i in range(n):
        if consumed[i]:
            continue
        match = ov[i] > threshold
        match[: i + 1] = False
        match &= ~consumed
        if not class_agnostic:
            match &= c == c[i]
        group = match.clone()
        group[i] = True
        g = b[group]
        out_b.append(torch.stack([g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()]))
        out_s.append(s[i])
        out_c.append(c[i])
        consumed |= group
        if len(out_b) >= max_det:
            break
    if not out_b:
        return b[:0], s[:0], c[:0]
    return torch.stack(out_b), torch.stack(out_s), torch.stack(out_c)


def preprocess_tiles(tiles_bgr_nchw: np.ndarray) -> torch.Tensor:
    """BGR [N,3,H,W] float[0,255] -> normalized RGB [N,3,H,W] torch tensor."""
    x = torch.from_numpy(tiles_bgr_nchw).float()
    x = x[:, [2, 1, 0], :, :]
    mean = torch.tensor(ATSS_PIXEL_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(ATSS_PIXEL_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    _, _, h, w = x.shape
    assert (
        h % ATSS_PAD_SIZE_DIVISOR == 0 and w % ATSS_PAD_SIZE_DIVISOR == 0
    ), f"Tile {h}x{w} not divisible by pad_size_divisor={ATSS_PAD_SIZE_DIVISOR}"
    return x


def _sharded_memory_configs(device, shape):
    """HEIGHT_SHARDED DRAM + L1 memory configs (mirrors test_atss_swin_l_dyhead_e2e_perf)."""
    ndim = len(shape)
    total_height = 1
    for i in range(ndim - 1):
        total_height *= shape[i]
    width = shape[-1]

    num_dram_cores = device.dram_grid_size().x
    while total_height % num_dram_cores != 0 and num_dram_cores > 1:
        num_dram_cores -= 1
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_cores - 1, 0))}),
        [total_height // num_dram_cores, width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)

    grid = device.compute_with_storage_grid_size()
    num_l1_cores = grid.x * grid.y
    while total_height % num_l1_cores != 0 and num_l1_cores > 1:
        num_l1_cores -= 1
    y = min(grid.y, num_l1_cores)
    while num_l1_cores % y != 0:
        y -= 1
    x = num_l1_cores // y
    l1_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))}),
        [total_height // num_l1_cores, width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec)

    return dram_mem_config, l1_mem_config


def run_4dev_inference(
    mesh_device,
    img_bgr: np.ndarray,
    checkpoint: str,
    use_trace: bool = True,
    num_command_queues: int = 2,
):
    """Slice img into 4 tiles, run on the 1x4 mesh with 2CQ+trace, return per-tile head outputs."""
    h, w = img_bgr.shape[:2]
    assert h == FRAME_SIZE and w == FRAME_SIZE, f"Demo expects {FRAME_SIZE}x{FRAME_SIZE} input (got {h}x{w})"

    tiles = build_overlap_grid(h, w, TILE_SIZE, TILE_SIZE)
    n_tiles = len(tiles)
    num_devices = mesh_device.get_num_devices()
    assert n_tiles == num_devices, (
        f"Expected {num_devices} tiles to match {num_devices} devices, got {n_tiles}. "
        f"Frame {w}x{h} with tile {TILE_SIZE}x{TILE_SIZE} produces {n_tiles} tiles."
    )

    logger.info(f"Tile grid {w}x{h} -> {n_tiles} tiles of {TILE_SIZE}x{TILE_SIZE}:")
    for i, ts in enumerate(tiles):
        logger.info(f"  tile {i}: start=({ts.col_start}, {ts.row_start})")

    tiles_bgr = slice_image_to_tiles(img_bgr, tiles, TILE_SIZE, TILE_SIZE)
    tiles_preproc = preprocess_tiles(tiles_bgr)  # [4, 3, 640, 640]

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(mesh_device)

    logger.info(f"Building TtATSSModel (input={TILE_SIZE}x{TILE_SIZE}, batch={n_tiles}, hybrid_dyhead='device')")
    model = TtATSSModel.from_checkpoint(
        checkpoint,
        device=mesh_device,
        input_h=TILE_SIZE,
        input_w=TILE_SIZE,
        hybrid_dyhead="device",
        inputs_mesh_mapper=inputs_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
    )

    tt_inputs_host = ttnn.from_torch(
        tiles_preproc,  # NCHW [4, 3, 640, 640]; ShardTensorToMesh(dim=0) gives each device [1, 3, 640, 640]
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )

    dram_mem_config, l1_mem_config = _sharded_memory_configs(mesh_device, tt_inputs_host.shape)

    def model_wrapper(input_on_device):
        x = ttnn.to_memory_config(input_on_device, ttnn.DRAM_MEMORY_CONFIG)
        return model.forward_device(x)

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=use_trace,
            num_command_queues=num_command_queues,
            all_transfers_on_separate_command_queue=False,
        ),
        model=model_wrapper,
        device=mesh_device,
        dram_input_memory_config=dram_mem_config,
        l1_input_memory_config=l1_mem_config,
    )

    logger.info("Compiling pipeline (warmup)...")
    t0 = time.perf_counter()
    pipeline.compile(tt_inputs_host)
    compile_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"  compile: {compile_ms:.1f} ms")

    pipeline.preallocate_output_tensors_on_host(1)
    logger.info("Running inference on 4 tiles in parallel...")
    t0 = time.perf_counter()
    results = pipeline.enqueue([tt_inputs_host]).pop_all()
    infer_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"  inference: {infer_ms:.1f} ms ({n_tiles / (infer_ms / 1000):.2f} tiles/s)")

    cls_scores_ttnn, bbox_preds_ttnn, centernesses_ttnn = results[0]

    def _to_nchw(level_tensors):
        out = []
        for t in level_tensors:
            torch_t = ttnn.to_torch(ttnn.from_device(t), mesh_composer=output_mesh_composer).float()
            # Tensor is [batch_total, H, W, C] in NHWC after mesh composition along dim 0.
            out.append(torch_t.permute(0, 3, 1, 2))
        return out

    cls_scores = _to_nchw(cls_scores_ttnn)
    bbox_preds = _to_nchw(bbox_preds_ttnn)
    centernesses = _to_nchw(centernesses_ttnn)

    pipeline.cleanup()
    return tiles, cls_scores, bbox_preds, centernesses, {"compile_ms": compile_ms, "infer_ms": infer_ms}


def postprocess_and_merge(
    tiles: List[TileSpec],
    cls_scores,
    bbox_preds,
    centernesses,
    score_thr: float = ATSS_SCORE_THR,
    per_tile_nms_iou: float = ATSS_NMS_IOU_THR,
    merge_iou_thr: float = 0.5,
    merge_match: str = "ios",
    class_agnostic: bool = False,
):
    """Per-tile atss_postprocess (anchors cached internally), shift boxes to frame
    coordinates, then greedy NMM across tiles."""
    n_tiles = len(tiles)
    all_boxes, all_scores, all_labels = [], [], []

    for i in range(n_tiles):
        ts = tiles[i]
        per_tile_cls = [t[i : i + 1] for t in cls_scores]
        per_tile_reg = [t[i : i + 1] for t in bbox_preds]
        per_tile_cent = [t[i : i + 1] for t in centernesses]

        res = atss_postprocess(
            per_tile_cls,
            per_tile_reg,
            per_tile_cent,
            img_shape=(TILE_SIZE, TILE_SIZE),
            score_thr=score_thr,
            nms_iou_thr=per_tile_nms_iou,
            max_per_img=ATSS_MAX_PER_IMG,
        )
        b = res["bboxes"]
        s = res["scores"]
        lb = res["labels"]
        if b.numel() == 0:
            continue
        shift = torch.tensor([ts.col_start, ts.row_start, ts.col_start, ts.row_start], dtype=b.dtype)
        all_boxes.append(b + shift)
        all_scores.append(s)
        all_labels.append(lb)

    if not all_boxes:
        return {
            "bboxes": torch.zeros(0, 4),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.long),
        }

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    boxes[:, 0::2].clamp_(0, FRAME_SIZE)
    boxes[:, 1::2].clamp_(0, FRAME_SIZE)

    mb, ms, mc = cross_tile_greedy_nmm(
        boxes,
        scores,
        labels.float(),
        threshold=merge_iou_thr,
        match_metric=merge_match,
        class_agnostic=class_agnostic,
        max_det=ATSS_MAX_PER_IMG,
    )
    return {"bboxes": mb, "scores": ms, "labels": mc.long()}


def main():
    parser = argparse.ArgumentParser(description="ATSS Swin-L DyHead SAHI 4-device demo (1280x1280)")
    parser.add_argument("--image", required=True, help="Input image (will be resized to 1280x1280 if needed)")
    parser.add_argument("--checkpoint", default=None, help="mmdet .pth checkpoint")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--merge-iou", type=float, default=0.5, help="Cross-tile greedy NMM IoU/IoS threshold")
    parser.add_argument("--merge-match", choices=["iou", "ios"], default="ios")
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--no-trace", action="store_true", help="Disable trace (2CQ only)")
    _default_out = str(Path(__file__).resolve().parent.parent / "results" / "sahi_4dev")
    parser.add_argument("--output-dir", default=_default_out)
    args = parser.parse_args()

    checkpoint = args.checkpoint or ATSS_CHECKPOINT
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    h, w = img_bgr.shape[:2]
    if (h, w) != (FRAME_SIZE, FRAME_SIZE):
        logger.info(f"Resizing input {w}x{h} -> {FRAME_SIZE}x{FRAME_SIZE}")
        img_bgr = cv2.resize(img_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_shape = ttnn.MeshShape(1, 4)
    logger.info(f"Opening 1x4 mesh (l1_small=32768, trace={not args.no_trace})")
    device_params = {
        "l1_small_size": 32768,
        "num_command_queues": 2,
    }
    if not args.no_trace:
        device_params["trace_region_size"] = 400_000_000
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)

    try:
        tiles, cls_scores, bbox_preds, centernesses, timings = run_4dev_inference(
            mesh_device,
            img_bgr,
            checkpoint,
            use_trace=(not args.no_trace),
            num_command_queues=2,
        )
        results = postprocess_and_merge(
            tiles,
            cls_scores,
            bbox_preds,
            centernesses,
            score_thr=ATSS_SCORE_THR,
            merge_iou_thr=args.merge_iou,
            merge_match=args.merge_match,
            class_agnostic=args.class_agnostic,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)

    n = int(results["bboxes"].shape[0])
    logger.info(f"Merged detections: {n}")
    for i in range(min(n, 20)):
        lb = int(results["labels"][i])
        name = COCO_CLASSES[lb] if lb < len(COCO_CLASSES) else str(lb)
        sc = float(results["scores"][i])
        b = results["bboxes"][i].tolist()
        logger.info(f"  {name:>14}  {sc:.3f}  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")

    vis = draw_detections(
        img_bgr,
        results,
        title=f"ATSS-SAHI 4dev (infer {timings['infer_ms']:.0f}ms)",
        score_thr=args.score_thr,
    )
    # Draw tile seams in faint white to make slicing visible.
    for ts in tiles:
        cv2.rectangle(
            vis,
            (ts.col_start, ts.row_start),
            (ts.col_start + TILE_SIZE - 1, ts.row_start + TILE_SIZE - 1),
            (200, 200, 200),
            1,
        )
    out_path = out_dir / "atss_sahi_4dev_detections.jpg"
    cv2.imwrite(str(out_path), vis)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
