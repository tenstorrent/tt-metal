#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sweep merge configurations for ATSS Swin-L DyHead 4-device slice demo.

Opens the mesh once, runs inference on each input image at multiple overlap
settings (model is fixed at 640x640 tiles regardless), then sweeps merge
configurations purely on CPU. Produces one annotated JPEG per (image, config)
pair, plus a CSV of detection counts.

Usage (from $TT_METAL_HOME with venv active and ARCH/TT_METAL/PYTHONPATH set):

    python3 models/experimental/atss_swin_l_dyhead/demo/sweep_merge_4dev.py \\
        --images /path/to/img1.jpg /path/to/img2.jpg \\
        --output-dir /path/to/sweep_out
"""

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from loguru import logger

import ttnn
from torchvision.ops import batched_nms

from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.experimental.atss_swin_l_dyhead.common import (
    ATSS_CHECKPOINT,
    ATSS_SCORE_THR,
    ATSS_NMS_IOU_THR,
)
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel
from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess
from models.experimental.atss_swin_l_dyhead.demo.demo_inference import draw_detections
from models.experimental.atss_swin_l_dyhead.demo.demo_slice_4dev import (
    TILE_SIZE,
    FRAME_SIZE,
    TileSpec,
    build_overlap_grid,
    slice_image_to_tiles,
    preprocess_tiles,
    cross_tile_greedy_nmm,
    merge_seam_adjacent,
    compute_seams,
    _sharded_memory_configs,
)


# ---------------------------------------------------------------------------
# Inference cache: head outputs per (image, overlap)
# ---------------------------------------------------------------------------


@dataclass
class TileHeads:
    image_path: str
    overlap: int
    infer_size: int
    tiles: List[TileSpec]
    cls_scores: List[torch.Tensor]
    bbox_preds: List[torch.Tensor]
    centernesses: List[torch.Tensor]
    img_bgr_1280: np.ndarray  # the 1280x1280 image used for visualization
    infer_ms: float


def run_inference_for_image(
    pipeline,
    output_mesh_composer,
    img_bgr_1280: np.ndarray,
    overlap: int,
) -> TileHeads:
    """Resize 1280x1280 image to (1280-overlap), tile into 4 x 640x640, run pipeline."""
    if overlap < 0 or overlap >= TILE_SIZE:
        raise ValueError(f"overlap must be in [0, {TILE_SIZE}), got {overlap}")
    infer_size = 2 * TILE_SIZE - overlap
    if infer_size != FRAME_SIZE:
        img_bgr_infer = cv2.resize(img_bgr_1280, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    else:
        img_bgr_infer = img_bgr_1280

    tiles = build_overlap_grid(infer_size, infer_size, TILE_SIZE, TILE_SIZE)
    tiles_bgr = slice_image_to_tiles(img_bgr_infer, tiles, TILE_SIZE, TILE_SIZE)
    tiles_preproc = preprocess_tiles(tiles_bgr)

    inputs_mesh_mapper, _, _ = (
        get_mesh_mappers(pipeline._device) if hasattr(pipeline, "_device") else (None, None, None)
    )
    # We need the original mapper that was used at pipeline construction; recompute from device.
    # The pipeline has the mesh device internally; reuse by name on caller.
    raise NotImplementedError("Use run_inference_for_image_via_callable instead")


def run_pipeline_on_tiles(
    pipeline,
    inputs_mesh_mapper,
    output_mesh_composer,
    tiles_preproc: torch.Tensor,
):
    """Run the pre-built pipeline on a [N,3,H,W] preprocessed tile batch."""
    tt_inputs_host = ttnn.from_torch(
        tiles_preproc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )
    t0 = time.perf_counter()
    results = pipeline.enqueue([tt_inputs_host]).pop_all()
    infer_ms = (time.perf_counter() - t0) * 1000.0
    cls_scores_ttnn, bbox_preds_ttnn, centernesses_ttnn = results[0]

    def _to_nchw(level_tensors):
        out = []
        for t in level_tensors:
            torch_t = ttnn.to_torch(ttnn.from_device(t), mesh_composer=output_mesh_composer).float()
            out.append(torch_t.permute(0, 3, 1, 2))
        return out

    return _to_nchw(cls_scores_ttnn), _to_nchw(bbox_preds_ttnn), _to_nchw(centernesses_ttnn), infer_ms


# ---------------------------------------------------------------------------
# Merge configurations
# ---------------------------------------------------------------------------


@dataclass
class MergeConfig:
    name: str  # short id, used for filename
    label: str  # printed in title
    overlap: int
    merge_mode: str  # "nmm" (greedy) or "nms" (plain torchvision NMS)
    merge_match: str  # "iou" or "ios" — only used when merge_mode == "nmm"
    merge_iou: float
    seam_merge: bool
    max_per_tile: int
    max_per_frame: int
    seam_tol: int = -1  # -1 -> max(overlap, 20)
    score_thr_post: float = ATSS_SCORE_THR  # per-tile score_thr fed to atss_postprocess
    vis_score_thr: float = 0.30  # visualization-only filter

    def effective_seam_tol(self) -> int:
        return max(self.overlap, 20) if self.seam_tol < 0 else self.seam_tol


SWEEP_CONFIGS: List[MergeConfig] = [
    MergeConfig(
        name="A_baseline",
        label="A baseline (ios merge, no overlap)",
        overlap=0,
        merge_mode="nmm",
        merge_match="ios",
        merge_iou=0.5,
        seam_merge=False,
        max_per_tile=100,
        max_per_frame=100,
    ),
    MergeConfig(
        name="B_iou",
        label="B IoU merge (overlap=0, caps=100/100)",
        overlap=0,
        merge_mode="nmm",
        merge_match="iou",
        merge_iou=0.55,
        seam_merge=False,
        max_per_tile=100,
        max_per_frame=100,
    ),
    MergeConfig(
        name="G_iou_overlap_caps_noseam",
        label="G IoU merge + overlap=128 + caps 300/500 (no seam-merge)",
        overlap=128,
        merge_mode="nmm",
        merge_match="iou",
        merge_iou=0.55,
        seam_merge=False,
        max_per_tile=300,
        max_per_frame=500,
    ),
    MergeConfig(
        name="H_g_plus_seam_tol20",
        label="H G + seam-merge tol=20 (conservative)",
        overlap=128,
        merge_mode="nmm",
        merge_match="iou",
        merge_iou=0.55,
        seam_merge=True,
        max_per_tile=300,
        max_per_frame=500,
        seam_tol=20,
    ),
    MergeConfig(
        name="I_g_plus_seam_tol40",
        label="I G + seam-merge tol=40 (moderate)",
        overlap=128,
        merge_mode="nmm",
        merge_match="iou",
        merge_iou=0.55,
        seam_merge=True,
        max_per_tile=300,
        max_per_frame=500,
        seam_tol=40,
    ),
    MergeConfig(
        name="J_iou_caps_no_overlap",
        label="J IoU merge + caps 300/500 (no overlap, no seam)",
        overlap=0,
        merge_mode="nmm",
        merge_match="iou",
        merge_iou=0.55,
        seam_merge=False,
        max_per_tile=300,
        max_per_frame=500,
    ),
    MergeConfig(
        name="K_plain_nms_overlap_caps",
        label="K plain NMS + overlap=128 + caps (no seam-merge)",
        overlap=128,
        merge_mode="nms",
        merge_match="iou",
        merge_iou=0.55,
        seam_merge=False,
        max_per_tile=300,
        max_per_frame=500,
    ),
]


# ---------------------------------------------------------------------------
# CPU-only postprocess + merge pass
# ---------------------------------------------------------------------------


def postprocess_and_merge_cpu(
    heads: TileHeads,
    cfg: MergeConfig,
):
    n_tiles = len(heads.tiles)
    all_boxes, all_scores, all_labels = [], [], []
    for i in range(n_tiles):
        ts = heads.tiles[i]
        per_tile_cls = [t[i : i + 1] for t in heads.cls_scores]
        per_tile_reg = [t[i : i + 1] for t in heads.bbox_preds]
        per_tile_cent = [t[i : i + 1] for t in heads.centernesses]
        res = atss_postprocess(
            per_tile_cls,
            per_tile_reg,
            per_tile_cent,
            img_shape=(TILE_SIZE, TILE_SIZE),
            score_thr=cfg.score_thr_post,
            nms_iou_thr=ATSS_NMS_IOU_THR,
            max_per_img=cfg.max_per_tile,
        )
        b, s, lb = res["bboxes"], res["scores"], res["labels"]
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
            "n_pre_merge": 0,
        }

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    boxes[:, 0::2].clamp_(0, heads.infer_size)
    boxes[:, 1::2].clamp_(0, heads.infer_size)
    n_pre_merge = int(boxes.shape[0])

    if cfg.merge_mode == "nmm":
        mb, ms, mc = cross_tile_greedy_nmm(
            boxes,
            scores,
            labels.float(),
            threshold=cfg.merge_iou,
            match_metric=cfg.merge_match,
            class_agnostic=False,
            max_det=cfg.max_per_frame,
        )
        mc = mc.long()
    elif cfg.merge_mode == "nms":
        keep = batched_nms(boxes, scores, labels, cfg.merge_iou)
        if cfg.max_per_frame > 0:
            keep = keep[: cfg.max_per_frame]
        mb, ms, mc = boxes[keep], scores[keep], labels[keep]
    else:
        raise ValueError(f"unknown merge_mode: {cfg.merge_mode}")

    if cfg.seam_merge:
        seams_x, seams_y = compute_seams(heads.tiles, heads.infer_size, heads.infer_size)
        if seams_x or seams_y:
            mb, ms, mc = merge_seam_adjacent(
                mb,
                ms,
                mc.float() if mc.dtype != torch.float else mc,
                seams_x=seams_x,
                seams_y=seams_y,
                tol=cfg.effective_seam_tol(),
            )
            mc = mc.long()

    return {"bboxes": mb, "scores": ms, "labels": mc, "n_pre_merge": n_pre_merge}


def make_visualization(
    img_bgr_1280: np.ndarray,
    heads: TileHeads,
    results: dict,
    cfg: MergeConfig,
    infer_ms: float,
) -> np.ndarray:
    final = deepcopy(results)
    if heads.infer_size != FRAME_SIZE and final["bboxes"].numel() > 0:
        scale = FRAME_SIZE / heads.infer_size
        final["bboxes"] = final["bboxes"] * scale
        final["bboxes"][:, 0::2].clamp_(0, FRAME_SIZE)
        final["bboxes"][:, 1::2].clamp_(0, FRAME_SIZE)

    title = (
        f"{cfg.label}  |  infer {infer_ms:.0f}ms  |  pre={results['n_pre_merge']}  post={int(final['bboxes'].shape[0])}"
    )
    vis = draw_detections(img_bgr_1280, final, title=title, score_thr=cfg.vis_score_thr)
    vis_scale = FRAME_SIZE / heads.infer_size
    for ts in heads.tiles:
        x0 = int(round(ts.col_start * vis_scale))
        y0 = int(round(ts.row_start * vis_scale))
        x1 = int(round((ts.col_start + TILE_SIZE) * vis_scale)) - 1
        y1 = int(round((ts.row_start + TILE_SIZE) * vis_scale)) - 1
        cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 200), 1)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument(
        "--configs", nargs="*", default=None, help="Optional subset of config names to run (default: all)."
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or ATSS_CHECKPOINT

    selected = SWEEP_CONFIGS if not args.configs else [c for c in SWEEP_CONFIGS if c.name in args.configs]
    if not selected:
        raise SystemExit(f"No configs matched: {args.configs}")
    overlaps_needed = sorted({c.overlap for c in selected})

    img_paths = [Path(p) for p in args.images]
    images_bgr_1280 = {}
    for p in img_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        h, w = bgr.shape[:2]
        if (h, w) != (FRAME_SIZE, FRAME_SIZE):
            bgr = cv2.resize(bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
        images_bgr_1280[p.stem] = (str(p), bgr)

    mesh_shape = ttnn.MeshShape(1, 4)
    device_params = {"l1_small_size": 32768, "num_command_queues": 2}
    if not args.no_trace:
        device_params["trace_region_size"] = 400_000_000
    logger.info(f"Opening 1x4 mesh (trace={not args.no_trace})")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)

    heads_cache: dict = {}  # (stem, overlap) -> TileHeads

    try:
        inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(mesh_device)
        logger.info("Building TtATSSModel @ 640x640")
        model = TtATSSModel.from_checkpoint(
            checkpoint,
            device=mesh_device,
            input_h=TILE_SIZE,
            input_w=TILE_SIZE,
            hybrid_dyhead="device",
            inputs_mesh_mapper=inputs_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )

        # Build a sample input to derive memory configs.
        sample = torch.zeros((4, 3, TILE_SIZE, TILE_SIZE))
        tt_sample = ttnn.from_torch(
            sample, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=inputs_mesh_mapper
        )
        dram_mc, l1_mc = _sharded_memory_configs(mesh_device, tt_sample.shape)

        def model_wrapper(input_on_device):
            x = ttnn.to_memory_config(input_on_device, ttnn.DRAM_MEMORY_CONFIG)
            return model.forward_device(x)

        pipeline = create_pipeline_from_config(
            config=PipelineConfig(
                use_trace=(not args.no_trace), num_command_queues=2, all_transfers_on_separate_command_queue=False
            ),
            model=model_wrapper,
            device=mesh_device,
            dram_input_memory_config=dram_mc,
            l1_input_memory_config=l1_mc,
        )
        logger.info("Compiling pipeline (warmup)")
        t0 = time.perf_counter()
        pipeline.compile(tt_sample)
        logger.info(f"  compile: {(time.perf_counter() - t0)*1000:.1f} ms")
        pipeline.preallocate_output_tensors_on_host(1)

        for stem, (path, img_1280) in images_bgr_1280.items():
            for overlap in overlaps_needed:
                infer_size = 2 * TILE_SIZE - overlap
                if infer_size != FRAME_SIZE:
                    img_infer = cv2.resize(img_1280, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
                else:
                    img_infer = img_1280
                tiles = build_overlap_grid(infer_size, infer_size, TILE_SIZE, TILE_SIZE)
                tiles_bgr = slice_image_to_tiles(img_infer, tiles, TILE_SIZE, TILE_SIZE)
                tiles_preproc = preprocess_tiles(tiles_bgr)
                logger.info(f"Inference: {stem} @ overlap={overlap} (infer_size={infer_size})")
                cs, bp, ct, infer_ms = run_pipeline_on_tiles(
                    pipeline, inputs_mesh_mapper, output_mesh_composer, tiles_preproc
                )
                heads_cache[(stem, overlap)] = TileHeads(
                    image_path=path,
                    overlap=overlap,
                    infer_size=infer_size,
                    tiles=tiles,
                    cls_scores=cs,
                    bbox_preds=bp,
                    centernesses=ct,
                    img_bgr_1280=img_1280,
                    infer_ms=infer_ms,
                )
                logger.info(f"  -> {infer_ms:.1f} ms")

        pipeline.cleanup()
    finally:
        ttnn.close_mesh_device(mesh_device)

    summary = []
    for stem, (path, img_1280) in images_bgr_1280.items():
        img_dir = out_dir / stem
        img_dir.mkdir(parents=True, exist_ok=True)
        for cfg in selected:
            heads = heads_cache[(stem, cfg.overlap)]
            results = postprocess_and_merge_cpu(heads, cfg)
            vis = make_visualization(img_1280, heads, results, cfg, heads.infer_ms)
            out_path = img_dir / f"{cfg.name}.jpg"
            cv2.imwrite(str(out_path), vis)
            row = {
                "image": stem,
                "config": cfg.name,
                "label": cfg.label,
                "overlap": cfg.overlap,
                "merge_mode": cfg.merge_mode,
                "merge_match": cfg.merge_match,
                "merge_iou": cfg.merge_iou,
                "seam_merge": cfg.seam_merge,
                "max_per_tile": cfg.max_per_tile,
                "max_per_frame": cfg.max_per_frame,
                "n_pre_merge": int(results["n_pre_merge"]),
                "n_post_merge": int(results["bboxes"].shape[0]),
                "infer_ms": float(heads.infer_ms),
                "out_path": str(out_path),
            }
            summary.append(row)
            logger.info(f"  {stem}/{cfg.name}: pre={row['n_pre_merge']} post={row['n_post_merge']}  -> {out_path}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
