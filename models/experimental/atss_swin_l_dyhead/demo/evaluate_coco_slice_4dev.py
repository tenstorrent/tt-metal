#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Evaluate ATSS Swin-L DyHead with the 4-device sliced pipeline on COCO bbox metrics.

The inference geometry intentionally matches ``demo_slice_4dev.py``:

1. Stretch the source image to 1280x1280.
2. Resize to a square overlap canvas (1152x1152 for the default 128 px overlap).
3. Run four 640x640 tiles in parallel on a 1x4 mesh.
4. Merge tile detections and map boxes back to the source image coordinates.

The model and traced pipeline are built once and reused for every dataset image.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

import ttnn

from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.atss_swin_l_dyhead.common import (
    ATSS_NMS_IOU_THR,
    ATSS_SCORE_THR,
    get_checkpoint_num_classes,
    get_checkpoint_path,
)
from models.experimental.atss_swin_l_dyhead.demo.demo_slice_4dev import (
    FRAME_SIZE,
    SLICE_MAX_PER_FRAME,
    SLICE_MAX_PER_TILE,
    TILE_SIZE,
    _sharded_memory_configs,
    build_overlap_grid,
    postprocess_and_merge,
    preprocess_tiles,
    slice_image_to_tiles,
)
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


COCO_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def remap_results_to_original(results, original_width: int, original_height: int, frame_size: int = FRAME_SIZE):
    """Map xyxy boxes from the square demo frame back to source-image coordinates."""
    remapped = {name: tensor.clone() for name, tensor in results.items()}
    boxes = remapped["bboxes"]
    if boxes.numel() == 0:
        return remapped

    boxes[:, 0::2] *= original_width / frame_size
    boxes[:, 1::2] *= original_height / frame_size
    boxes[:, 0::2].clamp_(0, original_width)
    boxes[:, 1::2].clamp_(0, original_height)
    return remapped


def results_to_coco(
    results,
    image_id: int,
    category_map: dict[int, int],
    score_threshold: float = ATSS_SCORE_THR,
):
    """Convert ATSS xyxy detections to COCO xywh result dictionaries."""
    coco_results = []
    boxes = results["bboxes"]
    scores = results["scores"]
    labels = results["labels"]

    for box, score, label_tensor in zip(boxes, scores, labels):
        score_value = float(score)
        if score_value < score_threshold:
            continue

        label = int(label_tensor)
        if label not in category_map:
            raise KeyError(f"No COCO category mapping for model label {label}")

        x1, y1, x2, y2 = (float(value) for value in box)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width == 0.0 or height == 0.0:
            continue

        coco_results.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_map[label]),
                "bbox": [x1, y1, width, height],
                "score": score_value,
            }
        )

    return coco_results


class Slice4DeviceRunner:
    """Reusable four-device traced pipeline for fixed 640x640 tile batches."""

    def __init__(self, checkpoint: str, use_trace: bool = True):
        self.mesh_device = None
        self.pipeline = None
        self.compile_ms = 0.0
        self.mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 4),
            l1_small_size=32768,
            num_command_queues=2,
            **({"trace_region_size": 400_000_000} if use_trace else {}),
        )

        try:
            self.inputs_mesh_mapper, _, self.output_mesh_composer = get_mesh_mappers(self.mesh_device)
            logger.info("Building TtATSSModel once for 640x640 tiles")
            model = TtATSSModel.from_checkpoint(
                checkpoint,
                device=self.mesh_device,
                input_h=TILE_SIZE,
                input_w=TILE_SIZE,
                hybrid_dyhead="device",
                inputs_mesh_mapper=self.inputs_mesh_mapper,
                output_mesh_composer=self.output_mesh_composer,
            )

            sample = torch.zeros((4, 3, TILE_SIZE, TILE_SIZE))
            tt_sample = ttnn.from_torch(
                sample,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.inputs_mesh_mapper,
            )
            dram_memory_config, l1_memory_config = _sharded_memory_configs(self.mesh_device, tt_sample.shape)

            def model_wrapper(input_on_device):
                x = ttnn.to_memory_config(input_on_device, ttnn.DRAM_MEMORY_CONFIG)
                return model.forward_device(x)

            self.pipeline = create_pipeline_from_config(
                config=PipelineConfig(
                    use_trace=use_trace,
                    num_command_queues=2,
                    all_transfers_on_separate_command_queue=False,
                ),
                model=model_wrapper,
                device=self.mesh_device,
                dram_input_memory_config=dram_memory_config,
                l1_input_memory_config=l1_memory_config,
            )

            logger.info("Compiling sliced pipeline")
            start = time.perf_counter()
            self.pipeline.compile(tt_sample)
            self.compile_ms = (time.perf_counter() - start) * 1000.0
            self.pipeline.preallocate_output_tensors_on_host(1)
            logger.info(f"Pipeline compile: {self.compile_ms:.1f} ms")
        except Exception:
            self.close()
            raise

    def infer_tiles(self, tiles_preprocessed: torch.Tensor):
        if tiles_preprocessed.shape != (4, 3, TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Expected four {TILE_SIZE}x{TILE_SIZE} tiles, got {tuple(tiles_preprocessed.shape)}")

        tt_inputs_host = ttnn.from_torch(
            tiles_preprocessed,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        start = time.perf_counter()
        results = self.pipeline.enqueue([tt_inputs_host]).pop_all()
        inference_ms = (time.perf_counter() - start) * 1000.0
        cls_ttnn, bbox_ttnn, centerness_ttnn = results[0]

        def to_nchw(level_tensors):
            converted = []
            for tensor in level_tensors:
                torch_tensor = ttnn.to_torch(
                    ttnn.from_device(tensor),
                    mesh_composer=self.output_mesh_composer,
                ).float()
                converted.append(torch_tensor.permute(0, 3, 1, 2))
            return converted

        return to_nchw(cls_ttnn), to_nchw(bbox_ttnn), to_nchw(centerness_ttnn), inference_ms

    def close(self):
        if self.pipeline is not None:
            self.pipeline.cleanup()
            self.pipeline = None
        if self.mesh_device is not None:
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None


class PyTorchSliceRunner:
    """Reusable PyTorch reference model that processes the same four tiles sequentially."""

    def __init__(self, checkpoint: str):
        from models.experimental.atss_swin_l_dyhead.reference.model import build_atss_model, load_mmdet_checkpoint

        logger.info("Building PyTorch reference model once")
        start = time.perf_counter()
        self.model = build_atss_model(num_classes=get_checkpoint_num_classes(checkpoint))
        load_mmdet_checkpoint(self.model, checkpoint)
        self.model.eval()
        self.setup_ms = (time.perf_counter() - start) * 1000.0
        self.compile_ms = 0.0
        logger.info(f"PyTorch model setup: {self.setup_ms:.1f} ms")

    def infer_tiles(self, tiles_preprocessed: torch.Tensor):
        if tiles_preprocessed.shape != (4, 3, TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Expected four {TILE_SIZE}x{TILE_SIZE} tiles, got {tuple(tiles_preprocessed.shape)}")

        branch_outputs = [[], [], []]
        start = time.perf_counter()
        with torch.no_grad():
            for tile_index in range(4):
                outputs = self.model(tiles_preprocessed[tile_index : tile_index + 1])
                for branch_index, levels in enumerate(outputs):
                    branch_outputs[branch_index].append(levels)
        inference_ms = (time.perf_counter() - start) * 1000.0

        merged_outputs = []
        for per_tile_branch in branch_outputs:
            merged_outputs.append(
                [torch.cat([tile_levels[level] for tile_levels in per_tile_branch], dim=0) for level in range(5)]
            )
        return *merged_outputs, inference_ms

    def close(self):
        self.model = None


def prepare_four_tiles(image_bgr: np.ndarray, overlap: int):
    """Apply the exact demo resize and tiling recipe."""
    if overlap < 0 or overlap >= TILE_SIZE:
        raise ValueError(f"overlap must be in [0, {TILE_SIZE}), got {overlap}")

    image_1280 = cv2.resize(image_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
    infer_size = 2 * TILE_SIZE - overlap
    if infer_size != FRAME_SIZE:
        image_infer = cv2.resize(image_1280, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    else:
        image_infer = image_1280

    tiles = build_overlap_grid(infer_size, infer_size, TILE_SIZE, TILE_SIZE)
    if len(tiles) != 4:
        raise RuntimeError(f"Expected exactly four tiles, got {len(tiles)}")
    tiles_bgr = slice_image_to_tiles(image_infer, tiles, TILE_SIZE, TILE_SIZE)
    return tiles, preprocess_tiles(tiles_bgr), infer_size


def evaluate(args):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError("pycocotools is required: pip install pycocotools") from error

    annotation_path = Path(args.annotations).resolve()
    images_dir = Path(args.images_dir).resolve()
    model_root = Path(__file__).resolve().parent.parent
    default_output_dir = model_root / "results" / "coco_eval_slice_4dev" / "test"
    if args.pytorch_only:
        default_output_dir /= "pytorch"
    output_dir = Path(args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = str(Path(args.checkpoint or get_checkpoint_path()).resolve())
    coco_gt = COCO(str(annotation_path))
    image_ids = sorted(coco_gt.getImgIds())
    if args.num_samples is not None:
        image_ids = image_ids[: args.num_samples]

    categories = {category["id"]: category["name"] for category in coco_gt.loadCats(coco_gt.getCatIds())}
    if args.category_id not in categories:
        raise ValueError(f"Category id {args.category_id} not found in annotations: {categories}")
    category_map = {0: args.category_id}

    logger.info(
        f"Evaluating {len(image_ids)} images, model label 0 -> "
        f"COCO category {args.category_id} ({categories[args.category_id]!r})"
    )

    backend = "pytorch" if args.pytorch_only else "ttnn"
    runner = (
        PyTorchSliceRunner(checkpoint)
        if args.pytorch_only
        else Slice4DeviceRunner(checkpoint, use_trace=not args.no_trace)
    )
    coco_predictions = []
    prep_times_ms = []
    inference_times_ms = []
    postprocess_times_ms = []

    try:
        for index, image_id in enumerate(image_ids, start=1):
            image_info = coco_gt.loadImgs([image_id])[0]
            image_path = images_dir / image_info["file_name"]
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read dataset image: {image_path}")

            original_height, original_width = image_bgr.shape[:2]
            if (original_width, original_height) != (image_info["width"], image_info["height"]):
                raise ValueError(
                    f"Image dimensions disagree for {image_path.name}: disk={original_width}x{original_height}, "
                    f"COCO={image_info['width']}x{image_info['height']}"
                )

            start = time.perf_counter()
            tiles, tiles_preprocessed, infer_size = prepare_four_tiles(image_bgr, args.overlap)
            prep_times_ms.append((time.perf_counter() - start) * 1000.0)

            cls_scores, bbox_preds, centernesses, inference_ms = runner.infer_tiles(tiles_preprocessed)
            inference_times_ms.append(inference_ms)

            start = time.perf_counter()
            results = postprocess_and_merge(
                tiles,
                cls_scores,
                bbox_preds,
                centernesses,
                frame_size=infer_size,
                score_thr=args.score_threshold,
                per_tile_nms_iou=ATSS_NMS_IOU_THR,
                merge_mode="nms",
                merge_iou_thr=args.merge_iou,
                seam_merge=False,
                max_per_tile=args.max_per_tile,
                max_per_frame=args.max_per_frame,
            )
            if infer_size != FRAME_SIZE and results["bboxes"].numel() > 0:
                results["bboxes"] *= FRAME_SIZE / infer_size
                results["bboxes"][:, 0::2].clamp_(0, FRAME_SIZE)
                results["bboxes"][:, 1::2].clamp_(0, FRAME_SIZE)
            results = remap_results_to_original(results, original_width, original_height)
            coco_predictions.extend(
                results_to_coco(
                    results,
                    image_id=image_id,
                    category_map=category_map,
                    score_threshold=args.score_threshold,
                )
            )
            postprocess_times_ms.append((time.perf_counter() - start) * 1000.0)
            logger.info(
                f"[{index}/{len(image_ids)}] {image_path.name}: "
                f"inference={inference_ms:.1f} ms, detections={results['bboxes'].shape[0]}"
            )
    finally:
        runner.close()

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(json.dumps(coco_predictions, indent=2))
    if not coco_predictions:
        raise RuntimeError(f"No detections were produced; empty results saved to {predictions_path}")

    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [args.category_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_values = {name: float(value) for name, value in zip(COCO_STAT_NAMES, coco_eval.stats)}
    metrics = {
        "dataset": {
            "annotations": str(annotation_path),
            "images_dir": str(images_dir),
            "num_images": len(image_ids),
            "category_id": args.category_id,
            "category_name": categories[args.category_id],
        },
        "checkpoint": checkpoint,
        "inference_config": {
            "backend": backend,
            "geometry": "stretch_to_1280_square_then_4x640_tiles",
            "overlap": args.overlap,
            "merge_mode": "nms",
            "merge_iou": args.merge_iou,
            "score_threshold": args.score_threshold,
            "max_per_tile": args.max_per_tile,
            "max_per_frame": args.max_per_frame,
            **({"trace": not args.no_trace} if not args.pytorch_only else {}),
        },
        "coco_bbox_metrics": metric_values,
        "timing_ms": {
            "compile": runner.compile_ms,
            **({"model_setup": runner.setup_ms} if args.pytorch_only else {}),
            "prep_mean": float(np.mean(prep_times_ms)),
            "inference_mean": float(np.mean(inference_times_ms)),
            "inference_min": float(np.min(inference_times_ms)),
            "inference_max": float(np.max(inference_times_ms)),
            "postprocess_mean": float(np.mean(postprocess_times_ms)),
        },
        "num_predictions": len(coco_predictions),
        "predictions_path": str(predictions_path),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved predictions: {predictions_path}")
    logger.info(f"Saved metrics: {metrics_path}")
    return metrics


def parse_args():
    model_root = Path(__file__).resolve().parent.parent
    test_dir = model_root / "boat-detection-marina.v2i.coco-segmentation" / "test"
    parser = argparse.ArgumentParser(description="COCO bbox evaluation for the ATSS 4-device sliced pipeline")
    parser.add_argument("--annotations", default=str(test_dir / "_annotations.coco.json"))
    parser.add_argument("--images-dir", default=str(test_dir))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--merge-iou", type=float, default=0.55)
    parser.add_argument("--score-threshold", type=float, default=ATSS_SCORE_THR)
    parser.add_argument("--max-per-tile", type=int, default=SLICE_MAX_PER_TILE)
    parser.add_argument("--max-per-frame", type=int, default=SLICE_MAX_PER_FRAME)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Run the PyTorch reference sequentially on the same four tiles.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
