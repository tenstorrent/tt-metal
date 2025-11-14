# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
COCO Evaluation Script for EfficientDet D0

This script evaluates the EfficientDet D0 model on COCO validation set
and computes standard COCO metrics (mAP, AP@0.5, AP@0.75, AR, etc.)

Usage:
    python models/experimental/efficientdetd0/tests/evaluate_coco.py \
        --coco-annotations path/to/annotations/instances_val2017.json \
        --coco-images path/to/val2017 \
        --weights-path path/to/efficientdet-d0.pth \
        --device-id 0 \
        --num-samples 5000
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import ttnn
from loguru import logger
from torchvision.ops.boxes import batched_nms

from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.efficient_det import TtEfficientDetBackbone
from models.experimental.efficientdetd0.common import load_torch_model_state
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    infer_torch_module_args,
    create_custom_mesh_preprocessor,
)
from models.demos.utils.common_demo_utils import get_mesh_mappers
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import disable_persistent_kernel_cache

# Try to import pycocotools for official COCO evaluation
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logger.warning("pycocotools not available. Install with: pip install pycocotools")


class BBoxTransform(torch.nn.Module):
    """Transform regression outputs to bounding boxes."""

    def forward(self, anchors, regression):
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.0
        xmin = x_centers - w / 2.0
        ymax = y_centers + h / 2.0
        xmax = x_centers + w / 2.0

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(torch.nn.Module):
    """Clip boxes to image boundaries."""

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)
        return boxes


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    """Resize image with aspect ratio preservation and padding."""
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, width, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return (
        canvas,
        new_w,
        new_h,
        old_w,
        old_h,
        padding_w,
        padding_h,
    )


def preprocess_image(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Preprocess image for EfficientDet inference."""
    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise ValueError(f"Could not load image from {image_path}")

    normalized_img = (ori_img[..., ::-1] / 255.0 - mean) / std
    img_meta = aspectaware_resize_padding(normalized_img, max_size, max_size, means=None)
    framed_img = img_meta[0]
    framed_meta = img_meta[1:]

    framed_img = torch.from_numpy(framed_img.transpose(2, 0, 1)).float()
    framed_img = framed_img.unsqueeze(0)

    return ori_img, framed_img, framed_meta


def postprocess(
    regression, classification, anchors, regressBoxes, clipBoxes, input_tensor, threshold=0.5, iou_threshold=0.5
):
    """Post-process model outputs to get bounding boxes."""
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, input_tensor)

    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]

    out = []
    for i in range(regression.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append(
                {
                    "rois": boxes_.cpu().numpy(),
                    "class_ids": classes_.cpu().numpy(),
                    "scores": scores_.cpu().numpy(),
                }
            )
        else:
            out.append(
                {
                    "rois": np.array(()),
                    "class_ids": np.array(()),
                    "scores": np.array(()),
                }
            )

    return out


def invert_affine(metas, preds):
    """
    Transform bounding boxes back to original image coordinates.

    This function transforms boxes from preprocessed image coordinates back to
    original image coordinates, accounting for aspect ratio preservation and padding.

    Args:
        metas: Metadata from preprocessing (new_w, new_h, old_w, old_h, padding_w, padding_h)
               or list of such tuples
        preds: List of prediction dictionaries with 'rois' key

    Returns:
        Modified preds with transformed boxes
    """
    for i in range(len(preds)):
        if len(preds[i]["rois"]) == 0:
            continue

        if isinstance(metas, float):
            # Simple scaling case
            preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / metas
            preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / metas
        else:
            # Aspect ratio preservation case
            if isinstance(metas, list) and len(metas) > i:
                meta = metas[i]
            else:
                meta = metas

            if isinstance(meta, (list, tuple)) and len(meta) >= 4:
                new_w, new_h, old_w, old_h = meta[0], meta[1], meta[2], meta[3]
                # Transform boxes back to original coordinates
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / (new_w / old_w)
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / (new_h / old_h)

    return preds


def convert_to_coco_format(predictions, image_ids, framed_metas):
    """
    Convert predictions to COCO format.

    Args:
        predictions: List of prediction dictionaries with 'rois', 'class_ids', 'scores'
        image_ids: List of COCO image IDs
        framed_metas: List of preprocessing metadata tuples (not used, kept for compatibility)

    Returns:
        List of COCO-format result dictionaries
    """
    coco_results = []
    for pred, img_id in zip(predictions, image_ids):
        if len(pred["rois"]) == 0:
            continue

        for roi, class_id, score in zip(pred["rois"], pred["class_ids"], pred["scores"]):
            x1, y1, x2, y2 = roi
            # Convert to COCO format: [x, y, width, height]
            # Note: COCO bbox format is [x_min, y_min, width, height]
            # Reference repo: rois[:, 2] -= rois[:, 0]  (x2 - x1 = width)
            #                  rois[:, 3] -= rois[:, 1]  (y2 - y1 = height)
            width = x2 - x1
            height = y2 - y1
            # COCO class IDs are 1-indexed (1-90), but our model outputs 0-indexed (0-89)
            # Reference repo: category_id = label + 1
            coco_class_id = int(class_id) + 1

            coco_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": coco_class_id,
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score),
                }
            )

    return coco_results


def evaluate_coco(
    device,
    coco_annotations_path,
    coco_images_path,
    weights_path=None,
    model_location_generator=None,
    num_samples=None,
    batch_size=1,
    height=512,
    width=512,
    num_classes=90,
    threshold=0.05,
    iou_threshold=0.5,
    use_ttnn=True,
):
    """
    Evaluate EfficientDet on COCO validation set.

    Args:
        device: TTNN device
        coco_annotations_path: Path to COCO annotations JSON file
        coco_images_path: Path to COCO validation images directory
        weights_path: Path to model weights
        num_samples: Number of samples to evaluate (None for all)
        use_ttnn: If True, use TTNN model; if False, use PyTorch reference
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools is required for COCO evaluation. Install with: pip install pycocotools")

    disable_persistent_kernel_cache()

    # Load COCO dataset
    coco = COCO(coco_annotations_path)
    img_ids = coco.getImgIds()
    if num_samples:
        img_ids = img_ids[:num_samples]

    logger.info("=" * 80)
    logger.info("COCO Evaluation for EfficientDet D0")
    logger.info("=" * 80)
    logger.info(f"Model: {'TTNN' if use_ttnn else 'PyTorch Reference'}, Images: {len(img_ids)}")

    # Initialize model
    torch_model = EfficientDetBackbone(
        num_classes=num_classes,
        compound_coef=0,
        load_weights=False,
    ).eval()
    load_torch_model_state(torch_model, model_location_generator=model_location_generator, weights_path=weights_path)

    ttnn_model = None
    if use_ttnn:
        _, weights_mesh_mapper, _ = get_mesh_mappers(device)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
            device=device,
        )
        # Get sample input for module args
        sample_input = torch.randn(batch_size, 3, height, width)
        module_args = infer_torch_module_args(model=torch_model, input=sample_input)

        ttnn_model = TtEfficientDetBackbone(
            device=device,
            parameters=parameters,
            conv_params=module_args,
            num_classes=num_classes,
            compound_coef=0,
        )

    # Post-processing modules
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Run inference
    all_predictions = []
    all_image_ids = []
    all_framed_metas = []

    inference_times = []

    for idx, img_id in enumerate(img_ids):
        if (idx + 1) % 100 == 0:
            logger.info(f"Processing image {idx + 1}/{len(img_ids)}")

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_images_path, img_info["file_name"])

        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        ori_img, torch_inputs, framed_meta = preprocess_image(img_path, max_size=max(height, width))

        # Run inference
        if use_ttnn and ttnn_model is not None:
            ttnn_input_tensor = ttnn.from_torch(
                torch_inputs,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            start_time = time.time()
            ttnn_features, ttnn_regression, ttnn_classification = ttnn_model(ttnn_input_tensor)
            ttnn.synchronize_device(device)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            ttnn_regression_torch = ttnn.to_torch(ttnn_regression, dtype=torch.float32)
            ttnn_classification_torch = ttnn.to_torch(ttnn_classification, dtype=torch.float32)

            regression = ttnn_regression_torch
            classification = ttnn_classification_torch

            ttnn.deallocate(ttnn_input_tensor)
            for feat in ttnn_features:
                ttnn.deallocate(feat)
            ttnn.deallocate(ttnn_regression)
            ttnn.deallocate(ttnn_classification)
        else:
            start_time = time.time()
            with torch.no_grad():
                torch_features, torch_regression, torch_classification = torch_model(torch_inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            regression = torch_regression
            classification = torch_classification

        # Post-process
        anchors = torch_model.anchors(torch_inputs, torch_inputs.dtype)
        preds = postprocess(
            regression,
            classification,
            anchors,
            regressBoxes,
            clipBoxes,
            torch_inputs,
            threshold=threshold,
            iou_threshold=iou_threshold,
        )

        # Transform boxes back to original image coordinates (matching reference repo)
        preds = invert_affine([framed_meta], preds)

        all_predictions.append(preds[0])  # preds is a list with one item per batch
        all_image_ids.append(img_id)
        all_framed_metas.append(framed_meta)

    # Convert to COCO format
    coco_results = convert_to_coco_format(all_predictions, all_image_ids, all_framed_metas)

    # Save results to temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco_results, f)
        results_file = f.name

    # Load results and run COCO evaluation
    coco_dt = coco.loadRes(results_file)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print results in the same format as benchmark
    logger.info("\n" + "=" * 80)
    logger.info("COCO Evaluation Results - mAP Summary")
    logger.info("=" * 80)
    logger.info(f"efficientdet-d0 ({'TTNN' if use_ttnn else 'PyTorch Reference'})")
    logger.info("")
    logger.info("KEY METRICS:")
    logger.info(f"  mAP@0.50:0.95 (IoU=0.50:0.95): {coco_eval.stats[0]:.3f}")
    logger.info(f"  mAP@0.50      (IoU=0.50):      {coco_eval.stats[1]:.3f}")
    logger.info(f"  mAP@0.75      (IoU=0.75):      {coco_eval.stats[2]:.3f}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("Detailed COCO Metrics")
    logger.info("=" * 80)
    logger.info(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {coco_eval.stats[0]:.3f}")
    logger.info(f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {coco_eval.stats[1]:.3f}")
    logger.info(f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {coco_eval.stats[2]:.3f}")
    logger.info(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {coco_eval.stats[3]:.3f}")
    logger.info(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {coco_eval.stats[4]:.3f}")
    logger.info(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {coco_eval.stats[5]:.3f}")
    logger.info(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {coco_eval.stats[6]:.3f}")
    logger.info(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {coco_eval.stats[7]:.3f}")
    logger.info(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {coco_eval.stats[8]:.3f}")
    logger.info(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {coco_eval.stats[9]:.3f}")
    logger.info(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {coco_eval.stats[10]:.3f}")
    logger.info(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {coco_eval.stats[11]:.3f}")

    # Performance metrics
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        logger.info(f"\nPerformance Metrics:")
        logger.info(f"  Average Inference Time: {avg_inference_time*1000:.2f} ms")

    # Cleanup
    os.unlink(results_file)

    return coco_eval.stats


def main():
    parser = argparse.ArgumentParser(description="COCO Evaluation for EfficientDet D0")
    parser.add_argument(
        "--coco-annotations",
        type=str,
        required=True,
        help="Path to COCO annotations JSON file (e.g., instances_val2017.json)",
    )
    parser.add_argument(
        "--coco-images",
        type=str,
        required=True,
        help="Path to COCO validation images directory",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to model weights file",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID to use (default: 0)",
    )
    parser.add_argument(
        "--l1-small-size",
        type=int,
        default=24576,
        help="L1 small size for device (default: 24576)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Input image height (default: 512)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Input image width (default: 512)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Score threshold for detections (default: 0.05, matching reference repo)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)",
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Use PyTorch reference model instead of TTNN",
    )

    args = parser.parse_args()

    if not PYCOCOTOOLS_AVAILABLE:
        logger.error("pycocotools is required. Install with: pip install pycocotools")
        return

    # Open device
    device = None
    if not args.pytorch_only:
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=args.l1_small_size)

    try:
        evaluate_coco(
            device=device,
            coco_annotations_path=args.coco_annotations,
            coco_images_path=args.coco_images,
            weights_path=args.weights_path,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            threshold=args.threshold,
            iou_threshold=args.iou_threshold,
            use_ttnn=not args.pytorch_only,
        )
    finally:
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
