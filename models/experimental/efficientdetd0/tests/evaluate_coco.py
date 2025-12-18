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

import numpy as np
import torch
import ttnn
from loguru import logger

from models.demos.utils.common_demo_utils import get_mesh_mappers
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import disable_persistent_kernel_cache
from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.efficientdetd0 import TtEfficientDetBackbone
from models.experimental.efficientdetd0.common import load_torch_model_state
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    infer_torch_module_args,
    create_custom_mesh_preprocessor,
)
from models.experimental.efficientdetd0.reference.utils import (
    ClipBoxes,
    BBoxTransform,
)
from models.experimental.efficientdetd0.demo.demo_utils import (
    postprocess,
    invert_affine,
    preprocess_image,
)

# Try to import pycocotools for official COCO evaluation
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logger.warning("pycocotools not available. Install with: pip install pycocotools")


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
            module_args=module_args,
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
    logger.info("=" * 80)
    logger.info("COCO Evaluation Results - mAP Summary")
    logger.info("=" * 80)
    logger.info(f"efficientdet-d0 ({'TTNN' if use_ttnn else 'PyTorch Reference'})")
    logger.info("")
    logger.info("KEY METRICS:")
    logger.info(f"  mAP@0.50:0.95 (IoU=0.50:0.95): {coco_eval.stats[0]:.3f}")
    logger.info(f"  mAP@0.50      (IoU=0.50):      {coco_eval.stats[1]:.3f}")
    logger.info(f"  mAP@0.75      (IoU=0.75):      {coco_eval.stats[2]:.3f}")
    logger.info("")

    # Performance metrics
    if inference_times:
        avg_inference_time = np.mean(inference_times[1:])
        logger.info(f" Performance Metrics:")
        logger.info(f" Average Inference Time: {avg_inference_time*1000:.2f} ms")
        logger.info(f" Average FPS: {1/avg_inference_time:.2f}")

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
