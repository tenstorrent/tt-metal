# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
EfficientDet Demo

This demo demonstrates how to run EfficientDet object detection model
on Tenstorrent hardware using TTNN.

Usage:
    python models/experimental/efficientdetd0/demo/demo.py
    python models/experimental/efficientdetd0/demo/demo.py --batch-size 1 --height 512 --width 512
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import ttnn
from loguru import logger
from torchvision.ops.boxes import batched_nms

from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.efficientdetd0 import TtEfficientDetBackbone
from models.experimental.efficientdetd0.common import load_torch_model_state
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    infer_torch_module_args,
    create_custom_mesh_preprocessor,
)
from models.demos.utils.common_demo_utils import get_mesh_mappers
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import disable_persistent_kernel_cache, comp_pcc

# Import COCO evaluation function
try:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
    from evaluate_coco import evaluate_coco

    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False


# COCO class names
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "",
    "backpack",
    "umbrella",
    "",
    "",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "",
    "dining table",
    "",
    "",
    "toilet",
    "",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


torch.manual_seed(0)


class BBoxTransform(nn.Module):
    """Transform regression outputs to bounding boxes."""

    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:
            boxes: [batchsize, boxes, (xmin, ymin, xmax, ymax)]
        """
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


class ClipBoxes(nn.Module):
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
    """
    Preprocess image for EfficientDet inference.

    Args:
        image_path: Path to input image
        max_size: Maximum size for resizing (default: 512)
        mean: Mean values for normalization (RGB order)
        std: Std values for normalization (RGB order)

    Returns:
        ori_img: Original image (BGR format)
        framed_img: Preprocessed image tensor (NCHW format)
        framed_meta: Metadata for coordinate transformation
    """
    # Load image
    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB and normalize
    normalized_img = (ori_img[..., ::-1] / 255.0 - mean) / std

    # Resize with aspect ratio preservation
    img_meta = aspectaware_resize_padding(normalized_img, max_size, max_size, means=None)
    framed_img = img_meta[0]
    framed_meta = img_meta[1:]

    # Convert to tensor and add batch dimension
    framed_img = torch.from_numpy(framed_img.transpose(2, 0, 1)).float()  # HWC to CHW
    framed_img = framed_img.unsqueeze(0)  # Add batch dimension

    return ori_img, framed_img, framed_meta


def postprocess(
    regression, classification, anchors, regressBoxes, clipBoxes, input_tensor, threshold=0.5, iou_threshold=0.5
):
    """
    Post-process model outputs to get bounding boxes.

    Args:
        regression: Regression outputs from model
        classification: Classification outputs from model
        anchors: Anchor boxes
        regressBoxes: BBoxTransform module
        clipBoxes: ClipBoxes module
        input_tensor: Input image tensor (for getting image dimensions)
        threshold: Score threshold for detections
        iou_threshold: IoU threshold for NMS

    Returns:
        List of detection dictionaries with 'rois', 'class_ids', and 'scores'
    """
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
    """Transform bounding boxes back to original image coordinates."""
    for i in range(len(preds)):
        if len(preds[i]["rois"]) == 0:
            continue
        else:
            if isinstance(metas, float):
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / metas
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]["rois"][:, [0, 2]] = preds[i]["rois"][:, [0, 2]] / (new_w / old_w)
                preds[i]["rois"][:, [1, 3]] = preds[i]["rois"][:, [1, 3]] / (new_h / old_h)
    return preds


def draw_bounding_boxes(image, preds, class_names, color=(0, 255, 0), label_color=(255, 255, 255)):
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR format)
        preds: List of prediction dictionaries with 'rois', 'class_ids', 'scores'
        class_names: List of class names
        color: Bounding box color (BGR format)
        label_color: Label text color (BGR format)

    Returns:
        Image with bounding boxes drawn
    """
    img = image.copy()

    for pred in preds:
        if len(pred["rois"]) == 0:
            continue

        for j in range(len(pred["rois"])):
            x1, y1, x2, y2 = pred["rois"][j].astype(int)
            class_id = int(pred["class_ids"][j])
            score = float(pred["scores"][j])

            # Clamp coordinates to image boundaries
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # Get class name
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            label = f"{class_name} {score:.2f}"

            # Draw bounding box
            thickness = max(1, int(0.001 * (h + w) / 2))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label background and text
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
            label_y = max(y1, text_height + 5)
            cv2.rectangle(
                img,
                (x1, label_y - text_height - 5),
                (x1 + text_width, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                thickness,
                cv2.LINE_AA,
            )

    return img


def run_efficient_det_demo(
    device,
    model_location_generator=None,
    batch_size=1,
    height=512,
    width=512,
    num_classes=90,
    image_path=None,
    threshold=0.5,
    iou_threshold=0.5,
    weights_path=None,
    output_dir=None,
):
    """
    Run EfficientDet demo on Tenstorrent device.

    Args:
        device: TTNN device
        model_location_generator: Model location generator for weights (optional)
        batch_size: Batch size for inference
        height: Input image height
        width: Input image width
        num_classes: Number of object classes
    """
    disable_persistent_kernel_cache()

    logger.info("=" * 80)
    logger.info("EfficientDet Demo")
    logger.info("=" * 80)
    logger.info(f"Input shape: ({batch_size}, 3, {height}, {width})")
    logger.info(f"Number of classes: {num_classes}")

    # Initialize PyTorch model
    logger.info("\n[1/5] Initializing PyTorch model...")
    torch_model = EfficientDetBackbone(
        num_classes=num_classes,
        compound_coef=0,
        load_weights=False,
    ).eval()
    load_torch_model_state(torch_model, model_location_generator=model_location_generator, weights_path=weights_path)

    # Load and preprocess image
    logger.info("\n[2/5] Loading and preprocessing image...")
    ori_img = None
    framed_meta = None

    # Use provided image path or search for image in resources folder
    if image_path is None:
        # Look for image in resources folder or source folder
        demo_dir = Path(__file__).parent
        resources_dir = demo_dir.parent / "resources"
        source_dir = demo_dir.parent / "source" / "Yet-Another-EfficientDet-Pytorch"

        # Try to find an image file
        for search_dir in [resources_dir, source_dir / "res", source_dir / "test"]:
            if search_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    images = list(search_dir.glob(ext))
                    if images:
                        image_path = str(images[0])
                        break
                if image_path:
                    break

    if image_path is None or not os.path.exists(image_path):
        logger.warning("No image found. Using random input tensor.")
        torch_inputs = torch.randn(batch_size, 3, height, width)
    else:
        logger.info(f"Loading image from: {image_path}")
        ori_img, torch_inputs, framed_meta = preprocess_image(image_path, max_size=max(height, width))
        logger.info(f"Original image shape: {ori_img.shape}")
        logger.info(f"Preprocessed tensor shape: {torch_inputs.shape}")

    # Run PyTorch forward pass for reference
    logger.info("\n[3/5] Running PyTorch reference forward pass...")
    with torch.no_grad():
        torch_features, torch_regression, torch_classification, torch_anchors = torch_model(torch_inputs)
    logger.info(f"PyTorch features: {len(torch_features)} feature maps")
    logger.info(f"PyTorch regression shape: {torch_regression.shape}")
    logger.info(f"PyTorch classification shape: {torch_classification.shape}")

    # Preprocess model parameters for TTNN
    logger.info("\n[4/5] Preprocessing model parameters for TTNN...")
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    module_args = infer_torch_module_args(model=torch_model, input=torch_inputs)

    # Create TTNN model
    logger.info("\n[5/5] Creating TTNN model and running inference...")
    ttnn_model = TtEfficientDetBackbone(
        device=device,
        parameters=parameters,
        conv_params=module_args,
        num_classes=num_classes,
        compound_coef=0,
    )

    # Convert inputs to TTNN format
    ttnn_input_tensor = ttnn.from_torch(
        torch_inputs,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN inference
    logger.info("Running TTNN inference...")
    ttnn_features, ttnn_regression, ttnn_classification = ttnn_model(ttnn_input_tensor)

    # Convert outputs back to PyTorch for comparison
    logger.info("\nConverting outputs to PyTorch format...")
    ttnn_regression_torch = ttnn.to_torch(ttnn_regression, dtype=torch.float32)
    ttnn_classification_torch = ttnn.to_torch(ttnn_classification, dtype=torch.float32)

    # Convert feature maps
    ttnn_features_torch = []
    for i, ttnn_feat in enumerate(ttnn_features):
        ttnn_feat_torch = ttnn.to_torch(ttnn_feat, dtype=torch.float32)
        # Reshape from NHWC to NCHW
        expected_batch, expected_channels, expected_h, expected_w = torch_features[i].shape
        ttnn_feat_torch = ttnn_feat_torch.reshape(expected_batch, expected_h, expected_w, expected_channels)
        ttnn_feat_torch = ttnn_feat_torch.permute(0, 3, 1, 2)
        ttnn_features_torch.append(ttnn_feat_torch)

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("Results Summary")
    logger.info("=" * 80)
    logger.info(f"Number of feature maps: {len(ttnn_features_torch)}")
    for i, (torch_feat, ttnn_feat) in enumerate(zip(torch_features, ttnn_features_torch)):
        logger.info(f"Feature map P{i+3}: PyTorch {torch_feat.shape} | TTNN {ttnn_feat.shape}")
    logger.info(f"Regression: PyTorch {torch_regression.shape} | TTNN {ttnn_regression_torch.shape}")
    logger.info(f"Classification: PyTorch {torch_classification.shape} | TTNN {ttnn_classification_torch.shape}")

    # Diagnostic: Compare output statistics
    logger.info("\n" + "=" * 80)
    logger.info("Output Statistics Comparison")
    logger.info("=" * 80)
    logger.info(
        f"PyTorch Regression - Min: {torch_regression.min().item():.6f}, Max: {torch_regression.max().item():.6f}, Mean: {torch_regression.mean().item():.6f}, Std: {torch_regression.std().item():.6f}"
    )
    logger.info(
        f"TTNN Regression    - Min: {ttnn_regression_torch.min().item():.6f}, Max: {ttnn_regression_torch.max().item():.6f}, Mean: {ttnn_regression_torch.mean().item():.6f}, Std: {ttnn_regression_torch.std().item():.6f}"
    )
    logger.info(
        f"PyTorch Classification - Min: {torch_classification.min().item():.6f}, Max: {torch_classification.max().item():.6f}, Mean: {torch_classification.mean().item():.6f}, Std: {torch_classification.std().item():.6f}"
    )
    logger.info(
        f"TTNN Classification    - Min: {ttnn_classification_torch.min().item():.6f}, Max: {ttnn_classification_torch.max().item():.6f}, Mean: {ttnn_classification_torch.mean().item():.6f}, Std: {ttnn_classification_torch.std().item():.6f}"
    )

    # Check max scores per anchor (before thresholding)
    torch_max_scores = torch.max(torch_classification, dim=2)[0]
    ttnn_max_scores = torch.max(ttnn_classification_torch, dim=2)[0]
    logger.info(
        f"\nPyTorch Max scores per anchor - Min: {torch_max_scores.min().item():.6f}, Max: {torch_max_scores.max().item():.6f}, Mean: {torch_max_scores.mean().item():.6f}"
    )
    logger.info(
        f"TTNN Max scores per anchor    - Min: {ttnn_max_scores.min().item():.6f}, Max: {ttnn_max_scores.max().item():.6f}, Mean: {ttnn_max_scores.mean().item():.6f}"
    )
    logger.info(f"Threshold: {threshold}")
    logger.info(
        f"PyTorch anchors above threshold: {(torch_max_scores > threshold).sum().item()} / {torch_max_scores.numel()}"
    )
    logger.info(
        f"TTNN anchors above threshold:    {(ttnn_max_scores > threshold).sum().item()} / {ttnn_max_scores.numel()}"
    )

    # PCC Comparisons
    logger.info("\n" + "=" * 80)
    logger.info("PCC (Pearson Correlation Coefficient) Comparisons")
    logger.info("=" * 80)
    PCC_THRESHOLD = 0.99  # Standard threshold used in tests

    # Compare BiFPN feature maps
    logger.info("\n--- BiFPN Feature Maps ---")
    for i, (torch_feat, ttnn_feat) in enumerate(zip(torch_features, ttnn_features_torch)):
        passing, pcc_value = comp_pcc(torch_feat, ttnn_feat, pcc=PCC_THRESHOLD)
        logger.info(f"Feature P{i+3}: PCC = {pcc_value:.6f} {'✓' if passing else '✗'} (threshold: {PCC_THRESHOLD})")

    # Compare Regression outputs
    logger.info("\n--- Regressor Outputs ---")
    passing, pcc_value = comp_pcc(torch_regression, ttnn_regression_torch, pcc=PCC_THRESHOLD)
    logger.info(f"Regression: PCC = {pcc_value:.6f} {'✓' if passing else '✗'} (threshold: {PCC_THRESHOLD})")

    # Compare Classification outputs
    logger.info("\n--- Classifier Outputs ---")
    passing, pcc_value = comp_pcc(torch_classification, ttnn_classification_torch, pcc=PCC_THRESHOLD)
    logger.info(f"Classification: PCC = {pcc_value:.6f} {'✓' if passing else '✗'} (threshold: {PCC_THRESHOLD})")

    # Post-process to get bounding boxes
    logger.info("\n" + "=" * 80)
    logger.info("Post-processing to get bounding boxes...")
    logger.info("=" * 80)

    # Initialize post-processing modules
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Post-process PyTorch reference outputs
    logger.info("\nPost-processing PyTorch reference outputs...")
    torch_preds = postprocess(
        torch_regression,
        torch_classification,
        torch_anchors,
        regressBoxes,
        clipBoxes,
        torch_inputs,
        threshold=threshold,
        iou_threshold=iou_threshold,
    )

    # Post-process TTNN outputs
    logger.info("Post-processing TTNN outputs...")
    ttnn_preds = postprocess(
        ttnn_regression_torch,
        ttnn_classification_torch,
        torch_anchors,
        regressBoxes,
        clipBoxes,
        torch_inputs,
        threshold=threshold,
        iou_threshold=iou_threshold,
    )

    # Transform boxes back to original image coordinates if we have image metadata
    if framed_meta is not None:
        # framed_meta is a tuple, but invert_affine expects a list of tuples (one per batch item)
        torch_preds = invert_affine([framed_meta], torch_preds)
        ttnn_preds = invert_affine([framed_meta], ttnn_preds)

    # Display bounding boxes
    logger.info("\n" + "=" * 80)
    logger.info("Detected Objects (Bounding Boxes)")
    logger.info("=" * 80)

    # Display PyTorch reference results
    logger.info("\n--- PyTorch Reference Results ---")
    if len(torch_preds) > 0 and len(torch_preds[0]["rois"]) > 0:
        for i, pred in enumerate(torch_preds):
            logger.info(f"\nImage {i+1}: Found {len(pred['rois'])} detections")
            for j in range(len(pred["rois"])):
                x1, y1, x2, y2 = pred["rois"][j].astype(int)
                class_id = int(pred["class_ids"][j])
                score = float(pred["scores"][j])
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                logger.info(f"  Detection {j+1}: {class_name} (score: {score:.3f}) - Box: [{x1}, {y1}, {x2}, {y2}]")
    else:
        logger.info("No objects detected.")

    # Display TTNN results
    logger.info("\n--- TTNN Results ---")
    if len(ttnn_preds) > 0 and len(ttnn_preds[0]["rois"]) > 0:
        for i, pred in enumerate(ttnn_preds):
            logger.info(f"\nImage {i+1}: Found {len(pred['rois'])} detections")
            for j in range(len(pred["rois"])):
                x1, y1, x2, y2 = pred["rois"][j].astype(int)
                class_id = int(pred["class_ids"][j])
                score = float(pred["scores"][j])
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                logger.info(f"  Detection {j+1}: {class_name} (score: {score:.3f}) - Box: [{x1}, {y1}, {x2}, {y2}]")
    else:
        logger.info("No objects detected.")

    # Visualize bounding boxes on image if we have an original image
    if ori_img is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Generating visualization images...")
        logger.info("=" * 80)

        # Draw PyTorch reference results (green boxes)
        torch_vis_img = draw_bounding_boxes(
            ori_img, torch_preds, COCO_CLASSES, color=(0, 255, 0), label_color=(0, 0, 0)
        )

        # Draw TTNN results (red boxes)
        ttnn_vis_img = draw_bounding_boxes(
            ori_img, ttnn_preds, COCO_CLASSES, color=(0, 0, 255), label_color=(255, 255, 255)
        )

        # Save visualization images
        if output_dir is None:
            demo_dir = Path(__file__).parent
            output_dir = demo_dir / "output"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get base image name
        if image_path:
            image_name = Path(image_path).stem
        else:
            image_name = "image"

        torch_output_path = output_dir / f"{image_name}_pytorch_reference.jpg"
        ttnn_output_path = output_dir / f"{image_name}_ttnn.jpg"

        cv2.imwrite(str(torch_output_path), torch_vis_img)
        cv2.imwrite(str(ttnn_output_path), ttnn_vis_img)

        logger.info(f"PyTorch reference visualization saved to: {torch_output_path}")
        logger.info(f"TTNN visualization saved to: {ttnn_output_path}")

    # Cleanup
    ttnn.deallocate(ttnn_input_tensor)
    for feat in ttnn_features:
        ttnn.deallocate(feat)
    ttnn.deallocate(ttnn_regression)
    ttnn.deallocate(ttnn_classification)

    logger.info("\n" + "=" * 80)
    logger.info("Demo completed successfully!")
    logger.info("=" * 80)


def main():
    """Main function to run the EfficientDet demo."""
    parser = argparse.ArgumentParser(description="EfficientDet Demo")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
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
        "--num-classes",
        type=int,
        default=90,
        help="Number of object classes (default: 90)",
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
        "--image-path",
        type=str,
        default=None,
        help="Path to input image (if not provided, will search in resources folder)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to model weights file (if not provided, will search in resources folder or default location)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output images with bounding boxes (default: demo/output/)",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Enable COCO evaluation mode (requires --coco-annotations and --coco-images)",
    )
    parser.add_argument(
        "--coco-annotations",
        type=str,
        default=None,
        help="Path to COCO annotations JSON file for evaluation (e.g., instances_val2017.json). Required if --eval-mode is set.",
    )
    parser.add_argument(
        "--coco-images",
        type=str,
        default=None,
        help="Path to COCO validation images directory for evaluation. Required if --eval-mode is set.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of COCO samples to evaluate (default: all if --eval-mode is set)",
    )

    args = parser.parse_args()

    # Check if evaluation mode is requested
    if args.eval_mode:
        # Validate required arguments for evaluation mode
        if not args.coco_annotations or not args.coco_images:
            logger.error("Evaluation mode requires both --coco-annotations and --coco-images")
            parser.print_help()
            return

        if not COCO_EVAL_AVAILABLE:
            logger.error(
                "COCO evaluation not available. Please ensure pycocotools is installed and evaluate_coco.py is accessible."
            )
            return

        # Run both PyTorch and TTNN evaluations
        logger.info("=" * 80)
        logger.info("COCO Evaluation - PyTorch Reference vs TTNN")
        logger.info("=" * 80)

        # First run PyTorch reference evaluation
        logger.info("\n" + "=" * 80)
        logger.info("Running COCO evaluation on PyTorch Reference model...")
        logger.info("=" * 80)
        pytorch_stats = evaluate_coco(
            device=None,  # Not needed for PyTorch
            coco_annotations_path=args.coco_annotations,
            coco_images_path=args.coco_images,
            weights_path=args.weights_path,
            model_location_generator=None,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            num_classes=args.num_classes,
            threshold=args.threshold,  # Lower threshold for evaluation
            # iou_threshold=0.2,
            iou_threshold=args.iou_threshold,
            use_ttnn=False,  # PyTorch reference
        )

        # Then run TTNN evaluation
        logger.info("\n" + "=" * 80)
        logger.info("Running COCO evaluation on TTNN model...")
        logger.info("=" * 80)
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=args.l1_small_size)

        try:
            ttnn_stats = evaluate_coco(
                device=device,
                coco_annotations_path=args.coco_annotations,
                coco_images_path=args.coco_images,
                weights_path=args.weights_path,
                model_location_generator=None,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                num_classes=args.num_classes,
                threshold=args.threshold,  # Lower threshold for evaluation
                iou_threshold=args.iou_threshold,
                use_ttnn=True,
            )

            # Print comparison summary
            logger.info("\n" + "=" * 80)
            logger.info("COCO Evaluation Results Summary")
            logger.info("=" * 80)

            # Print PyTorch results in benchmark format
            logger.info("\n--- PyTorch Reference Results ---")
            logger.info("efficientdet-d0 (PyTorch)")
            if pytorch_stats and len(pytorch_stats) >= 12:
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {pytorch_stats[0]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {pytorch_stats[1]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {pytorch_stats[2]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {pytorch_stats[3]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {pytorch_stats[4]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {pytorch_stats[5]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {pytorch_stats[6]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {pytorch_stats[7]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {pytorch_stats[8]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {pytorch_stats[9]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {pytorch_stats[10]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {pytorch_stats[11]:.3f}"
                )

            # Print TTNN results in benchmark format
            logger.info("\n--- TTNN Results ---")
            logger.info("efficientdet-d0 (TTNN)")
            if ttnn_stats and len(ttnn_stats) >= 12:
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ttnn_stats[0]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {ttnn_stats[1]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {ttnn_stats[2]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {ttnn_stats[3]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {ttnn_stats[4]:.3f}"
                )
                logger.info(
                    f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {ttnn_stats[5]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {ttnn_stats[6]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {ttnn_stats[7]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ttnn_stats[8]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {ttnn_stats[9]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {ttnn_stats[10]:.3f}"
                )
                logger.info(
                    f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {ttnn_stats[11]:.3f}"
                )

            # Print side-by-side comparison
            logger.info("\n" + "=" * 80)
            logger.info("Side-by-Side Comparison")
            logger.info("=" * 80)
            logger.info(f"{'Metric':<50} {'PyTorch':<12} {'TTNN':<12} {'Diff':<12}")
            logger.info("-" * 80)

            metric_names = [
                "AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
                "AP @[ IoU=0.50      | area=   all | maxDets=100 ]",
                "AP @[ IoU=0.75      | area=   all | maxDets=100 ]",
                "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
                "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
                "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
                "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
                "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
                "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
                "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
            ]

            for i, metric_name in enumerate(metric_names):
                pytorch_val = pytorch_stats[i] if pytorch_stats and len(pytorch_stats) > i else 0.0
                ttnn_val = ttnn_stats[i] if ttnn_stats and len(ttnn_stats) > i else 0.0
                diff = ttnn_val - pytorch_val
                logger.info(f"{metric_name:<50} {pytorch_val:>11.3f}  {ttnn_val:>11.3f}  {diff:>+11.3f}")

            logger.info("=" * 80)

        finally:
            ttnn.close_device(device)
    else:
        # Normal demo mode
        # Run regular demo
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=args.l1_small_size)

        try:
            run_efficient_det_demo(
                device=device,
                model_location_generator=None,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                num_classes=args.num_classes,
                image_path=args.image_path,
                threshold=args.threshold,
                iou_threshold=args.iou_threshold,
                weights_path=args.weights_path,
                output_dir=args.output_dir,
            )
        finally:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
