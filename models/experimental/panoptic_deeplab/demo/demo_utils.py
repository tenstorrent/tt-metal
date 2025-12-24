# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, Tuple
import cv2
import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import cdist
from models.experimental.panoptic_deeplab.reference.pytorch_postprocessing import get_panoptic_segmentation
from models.experimental.panoptic_deeplab.reference.pytorch_model import DEEPLAB_V3_PLUS
from models.experimental.panoptic_deeplab.tt.common import (
    get_panoptic_deeplab_images_path,
    get_panoptic_deeplab_weights_path,
    load_images,
)
from models.common.utility_functions import is_blackhole
import pytest

# Cityscapes dataset configuration
CITYSCAPES_CATEGORIES = [
    {"id": 0, "name": "road", "color": [128, 64, 128]},
    {"id": 1, "name": "sidewalk", "color": [244, 35, 232]},
    {"id": 2, "name": "building", "color": [70, 70, 70]},
    {"id": 3, "name": "wall", "color": [102, 102, 156]},
    {"id": 4, "name": "fence", "color": [190, 153, 153]},
    {"id": 5, "name": "pole", "color": [153, 153, 153]},
    {"id": 6, "name": "traffic light", "color": [250, 170, 30]},
    {"id": 7, "name": "traffic sign", "color": [220, 220, 0]},
    {"id": 8, "name": "vegetation", "color": [107, 142, 35]},
    {"id": 9, "name": "terrain", "color": [152, 251, 152]},
    {"id": 10, "name": "sky", "color": [70, 130, 180]},
    {"id": 11, "name": "person", "color": [220, 20, 60]},
    {"id": 12, "name": "rider", "color": [255, 0, 0]},
    {"id": 13, "name": "car", "color": [0, 0, 142]},
    {"id": 14, "name": "truck", "color": [0, 0, 70]},
    {"id": 15, "name": "bus", "color": [0, 60, 100]},
    {"id": 16, "name": "train", "color": [0, 80, 100]},
    {"id": 17, "name": "motorcycle", "color": [0, 0, 230]},
    {"id": 18, "name": "bicycle", "color": [119, 11, 32]},
]


def preprocess_input_params(output_dir, model_category, current_dir, model_location_generator=None):
    images_dir = get_panoptic_deeplab_images_path(model_location_generator, current_dir)
    weights_path = get_panoptic_deeplab_weights_path(model_location_generator, current_dir)
    output_dir = os.path.join(output_dir, model_category)
    images_paths = load_images(images_dir)
    if len(images_paths) == 0:
        logger.error(f"No images found in the specified path: {images_dir}")
        raise FileNotFoundError(f"No images found in the specified path: {images_dir}")
    return images_paths, weights_path, output_dir


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (512, 1024)) -> torch.Tensor:
    """
    Preprocess image for Panoptic DeepLab inference.

    Args:
        image_path: Path to input image
        target_size: Target size as (height, width)

    Returns:
        Preprocessed tensor in NCHW format (normalized to [0,1], ImageNet norm applied on device)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image = cv2.resize(image, (target_size[1], target_size[0]))  # cv2 expects (width, height)

    # Convert to tensor and normalize to [0, 1] only
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def merge_nearby_instances(panoptic_seg: np.ndarray, max_distance: int = 15) -> np.ndarray:
    """
    Merge nearby instances of the same class that are likely parts of the same object.

    Args:
        panoptic_seg: Panoptic segmentation map [H, W]
        max_distance: Maximum distance for merging instances

    Returns:
        Updated panoptic segmentation map
    """
    result = panoptic_seg.copy()
    unique_ids = np.unique(panoptic_seg)
    label_divisor = 1000

    # Group instances by class
    class_instances = {}
    for segment_id in unique_ids:
        if segment_id == 0 or segment_id == 255:  # Skip background and void
            continue

        category_id = segment_id // label_divisor
        instance_id = segment_id % label_divisor

        if category_id >= 11 and category_id <= 18:  # Only merge thing classes
            if category_id not in class_instances:
                class_instances[category_id] = []
            class_instances[category_id].append(segment_id)

    # For each class, find nearby instances and merge them
    for category_id, instance_ids in class_instances.items():
        if len(instance_ids) < 2:
            continue

        # Calculate centroids for each instance
        centroids = []
        for inst_id in instance_ids:
            mask = panoptic_seg == inst_id
            if np.sum(mask) == 0:
                continue
            y_coords, x_coords = np.where(mask)
            centroid = [np.mean(y_coords), np.mean(x_coords)]
            centroids.append(centroid)

        if len(centroids) < 2:
            continue

        # Calculate distances between centroids
        distances = cdist(centroids, centroids)

        # Find pairs to merge - use more aggressive iterative merging
        merged = set()
        for i in range(len(instance_ids)):
            if i in merged:
                continue

            # Merge all instances within distance to instance i
            for j in range(len(instance_ids)):
                if i == j or j in merged:
                    continue

                if distances[i, j] < max_distance:
                    # Merge instance j into instance i
                    mask_j = result == instance_ids[j]
                    result[mask_j] = instance_ids[i]
                    merged.add(j)
                    logger.debug(f"Merged {category_id} instance {instance_ids[j]} into {instance_ids[i]}")

    return result


def save_predictions(
    output_dir: str,
    image_name: str,
    original_image: np.ndarray,
    panoptic_vis: np.ndarray,
):
    """Save prediction results to files."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]

    # Save original image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

    # Save panoptic segmentation (blended)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_panoptic.jpg"), cv2.cvtColor(panoptic_vis, cv2.COLOR_RGB2BGR))

    logger.info(f"Saved predictions for {image_name} to {output_dir}")


def resolve_demo_paths(demo_file: str) -> Dict[str, str]:
    """
    Resolve all demo-related paths from the demo file location.

    Args:
        demo_file: Path to the demo file (usually __file__)

    Returns:
        Dictionary with resolved paths for images, weights, outputs
    """
    demo_dir = os.path.dirname(os.path.abspath(demo_file))
    base_dir = os.path.dirname(demo_dir)  # panoptic_deeplab/

    return {
        "images": os.path.join(base_dir, "images"),
        "weights": os.path.join(base_dir, "weights", "model_final_bd324a.pkl"),
        "outputs": os.path.join(base_dir, "demo_outputs"),
    }


def create_deeplab_v3plus_visualization(
    semantic_pred: np.ndarray,
    original_image: np.ndarray,
):
    # Calculate semantic classes
    semantic_classes = np.argmax(semantic_pred, axis=2)

    # ============================================================================
    # Semantic-Only Mode (when center_pred and offset_pred are None)
    # ============================================================================
    logger.info("Creating semantic segmentation visualization (no instance heads)")

    # Create semantic visualization using Cityscapes color map
    h, w = semantic_classes.shape
    vis_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(len(CITYSCAPES_CATEGORIES)):
        mask = semantic_classes == class_id
        if np.any(mask):
            color = CITYSCAPES_CATEGORIES[class_id]["color"]
            vis_image[mask] = color

    # Blend with original image if provided
    if original_image is not None:
        # Ensure original_image is uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        # Resize if needed
        if original_image.shape[:2] != (h, w):
            original_image = cv2.resize(original_image, (w, h))

        # Blend
        alpha = 0.6
        vis_image = cv2.addWeighted(original_image.astype(np.uint8), 1 - alpha, vis_image, alpha, 0)

    # Create minimal panoptic info DEEPLAB_V3_PLUS
    panoptic_info = {
        "mode": DEEPLAB_V3_PLUS,
        "num_classes": len(np.unique(semantic_classes)),
        "class_distribution": {
            int(cls): int(count) for cls, count in zip(*np.unique(semantic_classes, return_counts=True))
        },
    }

    return vis_image, panoptic_info


def create_panoptic_visualization(
    semantic_pred: np.ndarray,
    center_pred: np.ndarray = None,
    offset_pred: np.ndarray = None,
    original_image: np.ndarray = None,
    score_threshold: float = 0.05,
    center_threshold: float = 0.05,
    nms_kernel: int = 11,
    top_k: int = 1000,
    stuff_area: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """
    Create panoptic or semantic segmentation visualization from model outputs.

    Supports two modes:
    1. DeeplabV3Plus mode: When center_pred and offset_pred are None
    2. Full panoptic mode: When all predictions are provided

    Args:
        semantic_pred: Semantic segmentation predictions [H, W, num_classes]
        center_pred: Center heatmap predictions [H, W, 1] (optional, None for DeeplabV3Plus)
        offset_pred: Offset predictions [H, W, 2] (optional, None for DeeplabV3Plus)
        original_image: Original input image [H, W, 3]
        score_threshold: Minimum score for instance detection (panoptic mode only)
        center_threshold: Minimum center score for instance detection (panoptic mode only)
        nms_kernel: Kernel size for non-maximum suppression (panoptic mode only)
        top_k: Maximum number of instances to detect (panoptic mode only)
        stuff_area: Minimum area for stuff classes (panoptic mode only)

    Returns:
        Tuple of (visualization_image, panoptic_info)
    """
    # ============================================================================
    # Full Panoptic Mode (when center_pred and offset_pred are provided)
    # ============================================================================
    logger.info("Creating full panoptic segmentation visualization (with instance heads)")

    # Convert predictions to torch tensors and prepare for postprocessing
    # Semantic: take argmax to get class predictions [H, W]
    semantic_tensor = torch.from_numpy(semantic_pred)  # [H, W, C]
    semantic_tensor = torch.argmax(semantic_tensor, dim=2).unsqueeze(0)  # [1, H, W]

    # Center: handle center predictions - ensure proper 3D shape [1, H, W]
    center_tensor = torch.from_numpy(center_pred)  # [H, W, 1] or [H, W]
    if center_tensor.dim() == 3:
        center_tensor = center_tensor.squeeze(-1)  # [H, W]
    center_tensor = center_tensor.unsqueeze(0)  # [1, H, W]

    # Offset: rearrange to [2, H, W] format expected by the function
    offset_tensor = torch.from_numpy(offset_pred).permute(2, 0, 1)  # [2, H, W]

    # Get panoptic segmentation using reference implementation
    logger.info(f"Using thresholds - center: {center_threshold}, score: {score_threshold}, stuff_area: {stuff_area}")
    logger.debug(f"Max center prediction: {center_tensor.max().item():.6f}")
    logger.debug(f"Pixels above center threshold: {(center_tensor > center_threshold).sum().item()}")
    logger.debug(f"Pixels with center > 0.01: {(center_tensor > 0.01).sum().item()}")
    logger.debug(f"Pixels with center > 0.05: {(center_tensor > 0.05).sum().item()}")

    panoptic_seg, center_points = get_panoptic_segmentation(
        semantic_tensor,
        center_tensor,
        offset_tensor,
        thing_ids=set(range(11, 19)),  # Cityscapes thing classes (person, rider, car, etc.)
        label_divisor=1000,
        stuff_area=stuff_area,
        void_label=255,
        threshold=center_threshold,
        nms_kernel=nms_kernel,
        top_k=top_k,
        foreground_mask=None,
    )

    logger.debug(f"Panoptic segmentation returned {len(np.unique(panoptic_seg))} unique segments")

    # Create visualization
    panoptic_seg = panoptic_seg.squeeze(0).numpy()  # Remove batch dimension

    # Calculate semantic classes early for use in post-processing
    semantic_classes = np.argmax(semantic_pred, axis=2)
    # Get unique segment IDs
    unique_raw = np.unique(panoptic_seg)

    # Clean up any invalid segment IDs
    max_valid_id = 18999  # Max valid ID (class 18, instance 999)
    invalid_ids = unique_raw[unique_raw > max_valid_id]
    if len(invalid_ids) > 0:
        logger.warning(f"Found invalid segment IDs: {invalid_ids}")
        for invalid_id in invalid_ids:
            panoptic_seg[panoptic_seg == invalid_id] = 0

    # Clean up mixed semantic-panoptic assignments to fix color bleeding
    cleaned_panoptic = panoptic_seg.copy()

    for segment_id in unique_raw:
        if segment_id == 0 or segment_id == 255:
            continue
        category_id = segment_id // 1000
        mask = panoptic_seg == segment_id
        semantic_at_mask = semantic_classes[mask]

        if len(semantic_at_mask) > 0:
            semantic_counts = np.bincount(semantic_at_mask, minlength=19)
            most_common_semantic = np.argmax(semantic_counts)
            agreement_ratio = semantic_counts[most_common_semantic] / len(semantic_at_mask)

            # Remove pixels that don't match the panoptic segment's class
            if agreement_ratio < 0.85:  # Relaxed from 98% to 85% semantic agreement
                wrong_semantic_mask = mask & (semantic_classes != category_id)
                cleaned_panoptic[wrong_semantic_mask] = 0  # Set wrong pixels to background

    # Update panoptic segmentation with cleaned version
    panoptic_seg = cleaned_panoptic

    # Enable post-processing for TTNN to merge fragmented instances
    if len(np.unique(panoptic_seg)) > 15:  # Raised threshold to be less aggressive
        logger.info("Detected over-segmentation, applying post-processing to merge instances...")
        panoptic_seg = merge_nearby_instances(panoptic_seg, max_distance=40)  # Reduced from 80 to 40 pixels
        logger.info(f"After merging: {len(np.unique(panoptic_seg))} unique segments")

    # Create segments_info from panoptic segmentation
    segments_info = []
    unique_ids = np.unique(panoptic_seg)
    label_divisor = 1000

    logger.info(f"Panoptic segmentation shape: {panoptic_seg.shape}, unique IDs: {len(unique_ids)}")
    logger.info(f"Unique segment IDs: {unique_ids[:10]}...")  # Show first 10 IDs

    # Debug info about background coverage
    background_pixels = np.sum(panoptic_seg == 0)
    void_pixels = np.sum(panoptic_seg == 255)
    total_pixels = panoptic_seg.size
    logger.info(f"Background pixels: {background_pixels}/{total_pixels} ({background_pixels/total_pixels*100:.1f}%)")
    logger.info(f"Void pixels: {void_pixels}/{total_pixels} ({void_pixels/total_pixels*100:.1f}%)")

    # Show distribution of detected classes
    semantic_unique, semantic_counts = np.unique(semantic_classes, return_counts=True)
    for class_id, count in zip(semantic_unique, semantic_counts):
        if class_id < len(CITYSCAPES_CATEGORIES):
            class_name = CITYSCAPES_CATEGORIES[class_id]["name"]
            logger.info(f"Semantic class {class_id} ({class_name}): {count} pixels ({count/total_pixels*100:.1f}%)")
        else:
            logger.info(f"Unknown semantic class {class_id}: {count} pixels")

    for segment_id in unique_ids:
        if segment_id == 0:  # Skip background
            continue

        # Handle void label (255) - map to background instead of processing as segment
        if segment_id == 255:
            continue

        # Decode panoptic ID to get semantic category and instance ID
        category_id = segment_id // label_divisor
        instance_id = segment_id % label_divisor

        # Calculate segment area
        mask = panoptic_seg == segment_id
        area = np.sum(mask)

        segments_info.append(
            {
                "id": int(segment_id),
                "category_id": int(category_id),
                "area": int(area),
                "iscrowd": 0,
            }
        )

    # Create colored visualization
    height, width = panoptic_seg.shape
    vis_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Always use panoptic segmentation when we have good results
    logger.info(f"Using panoptic segmentation with {len(segments_info)} segments")

    # Fill in background regions with clean semantic segmentation
    background_mask = (panoptic_seg == 0) | (panoptic_seg == 255)

    # Create clean semantic fallback for unassigned regions
    for class_id in range(len(CITYSCAPES_CATEGORIES)):
        class_mask = (semantic_classes == class_id) & background_mask
        if np.any(class_mask):
            color = CITYSCAPES_CATEGORIES[class_id]["color"]

            # Skip creating additional instances from semantic fallback to avoid false positives
            # Just apply semantic color for all thing classes
            if class_id >= 11 and class_id <= 18:  # All thing classes
                vis_image[class_mask] = color
            else:
                # Stuff classes get semantic color (dimmer to distinguish from instances)
                color_dim = [int(c * 0.7) for c in color]
                vis_image[class_mask] = color_dim

    # Apply panoptic segmentation colors with instance-specific variations
    # Sort segments by area (largest first) to ensure proper layering
    segments_sorted = sorted(segments_info, key=lambda x: x["area"], reverse=True)

    for segment in segments_sorted:
        segment_id = segment["id"]
        category_id = segment["category_id"]
        instance_id = segment_id % label_divisor
        mask = panoptic_seg == segment_id

        # Get base color for this category
        if category_id < len(CITYSCAPES_CATEGORIES):
            base_color = np.array(CITYSCAPES_CATEGORIES[category_id]["color"], dtype=np.float32)
        else:
            # Generate stable color for unknown categories
            np.random.seed(segment_id)
            base_color = np.array(np.random.randint(128, 255, 3), dtype=np.float32)

        # Generate slightly different color for each instance
        # Use instance_id to create deterministic color variations
        # Apply small variations to RGB channels to create distinct but similar colors
        np.random.seed(instance_id)  # Deterministic based on instance_id
        variation = np.random.randint(-25, 26, 3)  # Small random variation per channel

        # Apply variation while keeping color in valid range and preserving category character
        color = base_color.astype(np.float32) + variation
        color = np.clip(color, 0, 255).astype(np.uint8)

        # Apply color
        vis_image[mask] = color

    # Add white borders around instances (fully surrounding each instance)
    # Use morphological operations to detect all boundaries of each instance
    # Use a 3x3 kernel with 2 iterations for slightly thicker borders
    kernel = np.ones((3, 3), np.uint8)

    # Create border mask that will surround all instances
    border_mask = np.zeros((height, width), dtype=bool)

    # For each unique segment, find its complete boundary
    unique_segments = np.unique(panoptic_seg)
    for seg_id in unique_segments:
        if seg_id == 0 or seg_id == 255:  # Skip background and void
            continue

        # Create mask for this segment
        seg_mask = (panoptic_seg == seg_id).astype(np.uint8) * 255

        # Dilate the segment to create a border region
        dilated_seg = cv2.dilate(seg_mask, kernel, iterations=2)

        # Border pixels are where dilated segment exists but original segment does not
        # This captures boundaries with background, void, and other instances
        border_pixels = (dilated_seg > 0) & (panoptic_seg != seg_id)
        border_mask |= border_pixels

    # Apply white borders
    vis_image[border_mask] = [255, 255, 255]

    # Add labels to instances with overlap avoidance
    # Draw labels on the visualization image before blending
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Track placed label bounding boxes to avoid overlaps
    placed_labels = []  # List of (x1, y1, x2, y2) bounding boxes

    def check_overlap(bbox, existing_boxes, margin=5):
        """Check if a bounding box overlaps with any existing boxes."""
        x1, y1, x2, y2 = bbox
        for ex_x1, ex_y1, ex_x2, ex_y2 in existing_boxes:
            # Check if boxes overlap (with margin)
            if not (x2 + margin < ex_x1 or x1 - margin > ex_x2 or y2 + margin < ex_y1 or y1 - margin > ex_y2):
                return True
        return False

    for segment in segments_sorted:
        segment_id = segment["id"]
        category_id = segment["category_id"]
        instance_id = segment_id % label_divisor
        mask = panoptic_seg == segment_id

        # Skip if segment is too small to label
        if segment["area"] < 100:  # Skip very small segments
            continue

        # Calculate centroid and get all pixel coordinates for alternative positions
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            continue

        centroid_y = int(np.mean(y_coords))
        centroid_x = int(np.mean(x_coords))

        # Get category name
        if category_id < len(CITYSCAPES_CATEGORIES):
            category_name = CITYSCAPES_CATEGORIES[category_id]["name"]
        else:
            category_name = f"class_{category_id}"

        # Create label text
        label_text = f"{category_name}#{instance_id}"

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Try multiple positions to avoid overlaps
        # Start with centroid, then try positions around it
        label_placed = False
        padding = 2
        box_width = text_width + 2 * padding
        box_height = text_height + baseline + 2 * padding

        # Generate candidate positions: centroid, then spiral outward
        candidate_positions = [
            (centroid_x, centroid_y),  # Centroid
            (centroid_x, centroid_y - box_height - 10),  # Above
            (centroid_x, centroid_y + box_height + 10),  # Below
            (centroid_x - box_width - 10, centroid_y),  # Left
            (centroid_x + box_width + 10, centroid_y),  # Right
            (centroid_x - box_width // 2, centroid_y - box_height // 2),  # Top-left
            (centroid_x + box_width // 2, centroid_y - box_height // 2),  # Top-right
            (centroid_x - box_width // 2, centroid_y + box_height // 2),  # Bottom-left
            (centroid_x + box_width // 2, centroid_y + box_height // 2),  # Bottom-right
        ]

        # Also try some random positions within the instance mask
        if len(x_coords) > 10:
            np.random.seed(segment_id)  # Deterministic
            for _ in range(5):
                idx = np.random.randint(0, len(x_coords))
                candidate_positions.append((int(x_coords[idx]), int(y_coords[idx])))

        # Try each candidate position
        for candidate_x, candidate_y in candidate_positions:
            # Calculate label position (centered on candidate point)
            label_x = max(padding, min(candidate_x - text_width // 2, width - text_width - padding))
            label_y = max(text_height + padding, min(candidate_y, height - baseline - padding))

            # Check if position is within image bounds
            if (
                label_x + text_width + padding > width
                or label_y + baseline + padding > height
                or label_x < padding
                or label_y < text_height + padding
            ):
                continue

            # Create bounding box
            bbox = (
                label_x - padding,
                label_y - text_height - padding,
                label_x + text_width + padding,
                label_y + baseline + padding,
            )

            # Check for overlap
            if not check_overlap(bbox, placed_labels):
                # Found a good position, place the label
                # Draw background rectangle for label
                cv2.rectangle(
                    vis_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 0, 0),  # Black background
                    -1,  # Filled
                )

                # Draw label text in white
                cv2.putText(
                    vis_image,
                    label_text,
                    (label_x, label_y),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness,
                    cv2.LINE_AA,
                )

                # Record this label position
                placed_labels.append(bbox)
                label_placed = True
                break

        # If no position found, skip this label (it's too crowded)
        if not label_placed:
            logger.debug(f"Skipping label for segment {segment_id} due to overlap constraints")

    # Create final blended visualization
    alpha = 0.6
    blended = cv2.addWeighted(original_image, 1 - alpha, vis_image, alpha, 0)

    return blended, {"segments": segments_info, "panoptic_seg": panoptic_seg, "pure_vis": vis_image}


def skip_if_not_blackhole_20_or_130_cores(device):
    """
    This function is meant to be run only inside pytest tests.
    """
    if not is_blackhole():
        pytest.skip("This test is intended to run only on Blackhole devices with 20 or 130 cores.")
    compute_grid = device.compute_with_storage_grid_size()
    if not ((compute_grid.x == 5 and compute_grid.y == 4) or (compute_grid.x == 13 and compute_grid.y == 10)):
        pytest.skip(
            f"This test is intended to run only on Blackhole devices with 20 or 130 cores. Core grid [{compute_grid.x},{compute_grid.y}] must be [5, 4] or [13, 10]."
        )
