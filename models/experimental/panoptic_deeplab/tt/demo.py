# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.reference.pytorch_postprocessing import get_panoptic_segmentation
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


def merge_nearby_instances(panoptic_seg: np.ndarray, max_distance: int = 15) -> np.ndarray:
    """
    Merge nearby instances of the same class that are likely parts of the same object.

    Args:
        panoptic_seg: Panoptic segmentation map [H, W]
        max_distance: Maximum distance for merging instances

    Returns:
        Updated panoptic segmentation map
    """
    from scipy.spatial.distance import cdist

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

        # Find pairs to merge
        merged = set()
        for i in range(len(instance_ids)):
            if i in merged:
                continue

            for j in range(i + 1, len(instance_ids)):
                if j in merged:
                    continue

                if distances[i, j] < max_distance:
                    # Merge instance j into instance i
                    mask_j = result == instance_ids[j]
                    result[mask_j] = instance_ids[i]
                    merged.add(j)

    return result


def expand_instances_to_semantic(
    panoptic_seg: np.ndarray, semantic_classes: np.ndarray, expand_radius: int = 5
) -> np.ndarray:
    """
    Expand existing instances to cover nearby pixels of the same semantic class.
    Uses moderate expansion to avoid over-processing.

    Args:
        panoptic_seg: Panoptic segmentation map [H, W]
        semantic_classes: Semantic segmentation map [H, W]
        expand_radius: Radius for expanding instances

    Returns:
        Updated panoptic segmentation map
    """
    result = panoptic_seg.copy()
    unique_ids = np.unique(panoptic_seg)
    label_divisor = 1000

    # For each instance, try to expand it moderately
    for segment_id in unique_ids:
        if segment_id == 0 or segment_id == 255:  # Skip background and void
            continue

        category_id = segment_id // label_divisor
        if category_id < 11 or category_id > 18:  # Only expand thing classes
            continue

        # Get current instance mask
        instance_mask = panoptic_seg == segment_id
        if np.sum(instance_mask) == 0:
            continue

        # Use moderate expansion - same for all objects to avoid complexity
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_radius * 2 + 1, expand_radius * 2 + 1))
        expanded_mask = cv2.dilate(instance_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Find pixels that are:
        # 1. In the expanded region
        # 2. Same semantic class
        # 3. Currently background (0) or void (255)
        background_mask = (result == 0) | (result == 255)
        same_semantic = semantic_classes == category_id
        expandable_pixels = expanded_mask & same_semantic & background_mask

        # Add these pixels to the instance
        result[expandable_pixels] = segment_id

    return result


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


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (512, 1024)) -> torch.Tensor:
    """
    Preprocess image for Panoptic DeepLab inference.

    Args:
        image_path: Path to input image
        target_size: Target size as (height, width)

    Returns:
        Preprocessed tensor in NCHW format
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image = cv2.resize(image, (target_size[1], target_size[0]))  # cv2 expects (width, height)

    # Convert to tensor and normalize to [0, 1]
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def create_panoptic_visualization(
    semantic_pred: np.ndarray,
    center_pred: np.ndarray,
    offset_pred: np.ndarray,
    original_image: np.ndarray,
    score_threshold: float = 0.05,
    center_threshold: float = 0.05,
    nms_kernel: int = 11,
    top_k: int = 1000,
    stuff_area: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """
    Create panoptic segmentation visualization from model outputs.

    Args:
        semantic_pred: Semantic segmentation predictions [H, W, num_classes]
        center_pred: Center heatmap predictions [H, W, 1]
        offset_pred: Offset predictions [H, W, 2]
        original_image: Original input image [H, W, 3]
        score_threshold: Minimum score for instance detection
        center_threshold: Minimum center score for instance detection
        nms_kernel: Kernel size for non-maximum suppression
        top_k: Maximum number of instances to detect
        stuff_area: Minimum area for stuff classes

    Returns:
        Tuple of (visualization_image, panoptic_info)
    """
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
    # Add debug logging to understand what's happening
    logger.info(
        f"DEBUG: Using thresholds - center: {center_threshold}, score: {score_threshold}, stuff_area: {stuff_area}"
    )
    logger.info(f"DEBUG: Max center prediction: {center_tensor.max().item():.6f}")
    logger.info(f"DEBUG: Number of pixels above center threshold: {(center_tensor > center_threshold).sum().item()}")

    # Additional debug: show distribution of center values
    center_flat = center_tensor.flatten()
    center_sorted = torch.sort(center_flat, descending=True)[0]
    logger.info(f"DEBUG: Top 10 center values: {center_sorted[:10].tolist()}")
    logger.info(f"DEBUG: Center values > 0.01: {(center_tensor > 0.01).sum().item()}")
    logger.info(f"DEBUG: Center values > 0.05: {(center_tensor > 0.05).sum().item()}")

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

    logger.info(f"DEBUG: get_panoptic_segmentation returned {len(np.unique(panoptic_seg))} unique segments")

    # Create visualization
    panoptic_seg = panoptic_seg.squeeze(0).numpy()  # Remove batch dimension

    # Calculate semantic classes early for use in post-processing
    semantic_classes = np.argmax(semantic_pred, axis=2)

    # Skip post-processing to avoid artifacts - use raw panoptic segmentation
    # panoptic_seg = merge_nearby_instances(panoptic_seg, max_distance=15)
    # panoptic_seg = expand_instances_to_semantic(panoptic_seg, semantic_classes, expand_radius=6)

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

    # Apply panoptic segmentation with clean, stable colors - NO color variation
    for segment in segments_info:
        segment_id = segment["id"]
        category_id = segment["category_id"]
        mask = panoptic_seg == segment_id

        # Get exact base color for this category - no modifications
        if category_id < len(CITYSCAPES_CATEGORIES):
            color = CITYSCAPES_CATEGORIES[category_id]["color"]
        else:
            # Generate stable color for unknown categories
            np.random.seed(segment_id)
            color = np.random.randint(128, 255, 3).tolist()

        # Apply color directly without any modifications
        vis_image[mask] = color

    # Blend with original image for better visualization
    alpha = 0.6
    blended = cv2.addWeighted(original_image, 1 - alpha, vis_image, alpha, 0)

    return blended, {"segments": segments_info, "panoptic_seg": panoptic_seg}


def save_predictions(
    output_dir: str,
    image_name: str,
    original_image: np.ndarray,
    semantic_pred: np.ndarray,
    panoptic_vis: np.ndarray,
    panoptic_info: Dict,
):
    """Save prediction results to files."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]

    # Save original image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

    # Save semantic segmentation
    semantic_colored = np.zeros((semantic_pred.shape[0], semantic_pred.shape[1], 3), dtype=np.uint8)
    for i, category in enumerate(CITYSCAPES_CATEGORIES):
        if i < semantic_pred.shape[2]:
            mask = np.argmax(semantic_pred, axis=2) == i
            semantic_colored[mask] = category["color"]

    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_semantic.jpg"), cv2.cvtColor(semantic_colored, cv2.COLOR_RGB2BGR)
    )

    # Save panoptic segmentation
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_panoptic.jpg"), cv2.cvtColor(panoptic_vis, cv2.COLOR_RGB2BGR))

    # Save segmentation info
    with open(os.path.join(output_dir, f"{base_name}_info.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_info = {
            "segments": panoptic_info["segments"],
            "num_segments": len(panoptic_info["segments"]),
        }
        json.dump(serializable_info, f, indent=2)

    logger.info(f"Saved predictions for {image_name} to {output_dir}")


def run_panoptic_deeplab_demo(
    device: ttnn.Device,
    image_path: str,
    weights_path: str,
    output_dir: str = "panoptic_deeplab_predictions",
    target_size: Tuple[int, int] = (512, 1024),
):
    """
    Run Panoptic DeepLab inference on a single image.

    Args:
        device: TTNN device
        image_path: Path to input image
        weights_path: Path to model weights (.pkl file)
        output_dir: Directory to save outputs
        target_size: Input size as (height, width)
    """
    disable_persistent_kernel_cache()

    logger.info(f"Running Panoptic DeepLab demo on {image_path}")
    logger.info(f"Target size: {target_size}")

    # Model configuration
    batch_size = 1
    num_classes = 19
    project_channels = [32, 64]
    decoder_channels = [256, 256, 256]
    sem_seg_head_channels = 256
    ins_embed_head_channels = 32
    common_stride = 4

    # Preprocess image
    logger.info("Preprocessing image...")
    input_tensor = preprocess_image(image_path, target_size)

    # Load original image for visualization
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (target_size[1], target_size[0]))

    # Convert input to TTNN format (NHWC)
    ttnn_input_torch = input_tensor.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(ttnn_input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    try:
        # Load PyTorch model with weights
        logger.info("Loading PyTorch model...")
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            norm="SyncBN",
            train_size=target_size,
            weights_path=weights_path,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        # Create TTNN parameters
        logger.info("Creating TTNN parameters...")
        ttnn_parameters = create_panoptic_deeplab_parameters(pytorch_model, device)

        # Apply Conv+BatchNorm fusion
        logger.info("Applying Conv+BatchNorm fusion...")
        fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)

        # Create TTNN model
        logger.info("Creating TTNN model...")
        ttnn_model = TtPanopticDeepLab(
            device=device,
            parameters=fused_parameters,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            norm="",
            train_size=target_size,
        )

    except FileNotFoundError:
        logger.error(f"Weights file not found: {weights_path}")
        logger.error("Please download the Panoptic DeepLab weights and place them at the specified path.")
        return

    # Priprema ulaza
    pytorch_input = preprocess_image(image_path, target_size).to(dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        pytorch_input.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    logger.info("Running PyTorch inference...")
    with torch.no_grad():
        pytorch_semantic_logits, pytorch_center_logits, pytorch_offset_logits, _ = pytorch_model.forward(pytorch_input)

    # Run inference
    logger.info("Running TTNN inference...")
    ttnn_semantic_logits, ttnn_center_logits, ttnn_offset_logits, _ = ttnn_model.forward(ttnn_input)

    # a) Obrada i čuvanje TTNN rezultata
    logger.info("--- Processing TTNN Results ---")
    # Konverzija TTNN izlaza u numpy
    semantic_np_ttnn = ttnn.to_torch(ttnn_semantic_logits).float().squeeze(0).numpy()  # (1,H,W,C) -> (H,W,C)
    center_np_ttnn = ttnn.to_torch(ttnn_center_logits).float().squeeze(0).numpy()  # (1,H,W,1) -> (H,W,1)
    offset_np_ttnn = ttnn.to_torch(ttnn_offset_logits).float().squeeze(0).numpy()  # (1,H,W,2) -> (H,W,2)

    logger.debug(
        f"TTNN numpy shapes -> semantic: {semantic_np_ttnn.shape}, center: {center_np_ttnn.shape}, offset: {offset_np_ttnn.shape}"
    )

    # Kreiranje vizualizacije
    panoptic_vis_ttnn, panoptic_info_ttnn = create_panoptic_visualization(
        semantic_np_ttnn, center_np_ttnn, offset_np_ttnn, original_image
    )

    # Čuvanje slika
    image_name = os.path.basename(image_path)
    ttnn_output_dir = os.path.join(output_dir, "ttnn_output")
    save_predictions(
        ttnn_output_dir, image_name, original_image, semantic_np_ttnn, panoptic_vis_ttnn, panoptic_info_ttnn
    )

    # b) Obrada i čuvanje PyTorch rezultata
    logger.info("--- Processing PyTorch Results ---")
    # Konverzija PyTorch izlaza (NCHW) u numpy (HWC)
    semantic_np_pytorch = (
        pytorch_semantic_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    )  # (1,C,H,W) -> (C,H,W) -> (H,W,C)
    center_np_pytorch = pytorch_center_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    offset_np_pytorch = (
        pytorch_offset_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    )  # (1,2,H,W) -> (2,H,W) -> (H,W,2)

    logger.debug(
        f"PyTorch numpy shapes -> semantic: {semantic_np_pytorch.shape}, center: {center_np_pytorch.shape}, offset: {offset_np_pytorch.shape}"
    )

    # Kreiranje vizualizacije
    panoptic_vis_pytorch, panoptic_info_pytorch = create_panoptic_visualization(
        semantic_np_pytorch, center_np_pytorch, offset_np_pytorch, original_image
    )

    # Čuvanje slika u odvojenom direktorijumu
    pytorch_output_dir = os.path.join(output_dir, "pytorch_output")
    save_predictions(
        pytorch_output_dir, image_name, original_image, semantic_np_pytorch, panoptic_vis_pytorch, panoptic_info_pytorch
    )

    logger.info(f"Demo completed! Results for both models saved to {output_dir}")


def run_panoptic_deeplab_batch_demo(
    device: ttnn.Device,
    input_dir: str,
    weights_path: str,
    output_dir: str = "panoptic_deeplab_predictions",
    target_size: Tuple[int, int] = (512, 1024),
    max_images: int = 10,
):
    """
    Run Panoptic DeepLab inference on multiple images from a directory.

    Args:
        device: TTNN device
        input_dir: Directory containing input images
        weights_path: Path to model weights (.pkl file)
        output_dir: Directory to save outputs
        target_size: Input size as (height, width)
        max_images: Maximum number of images to process
    """
    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in image_extensions]

    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return

    # Limit number of images
    image_files = image_files[:max_images]
    logger.info(f"Processing {len(image_files)} images from {input_dir}")

    # Process each image
    for i, image_file in enumerate(image_files):
        logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file}")
        image_path = os.path.join(input_dir, image_file)

        try:
            run_panoptic_deeplab_demo(device, image_path, weights_path, output_dir, target_size)
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            continue

    logger.info("Batch processing completed!")


# Test functions for pytest
@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize(
    "image_path, target_size",
    [
        ("models/experimental/panoptic_deeplab/resources/sample_image.jpg", (512, 1024)),
    ],
)
def test_panoptic_deeplab_demo(device, image_path, target_size):
    """Test Panoptic DeepLab demo with a single image."""
    weights_path = "models/experimental/panoptic_deeplab/weights/model_final_bd324a.pkl"

    run_panoptic_deeplab_demo(
        device=device,
        image_path=image_path,
        weights_path=weights_path,
        target_size=target_size,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize(
    "input_dir, target_size, max_images",
    [
        ("models/experimental/panoptic_deeplab/resources", (512, 1024), 5),
    ],
)
def test_panoptic_deeplab_batch_demo(device, input_dir, target_size, max_images):
    """Test Panoptic DeepLab demo with multiple images."""
    weights_path = "models/experimental/panoptic_deeplab/weights/model_final_bd324a.pkl"

    run_panoptic_deeplab_batch_demo(
        device=device,
        input_dir=input_dir,
        weights_path=weights_path,
        target_size=target_size,
        max_images=max_images,
    )


if __name__ == "__main__":
    # Example usage for standalone execution
    import sys

    if len(sys.argv) < 3:
        print("Usage: python demo.py <image_path> <weights_path> [output_dir]")
        print("   or: python demo.py <input_dir> <weights_path> [output_dir] --batch")
        sys.exit(1)

    device = ttnn.open_device(device_id=0, l1_small_size=65536)

    try:
        input_path = sys.argv[1]
        weights_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "panoptic_deeplab_predictions"
        batch_mode = "--batch" in sys.argv

        if batch_mode:
            run_panoptic_deeplab_batch_demo(device, input_path, weights_path, output_dir)
        else:
            run_panoptic_deeplab_demo(device, input_path, weights_path, output_dir)

    finally:
        ttnn.close_device(device)
