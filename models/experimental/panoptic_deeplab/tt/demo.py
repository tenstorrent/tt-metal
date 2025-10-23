# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import argparse
from loguru import logger
from scipy.spatial.distance import cdist
import ttnn
from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab, create_resnet_dtype_config
from models.experimental.panoptic_deeplab.tt.tt_normalization import TtImageNetNormalization
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.common import get_panoptic_deeplab_config, PDL_L1_SMALL_SIZE
from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations
from models.experimental.panoptic_deeplab.reference.pytorch_postprocessing import get_panoptic_segmentation
from models.common.utility_functions import disable_persistent_kernel_cache


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


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (512, 1024), use_imagenet_norm: bool = True
) -> torch.Tensor:
    """
    Preprocess image for Panoptic DeepLab inference.

    Args:
        image_path: Path to input image
        target_size: Target size as (height, width)
        use_imagenet_norm: Whether to apply ImageNet normalization (will be done on device if True)

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

    # Enable post-processing for TTNN to merge fragmented instances - use raw panoptic segmentation for PyTorch
    # Check if this is TTNN output (has more fragmentation) vs PyTorch (cleaner)
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

    # Create colored visualization with clean, stable colors

    # Apply panoptic segmentation colors
    # Sort segments by area (largest first) to ensure proper layering
    segments_sorted = sorted(segments_info, key=lambda x: x["area"], reverse=True)
    for segment in segments_sorted:
        segment_id = segment["id"]
        category_id = segment["category_id"]
        mask = panoptic_seg == segment_id

        # Get color for this category
        if category_id < len(CITYSCAPES_CATEGORIES):
            color = CITYSCAPES_CATEGORIES[category_id]["color"]
        else:
            # Generate stable color for unknown categories
            np.random.seed(segment_id)
            color = np.random.randint(128, 255, 3).tolist()

        # Apply color
        vis_image[mask] = color

    # Create final blended visualization
    alpha = 0.6
    blended = cv2.addWeighted(original_image, 1 - alpha, vis_image, alpha, 0)

    return blended, {"segments": segments_info, "panoptic_seg": panoptic_seg, "pure_vis": vis_image}


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


def run_panoptic_deeplab_demo(
    device: ttnn.Device,
    image_path: str,
    weights_path: str,
    output_dir: str = "panoptic_deeplab_predictions",
    target_size: Tuple[int, int] = (512, 1024),
    resnet_dtype_config: str = None,
    center_threshold: float = 0.05,
    use_imagenet_norm: bool = True,
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

    # Get model configuration
    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    project_channels = config["project_channels"]
    decoder_channels = config["decoder_channels"]
    sem_seg_head_channels = config["sem_seg_head_channels"]
    ins_embed_head_channels = config["ins_embed_head_channels"]
    common_stride = config["common_stride"]

    # Preprocess image
    logger.info("Preprocessing image...")
    input_tensor = preprocess_image(image_path, target_size, use_imagenet_norm)

    # Load original image for visualization
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (target_size[1], target_size[0]))

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

        # Create model configurations
        logger.info("Creating model configurations...")
        model_configs = ModelOptimisations()
        model_configs.setup_all_layer_overrides()

        # Create TTNN model
        logger.info(f"Creating TTNN model with ResNet dtype config: {resnet_dtype_config}")
        layer_dtypes = create_resnet_dtype_config(resnet_dtype_config)
        logger.info(f"ResNet layer dtypes: {layer_dtypes}")

        ttnn_model = TtPanopticDeepLab(
            device=device,
            parameters=fused_parameters,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            train_size=target_size,
            model_configs=model_configs,
            resnet_layer_dtypes=layer_dtypes,
        )

    except FileNotFoundError:
        logger.error(f"Weights file not found: {weights_path}")
        logger.error("Please download the Panoptic DeepLab weights and place them at the specified path.")
        return

    # Initialize ImageNet normalization module once for this inference
    imagenet_normalizer = None
    if use_imagenet_norm:
        logger.info("Initializing ImageNet normalization module...")
        imagenet_normalizer = TtImageNetNormalization(device, target_size)

    # Prepare inputs for both models
    # Convert to TTNN format (values still in [0,1] range)
    ttnn_input = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Apply ImageNet normalization
    if use_imagenet_norm:
        logger.info("Applying ImageNet normalization...")

        # Apply normalization on device for TTNN using the efficient module
        ttnn_input = imagenet_normalizer.forward(ttnn_input)

        # For PyTorch: Apply normalization on CPU
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        pytorch_input = (input_tensor - mean) / std
        pytorch_input = pytorch_input.to(dtype=torch.bfloat16)
    else:
        # No normalization case - just convert PyTorch input to bfloat16
        pytorch_input = input_tensor.to(dtype=torch.bfloat16)

    logger.info("Running PyTorch inference...")
    with torch.no_grad():
        pytorch_semantic_logits, pytorch_center_logits, pytorch_offset_logits, _ = pytorch_model.forward(pytorch_input)

    # Run inference
    logger.info("Running TTNN inference...")
    ttnn_semantic_logits, ttnn_center_logits, ttnn_offset_logits, _ = ttnn_model.forward(ttnn_input)

    # Process TTNN results
    logger.info("Processing TTNN results...")

    # Handle semantic output - convert from NHWC to NCHW and slice padding if needed
    ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic_logits).permute(0, 3, 1, 2)
    semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
    if semantic_original_channels is not None:
        logger.info(
            f"Slicing semantic output from {ttnn_semantic_torch.shape[1]} to {semantic_original_channels} channels"
        )
        ttnn_semantic_torch = ttnn_semantic_torch[:, :semantic_original_channels, :, :]

    # Handle center output - convert from NHWC to NCHW and slice padding if needed
    ttnn_center_torch = ttnn.to_torch(ttnn_center_logits).permute(0, 3, 1, 2)
    center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
    if center_original_channels is not None:
        logger.info(f"Slicing center output from {ttnn_center_torch.shape[1]} to {center_original_channels} channels")
        ttnn_center_torch = ttnn_center_torch[:, :center_original_channels, :, :]

    # Handle offset output - convert from NHWC to NCHW and slice padding if needed
    ttnn_offset_torch = ttnn.to_torch(ttnn_offset_logits).permute(0, 3, 1, 2)
    offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()
    if offset_original_channels is not None:
        logger.info(f"Slicing offset output from {ttnn_offset_torch.shape[1]} to {offset_original_channels} channels")
        ttnn_offset_torch = ttnn_offset_torch[:, :offset_original_channels, :, :]

    # Convert to numpy in HWC format for visualization
    semantic_np_ttnn = ttnn_semantic_torch.float().squeeze(0).permute(1, 2, 0).numpy()
    center_np_ttnn = ttnn_center_torch.float().squeeze(0).permute(1, 2, 0).numpy()
    offset_np_ttnn = ttnn_offset_torch.float().squeeze(0).permute(1, 2, 0).numpy()

    panoptic_vis_ttnn, panoptic_info_ttnn = create_panoptic_visualization(
        semantic_np_ttnn,
        center_np_ttnn,
        offset_np_ttnn,
        original_image,
        center_threshold=center_threshold,  # Use parameter
        score_threshold=center_threshold,  # Use same value for consistency
        stuff_area=1,  # Match PyTorch defaults
        top_k=1000,  # Match PyTorch defaults
        nms_kernel=11,  # Match PyTorch defaults
    )

    # Save TTNN results
    image_name = os.path.basename(image_path)
    ttnn_output_dir = os.path.join(output_dir, "ttnn_output")
    save_predictions(ttnn_output_dir, image_name, original_image, panoptic_vis_ttnn)

    # Process PyTorch results
    logger.info("Processing PyTorch results...")
    semantic_np_pytorch = pytorch_semantic_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    center_np_pytorch = pytorch_center_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    offset_np_pytorch = pytorch_offset_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    panoptic_vis_pytorch, panoptic_info_pytorch = create_panoptic_visualization(
        semantic_np_pytorch, center_np_pytorch, offset_np_pytorch, original_image
    )

    # Save PyTorch results
    pytorch_output_dir = os.path.join(output_dir, "pytorch_output")
    save_predictions(pytorch_output_dir, image_name, original_image, panoptic_vis_pytorch)

    logger.info(f"Demo completed! Results saved to {output_dir}")
    logger.info("Output includes original and panoptic images for both TTNN and PyTorch models")


def run_panoptic_deeplab_batch_demo(
    device: ttnn.Device,
    input_dir: str,
    weights_path: str,
    output_dir: str = "panoptic_deeplab_predictions",
    target_size: Tuple[int, int] = (512, 1024),
    max_images: int = 10,
    resnet_dtype_config: str = None,
    center_threshold: float = 0.05,
    use_imagenet_norm: bool = True,
):
    """
    Run Panoptic DeepLab inference on multiple images from a directory.
    Optimized for batch processing - loads models once and reuses them.

    Args:
        device: TTNN device
        input_dir: Directory containing input images
        weights_path: Path to model weights (.pkl file)
        output_dir: Directory to save outputs
        target_size: Input size as (height, width)
        max_images: Maximum number of images to process
    """
    disable_persistent_kernel_cache()

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in image_extensions]

    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        return

    # Limit number of images
    image_files = image_files[:max_images]
    logger.info(f"Processing {len(image_files)} images from {input_dir}")

    # Get model configuration (do this once)
    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    project_channels = config["project_channels"]
    decoder_channels = config["decoder_channels"]
    sem_seg_head_channels = config["sem_seg_head_channels"]
    ins_embed_head_channels = config["ins_embed_head_channels"]
    common_stride = config["common_stride"]

    try:
        # Load PyTorch model with weights (do this once)
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

        # Create TTNN parameters (do this once)
        logger.info("Creating TTNN parameters...")
        ttnn_parameters = create_panoptic_deeplab_parameters(pytorch_model, device)

        # Apply Conv+BatchNorm fusion (do this once)
        logger.info("Applying Conv+BatchNorm fusion...")
        fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)

        # Create model configurations (do this once)
        logger.info("Creating model configurations...")
        model_configs = ModelOptimisations()
        model_configs.setup_all_layer_overrides()

        # Create TTNN model (do this once)
        logger.info(f"Creating TTNN model with ResNet dtype config: {resnet_dtype_config}")
        layer_dtypes = create_resnet_dtype_config(resnet_dtype_config)
        logger.info(f"ResNet layer dtypes: {layer_dtypes}")

        ttnn_model = TtPanopticDeepLab(
            device=device,
            parameters=fused_parameters,
            num_classes=num_classes,
            common_stride=common_stride,
            project_channels=project_channels,
            decoder_channels=decoder_channels,
            sem_seg_head_channels=sem_seg_head_channels,
            ins_embed_head_channels=ins_embed_head_channels,
            train_size=target_size,
            model_configs=model_configs,
            resnet_layer_dtypes=layer_dtypes,
        )

        # Initialize ImageNet normalization module once for batch processing
        imagenet_normalizer = None
        if use_imagenet_norm:
            logger.info("Initializing ImageNet normalization module...")
            imagenet_normalizer = TtImageNetNormalization(device, target_size)

    except FileNotFoundError:
        logger.error(f"Weights file not found: {weights_path}")
        logger.error("Please download the Panoptic DeepLab weights and place them at the specified path.")
        return

    # Process each image with pre-loaded models
    for i, image_file in enumerate(image_files):
        logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file}")
        image_path = os.path.join(input_dir, image_file)

        try:
            # Preprocess image
            logger.info("Preprocessing image...")
            input_tensor = preprocess_image(image_path, target_size, use_imagenet_norm)

            # Load original image for visualization
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = cv2.resize(original_image, (target_size[1], target_size[0]))

            # Prepare inputs for both models
            # Convert to TTNN format (values still in [0,1] range)
            ttnn_input = ttnn.from_torch(
                input_tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

            # Apply ImageNet normalization
            if use_imagenet_norm:
                logger.info("Applying ImageNet normalization...")

                # Apply normalization on device for TTNN using the pre-initialized module
                ttnn_input = imagenet_normalizer.forward(ttnn_input)

                # For PyTorch: Apply normalization on CPU
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                pytorch_input = (input_tensor - mean) / std
                pytorch_input = pytorch_input.to(dtype=torch.bfloat16)
            else:
                # No normalization case - just convert PyTorch input to bfloat16
                pytorch_input = input_tensor.to(dtype=torch.bfloat16)

            logger.info("Running PyTorch inference...")
            with torch.no_grad():
                pytorch_semantic_logits, pytorch_center_logits, pytorch_offset_logits, _ = pytorch_model.forward(
                    pytorch_input
                )

            # Run inference
            logger.info("Running TTNN inference...")
            ttnn_semantic_logits, ttnn_center_logits, ttnn_offset_logits, _ = ttnn_model.forward(ttnn_input)

            # Process TTNN results
            logger.info("Processing TTNN results...")

            # Handle semantic output - convert from NHWC to NCHW and slice padding if needed
            ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic_logits).permute(0, 3, 1, 2)
            semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
            if semantic_original_channels is not None:
                logger.info(
                    f"Slicing semantic output from {ttnn_semantic_torch.shape[1]} to {semantic_original_channels} channels"
                )
                ttnn_semantic_torch = ttnn_semantic_torch[:, :semantic_original_channels, :, :]

            # Handle center output - convert from NHWC to NCHW and slice padding if needed
            ttnn_center_torch = ttnn.to_torch(ttnn_center_logits).permute(0, 3, 1, 2)
            center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
            if center_original_channels is not None:
                logger.info(
                    f"Slicing center output from {ttnn_center_torch.shape[1]} to {center_original_channels} channels"
                )
                ttnn_center_torch = ttnn_center_torch[:, :center_original_channels, :, :]

            # Handle offset output - convert from NHWC to NCHW and slice padding if needed
            ttnn_offset_torch = ttnn.to_torch(ttnn_offset_logits).permute(0, 3, 1, 2)
            offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()
            if offset_original_channels is not None:
                logger.info(
                    f"Slicing offset output from {ttnn_offset_torch.shape[1]} to {offset_original_channels} channels"
                )
                ttnn_offset_torch = ttnn_offset_torch[:, :offset_original_channels, :, :]

            # Convert to numpy in HWC format for visualization
            semantic_np_ttnn = ttnn_semantic_torch.float().squeeze(0).permute(1, 2, 0).numpy()
            center_np_ttnn = ttnn_center_torch.float().squeeze(0).permute(1, 2, 0).numpy()
            offset_np_ttnn = ttnn_offset_torch.float().squeeze(0).permute(1, 2, 0).numpy()

            panoptic_vis_ttnn, panoptic_info_ttnn = create_panoptic_visualization(
                semantic_np_ttnn,
                center_np_ttnn,
                offset_np_ttnn,
                original_image,
                center_threshold=center_threshold,
                score_threshold=center_threshold,
                stuff_area=1,
                top_k=1000,
                nms_kernel=11,
            )

            # Save TTNN results
            image_name = os.path.basename(image_path)
            ttnn_output_dir = os.path.join(output_dir, "ttnn_output")
            save_predictions(ttnn_output_dir, image_name, original_image, panoptic_vis_ttnn)

            # Process PyTorch results
            logger.info("Processing PyTorch results...")
            semantic_np_pytorch = pytorch_semantic_logits.float().squeeze(0).permute(1, 2, 0).numpy()
            center_np_pytorch = pytorch_center_logits.float().squeeze(0).permute(1, 2, 0).numpy()
            offset_np_pytorch = pytorch_offset_logits.float().squeeze(0).permute(1, 2, 0).numpy()
            panoptic_vis_pytorch, panoptic_info_pytorch = create_panoptic_visualization(
                semantic_np_pytorch, center_np_pytorch, offset_np_pytorch, original_image
            )

            # Save PyTorch results
            pytorch_output_dir = os.path.join(output_dir, "pytorch_output")
            save_predictions(pytorch_output_dir, image_name, original_image, panoptic_vis_pytorch)

        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            continue

    logger.info("Batch processing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Panoptic DeepLab inference on images using TTNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Single image with custom output directory
            python models/experimental/panoptic_deeplab/tt/demo.py \
            <image_path> \
            <weight_path> \
            <output_dir>

            # Process all images in a directory
            python models/experimental/panoptic_deeplab/tt/demo.py \
            <image_directory> \
            <weight_path> \
            <output_dir> \
            --batch
            """,
    )

    parser.add_argument("input_path", help="Path to input image or directory (for batch mode)")

    parser.add_argument("weights_path", help="Path to model weights (.pkl file)")

    parser.add_argument("output_dir", help="Directory to save prediction outputs")

    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode for directory of images")

    parser.add_argument(
        "--resnet-dtype-config",
        default="all_bfloat16",
        choices=[
            "all_bfloat16",
            "all_bfloat8",
            "mixed_early_bf16",
            "mixed_late_bf16",
            "stem_only_bf16",
            "res5_only_bf16",
            "res4_only_bf16",
        ],
        help="ResNet layer dtype configuration. Options: all_bfloat16 (highest accuracy), all_bfloat8 (fastest), mixed_early_bf16, mixed_late_bf16, stem_only_bf16, res5_only_bf16, res4_only_bf16",
    )

    parser.add_argument(
        "--center-threshold",
        type=float,
        default=0.05,
        help="Center detection threshold (default: 0.05). Lower values detect more instances but may increase false positives.",
    )

    parser.add_argument(
        "--no-imagenet-norm",
        action="store_true",
        help="Disable ImageNet normalization (use simple [0,1] normalization instead)",
    )

    args = parser.parse_args()

    device = ttnn.open_device(device_id=0, l1_small_size=PDL_L1_SMALL_SIZE)

    try:
        if args.batch:
            run_panoptic_deeplab_batch_demo(
                device,
                args.input_path,
                args.weights_path,
                args.output_dir,
                resnet_dtype_config=args.resnet_dtype_config,
                center_threshold=args.center_threshold,
                use_imagenet_norm=not args.no_imagenet_norm,
            )
        else:
            run_panoptic_deeplab_demo(
                device,
                args.input_path,
                args.weights_path,
                args.output_dir,
                resnet_dtype_config=args.resnet_dtype_config,
                center_threshold=args.center_threshold,
                use_imagenet_norm=not args.no_imagenet_norm,
            )

    finally:
        ttnn.close_device(device)
