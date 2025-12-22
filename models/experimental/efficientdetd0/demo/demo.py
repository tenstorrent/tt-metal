# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path

import cv2
import torch
import ttnn
from loguru import logger

from models.demos.utils.common_demo_utils import get_mesh_mappers
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc
from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.efficientdetd0 import TtEfficientDetBackbone
from models.experimental.efficientdetd0.common import load_torch_model_state
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    infer_torch_module_args,
    create_custom_mesh_preprocessor,
)
from models.experimental.efficientdetd0.reference.utils import (
    Anchors,
    ClipBoxes,
    BBoxTransform,
)
from models.experimental.efficientdetd0.demo.demo_utils import (
    COCO_CLASSES,
    postprocess,
    invert_affine,
    preprocess_image,
    draw_bounding_boxes,
)


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
    use_torch_maxpool=False,
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

    logger.info("=" * 80)
    logger.info("EfficientDet Demo")
    logger.info("=" * 80)

    # Initialize PyTorch model
    torch_model = EfficientDetBackbone(
        num_classes=num_classes,
        compound_coef=0,
    ).eval()
    load_torch_model_state(torch_model, model_location_generator=model_location_generator, weights_path=weights_path)

    # Load and preprocess image
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
        ori_img, torch_inputs, framed_meta = preprocess_image(image_path, max_size=max(height, width))

    # Run PyTorch forward pass for reference
    with torch.no_grad():
        torch_features, torch_regression, torch_classification, torch_anchors = torch_model(torch_inputs)

    # Preprocess model parameters for TTNN
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    module_args = infer_torch_module_args(model=torch_model, input=torch_inputs)

    # Create TTNN model
    ttnn_model = TtEfficientDetBackbone(
        device=device,
        parameters=parameters,
        module_args=module_args,
        num_classes=num_classes,
        compound_coef=0,
        use_torch_maxpool=use_torch_maxpool,
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
    ttnn_features, ttnn_regression, ttnn_classification = ttnn_model(ttnn_input_tensor)

    # Convert outputs back to PyTorch for comparison
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

    # PCC Comparisons
    logger.info("=" * 80)
    logger.info("PCC (Pearson Correlation Coefficient) Comparisons")
    logger.info("=" * 80)
    PCC_THRESHOLD = 0.97

    # Compare BiFPN feature maps
    for i, (torch_feat, ttnn_feat) in enumerate(zip(torch_features, ttnn_features_torch)):
        passing, pcc_value = comp_pcc(torch_feat, ttnn_feat, pcc=PCC_THRESHOLD)
        logger.info(f"Feature P{i+3}: PCC = {pcc_value:.6f} {'✓' if passing else '✗'}")

    # Compare Regression and Classification outputs
    passing, pcc_value = comp_pcc(torch_regression, ttnn_regression_torch, pcc=PCC_THRESHOLD)
    logger.info(f"Regression: PCC = {pcc_value:.6f} {'✓' if passing else '✗'}")
    passing, pcc_value = comp_pcc(torch_classification, ttnn_classification_torch, pcc=PCC_THRESHOLD)
    logger.info(f"Classification: PCC = {pcc_value:.6f} {'✓' if passing else '✗'}")

    # Post-process to get bounding boxes
    logger.info("=" * 80)
    logger.info("Post-processing Results")
    logger.info("=" * 80)

    # Initialize post-processing modules
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Post-process torch outputs
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

    # Generating anchors for TTNN outputs
    gen_anchors = Anchors(
        anchor_scale=4.0,
        pyramid_levels=(torch.arange(5) + 3).tolist(),
    )
    anchors = gen_anchors(torch_inputs, torch_inputs.dtype)

    # Post-process TTNN outputs
    ttnn_preds = postprocess(
        ttnn_regression_torch,
        ttnn_classification_torch,
        anchors,
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

    # Display detection summary
    torch_detections = len(torch_preds[0]["rois"]) if len(torch_preds) > 0 and len(torch_preds[0]["rois"]) > 0 else 0
    ttnn_detections = len(ttnn_preds[0]["rois"]) if len(ttnn_preds) > 0 and len(ttnn_preds[0]["rois"]) > 0 else 0
    logger.info(f"\nDetections: PyTorch={torch_detections}, TTNN={ttnn_detections}")

    # Visualize bounding boxes on image if we have an original image
    if ori_img is not None:
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

        logger.info(f"Visualizations saved: {torch_output_path}, {ttnn_output_path}")

    # Cleanup
    ttnn.deallocate(ttnn_input_tensor)
    for feat in ttnn_features:
        ttnn.deallocate(feat)
    ttnn.deallocate(ttnn_regression)
    ttnn.deallocate(ttnn_classification)


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
        default=16384,
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
        "--use_torch_maxpool",
        type=bool,
        default=True,
        help="Run MaxPool in torch (default: True)",
    )

    args = parser.parse_args()

    # Run demo
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
            use_torch_maxpool=args.use_torch_maxpool,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
