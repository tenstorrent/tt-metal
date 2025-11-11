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
import torch
import ttnn
from loguru import logger

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


torch.manual_seed(0)


def run_efficient_det_demo(
    device,
    model_location_generator=None,
    batch_size=1,
    height=512,
    width=512,
    num_classes=90,
    compound_coef=0,
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
        compound_coef: EfficientDet compound coefficient (0-8)
    """
    disable_persistent_kernel_cache()

    logger.info("=" * 80)
    logger.info("EfficientDet Demo")
    logger.info("=" * 80)
    logger.info(f"Input shape: ({batch_size}, 3, {height}, {width})")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Compound coefficient: {compound_coef}")

    # Initialize PyTorch model
    logger.info("\n[1/5] Initializing PyTorch model...")
    torch_model = EfficientDetBackbone(
        num_classes=num_classes,
        compound_coef=compound_coef,
        load_weights=False,
    ).eval()
    load_torch_model_state(torch_model, model_location_generator=model_location_generator)

    # Create random input tensor
    logger.info("\n[2/5] Creating input tensor...")
    torch_inputs = torch.randn(batch_size, 3, height, width)
    logger.info(f"Input tensor shape: {torch_inputs.shape}")

    # Run PyTorch forward pass for reference
    logger.info("\n[3/5] Running PyTorch reference forward pass...")
    with torch.no_grad():
        torch_features, torch_regression, torch_classification = torch_model(torch_inputs)
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
        compound_coef=compound_coef,
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
        "--compound-coef",
        type=int,
        default=0,
        choices=range(0, 9),
        help="EfficientDet compound coefficient 0-8 (default: 0)",
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

    args = parser.parse_args()

    # Open device
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=args.l1_small_size)

    try:
        run_efficient_det_demo(
            device=device,
            model_location_generator=None,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            num_classes=args.num_classes,
            compound_coef=args.compound_coef,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
