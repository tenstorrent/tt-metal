# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Tuple
import pytest
import cv2
import torch
from loguru import logger
import ttnn
from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS
from models.experimental.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from models.experimental.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab, create_resnet_dtype_config
from models.experimental.panoptic_deeplab.tt.tt_normalization import TtImageNetNormalization
from models.experimental.panoptic_deeplab.reference.pytorch_model import PytorchPanopticDeepLab
from models.experimental.panoptic_deeplab.tt.common import (
    get_panoptic_deeplab_config,
    PDL_L1_SMALL_SIZE,
)
from models.experimental.panoptic_deeplab.tt.model_configs import ModelOptimisations
from models.common.utility_functions import disable_persistent_kernel_cache
from models.experimental.panoptic_deeplab.demo.demo_utils import (
    preprocess_image,
    create_panoptic_visualization,
    create_deeplab_v3plus_visualization,
    save_predictions,
    preprocess_input_params,
)


def run_panoptic_deeplab_demo(
    device: ttnn.Device,
    image_path: str,
    weights_path: str,
    output_dir: str = "panoptic_deeplab_predictions",
    target_size: Tuple[int, int] = (512, 1024),
    resnet_dtype_config: str = "all_bfloat16",
    center_threshold: float = 0.05,
    use_imagenet_norm: bool = True,
    model_category=PANOPTIC_DEEPLAB,
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
    original_image = cv2.resize(original_image, (int(target_size[1]), int(target_size[0])))

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
            train_size=target_size,
            weights_path=weights_path,
            model_category=model_category,
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

        # Create TTNN parameters
        logger.info("Creating TTNN parameters...")
        ttnn_parameters = create_panoptic_deeplab_parameters(
            pytorch_model, device, input_height=int(target_size[0]), input_width=int(target_size[1]), batch_size=1
        )

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
            model_category=model_category,
        )

    except FileNotFoundError:
        logger.error(f"Weights file not found: {weights_path}")
        logger.error("Please download the Panoptic DeepLab weights and place them at the specified path.")
        return

    # Prepare inputs for both models
    # Convert to TTNN format (values still in [0,1] range)
    ttnn_input = ttnn.from_torch(
        input_tensor.permute(0, 2, 3, 1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    # Initialize ImageNet normalization module once for this inference
    imagenet_normalizer = None

    # Apply ImageNet normalization
    if use_imagenet_norm:
        logger.info("Initializing ImageNet normalization module...")
        imagenet_normalizer = TtImageNetNormalization(device, target_size)

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
    ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic_logits)
    semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
    if semantic_original_channels is not None:
        logger.info(
            f"Slicing semantic output from {ttnn_semantic_torch.shape[1]} to {semantic_original_channels} channels"
        )
        ttnn_semantic_torch = ttnn_semantic_torch[:, :semantic_original_channels, :, :]

    if ttnn_model.model_category == PANOPTIC_DEEPLAB:
        # Handle center output - convert from NHWC to NCHW and slice padding if needed
        ttnn_center_torch = ttnn.to_torch(ttnn_center_logits)
        center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
        if center_original_channels is not None:
            logger.info(
                f"Slicing center output from {ttnn_center_torch.shape[1]} to {center_original_channels} channels"
            )
            ttnn_center_torch = ttnn_center_torch[:, :center_original_channels, :, :]

        # Handle offset output - convert from NHWC to NCHW and slice padding if needed
        ttnn_offset_torch = ttnn.to_torch(ttnn_offset_logits)
        offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()
        if offset_original_channels is not None:
            logger.info(
                f"Slicing offset output from {ttnn_offset_torch.shape[1]} to {offset_original_channels} channels"
            )
            ttnn_offset_torch = ttnn_offset_torch[:, :offset_original_channels, :, :]

    # Convert to numpy in HWC format for visualization
    semantic_np_ttnn = ttnn_semantic_torch.float().squeeze(0).permute(1, 2, 0).numpy()
    center_np_ttnn = (
        ttnn_center_torch.float().squeeze(0).permute(1, 2, 0).numpy()
        if ttnn_model.model_category == PANOPTIC_DEEPLAB
        else None
    )
    offset_np_ttnn = (
        ttnn_offset_torch.float().squeeze(0).permute(1, 2, 0).numpy()
        if ttnn_model.model_category == PANOPTIC_DEEPLAB
        else None
    )

    if ttnn_model.model_category == PANOPTIC_DEEPLAB:
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
    else:
        panoptic_vis_ttnn, panoptic_info_ttnn = create_deeplab_v3plus_visualization(
            semantic_np_ttnn,
            original_image=original_image,
        )
    # Save TTNN results
    image_name = os.path.basename(image_path)
    ttnn_output_dir = os.path.join(output_dir, "ttnn_output")
    save_predictions(ttnn_output_dir, image_name, original_image, panoptic_vis_ttnn)

    # Process PyTorch results
    logger.info("Processing PyTorch results...")
    semantic_np_pytorch = pytorch_semantic_logits.float().squeeze(0).permute(1, 2, 0).numpy()
    center_np_pytorch = (
        pytorch_center_logits.float().squeeze(0).permute(1, 2, 0).numpy() if pytorch_center_logits is not None else None
    )
    offset_np_pytorch = (
        pytorch_offset_logits.float().squeeze(0).permute(1, 2, 0).numpy() if pytorch_offset_logits is not None else None
    )
    if pytorch_model.model_category == PANOPTIC_DEEPLAB:
        panoptic_vis_pytorch, panoptic_info_pytorch = create_panoptic_visualization(
            semantic_np_pytorch, center_np_pytorch, offset_np_pytorch, original_image
        )
    else:
        panoptic_vis_pytorch, panoptic_info_pytorch = create_deeplab_v3plus_visualization(
            semantic_np_pytorch,
            original_image=original_image,
        )

    # Save PyTorch results
    pytorch_output_dir = os.path.join(output_dir, "pytorch_output")
    save_predictions(pytorch_output_dir, image_name, original_image, panoptic_vis_pytorch)

    logger.info(f"Demo completed! Results saved to {output_dir}")
    logger.info("Output includes original and panoptic images for both TTNN and PyTorch models")


@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "output_dir, use_imagenet_norm",
    [
        (
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../demo_outputs")),
            True,
        ),
    ],
)
@pytest.mark.parametrize("model_category", [PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS])
def test_panoptic_deeplab_demo(device, output_dir, model_category, use_imagenet_norm, model_location_generator):
    skip_if_not_blackhole_20_cores(device)
    images, weights_path, output_dir = preprocess_input_params(
        output_dir, model_category, current_dir=__file__, model_location_generator=model_location_generator
    )
    run_panoptic_deeplab_demo(
        device, images[0], weights_path, output_dir, model_category=model_category, use_imagenet_norm=use_imagenet_norm
    )
