# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Demo script for Attention DenseUNet inference.

This demo shows how to run the Attention DenseUNet model on sample images
and visualize the segmentation results.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.attention_denseunet.reference.model import create_attention_denseunet
from models.demos.attention_denseunet.tt.common import (
    ATTENTION_DENSEUNET_L1_SMALL_SIZE,
    ATTENTION_DENSEUNET_TRACE_SIZE,
    create_preprocessor,
)
from models.demos.attention_denseunet.tt.config import OptimizationLevel, create_configs_from_parameters
from models.demos.attention_denseunet.tt.model import create_model_from_configs

DEFAULT_RESOLUTION = (256, 256)
INPUT_CHANNELS = 3
DEMO_DIR = "models/demos/attention_denseunet/demo"
DEMO_IMAGE_DIR = os.path.join(DEMO_DIR, "images")
PRED_DIR = os.path.join(DEMO_DIR, "pred")


def create_sample_input(batch_size: int = 1, resolution: tuple = DEFAULT_RESOLUTION):
    """
    Create a sample input tensor for demo purposes.
    Args:
        batch_size: Number of images in batch
        resolution: (height, width) of input images

    Returns:
        PyTorch tensor of shape (batch, channels, height, width)
    """
    height, width = resolution
    sample_input = torch.randn(batch_size, INPUT_CHANNELS, height, width)
    return sample_input


def load_image_as_tensor(image_path: str, resolution: tuple = DEFAULT_RESOLUTION) -> torch.Tensor:
    """
    Load an RGB image and convert to normalized NCHW tensor.
    """
    height, width = resolution
    image = Image.open(image_path).convert("RGB").resize((width, height))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def prepare_ttnn_input(
    x: torch.Tensor, batch_size: int, resolution: tuple, device: ttnn.Device, memory_config: ttnn.MemoryConfig
) -> ttnn.Tensor:
    """
    Convert PyTorch input tensor to TTNN format.

    Args:
        x: Input tensor in NCHW format
        batch_size: Batch size
        resolution: (height, width)
        device: TTNN device
        memory_config: Memory configuration for tensor placement

    Returns:
        TTNN tensor ready for model inference
    """
    height, width = resolution

    ttnn_input = x.reshape(batch_size, 1, INPUT_CHANNELS, height * width)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device, memory_config=memory_config)
    return ttnn_input


def run_pytorch_inference(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Run inference using PyTorch reference model.

    Args:
        model: PyTorch model
        x: Input tensor

    Returns:
        Output segmentation mask
    """
    start_time = time.time()
    logger.info("Running PyTorch inference...")

    with torch.no_grad():
        y_pred = model(x)

    elapsed = time.time() - start_time
    logger.info(f"PyTorch inference completed in {elapsed:.3f}s")

    return y_pred


def run_ttnn_inference(
    model, x: torch.Tensor, batch_size: int, resolution: tuple, device: ttnn.Device, configs
) -> torch.Tensor:
    """
    Run inference using TTNN model.

    Args:
        model: TTNN model
        x: PyTorch input tensor
        batch_size: Batch size
        resolution: (height, width)
        device: TTNN device
        configs: Model configurations

    Returns:
        Output segmentation mask as PyTorch tensor
    """
    start_time = time.time()
    logger.info("Running TTNN inference...")

    ttnn_input = prepare_ttnn_input(x, batch_size, resolution, device, configs.l1_input_memory_config)

    y_pred = model(ttnn_input)

    y_pred = ttnn.to_torch(y_pred)
    y_pred = y_pred.reshape(batch_size, 1, resolution[0], resolution[1])
    y_pred = y_pred.to(torch.float)

    elapsed = time.time() - start_time
    logger.info(f"TTNN inference completed in {elapsed:.3f}s")

    return y_pred


def save_segmentation_visualization(input_tensor: torch.Tensor, prediction: torch.Tensor, output_path: str):
    """
    Save visualization of segmentation result.

    Args:
        input_tensor: Original input image
        prediction: Predicted segmentation mask
        output_path: Path to save visualization
    """
    pred_np = prediction.detach().cpu().numpy()[0, 0]
    probs = 1.0 / (1.0 + np.exp(-pred_np))
    mask_05 = (probs > 0.5).astype(np.uint8) * 255
    adaptive_threshold = float(probs.mean())
    mask_adaptive = (probs > adaptive_threshold).astype(np.uint8) * 255
    prob_img = (np.clip(probs, 0.0, 1.0) * 255.0).astype(np.uint8)

    output_base = Path(output_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask_05).save(output_base.with_name(output_base.stem + "_mask_0p5.png"))
    Image.fromarray(mask_adaptive).save(output_base.with_name(output_base.stem + "_mask_adaptive.png"))
    Image.fromarray(prob_img).save(output_base.with_name(output_base.stem + "_prob.png"))

    input_rgb = (input_tensor[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    overlay = input_rgb.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], mask_adaptive)
    Image.fromarray(overlay).save(output_base.with_name(output_base.stem + "_overlay.png"))

    logger.info(
        f"Saved outputs to {output_base.parent}: "
        f"{output_base.stem}_mask_0p5.png, {output_base.stem}_mask_adaptive.png, "
        f"{output_base.stem}_prob.png, {output_base.stem}_overlay.png"
    )
    logger.info(f"Probability stats min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")


def run_attention_denseunet_demo(
    device: ttnn.Device,
    reset_seeds,
    use_pytorch: bool = False,
    optimization_level: OptimizationLevel = OptimizationLevel.STAGE2,
    batch_size: int = 1,
    resolution: tuple = DEFAULT_RESOLUTION,
    input_tensor: torch.Tensor | None = None,
    output_name: str = "demo_result.png",
):
    """
    Run the Attention DenseUNet demo.

    Args:
        device: TTNN device
        reset_seeds: Pytest fixture for seed reset
        use_pytorch: If True, use PyTorch model instead of TTNN
        batch_size: Batch size
        resolution: Input resolution (height, width)
    """
    logger.info(f"Starting Attention DenseUNet demo")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Backend: {'PyTorch' if use_pytorch else 'TTNN'}")
    os.makedirs(PRED_DIR, exist_ok=True)
    if input_tensor is None:
        logger.info("Creating sample input...")
        sample_input = create_sample_input(batch_size, resolution)
    else:
        sample_input = input_tensor
    start_time = time.time()
    logger.info("Loading reference model...")
    reference_model = create_attention_denseunet()
    reference_model.eval()
    logger.info(f"Model loaded in {time.time() - start_time:.2f}s")

    if use_pytorch:
        prediction = run_pytorch_inference(reference_model, sample_input)
    else:
        start_time = time.time()
        logger.info("Initializing TTNN model...")

        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model, custom_preprocessor=create_preprocessor(device), device=None
        )

        configs = create_configs_from_parameters(
            parameters=parameters,
            in_channels=INPUT_CHANNELS,
            out_channels=1,
            input_height=resolution[0],
            input_width=resolution[1],
            batch_size=batch_size,
            optimization_level=optimization_level,
        )

        ttnn_model = create_model_from_configs(configs, device)
        logger.info(f"TTNN model initialized in {time.time() - start_time:.2f}s")
        prediction = run_ttnn_inference(ttnn_model, sample_input, batch_size, resolution, device, configs)

    output_path = os.path.join(PRED_DIR, output_name)
    save_segmentation_visualization(sample_input, prediction, output_path)

    logger.info("Demo completed successfully!")

    return prediction


@run_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": ATTENTION_DENSEUNET_L1_SMALL_SIZE,
            "trace_region_size": ATTENTION_DENSEUNET_TRACE_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_pytorch", [False, True])
def test_attention_denseunet_demo(device: ttnn.Device, reset_seeds, batch_size: int, use_pytorch: bool):
    """
    Pytest entry point for running the demo.
    """
    return run_attention_denseunet_demo(
        device=device, reset_seeds=reset_seeds, use_pytorch=use_pytorch, batch_size=batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention DenseUNet Demo")
    parser.add_argument("--pytorch", action="store_true", help="Use PyTorch model instead of TTNN")
    parser.add_argument(
        "--optimization-level",
        type=str,
        choices=[level.value for level in OptimizationLevel],
        default=OptimizationLevel.STAGE2.value,
        help="Optimization preset for TTNN backend",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256], help="Input resolution (height width)")
    parser.add_argument("--image", type=str, default=None, help="Input image path for single-image demo")
    parser.add_argument("--output-name", type=str, default="demo_result.png", help="Base output filename")
    args = parser.parse_args()
    resolution = tuple(args.resolution)
    image_tensor = load_image_as_tensor(args.image, resolution) if args.image else None

    if args.pytorch:
        logger.info("Running in PyTorch-only mode (no device required)")
        run_attention_denseunet_demo(
            device=None,
            reset_seeds=None,
            use_pytorch=True,
            batch_size=args.batch_size,
            resolution=resolution,
            input_tensor=image_tensor,
            output_name=args.output_name,
        )
    else:
        logger.info("Running TTNN demo directly")
        device = ttnn.open_device(
            device_id=0,
            l1_small_size=ATTENTION_DENSEUNET_L1_SMALL_SIZE,
            trace_region_size=ATTENTION_DENSEUNET_TRACE_SIZE,
            num_command_queues=2,
        )
        try:
            run_attention_denseunet_demo(
                device=device,
                reset_seeds=None,
                use_pytorch=False,
                optimization_level=OptimizationLevel(args.optimization_level),
                batch_size=args.batch_size,
                resolution=resolution,
                input_tensor=image_tensor,
                output_name=args.output_name,
            )
        finally:
            ttnn.close_device(device)
