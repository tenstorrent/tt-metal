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

import numpy as np
import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.attention_denseunet.reference.model import create_attention_denseunet
from models.demos.attention_denseunet.tt.common import (
    ATTENTION_DENSEUNET_L1_SMALL_SIZE,
    ATTENTION_DENSEUNET_TRACE_SIZE,
    create_preprocessor,
)
from models.demos.attention_denseunet.tt.config import create_configs_from_parameters
from models.demos.attention_denseunet.tt.model import create_model_from_configs

DEFAULT_RESOLUTION = (256, 256)
INPUT_CHANNELS = 3
DEMO_DIR = "models/demos/attention_denseunet/demo"
DEMO_IMAGE_DIR = os.path.join(DEMO_DIR, "images")
PRED_DIR = os.path.join(DEMO_DIR, "pred")


def create_sample_input(batch_size: int = 1, resolution: tuple = DEFAULT_RESOLUTION):
    """
    Create a sample input tensor for demo purposes.

    In a real scenario, this would load an actual image.

    Args:
        batch_size: Number of images in batch
        resolution: (height, width) of input images

    Returns:
        PyTorch tensor of shape (batch, channels, height, width)
    """
    height, width = resolution
    sample_input = torch.randn(batch_size, INPUT_CHANNELS, height, width)
    return sample_input


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
    try:
        from skimage.io import imsave

        pred_np = prediction.detach().cpu().numpy()
        mask = (pred_np[0, 0] > 0.5).astype(np.uint8) * 255
        imsave(output_path, mask)
        logger.info(f"Saved segmentation result to: {output_path}")

    except ImportError:
        logger.warning("skimage not available, skipping visualization save")
        np_path = output_path.replace(".png", ".npy")
        np.save(np_path, prediction.detach().cpu().numpy())
        logger.info(f"Saved prediction as numpy to: {np_path}")


def run_attention_denseunet_demo(
    device: ttnn.Device,
    reset_seeds,
    use_pytorch: bool = False,
    batch_size: int = 1,
    resolution: tuple = DEFAULT_RESOLUTION,
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
    logger.info("Creating sample input...")
    sample_input = create_sample_input(batch_size, resolution)
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
        )

        ttnn_model = create_model_from_configs(configs, device)
        logger.info(f"TTNN model initialized in {time.time() - start_time:.2f}s")
        prediction = run_ttnn_inference(ttnn_model, sample_input, batch_size, resolution, device, configs)

    output_path = os.path.join(PRED_DIR, "demo_result.png")
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
@pytest.mark.parametrize("use_pytorch", [True])
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256], help="Input resolution (height width)")
    args = parser.parse_args()
    if args.pytorch:
        logger.info("Running in PyTorch-only mode (no device required)")
        resolution = tuple(args.resolution)
        sample_input = create_sample_input(args.batch_size, resolution)
        model = create_attention_denseunet()
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        logger.info(f"Input shape: {sample_input.shape}")
        logger.info(f"Output shape: {output.shape}")
        logger.info("Demo completed!")
    else:
        logger.info("For TTNN execution, run via pytest:")
        logger.info("  pytest models/demos/attention_denseunet/demo/demo.py -v")
