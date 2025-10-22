# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time

import pytest
import torch
from loguru import logger
from skimage.io import imsave
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.vanilla_unet_new.demo import demo_utils
from models.demos.vanilla_unet_new.tt.common import (
    VANILLA_UNET_L1_SMALL_SIZE,
    VANILLA_UNET_TRACE_SIZE,
    create_unet_preprocessor,
    load_reference_model,
)
from models.demos.vanilla_unet_new.tt.config import create_unet_configs_from_parameters
from models.demos.vanilla_unet_new.tt.model import create_unet_from_configs

# Constants
DEFAULT_RESOLUTION = (480, 640)
INPUT_CHANNELS = 3
DEMO_IMAGE_DIR = "models/demos/vanilla_unet_new/demo/images"
DEMO_IMAGE_NAME = "TCGA_CS_4944_20010208_1.tif"
DEMO_MASK_NAME = "TCGA_CS_4944_20010208_1_mask.tif"
WEIGHTS_PATH = "models/demos/vanilla_unet_new/unet.pt"
WEIGHTS_DOWNLOAD_SCRIPT = "models/demos/vanilla_unet_new/weights_download.sh"
PRED_DIR = "models/demos/vanilla_unet_new/demo/pred"


def get_weights_path(model_location_generator):
    """Get the path to model weights, downloading if necessary."""
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        if not os.path.exists(WEIGHTS_PATH):
            logger.info("Downloading weights...")
            os.system(f"bash {WEIGHTS_DOWNLOAD_SCRIPT}")
        return WEIGHTS_PATH
    else:
        return (
            model_location_generator("vision-models/unet_vanilla", model_subdir="", download_if_ci_v2=True) / "unet.pt"
        )


def prepare_ttnn_input(x, batch_size, resolution, device, memory_config):
    """Convert PyTorch input tensor to TT-NN format."""
    ttnn_input = x.reshape(batch_size, 1, INPUT_CHANNELS, resolution[0] * resolution[1])
    ttnn_input = ttnn.from_torch(
        ttnn_input,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=memory_config,
    )
    return ttnn_input


def run_ttnn_inference(model, x, batch_size, resolution, device, configs):
    """Run inference using TT-NN model."""
    start_time = time.time()
    logger.info("Running model compilation and inference...")

    ttnn_input = prepare_ttnn_input(x, batch_size, resolution, device, configs.l1_input_memory_config)
    y_pred = model(ttnn_input)

    y_pred = ttnn.to_torch(y_pred)
    y_pred = y_pred.reshape(batch_size, 1, resolution[0], resolution[1])
    y_pred = y_pred.to(torch.float)

    logger.info(f"Model compilation and inference completed in {time.time() - start_time:.2f}s")
    return y_pred


def run_torch_inference(model, x):
    """Run inference using PyTorch reference model."""
    start_time = time.time()
    logger.info("Running inference...")
    y_pred = model(x)
    logger.info(f"Inference completed in {time.time() - start_time:.2f}s")
    return y_pred


def save_visualization(y_pred_np, y_true_np, output_path):
    """Create and save visualization with predicted and ground truth outlines."""
    image = demo_utils.gray2rgb(y_pred_np[0, 0])
    image = demo_utils.outline(image, y_pred_np[0, 0], color=[255, 0, 0])  # Predicted (red)
    image = demo_utils.outline(image, y_true_np[0, 0], color=[0, 255, 0])  # Ground truth (green)
    imsave(output_path, image)
    logger.info(f"Saved result to: {output_path}")


def run_unet_demo_single_image(
    device,
    reset_seeds,
    model_location_generator,
    use_torch_model,
    batch_size,
    resolution=DEFAULT_RESOLUTION,
    filename="result_ttnn_1.png",
):
    logger.info(f"Starting Vanilla UNet demo with resolution=({resolution[0]}x{resolution[1]}) and batch={batch_size}")

    weights_path = get_weights_path(model_location_generator)
    os.makedirs(PRED_DIR, exist_ok=True)

    image_path = os.path.join(DEMO_IMAGE_DIR, DEMO_IMAGE_NAME)
    mask_path = os.path.join(DEMO_IMAGE_DIR, DEMO_MASK_NAME)

    logger.info(f"Loading test image: {image_path}")
    args = argparse.Namespace(
        image=image_path,
        mask=mask_path,
        image_size=resolution,
        batch_size=batch_size,
    )
    loader = demo_utils.data_loader(args)

    start_time = time.time()
    logger.info("Loading reference model...")
    reference_model = load_reference_model(model_location_generator)
    logger.info(f"Reference model loaded in {time.time() - start_time:.2f}s")

    ttnn_model = None
    ttnn_configs = None
    if not use_torch_model:
        start_time = time.time()
        logger.info("Initializing TT-NN model...")
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_unet_preprocessor(device),
            device=None,
        )
        ttnn_configs = create_unet_configs_from_parameters(
            parameters=parameters,
            input_height=resolution[0],
            input_width=resolution[1],
            batch_size=batch_size,
        )
        ttnn_model = create_unet_from_configs(ttnn_configs, device)
        logger.info(f"TT-NN model initialized in {time.time() - start_time:.2f}s")

    for data in loader:
        x, y_true = data
        x = x.squeeze(1)

        if use_torch_model:
            y_pred = run_torch_inference(reference_model, x)
        else:
            y_pred = run_ttnn_inference(ttnn_model, x, batch_size, resolution, device, ttnn_configs)

        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        output_path = os.path.join(PRED_DIR, filename)
        save_visualization(y_pred_np, y_true_np, output_path)

    logger.info("Demo completed successfully!")


@run_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": VANILLA_UNET_L1_SMALL_SIZE,
            "trace_region_size": VANILLA_UNET_TRACE_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_torch_model", [False])
def test_unet_demo_single_image(device, reset_seeds, model_location_generator, use_torch_model, batch_size):
    return run_unet_demo_single_image(device, reset_seeds, model_location_generator, use_torch_model, batch_size)
