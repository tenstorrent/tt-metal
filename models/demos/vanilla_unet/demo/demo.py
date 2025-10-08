# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import pytest
import torch
from loguru import logger
from skimage.io import imsave
from tqdm import tqdm

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.vanilla_unet.common import VANILLA_UNET_L1_SMALL_SIZE, load_torch_model
from models.demos.vanilla_unet.demo import demo_utils
from models.demos.vanilla_unet.runner.performant_runner import VanillaUNetPerformantRunner


def run_unet_demo_single_image(
    device,
    reset_seeds,
    model_location_generator,
    use_torch_model,
    batch_size,
    act_dtype,
    weight_dtype,
    resolution=(480, 640),
    filename="result_ttnn_1.png",
):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/vanilla_unet/unet.pt"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/vanilla_unet/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/unet_vanilla", model_subdir="", download_if_ci_v2=True) / "unet.pt"
        )

    pred_dir = "models/demos/vanilla_unet/demo/pred"
    # Create the directory if it doesn't exist
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    args = argparse.Namespace(
        device="cpu",  # Choose "cpu" or "cuda:0" based on your setup
        batch_size=1,
        weights=weights_path,  # Path to the pre-trained model weights
        image="models/demos/vanilla_unet/demo/images/TCGA_CS_4944_20010208_1.tif",  # Path to your input image
        mask="models/demos/vanilla_unet/demo/images/TCGA_CS_4944_20010208_1_mask.tif",  # Path to your input mask
        image_size=resolution,  # Resize input image to this size
        predictions="models/demos/vanilla_unet/demo/pred",  # Directory to save prediction results
    )

    loader = demo_utils.data_loader(args)  # loader will load just a single image
    reference_model = load_torch_model(model_location_generator)

    performant_runner = VanillaUNetPerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )

    # Processing the data
    for data in tqdm(loader):
        x, y_true = data
        x = x.squeeze(1)
        # Get the prediction
        if use_torch_model:
            y_pred = reference_model(x)
        else:
            y_pred = performant_runner.run(x)
            y_pred = ttnn.to_torch(y_pred, mesh_composer=performant_runner.runner_infra.output_mesh_composer)
            y_pred = y_pred.permute(0, 3, 1, 2)
            y_pred = y_pred.reshape(batch_size, 1, resolution[0], resolution[1])
            y_pred = y_pred.to(torch.float)

        # Convert predictions to numpy
        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()

        # Save the result
        image = demo_utils.gray2rgb(y_pred_np[0, 0])  # Grayscale to RGB
        image = demo_utils.outline(image, y_pred_np[0, 0], color=[255, 0, 0])  # Predicted outline (red)
        image = demo_utils.outline(image, y_true_np[0, 0], color=[0, 255, 0])  # True outline (green)

        filepath = os.path.join(args.predictions, filename)
        imsave(filepath, image)

    logger.info(f"All Predictions are saved to:{pred_dir} ")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE, "trace_region_size": 1605632, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("use_torch_model", [False])
@run_for_wormhole_b0()
def test_unet_demo_single_image(
    device, reset_seeds, model_location_generator, use_torch_model, batch_size, act_dtype, weight_dtype
):
    return run_unet_demo_single_image(
        device, reset_seeds, model_location_generator, use_torch_model, batch_size, act_dtype, weight_dtype
    )
