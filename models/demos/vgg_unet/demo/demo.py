# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from warnings import filterwarnings

import kagglehub
import pytest
from loguru import logger

from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.vgg_unet.common import load_torch_model
from models.demos.vgg_unet.demo.demo_utils import postprocess, prediction, preprocess, process_single_image
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
from models.demos.vgg_unet.runner.performant_runner import VggUnetTrace2CQ

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        logger.info(os.path.join(dirname, filename))
filterwarnings("ignore")


def run_demo(
    device, model_location_generator, reset_seeds, demo_type, model_type, use_pretrained_weight, device_batch_size
):
    # The below line is commented due to the issue https://github.com/tenstorrent/tt-metal/issues/23270
    # device.disable_and_clear_program_cache()
    disable_persistent_kernel_cache()
    # Download latest version of the dataset
    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    batch_size = device_batch_size * device.get_num_devices()
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            logger.info(os.path.join(dirname, filename))
    model_seg = UNetVGG19()
    if use_pretrained_weight:
        model_seg = load_torch_model(model_seg, model_location_generator)
    model_seg.eval()  # Set to evaluation mode
    if model_type == "ttnn_model":
        vgg_unet_trace_2cq = VggUnetTrace2CQ()

        vgg_unet_trace_2cq.initialize_vgg_unet_trace_2cqs_inference(
            device,
            model_location_generator=model_location_generator,
            use_pretrained_weight=use_pretrained_weight,
            device_batch_size=device_batch_size,
        )
    if demo_type == "multi":
        X_test = preprocess(path)
        # making prediction
        if model_type == "torch_model":
            df_pred = prediction(X_test, model_seg, model_type, batch_size=batch_size)
        else:
            df_pred = prediction(
                X_test,
                vgg_unet_trace_2cq,
                model_type,
                batch_size=batch_size,
            )
        postprocess(df_pred, X_test, model_type)
    else:
        base_path = path
        img_relative_path = "lgg-mri-segmentation/kaggle_3m/TCGA_FG_6690_20020226/TCGA_FG_6690_20020226_31.tif"
        mask_relative_path = "lgg-mri-segmentation/kaggle_3m/TCGA_FG_6690_20020226/TCGA_FG_6690_20020226_31_mask.tif"

        img_path = os.path.join(base_path, img_relative_path)
        mask_path = os.path.join(base_path, mask_relative_path)

        if model_type == "torch_model":
            process_single_image(
                img_path,
                mask_path,
                model_seg,
                output_dir="models/demos/vgg_unet/demo/output_single_image",
                model_type=model_type,
            )
        else:
            process_single_image(
                img_path,
                mask_path,
                vgg_unet_trace_2cq,
                output_dir="models/demos/vgg_unet/demo/output_single_image_ttnn",
                model_type=model_type,
            )


@pytest.mark.parametrize(
    "demo_type",
    [
        "single",
        "multi",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "torch_model",
        "ttnn_model",
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
def test_demo(device, model_location_generator, reset_seeds, demo_type, model_type, use_pretrained_weight, batch_size):
    return run_demo(
        device, model_location_generator, reset_seeds, demo_type, model_type, use_pretrained_weight, batch_size
    )


@pytest.mark.parametrize(
    "demo_type",
    [
        "multi",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "torch_model",
        "ttnn_model",
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "device_batch_size",
    ((1),),
)
def test_demo_dp(
    mesh_device, model_location_generator, reset_seeds, demo_type, model_type, use_pretrained_weight, device_batch_size
):
    return run_demo(
        mesh_device,
        model_location_generator,
        reset_seeds,
        demo_type,
        model_type,
        use_pretrained_weight,
        device_batch_size,
    )
