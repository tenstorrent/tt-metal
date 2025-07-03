# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from warnings import filterwarnings

import kagglehub
import pytest
import torch

from models.demos.vgg_unet.demo.demo_utils import postprocess, prediction, preprocess, process_single_image
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
from models.demos.vgg_unet.tests.vgg_unet_e2e_performant import VggUnetTrace2CQ
from models.utility_functions import disable_persistent_kernel_cache

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
filterwarnings("ignore")


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
def test_demo(device, model_location_generator, reset_seeds, demo_type, model_type, use_pretrained_weight):
    # The below line is commented due to the issue https://github.com/tenstorrent/tt-metal/issues/23270
    # device.disable_and_clear_program_cache()
    disable_persistent_kernel_cache()
    # Download latest version of the dataset
    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    model_seg = UNetVGG19()
    if use_pretrained_weight:
        if not os.path.exists("models/demos/vgg_unet/vgg_unet_torch.pth"):
            os.system("bash models/demos/vgg_unet/weights_download.sh")
        model_seg.load_state_dict(torch.load("models/demos/vgg_unet/vgg_unet_torch.pth"))
    model_seg.eval()  # Set to evaluation mode

    if model_type == "ttnn_model":
        vgg_unet_trace_2cq = VggUnetTrace2CQ()

        vgg_unet_trace_2cq.initialize_vgg_unet_trace_2cqs_inference(
            device,
            model_location_generator,
            use_pretrained_weight=use_pretrained_weight,
        )

    if demo_type == "multi":
        X_test = preprocess(path)
        # making prediction
        if model_type == "torch_model":
            df_pred = prediction(X_test, model_seg, model_type)
        else:
            df_pred = prediction(X_test, vgg_unet_trace_2cq, model_type)
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
