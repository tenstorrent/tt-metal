# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from warnings import filterwarnings

import kagglehub
import pytest
import torch

from models.demos.vgg_unet.demo.demo_utils import postprocess, prediction, preprocess, process_single_image
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
from models.demos.vgg_unet.ttnn.model_preprocessing import create_vgg_unet_model_parameters
from models.demos.vgg_unet.ttnn.ttnn_vgg_unet import Tt_vgg_unet

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
        False,
        # True,
    ],
    ids=[
        "pretrained_weight_false",
        # "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_demo(device, reset_seeds, demo_type, model_type, use_pretrained_weight):
    # Download latest version of the dataset
    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    model_seg = UNetVGG19()
    if use_pretrained_weight:
        model_seg.load_state_dict(torch.load("models/experimental/vgg_unet/vgg_unet_torch.pth"))
    model_seg.eval()  # Set to evaluation mode

    if model_type == "ttnn_model":
        # Weights pre-processing
        torch_input = torch.randn((1, 3, 256, 256), dtype=torch.float)
        parameters = create_vgg_unet_model_parameters(model_seg, torch_input, device)
        ttnn_model = Tt_vgg_unet(device, parameters, parameters.conv_args)

    if demo_type == "multi":
        X_test = preprocess(path)
        # making prediction
        if model_type == "torch_model":
            df_pred = prediction(X_test, model_seg, model_type)
        else:
            df_pred = prediction(X_test, ttnn_model, model_type)
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
                output_dir="models/experimental/vgg_unet/demo/output_single_image",
                model_type=model_type,
            )
        else:
            process_single_image(
                img_path,
                mask_path,
                ttnn_model,
                output_dir="models/experimental/vgg_unet/demo/output_single_image_ttnn",
                model_type=model_type,
            )
