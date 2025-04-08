# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import skip_for_grayskull
from models.experimental.functional_vanilla_unet.reference.unet import UNet
from models.experimental.functional_vanilla_unet.ttnn.ttnn_unet import TtUnet
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_vanilla_unet.ttnn.model_preprocesser import create_custom_preprocessor
import argparse
from models.experimental.functional_vanilla_unet.demo import demo_utils
from tqdm import tqdm
import os
from skimage.io import imsave


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("use_torch_model", [False])
@skip_for_grayskull()
def test_unet_demo_single_image(device, reset_seeds, model_location_generator, use_torch_model):
    args = argparse.Namespace(
        device="cpu",  # Choose "cpu" or "cuda:0" based on your setup
        batch_size=1,
        weights="models/experimental/functional_vanilla_unet/unet.pt",  # Path to the pre-trained model weights
        image="models/experimental/functional_vanilla_unet/demo/images/TCGA_CS_4944_20010208_1.tif",  # Path to your input image
        mask="models/experimental/functional_vanilla_unet/demo/images/TCGA_CS_4944_20010208_1_mask.tif",  # Path to your input mask
        image_size=(480, 640),  # Resize input image to this size
        predictions="models/experimental/functional_vanilla_unet/demo/pred",  # Directory to save prediction results
    )

    loader = demo_utils.data_loader(args)  # loader will load just a single image

    state_dict = torch.load(
        "models/experimental/functional_vanilla_unet/unet.pt",
        map_location=torch.device("cpu"),
    )
    ds_state_dict = {k: v for k, v in state_dict.items()}

    reference_model = UNet()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    # Processing the data
    for data in tqdm(loader):
        x, y_true = data
        # x, y_true = x.to(device), y_true.to(device)
        x = x.squeeze(1)
        # Get the prediction
        if use_torch_model:
            y_pred = reference_model(x)
        else:
            parameters = preprocess_model_parameters(
                initialize_model=lambda: reference_model,
                custom_preprocessor=create_custom_preprocessor(None),
                device=None,
            )
            ttnn_model = TtUnet(device=device, parameters=parameters, model=reference_model)

            ttnn_input_tensor = ttnn.from_torch(
                x.permute(0, 2, 3, 1), device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )

            y_pred = ttnn_model(device, ttnn_input_tensor)

        # Convert predictions to numpy
        y_pred_np = y_pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()

        # Save the result
        image = demo_utils.gray2rgb(y_pred_np[0, 0])  # Grayscale to RGB
        image = demo_utils.outline(image, y_pred_np[0, 0], color=[255, 0, 0])  # Predicted outline (red)
        image = demo_utils.outline(image, y_true_np[0, 0], color=[0, 255, 0])  # True outline (green)

        filename = "result_ttnn_1.png"
        filepath = os.path.join(args.predictions, filename)
        imsave(filepath, image)

    print("Prediction saved to:", filepath)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("use_torch_model", [False])
@skip_for_grayskull()
def test_unet_demo_imageset(device, reset_seeds, model_location_generator, use_torch_model):
    args = argparse.Namespace(
        device="cpu",  # Choose "cpu" or "cuda:0" based on your setup
        batch_size=1,
        weights="models/experimental/functional_vanilla_unet/unet.pt",  # Path to the pre-trained model weights
        images="models/experimental/functional_vanilla_unet/demo/imageset",  # Path to your input image
        image_size=(480, 640),  # Resize input image to this size
        predictions="models/experimental/functional_vanilla_unet/demo/pred_image_set",  # Directory to save prediction results
    )

    loader = demo_utils.data_loader_imageset(args)
    state_dict = torch.load(
        "models/experimental/functional_vanilla_unet/unet.pt",
        map_location=torch.device("cpu"),
    )
    ds_state_dict = {k: v for k, v in state_dict.items()}

    reference_model = UNet()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    unet = reference_model
    if not use_torch_model:
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(None), device=None
        )
        ttnn_model = TtUnet(device=device, parameters=parameters, model=reference_model)

    input_list = []
    pred_list = []
    true_list = []

    for i, data in tqdm(enumerate(loader)):
        x, y_true = data

        if use_torch_model:
            y_pred = unet(x)
        else:
            ttnn_input_tensor = ttnn.from_torch(
                x.permute(0, 2, 3, 1), device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            y_pred = ttnn_model(device, ttnn_input_tensor)
        y_pred_np = y_pred.detach().cpu().numpy()
        pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

        y_true_np = y_true.detach().cpu().numpy()
        true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

        x_np = x.detach().cpu().numpy()
        input_list.extend([x_np[s] for s in range(x_np.shape[0])])

    volumes = demo_utils.postprocess_per_volume(
        input_list,
        pred_list,
        true_list,
        loader.dataset.patient_slice_index,
        loader.dataset.patients,
    )

    for p in volumes:
        x = volumes[p][0]
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        for s in range(x.shape[0]):
            image = demo_utils.gray2rgb(x[s, 1])  # channel 1 is for FLAIR
            image = demo_utils.outline(image, y_pred[s, 0], color=[255, 0, 0])
            image = demo_utils.outline(image, y_true[s, 0], color=[0, 255, 0])
            filename = "{}-{}.png".format(p, str(s).zfill(2))
            filepath = os.path.join(args.predictions, filename)
            imsave(filepath, image)
