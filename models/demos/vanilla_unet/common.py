# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.vanilla_unet.reference.unet import UNet

VANILLA_UNET_L1_SMALL_SIZE = (7 * 8192) + 2592


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/vanilla_unet/unet.pt"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/vanilla_unet/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/unet_vanilla", model_subdir="", download_if_ci_v2=True) / "unet.pt"
        )

    state_dict = torch.load(
        weights_path,
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
    return reference_model
