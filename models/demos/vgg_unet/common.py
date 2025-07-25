# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch


def load_torch_model(torch_model, model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/vgg_unet/vgg_unet_torch.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/vgg_unet/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/unet_vgg", model_subdir="", download_if_ci_v2=True)
            / "vgg_unet_torch.pth"
        )

    torch_dict = torch.load(weights_path)
    new_state_dict = dict(zip(torch_model.state_dict().keys(), torch_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    return torch_model
