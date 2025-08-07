# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from efficientnet_pytorch import EfficientNet
from models.experimental.efficientnetb0.reference import efficientnetb0

EFFICIENTNETB0_L1_SMALL_SIZE = 79104


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = EfficientNet.from_pretrained("efficientnet-b0").eval()
        state_dict = model.state_dict()
    else:
        weights_path = (
            model_location_generator("vision-models/efficientnet", model_subdir="", download_if_ci_v2=True)
            / "efficientnet_b0.pth"
        )
        state_dict = torch.load(weights_path)

    torch_model = efficientnetb0.Efficientnetb0()

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model
