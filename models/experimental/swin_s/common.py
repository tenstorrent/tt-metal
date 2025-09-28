# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torchvision import models

SWIN_S_L1_SMALL_SIZE = 32768


def load_torch_model(torch_model, i=0, j=0, module="model", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = models.swin_s(weights="IMAGENET1K_V1")
        state_dict = model.state_dict()
    else:
        weights_path = (
            model_location_generator("vision-models/swin_s", model_subdir="", download_if_ci_v2=True)
            / "swin_s-5e29d889.pth"
        )
        state_dict = torch.load(weights_path)

    if module == "mlp":
        mlp_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}.mlp."))}
        new_state_dict = {}
        keys = [name for name, parameter in torch_model.state_dict().items()]
        values = [parameter for name, parameter in mlp_state_dict.items()]

        for i in range(len(keys)):
            new_state_dict[keys[i]] = values[i]

    elif module == "patchmerging":
        patchmerging_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}."))}
        new_state_dict = {}
        keys = [name for name, parameter in torch_model.state_dict().items()]
        values = [parameter for name, parameter in patchmerging_state_dict.items()]

        for i in range(len(keys)):
            new_state_dict[keys[i]] = values[i]

    elif module == "attention":
        ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}.attn."))}
        new_state_dict = {}
        keys = [name for name, parameter in torch_model.state_dict().items()]
        values = [parameter for name, parameter in ds_state_dict.items()]

        for i in range(len(keys)):
            new_state_dict[keys[i]] = values[i]

    elif module == "transformer_block":
        ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}."))}
        new_state_dict = {}
        keys = [name for name, parameter in torch_model.state_dict().items()]
        values = [parameter for name, parameter in ds_state_dict.items()]

        for i in range(len(keys)):
            new_state_dict[keys[i]] = values[i]

    else:
        new_state_dict = state_dict

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model
