# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torchvision import models

SWIN_V2_L1_SMALL_SIZE = 24576


def load_torch_model(torch_model, i=0, j=0, module="model", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = models.swin_v2_s(weights="IMAGENET1K_V1")
        state_dict = model.state_dict()
    else:
        weights_path = (
            model_location_generator("vision-models/swin_s_v2", model_subdir="", download_if_ci_v2=True)
            / "swin_v2_s-637d8ceb.pth"
        )
        state_dict = torch.load(weights_path)

    if module == "mlp":
        mlp_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}.mlp."))}
        new_state_dict = {}
        new_torch_state_dic = {}
        for k, v in mlp_state_dict.items():
            new_state_dict[k] = mlp_state_dict[k]
            new_torch_state_dic[k.replace(f"features.{i}.{j}.mlp.", "")] = mlp_state_dict[k]
    elif module == "transformer_block":
        ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}."))}
        new_state_dict = {}
        new_torch_state_dic = {}
        for k, v in ds_state_dict.items():
            if "cbp_mlp" not in k:
                new_state_dict[k] = ds_state_dict[k]
            new_torch_state_dic[k.replace(f"features.{i}.{j}.", "")] = ds_state_dict[k]
    else:
        new_torch_state_dic = state_dict

    torch_model.load_state_dict(new_torch_state_dic)
    torch_model.eval()
    return torch_model
