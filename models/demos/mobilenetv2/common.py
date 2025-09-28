# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

MOBILENETV2_L1_SMALL_SIZE = 8 * 1024  # 8 KiB
MOBILENETV2_BATCH_SIZE = 10


def load_torch_model(torch_model, model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/mobilenetv2/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/mobilenetv2", model_subdir="", download_if_ci_v2=True)
            / "mobilenet_v2-b0353104.pth"
        )

    state_dict = torch.load(weights_path)
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {
        name1: parameter2
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
        if isinstance(parameter2, torch.FloatTensor)
    }
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    return torch_model
