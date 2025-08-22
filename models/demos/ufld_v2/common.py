# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34

UFLD_V2_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None, use_pretrained_weight=True):
    torch_model = TuSimple34(input_height=320, input_width=800)
    torch_model.eval()
    if not use_pretrained_weight:
        return torch_model

    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/ufld_v2/tusimple_res34.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/ufld_v2/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/ufldv2", model_subdir="", download_if_ci_v2=True)
            / "tusimple_res34.pth"
        )

    state_dict = torch.load(weights_path)
    new_state_dict = {}
    for key, value in state_dict["model"].items():
        new_key = key.replace("model.", "res_model.")
        new_state_dict[new_key] = value
    torch_model.load_state_dict(new_state_dict)
    return torch_model
