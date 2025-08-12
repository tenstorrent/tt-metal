# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch


def load_torch_model(torch_model, layer="", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/experimental/vadv2/vadv2_weights_1.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/experimental/vadv2/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/vad_v2", model_subdir="", download_if_ci_v2=True)
            / "vadv2_weights_1.pth"
        )

    torch_dict = torch.load(weights_path)

    if layer == "":
        new_state_dict = {}
        for k, v in torch_dict.items():
            k_new = k.replace("lateral_convs.0.", "lateral_convs.")
            k_new = k_new.replace("fpn_convs.0.", "fpn_convs.")
            new_state_dict[k_new] = v
    else:
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith(layer))}
        new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model
