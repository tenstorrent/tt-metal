# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import timm
import torch

VOVNET_L1_SMALL_SIZE = 16384


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()
        return model
    else:
        model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=False).eval()
        weights = (
            model_location_generator("vision-models/vovnet", model_subdir="", download_if_ci_v2=True)
            / "ese_vovnet19b_dw_ra_in1k.pth"
        )
        state_dict = torch.load(weights)

        model.load_state_dict(state_dict)
        return model.eval()
