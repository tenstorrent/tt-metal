# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import transformers


def load_torch_model(model_location_generator=None, embedding=False):
    config = transformers.ViTConfig.from_pretrained("google/vit-base-patch16-224")
    if not embedding:
        config.num_hidden_layers = 12
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", config=config)
        return model.eval()
    else:
        model = transformers.ViTForImageClassification.from_pretrained(config=config)
        weights_path = (
            model_location_generator("vision-models/vit", model_subdir="", download_if_ci_v2=True)
            / "vit_base_patch16_224.pth"
        )
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        return model.eval()
