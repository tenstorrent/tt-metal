# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from models.experimental.swin.tt.swin_for_image_classification import (
    TtSwinForImageClassification,
)
from transformers import SwinForImageClassification as HF_SwinForImageClassification


def _swin(config, state_dict, base_address, device):
    return TtSwinForImageClassification(
        config=config,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
    )


def swin_for_image_classification(device) -> TtSwinForImageClassification:
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    model = HF_SwinForImageClassification.from_pretrained(model_name)
    model.eval()
    state_dict = model.state_dict()
    config = model.config
    base_address = f"swin."
    model = _swin(config, state_dict, base_address, device)
    return model
