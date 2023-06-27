import torch
import torch.nn as nn

import tt_lib
from python_api_testing.models.swin.tt.swin_for_image_classification import (
    TtSwinForImageClassification,
)
from transformers import SwinForImageClassification as HF_SwinForImageClassification


def _swin(config, state_dict, base_address, device, host):
    return TtSwinForImageClassification(
        config=config,
        state_dict=state_dict,
        base_address=base_address,
        device=device,
        host=host,
    )


def swin_for_image_classification(device, host) -> TtSwinForImageClassification:
    model_name = "microsoft/swin-tiny-patch4-window7-224"
    model = HF_SwinForImageClassification.from_pretrained(model_name)
    model.eval()
    state_dict = model.state_dict()
    config = model.config
    base_address = f"swin."
    model = _swin(config, state_dict, base_address, device, host)
    return model
