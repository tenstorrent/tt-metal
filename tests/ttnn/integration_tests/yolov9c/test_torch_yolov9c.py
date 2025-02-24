# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import torch
import pytest
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov9c.reference import yolov9c

try:
    sys.modules["ultralytics"] = yolov9c
    sys.modules["ultralytics.nn.tasks"] = yolov9c
    sys.modules["ultralytics.nn.modules.conv"] = yolov9c
    sys.modules["ultralytics.nn.modules.block"] = yolov9c
    sys.modules["ultralytics.nn.modules.head"] = yolov9c

except KeyError:
    print("models.experimental.functional_yolov9c.reference.YoloV9 not found.")


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = "models/experimental/functional_yolov9c/reference/yolov9c.pt"
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9(device, use_program_cache, reset_seeds):
    torch_input = torch.load("models/experimental/functional_yolov9c/reference/input.pt")
    torch_model = attempt_load("yolov9c.pt", map_location="cpu")
    state_dict = torch_model.state_dict()
    torch_model = yolov9c.YoloV9()
    print(torch_model)
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_output = torch_model(torch_input)

    ttnn_output = torch.load("models/experimental/functional_yolov9c/output.pt")
    assert_with_pcc(torch_output, ttnn_output, 0.9999999999999)
