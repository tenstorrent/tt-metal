# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
import sys

from models.experimental.functional_yolov11.reference import yolov11

from models.experimental.functional_yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.experimental.functional_yolov11.tt import ttnn_yolov11
import torch.nn as nn
from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_yolov11.test.yolov11_perfomant import run_yolov11_trace_inference

try:
    sys.modules["ultralytics"] = yolov11
    sys.modules["ultralytics.nn.tasks"] = yolov11
    sys.modules["ultralytics.nn.modules.conv"] = yolov11
    sys.modules["ultralytics.nn.modules.block"] = yolov11
    sys.modules["ultralytics.nn.modules.head"] = yolov11

except KeyError:
    print("models.experimental.functional_yolov11.reference.yolov11 not found.")


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
        w = "models/experimental/functional_yolov11/reference/yolo11n.pt"
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
def test_yolov11(device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_yolov11_input_tensors(device)

    torch_model = attempt_load("yolov11n.pt", map_location="cpu")
    state_dict = torch_model.state_dict()
    torch_model = yolov11.YoloV11()
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_output = torch_model(torch_input)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11.YoloV11(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)
    print(ttnn_output.shape, torch_output.shape)
    ttnn_output = ttnn_output.reshape(torch_output.shape)

    assert_with_pcc(torch_output, ttnn_output, 0.99999)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 1843200}], indirect=True)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_yolov11_trace_inference(
    device,
    use_program_cache,
    enable_async_mode,
    model_location_generator,
):
    run_yolov11_trace_inference(
        device,
        model_location_generator,
    )
