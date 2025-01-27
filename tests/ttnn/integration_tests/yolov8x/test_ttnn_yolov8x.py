# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import torch
import pytest
import torch.nn as nn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x import YOLOv8x
from models.experimental.functional_yolov8x.reference import yolov8x_utils

from models.experimental.functional_yolov8x.tt.ttnn_yolov8x import Conv, C2f, SPPF, Detect_cv2, Detect, DFL
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x_utils import (
    ttnn_decode_bboxes,
    custom_preprocessor,
)

try:
    sys.modules["ultralytics"] = yolov8x_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8x_utils

except KeyError:
    print("models.experimental.functional_yolov8x.reference.yolov8x_utils not found.")


# For testing reference
def decode_bboxes(distance, anchor_points, xywh=True, dim=1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset

        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


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
        w = "models/experimental/functional_yolov8x/demo/yolov8x.pt"
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


def run_submodule(x, submodule):
    y = []
    for m in submodule:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x)
    return x


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor1"])
def test_demo(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))

    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        ttnn_model_output = YOLOv8x(device, ttnn_input, parameters)[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    with torch.inference_mode():
        torch_model_output = torch_model(input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 3, 640, 640)))], ids=["input_tensor1"])
def test_Conv(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        ttnn_model_output, out_h, out_w = Conv(
            device,
            ttnn_input,
            parameters,
            "model.0",
            ttnn_input.shape[1],
            ttnn_input.shape[2],
            3,
            2,
            1,
            change_shard=True,
            deallocate_activation=True,
        )
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.0")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 1280, 20, 20)))], ids=["input_tensor1"])
def test_C2f(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        ttnn_model_output, out_h, out_w = C2f(
            device,
            ttnn_input,
            parameters,
            "model.21",
            ttnn_input.shape[1],
            ttnn_input.shape[2],
            n=3,
            shortcut=False,
            change_shard=True,
        )
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.21")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 640, 20, 20)))], ids=["input_tensor1"])
def test_SPPF(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        ttnn_model_output, out_h, out_w = SPPF(
            device, ttnn_input, parameters, "model.9", ttnn_input.shape[1], ttnn_input.shape[2]
        )
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.9")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor, c1, c2, k, reg_max, idx",
    [
        (torch.rand((1, 320, 80, 80)), 320, 80, 3, 64, 0),
        (torch.rand((1, 640, 40, 40)), 640, 80, 3, 64, 1),
        (torch.rand((1, 640, 20, 20)), 640, 80, 3, 64, 2),
    ],
    ids=["input_tensor1", "input_tensor2", "input_tensor3"],
)
def test_Detect_cv2(device, input_tensor, c1, c2, k, reg_max, idx):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        ttnn_model_output, out_h, out_w = Detect_cv2(
            device, ttnn_input, parameters, f"model.22.cv2.{idx}", ttnn_input.shape[1], ttnn_input.shape[2], k, reg_max
        )
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule(f"model.22.cv2.{idx}")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor, c1, c2, k, reg_max, idx",
    [
        (torch.rand((1, 320, 80, 80)), 320, 320, 3, 80, 0),
        (torch.rand((1, 640, 40, 40)), 640, 320, 3, 80, 1),
        (torch.rand((1, 640, 20, 20)), 640, 320, 3, 80, 2),
    ],
    ids=["input_tensor1", "input_tensor2", "input_tensor3"],
)
def test_Detect_cv3(device, input_tensor, c1, c2, k, reg_max, idx):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        ttnn_model_output, out_h, out_w = Detect_cv2(
            device,
            ttnn_input,
            parameters,
            f"model.22.cv3.{idx}",
            ttnn_input.shape[1],
            ttnn_input.shape[2],
            k,
            reg_max=reg_max,
        )
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule(f"model.22.cv3.{idx}")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [([torch.rand((1, 320, 80, 80)), torch.rand((1, 640, 40, 40)), torch.rand((1, 640, 20, 20))])],
    ids=["input_tensor1"],
)
def test_last_detect(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = []
    for i in range(len(input_tensor)):
        x = ttnn.from_torch(input_tensor[i], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        x = ttnn.permute(x, (0, 2, 3, 1))
        ttnn_input.append(x)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        ttnn_model_output = Detect(device, ttnn_input, parameters, "model.22", nc=80, ch=(320, 640, 640))[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    submodule = torch_model.get_submodule(f"model.22")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 64, 8400)))], ids=["input_tensor1"])
def test_DFL(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        ttnn_model_output = DFL(device, ttnn_input, parameters, "model.22.dfl")
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    submodule = torch_model.get_submodule("model.22.dfl")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "distance, anchors", [(torch.rand((1, 4, 8400)), torch.rand((1, 2, 8400)))], ids=["input_tensor"]
)
def test_dist2bbox(device, distance, anchors):
    disable_persistent_kernel_cache()

    ttnn_distance = ttnn.from_torch(distance, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_anchors = ttnn.from_torch(anchors, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_model_output = ttnn_decode_bboxes(device, ttnn_distance, ttnn_anchors)
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    torch_model_output = decode_bboxes(distance, anchors)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
