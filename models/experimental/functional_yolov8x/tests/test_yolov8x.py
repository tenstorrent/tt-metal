# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import time
import torch
import pytest
import torch.nn as nn
from loguru import logger
from models.utility_functions import is_wormhole_b0
from models.perf.perf_utils import prep_perf_report
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x import YOLOv8x
from models.experimental.functional_yolov8x.reference import yolov8x_utils
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x_utils import custom_preprocessor
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

try:
    sys.modules["ultralytics"] = yolov8x_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8x_utils

except KeyError:
    print("models.experimental.functional_yolov8x.reference.yolov8x_utils not found.")


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


def get_expected_times(name):
    base = {"yolov8x": (177.47, 4.81)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [(1)])
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor"])
def test_yolov8x(device, input_tensor, batch_size):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output = YOLOv8x(device, ttnn_input, parameters)[0]
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov8x")

    prep_perf_report(
        model_name="models/experimental/functional_yolov8x",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 10.94],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov8x(batch_size, expected_perf):
    subdir = "ttnn_yolov8x"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 10.94

    command = f"pytest tests/ttnn/integration_tests/yolov8x/test_ttnn_yolov8x.py::test_demo"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov8x{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
