# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import ttnn
import torch
import pytest
import torch.nn as nn
from pathlib import Path
from loguru import logger
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x_utils import custom_preprocessor
from models.utility_functions import is_wormhole_b0
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import (
    profiler,
)
from models.experimental.functional_yolov8x.tt.ttnn_yolov8x import YOLOv8xModel
from models.experimental.functional_yolov8x.reference import yolov8x_utils
from models.experimental.functional_yolov8x.reference import yolov8x

try:
    sys.modules["ultralytics"] = yolov8x_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8x_utils

except KeyError:
    logger.info("models.experimental.functional_yolov8x.reference.yolov8x_utils not found.")


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_download(file, repo="ultralytics/assets"):
    tests = Path(__file__).parent.parent / "reference"
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolov8x.pt"
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"
        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"
        except Exception as e:
            logger.info(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                logger.info(f"ERROR: Download failure for {msg}")
            else:
                logger.info(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        logger.info("Loading weights from:", weight_path)
        ckpt = torch.load(weight_path, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def get_expected_times(name):
    base = {"yolov8x": (153.35, 0.6)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 640, 640, 3))], ids=["input_tensor"])
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_yolov8x(device, input_tensor, use_pretrained_weight):
    disable_persistent_kernel_cache()
    profiler.clear()
    batch_size = input_tensor.shape[0]
    if use_pretrained_weight:
        torch_model = attempt_load("yolov8x.pt", map_location="cpu")
        state_dict = torch_model.state_dict()
    else:
        torch_model = yolov8x.DetectionModel()
        state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict)

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_model = YOLOv8xModel(device, parameters)
    logger.info(f"Compiling model with warmup run")
    profiler.start(f"inference_and_compile_time")
    ttnn_output_tensor = ttnn_model(ttnn_input)

    profiler.end(f"inference_and_compile_time")

    inference_and_compile_time = profiler.get("inference_and_compile_time")
    logger.info(f"Model compiled with warmup run in {(inference_and_compile_time):.2f} s")

    iterations = 16
    outputs = []
    logger.info(f"Running inference for {iterations} iterations")
    for idx in range(iterations):
        profiler.start("inference_time")
        profiler.start(f"inference_time_{idx}")
        ttnn_output_tensor = ttnn_model(ttnn_input)

        profiler.end(f"inference_time_{idx}")
        profiler.end("inference_time")

    mean_inference_time = profiler.get("inference_time")
    inference_time = profiler.get(f"inference_time_{iterations - 1}")
    compile_time = inference_and_compile_time - inference_time
    logger.info(f"Model compilation took {compile_time:.1f} s")
    logger.info(f"Inference time on last iterations was completed in {(inference_time * 1000.0):.2f} ms")
    logger.info(
        f"Mean inference time for {batch_size} (batch) images was {(mean_inference_time * 1000.0):.2f} ms ({batch_size / mean_inference_time:.2f} fps)"
    )

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
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit Yolov8x perf test")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 31],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov8x(batch_size, expected_perf):
    subdir = "ttnn_yolov8x"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 10.94
    device_params_str = "{'l1_small_size': 32768}"
    command = f"pytest tests/ttnn/integration_tests/yolov8x/test_ttnn_yolov8x.py::test_yolov8x_640"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov8x{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
