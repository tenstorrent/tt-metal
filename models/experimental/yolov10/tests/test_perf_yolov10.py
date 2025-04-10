# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import torch
import os
import sys
import torch.nn as nn
from pathlib import Path
import pytest
from models.perf.perf_utils import prep_perf_report
from loguru import logger
from models.utility_functions import run_for_wormhole_b0
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.experimental.yolov10.tt.model_preprocessing import (
    create_yolov10x_input_tensors,
    create_yolov10x_model_parameters,
)
from models.experimental.yolov10.reference.yolov10 import YOLOv10
from models.experimental.yolov10.tt.yolov10 import TtnnYolov10
from models.experimental.yolov10.reference import yolov10

try:
    sys.modules["ultralytics"] = yolov10
    sys.modules["ultralytics.nn.tasks"] = yolov10
    sys.modules["ultralytics.nn.modules.conv"] = yolov10
    sys.modules["ultralytics.nn.modules.block"] = yolov10
    sys.modules["ultralytics.nn.modules.head"] = yolov10
except KeyError:
    logger.error("models.experimental.yolov10x.reference.YoloV10x not found.")


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
    tests = Path(__file__).parent.parent / "tests"
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolov10x.pt"  # file_path.name
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"

        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"

            print(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            # Validate the file
            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"

        except Exception as e:
            print(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            print(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                print(f"ERROR: Download failure for {msg}")
            else:
                print(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
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
    base = {"yolov10x": (159.66, 10.38)}
    return base[name]


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        "False",
        # "True",
    ],
)
def test_perf(device, use_pretrained_weight):
    disable_persistent_kernel_cache()
    torch_input, ttnn_input = create_yolov10x_input_tensors(device)
    batch_size = torch_input.shape[0]
    state_dict = None

    if use_pretrained_weight:
        torch_model = attempt_load("yolov10x.pt", map_location="cpu")
        state_dict = torch_model.state_dict()

    torch_model = YOLOv10()
    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device)

    torch_model_output = torch_model(torch_input)[0]
    ttnn_model = TtnnYolov10(
        device=device,
        parameters=parameters,
        conv_pt=parameters,
    )

    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output = ttnn_model(ttnn_input)
        end = time.time()
        durations.append(end - start)
        ttnn.deallocate(ttnn_model_output)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov10x")

    prep_perf_report(
        model_name="models/experimental/yolov10",
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


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 10.81],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov10x(batch_size, expected_perf):
    subdir = "ttnn_yolov10"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/yolov10/test_ttnn_yolov10.py::test_yolov10x"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov10x{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
