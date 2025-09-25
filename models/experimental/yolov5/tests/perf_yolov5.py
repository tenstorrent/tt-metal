# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.tt.yolov5_detection_model import (
    yolov5s_detection_model,
)
from models.utility_functions import (
    torch2tt_tensor,
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

BATCH_SIZE = 1


def test_perf(device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    refence_model = DetectMultiBackend(
        weights="models/experimental/yolov5/reference/yolov5s.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
    )

    refence_module = refence_model.model
    tt_model = yolov5s_detection_model(device)

    torch.manual_seed(0)
    test_input = torch.rand(1, 3, 640, 480)
    tt_inputs = torch2tt_tensor(test_input, device)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = refence_module(test_input)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_inputs)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_inputs)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    # TODO: expected compile time (100 s) and expected inference time (100 s) are not real values
    # update to real time and add to CI pipeline
    prep_perf_report("yolov5", BATCH_SIZE, first_iter_time, second_iter_time, 100, 100, "yolov5s", cpu_time)
