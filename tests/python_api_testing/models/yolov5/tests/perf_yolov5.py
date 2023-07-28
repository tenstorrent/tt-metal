from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib

from python_api_testing.models.yolov5.reference.models.common import DetectMultiBackend
from python_api_testing.models.yolov5.tt.yolov5_detection_model import (
    yolov5s_detection_model,
)
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    prep_report,
)

BATCH_SIZE = 1


def test_perf():
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    refence_model = DetectMultiBackend(
        weights="tests/python_api_testing/models/yolov5/reference/yolov5s.pt",
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

    prep_report(
        "yolov5", BATCH_SIZE, first_iter_time, second_iter_time, "yolov5s", cpu_time
    )
    tt_lib.device.CloseDevice(device)
