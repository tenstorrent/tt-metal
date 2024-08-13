# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from loguru import logger


import ttnn
from models.experimental.yolov3.reference.models.common import DetectMultiBackend
from models.experimental.yolov3.tt.yolov3_detection_model import TtDetectionModel
from models.experimental.yolov3.reference.models.common import autopad
from models.experimental.yolov3.reference.utils.dataloaders import LoadImages
from models.experimental.yolov3.reference.utils.general import check_img_size
from models.utility_functions import (
    torch2tt_tensor,
    profiler,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

BATCH_SIZE = 1


def test_perf(device, use_program_cache, model_location_generator):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "yolov3-fused"

    torch.manual_seed(1234)

    # Load yolo
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)
    state_dict = reference_model.state_dict()
    reference_model = reference_model.model

    tt_module = TtDetectionModel(
        cfg=model_config_path,
        state_dict=state_dict,
        base_address="model.model",
        device=device,
    )

    with torch.no_grad():
        tt_module.eval()
        reference_model.eval()

        # Load data
        stride = max(int(max(reference_model.stride)), 32)  # model stride
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

        path, im, _, _, _ = next(iter(dataset))
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        tt_im = torch2tt_tensor(im, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        profiler.start(cpu_key)
        pt_out = reference_model(im)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_out = tt_module(tt_im)
        profiler.end(first_key)

        profiler.start(second_key)
        tt_out = tt_module(tt_im)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    # TODO: expected compile time (100 s) and expected inference time (100 s) are not real values
    # update to real time and add to CI pipeline
    prep_perf_report("yolov3", BATCH_SIZE, first_iter_time, second_iter_time, 100, 100, comments, cpu_time)
