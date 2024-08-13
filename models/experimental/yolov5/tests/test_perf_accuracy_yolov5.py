# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import torch
import ttnn
import pytest
import numpy as np

from loguru import logger
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.perf.perf_utils import prep_perf_report
from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.tt.yolov5_detection_model import yolov5s_detection_model
from models.utility_functions import (
    torch2tt_tensor,
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.experimental.yolov5.reference.utils.metrics import ap_per_class
from models.experimental.yolov5.reference.utils.general import check_img_size
from models.experimental.yolov5.reference.utils.dataloaders import LoadImages
from models.experimental.yolov5.reference.utils.general import (
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)


BATCH_SIZE = 1


def run_perf_yolov5s(
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
    device,
):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = f"first_iter"
    second_key = f"second_iter"
    third_key = f"third_iter"
    cpu_key = f"ref_key"
    comments = f"yolov5s"

    refence_model = DetectMultiBackend(
        ROOT / "yolov5s.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
    )

    refence_module = refence_model.model
    tt_module = yolov5s_detection_model(device)

    test_input = torch.rand(1, 3, 640, 480)
    tt_inputs = torch2tt_tensor(test_input, device)

    data_images_path = "/mnt/MLPerf/tt_dnn-models/ssd/coco128/images/train2017"
    data_labels_path = "/mnt/MLPerf/tt_dnn-models/ssd/coco128/labels/train2017"
    image_files = os.listdir(data_images_path)

    stride = max(int(max(refence_module.stride)), 32)
    imgsz = check_img_size((640, 640), s=stride)

    with torch.no_grad():
        tt_module.eval()
        refence_module.eval()

        profiler.start(cpu_key)
        logits = refence_module(test_input)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_module(tt_inputs)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_module(tt_inputs)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_output

        iteration = 0
        ap_list = []
        all_predictions = []

        profiler.start(third_key)
        while iteration < iterations:
            image_file = image_files[iteration]
            image_path = os.path.join(data_images_path, image_file)
            dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=True)
            names = refence_model.names

            for path, im, im0s, _, s in dataset:
                im = torch.from_numpy(im)
                im = im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]

                image_file = Path(path).stem
                label_file = image_file + ".txt"
                label_path = os.path.join(data_labels_path, label_file)

                if os.path.exists(label_path):
                    all_ground_truths = []
                    lines = [l.strip().split() for l in open(label_path, "r").readlines()]
                    reference_labels = [{"class": int(line[0]), "bbox": list(map(float, line[1:]))} for line in lines]
                    gt_boxes = [label["bbox"] for label in reference_labels]
                    gt_classes = [label["class"] for label in reference_labels]
                    all_ground_truths.append(reference_labels)

                    tt_im = torch2tt_tensor(im, device)
                    pred = tt_module(tt_im)

                    conf_thres, iou_thres = 0.25, 0.45
                    classes = None
                    agnostic_nms = False

                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)
                    for i, det in enumerate(pred):
                        s += "%gx%g " % im.shape[2:]
                        gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                            for *xyxy, conf, cls in reversed(det):
                                if True:
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    line = (cls, *xywh)
                                c = int(cls)
                                label = None if False else (f"{names[c]} {conf:.2f}")
                                prediction = {"class": c, "confidence": f"{conf:.2f}", "bbox": xywh}
                                all_predictions.append(prediction)

            iteration += 1

        _, _, _, _, _, ap, _ = ap_per_class(
            tp=[pred["bbox"] for pred in all_predictions],
            conf=[float(pred["confidence"]) for pred in all_predictions],
            pred_cls=[int(pred["class"]) for pred in all_predictions],
            target_cls=gt_classes,
        )
        ap_list.append(ap)

        ttnn.synchronize_device(device)
        profiler.end(third_key)

    mAP = np.mean(ap_list)
    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_perf_report(
        model_name="yolov5s",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"yolov5 mAP: {mAP}")
    logger.info(f"{comments} inference time: {second_iter_time}")
    logger.info(f"yolov5 compile time: {compile_time}")
    logger.info(f"yolov5 inference time for {iterations} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    ((2.5, 7.8, 5),),
)
def test_perf_bare_metal(
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
    device,
    reset_seeds,
):
    run_perf_yolov5s(
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        iterations,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    ((2.3, 0.85, 5),),
)
def test_perf_virtual_machine(
    model_location_generator,
    expected_inference_time,
    expected_compile_time,
    iterations,
    device,
    reset_seeds,
):
    run_perf_yolov5s(
        model_location_generator,
        expected_inference_time,
        expected_compile_time,
        iterations,
        device,
    )
