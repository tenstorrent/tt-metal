# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import pytest
import numpy as np

from pathlib import Path
from loguru import logger
from collections import defaultdict

from models.perf.perf_utils import prep_perf_report
from models.experimental.yolov3.reference.models.common import DetectMultiBackend
from models.experimental.yolov3.tt.yolov3_detection_model import TtDetectionModel
from models.experimental.yolov3.reference.utils.dataloaders import LoadImages
from models.utility_functions import (
    torch2tt_tensor,
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.experimental.yolov3.reference.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
    check_img_size,
)
from models.experimental.yolov3.reference.utils.metrics import *


BATCH_SIZE = 1


def run_perf_yolov3(expected_inference_time, expected_compile_time, model_location_generator, device, iterations):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"
    comments = "yolov3-fused"

    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)
    state_dict = reference_model.state_dict()
    reference_model = reference_model.model
    reference_model.eval()

    tt_module = TtDetectionModel(
        cfg=model_config_path,
        state_dict=state_dict,
        base_address="model.model",
        device=device,
    )
    tt_module.eval()
    stride = max(int(max(reference_model.stride)), 32)
    imgsz = check_img_size((640, 640), s=stride)
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)

    path, im, _, _, _ = next(iter(dataset))
    im = torch.from_numpy(im)
    im = im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]

    tt_im = torch2tt_tensor(im, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

    with torch.no_grad():
        profiler.start(cpu_key)
        pt_out = reference_model(im)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_out = tt_module(tt_im)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_out

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_out = tt_module(tt_im)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_out

        data_images_path = "/mnt/MLPerf/tt_dnn-models/ssd/coco128/images/train2017"
        data_labels_path = "/mnt/MLPerf/tt_dnn-models/ssd/coco128/labels/train2017"
        image_files = os.listdir(data_images_path)
        iteration = 0
        ap_list = []
        all_predictions = []

        profiler.start(third_key)
        while iteration < iterations:
            image_file = image_files[iteration]
            image_path = os.path.join(data_images_path, image_file)
            dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=True)
            names = reference_model.names

            path, im, im0s, _, s = next(iter(dataset))
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

                pred = non_max_suppression(
                    prediction=pred,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    classes=classes,
                    agnostic=agnostic_nms,
                    max_det=1000,
                )

                for i, det in enumerate(pred):
                    s += "%gx%g " % im.shape[2:]
                    gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]

                    if len(det):
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                        class_confidence = defaultdict(float)
                        class_bbox = {}

                        for bbox in det:
                            label = int(bbox[5])
                            confidence = float(bbox[4])
                            bbox_info = {
                                "x_center": float(bbox[0]),
                                "y_center": float(bbox[1]),
                                "width": float(bbox[2] - bbox[0]),
                                "height": float(bbox[3] - bbox[1]),
                            }

                            if confidence > class_confidence[label]:
                                class_confidence[label] = confidence
                                class_bbox[label] = bbox_info

                        for *xyxy, conf, cls in reversed(det):
                            if True:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

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

    prep_perf_report(
        model_name="yolov3",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time

    logger.info(f"yolov3 mAP: {mAP}")
    logger.info(f"yolov3 {comments} inference time: {second_iter_time}")
    logger.info(f"yolov3 compile time: {compile_time}")
    logger.info(f"yolov3 inference time for {iterations} Samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            5.86,
            9.47,
            10,
        ),
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    model_location_generator,
    iterations,
    reset_seeds,
):
    run_perf_yolov3(expected_inference_time, expected_compile_time, model_location_generator, device, iterations)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time, iterations",
    (
        (
            5.8,
            0.7,
            10,
        ),
    ),
)
def test_perf_virtual_machine(
    device,
    use_program_cache,
    expected_inference_time,
    expected_compile_time,
    model_location_generator,
    iterations,
    reset_seeds,
):
    run_perf_yolov3(expected_inference_time, expected_compile_time, model_location_generator, device, iterations)
