# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Gate OCR on one Blackhole chip: camera frame -> YOLO11-small (vehicles) -> crop from the
# full-resolution frame -> PP-OCRv6 small detector + recognizer -> text.
# All three models are traced on the same device; pre/post-processing runs on the host.

import time
from dataclasses import dataclass, field

import numpy as np
import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.common import DET_L1_SMALL_SIZE, TRACE_REGION_SIZE
from models.demos.blackhole.pp_ocrv6.demo.pipeline import PPOCRv6Pipeline, TtnnBackend
from models.demos.utils.common_demo_utils import non_max_suppression, preprocess, scale_boxes
from models.demos.yolov11s.common import YOLOV11S_L1_SMALL_SIZE
from models.demos.yolov11s.runner.performant_runner import YOLOv11sPerformantRunner

YOLO_SIZE = 640
YOLO_TRACE_REGION_SIZE = 6434816
DEVICE_PARAMS = {
    "l1_small_size": DET_L1_SMALL_SIZE + YOLOV11S_L1_SMALL_SIZE,  # both models' conv configs stay resident
    "trace_region_size": TRACE_REGION_SIZE + YOLO_TRACE_REGION_SIZE,
    "num_command_queues": 2,
}
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids
YOLO_CONF = 0.25
YOLO_IOU = 0.7
CROP_MARGIN = 0.02  # of the box size, so text on the vehicle's edge is kept


@dataclass
class VehicleRead:
    label: str
    confidence: float
    box: tuple  # x1, y1, x2, y2 in frame pixels
    lines: list  # TextLine, boxes in frame pixels
    timings_ms: dict


@dataclass
class FrameResult:
    vehicles: list
    timings_ms: dict = field(default_factory=dict)


class GateOCR:
    def __init__(self, device, full_frame_fallback=True):
        self.device = device
        self.full_frame_fallback = full_frame_fallback
        # Both models' weights are prepared before either captures a trace (see TtnnBackend.prepare).
        backend = TtnnBackend(device, traced=False)
        backend.prepare()
        self.yolo = YOLOv11sPerformantRunner(
            device, 1, act_dtype=ttnn.bfloat8_b, weight_dtype=ttnn.bfloat8_b, resolution=(YOLO_SIZE, YOLO_SIZE)
        )
        backend.capture()
        self.ocr = PPOCRv6Pipeline(backend)

    def detect_vehicles(self, frame_bgr):
        x = preprocess([frame_bgr], res=(YOLO_SIZE, YOLO_SIZE))
        t_pre = time.perf_counter()
        preds = ttnn.to_torch(self.yolo.run(x), dtype=torch.float32)
        t_yolo = time.perf_counter()
        det = non_max_suppression(preds, YOLO_CONF, YOLO_IOU)[0]
        det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], frame_bgr.shape)
        keep = [int(c) in VEHICLE_CLASSES for c in det[:, 5]]
        return det[keep], t_pre, t_yolo

    def _read(self, frame_bgr, label, confidence, box):
        start = time.perf_counter()
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
        h, w = frame_bgr.shape[:2]
        x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
        crop = np.ascontiguousarray(frame_bgr[y1:y2, x1:x2])
        crop_ms = (time.perf_counter() - start) * 1000
        result = self.ocr(crop)
        for line in result.lines:
            line.box = line.box + np.array([x1, y1], np.float32)
        timings = {"crop": crop_ms, **{f"ocr_{k}": v for k, v in result.timings_ms.items()}}
        timings["total"] = crop_ms + result.timings_ms["total"]
        return VehicleRead(label, confidence, (x1, y1, x2, y2), result.lines, timings)

    def __call__(self, frame_bgr):
        start = time.perf_counter()
        det, t_pre, t_yolo = self.detect_vehicles(frame_bgr)
        t_nms = time.perf_counter()

        vehicles = [
            self._read(frame_bgr, VEHICLE_CLASSES[int(c)], float(conf), tuple(float(v) for v in xyxy))
            for *xyxy, conf, c in det[:, :6].tolist()
        ]
        if not vehicles and self.full_frame_fallback:
            h, w = frame_bgr.shape[:2]
            vehicles = [self._read(frame_bgr, "no vehicle: full frame", 0.0, (0, 0, w, h))]
        end = time.perf_counter()

        timings = {
            "yolo_pre": (t_pre - start) * 1000,
            "yolo": (t_yolo - t_pre) * 1000,
            "yolo_nms": (t_nms - t_yolo) * 1000,
            "ocr_all_crops": (end - t_nms) * 1000,
            "total": (end - start) * 1000,
        }
        return FrameResult(vehicles, timings)

    def release(self):
        self.yolo.release()
        for runner in self.ocr.backend.runners.values():
            runner.release()
