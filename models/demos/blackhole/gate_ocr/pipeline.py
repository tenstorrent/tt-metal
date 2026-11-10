# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Gate OCR on one Blackhole chip: camera frame -> YOLO11-small (vehicles) -> crop from the
# full-resolution frame -> PP-OCRv6 small detector + recognizer -> text.
# All three models are traced on the same device; pre/post-processing runs on the host.

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.common import DET_L1_SMALL_SIZE, TRACE_REGION_SIZE
from models.demos.blackhole.pp_ocrv6.demo.pipeline import PPOCRv6Pipeline, TtnnBackend
from models.demos.utils.common_demo_utils import letterbox, non_max_suppression, scale_boxes
from models.demos.yolov11s.common import YOLOV11S_L1_SMALL_SIZE
from models.demos.yolov11s.runner.performant_runner import YOLOv11sPerformantRunner

YOLO_SIZE = 640
YOLO_TRACE_REGION_SIZE = 6434816
# PP-OCR detector batches; a frame's vehicle crops go through in pairs. Batch 4 reads wrong once an
# L1 buffer (YOLO's input) is allocated after its first run, so it is not traced.
DET_BATCHES = (1, 2)
DET_EXTRA_BATCH_L1_SMALL = 12 << 10  # conv and pool configs of one more detector batch size
DEVICE_PARAMS = {
    # Every traced shape's conv configs stay resident, for both models.
    "l1_small_size": DET_L1_SMALL_SIZE + YOLOV11S_L1_SMALL_SIZE + DET_EXTRA_BATCH_L1_SMALL * (len(DET_BATCHES) - 1),
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


@dataclass
class FrameResult:
    vehicles: list
    timings_ms: dict = field(default_factory=dict)


class GateOCR:
    def __init__(self, device, full_frame_fallback=True):
        self.device = device
        self.full_frame_fallback = full_frame_fallback
        # One lock around every device call and its readback: frames can be processed on several
        # threads, and the host work of one overlaps the device work of another.
        self.lock = threading.Lock()
        # Both models' weights are prepared before either captures a trace (see TtnnBackend.prepare).
        backend = TtnnBackend(device, traced=False, det_batches=DET_BATCHES)
        backend.prepare()
        self.yolo = YOLOv11sPerformantRunner(
            device, 1, act_dtype=ttnn.bfloat8_b, weight_dtype=ttnn.bfloat8_b, resolution=(YOLO_SIZE, YOLO_SIZE)
        )
        backend.capture()
        backend.lock = self.lock
        self.ocr = PPOCRv6Pipeline(backend)

    def _yolo_host_input(self, frame_bgr):
        img = letterbox(frame_bgr, (YOLO_SIZE, YOLO_SIZE))
        x = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)))[None]
        # torch's bf16 cast is much faster than from_torch converting float32.
        x = (x.float() / 255).to(torch.bfloat16)
        return ttnn.from_host_shards(
            [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)], self.device.shape
        )

    def detect_vehicles(self, frame_bgr):
        host = self._yolo_host_input(frame_bgr)
        t_pre = time.perf_counter()
        with self.lock:
            out = self.yolo.run(tt_inputs_host=host)
            ttnn.event_synchronize(ttnn.record_event(self.device, 0))  # releases the GIL while waiting
            preds = ttnn.to_torch(out, dtype=torch.float32)
        t_yolo = time.perf_counter()
        det = non_max_suppression(preds, YOLO_CONF, YOLO_IOU)[0]
        det[:, :4] = scale_boxes((YOLO_SIZE, YOLO_SIZE), det[:, :4], frame_bgr.shape)
        keep = [int(c) in VEHICLE_CLASSES for c in det[:, 5]]
        return det[keep], t_pre, t_yolo

    @staticmethod
    def _crop_box(frame_bgr, box):
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
        h, w = frame_bgr.shape[:2]
        return int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, x2 + mx)), int(min(h, y2 + my))

    def __call__(self, frame_bgr):
        start = time.perf_counter()
        det, t_pre, t_yolo = self.detect_vehicles(frame_bgr)
        t_nms = time.perf_counter()

        found = [(VEHICLE_CLASSES[int(c)], float(conf), xyxy) for *xyxy, conf, c in det[:, :6].tolist()]
        if not found and self.full_frame_fallback:
            h, w = frame_bgr.shape[:2]
            found = [("no vehicle: full frame", 0.0, (0, 0, w, h))]
        boxes = [self._crop_box(frame_bgr, xyxy) for _, _, xyxy in found]
        # Views into the frame; cv2 reads row-strided arrays without a copy.
        results = self.ocr.read_many([frame_bgr[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes])
        vehicles = []
        for (label, conf, _), box, result in zip(found, boxes, results):
            for line in result.lines:
                line.box = line.box + np.array(box[:2], np.float32)
            vehicles.append(VehicleRead(label, conf, box, result.lines))
        end = time.perf_counter()

        timings = {
            "yolo_pre": (t_pre - start) * 1000,
            "yolo": (t_yolo - t_pre) * 1000,
            "yolo_nms": (t_nms - t_yolo) * 1000,
        }
        if results:
            timings.update({f"ocr_{k}": v for k, v in results[0].timings_ms.items()})
        timings["total"] = (end - start) * 1000
        return FrameResult(vehicles, timings)

    def release(self):
        self.yolo.release()
        for runner in self.ocr.backend.runners.values():
            runner.release()
