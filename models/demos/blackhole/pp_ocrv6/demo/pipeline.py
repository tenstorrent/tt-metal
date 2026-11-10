# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# PP-OCRv6 small text pipeline: crop -> detector -> boxes -> recognizer -> text.
# Pre/post-processing runs on the host; detector and recognizer run on the chosen backend.

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from models.demos.blackhole.pp_ocrv6.common import (
    REC_HEIGHT,
    REC_WIDTH,
    ctc_decode_indices,
    det_normalize_bgr,
    load_rec_charset,
    load_torch_det,
    load_torch_rec,
    rec_normalize_bgr,
)

DET_SIZE = 640
DB_THRESH = 0.2
DB_BOX_THRESH = 0.45
DB_UNCLIP_RATIO = 1.4
MIN_BOX_SIDE = 3
REC_BATCH_BUCKETS = (1, 2, 4, 8)


@dataclass
class TextLine:
    text: str
    confidence: float
    box: np.ndarray  # 4x2 corner points in input-crop pixels


@dataclass
class Result:
    lines: list
    timings_ms: dict = field(default_factory=dict)


class TorchBackend:
    name = "torch-cpu"

    def __init__(self):
        self.det = load_torch_det()
        self.rec = load_torch_rec()

    @torch.no_grad()
    def detect(self, x):
        return self.det(x)[0, 0]

    @torch.no_grad()
    def recognize(self, x):
        """[B, 3, 48, W] -> (argmax indices [B, T], their probabilities [B, T])."""
        conf, idx = self.rec(x).max(-1)
        return idx, conf


class TtnnBackend:
    """Detector and recognizer on device. traced=True replays one captured trace per input shape."""

    name = "ttnn-p150"

    def __init__(self, device, traced=True):
        import ttnn
        from models.demos.blackhole.pp_ocrv6.tt.ttnn_det import TtPPOCRv6SmallDet
        from models.demos.blackhole.pp_ocrv6.tt.ttnn_rec import TtPPOCRv6SmallRec

        self.ttnn = ttnn
        self.device = device
        self.traced = traced
        self.det = TtPPOCRv6SmallDet(device, load_torch_det())
        self.rec = TtPPOCRv6SmallRec(device, load_torch_rec())
        self.runners = {}
        if traced:
            self.prepare()
            self.capture()

    def _host(self, x):
        # 2x2 space_to_depth (12 channels; the stem's first conv takes it as is), then one
        # folded image row per page: NHWC row-major with few channels has tiny pages, which
        # makes the host-to-device copy ~60x slower. Unpacked on device by _nhwc.
        # torch's own bf16 cast is ~10x faster than letting from_torch convert float32.
        from models.demos.blackhole.pp_ocrv6.tt.common import space_to_depth

        b, c, h, w = x.shape
        x = space_to_depth(x.permute(0, 2, 3, 1)).reshape(1, 1, b * h // 2, w // 2 * 4 * c).to(torch.bfloat16)
        return self.ttnn.from_torch(x, dtype=self.ttnn.bfloat16)

    def _nhwc(self, t, b, h, w):
        return self.ttnn.reshape(t, (b, h // 2, w // 2, 12))

    def _det_fn(self, b, h, w):
        return lambda t: self.det(self._nhwc(t, b, h, w), b, h, w)[0]

    def _rec_fn(self, b):
        return lambda t: self.rec(self._nhwc(t, b, REC_HEIGHT, REC_WIDTH), b, REC_HEIGHT, REC_WIDTH, decode=True)

    def _trace_shapes(self):
        shapes = {("det", 1, DET_SIZE, DET_SIZE): (self._det_fn(1, DET_SIZE, DET_SIZE), (1, 3, DET_SIZE, DET_SIZE))}
        for b in REC_BATCH_BUCKETS:
            shapes[("rec", b)] = (self._rec_fn(b), (b, 3, REC_HEIGHT, REC_WIDTH))
        return {key: (fn, self._host(torch.zeros(shape))) for key, (fn, shape) in shapes.items()}

    def prepare(self):
        """Run every traced shape once untraced, which prepares its conv weights on device.

        Must happen before any trace is captured on the device, by this model or another one
        sharing it: buffers allocated after a capture can land in that trace's scratch memory.
        """
        for fn, host in self._trace_shapes().values():
            fn(host)
        self.ttnn.synchronize_device(self.device)

    def capture(self):
        from models.demos.blackhole.pp_ocrv6.tt.runner import TracedRunner

        for key, (fn, host) in self._trace_shapes().items():
            self.runners[key] = TracedRunner(self.device, fn, host)
        self.traced = True

    def _run(self, key, fn, host):
        if key in self.runners:
            return self.runners[key](host)
        if self.traced:
            raise ValueError(
                f"no trace captured for {key}; traced mode supports det {DET_SIZE}x{DET_SIZE}, rec {REC_BATCH_BUCKETS}"
            )
        return fn(host)

    def detect(self, x):
        from models.demos.blackhole.pp_ocrv6.tt.ttnn_det import unshuffle_probability

        b, _, h, w = x.shape
        out = self._run(("det", b, h, w), self._det_fn(b, h, w), self._host(x))
        return unshuffle_probability(self.ttnn.to_torch(out).float(), b, h // 4, w // 4)[0, 0]

    def recognize(self, x):
        """[B, 3, 48, W] -> (argmax indices [B, T], their probabilities [B, T])."""
        from models.demos.blackhole.pp_ocrv6.tt.ttnn_rec import rec_decode_to_torch

        b = x.shape[0]
        idx, conf = self._run(("rec", b), self._rec_fn(b), self._host(x))
        return rec_decode_to_torch(idx, conf, b, REC_WIDTH // 8)


def letterbox(bgr, size=DET_SIZE):
    """Fit the longest side to `size`, pad bottom/right. Returns padded image and the scale applied."""
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))))
    out = np.zeros((size, size, 3), np.uint8)
    out[: resized.shape[0], : resized.shape[1]] = resized
    return out, scale


def _order_points(pts):
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2], pts[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], np.float32)


def db_boxes(prob, thresh=DB_THRESH, box_thresh=DB_BOX_THRESH, unclip_ratio=DB_UNCLIP_RATIO):
    """Differentiable-binarization post-processing: probability map -> rotated text boxes."""
    prob = prob.numpy() if isinstance(prob, torch.Tensor) else prob
    mask = (prob > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) < MIN_BOX_SIDE:
            continue
        poly = np.zeros_like(mask)
        cv2.fillPoly(poly, [contour], 1)
        if prob[poly > 0].mean() < box_thresh:
            continue
        area = cv2.contourArea(contour)
        dist = area * unclip_ratio / max(cv2.arcLength(contour, True), 1e-6)
        (cx, cy), (rw, rh), angle = rect
        expanded = ((cx, cy), (rw + 2 * dist, rh + 2 * dist), angle)
        if min(expanded[1]) < MIN_BOX_SIDE + 2:
            continue
        boxes.append(_order_points(cv2.boxPoints(expanded)))
    return sorted(boxes, key=lambda b: (round(b[0, 1] / 10), b[0, 0]))


def crop_text_line(bgr, box):
    tl, tr, br, bl = box
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    crop = cv2.warpPerspective(
        bgr, cv2.getPerspectiveTransform(box, dst), (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if h / max(w, 1) >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


def _bucket(n):
    for b in REC_BATCH_BUCKETS:
        if n <= b:
            return b
    return REC_BATCH_BUCKETS[-1]


class PPOCRv6Pipeline:
    def __init__(self, backend, charset=None):
        self.backend = backend
        self.charset = charset or load_rec_charset()

    def __call__(self, bgr):
        t = {}
        start = time.perf_counter()

        img, scale = letterbox(bgr)
        x = det_normalize_bgr(img)
        t["det_pre"] = time.perf_counter()

        prob = self.backend.detect(x)
        t["det"] = time.perf_counter()

        boxes = db_boxes(prob)
        boxes = [np.clip(b / scale, 0, [bgr.shape[1] - 1, bgr.shape[0] - 1]).astype(np.float32) for b in boxes]
        crops = [rec_normalize_bgr(crop_text_line(bgr, b)) for b in boxes]
        t["det_post_rec_pre"] = time.perf_counter()

        steps = []
        i = 0
        while i < len(crops):
            chunk = crops[i : i + REC_BATCH_BUCKETS[-1]]
            bucket = _bucket(len(chunk))
            batch = torch.cat(chunk + [torch.zeros_like(chunk[0])] * (bucket - len(chunk)))
            idx, conf = self.backend.recognize(batch)
            steps.extend(zip(idx[: len(chunk)], conf[: len(chunk)]))
            i += len(chunk)
        t["rec"] = time.perf_counter()

        lines = []
        for box, (idx, conf) in zip(boxes, steps):
            text, conf = ctc_decode_indices(idx, conf, self.charset)
            if text:
                lines.append(TextLine(text, conf, box))
        t["rec_post"] = time.perf_counter()

        timings, prev = {}, start
        for k, v in t.items():
            timings[k] = (v - prev) * 1000
            prev = v
        timings["total"] = (prev - start) * 1000
        return Result(lines, timings)
