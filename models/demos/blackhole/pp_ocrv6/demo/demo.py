# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Usage:
#   python models/demos/blackhole/pp_ocrv6/demo/demo.py --image crop.jpg [--backend ttnn|torch] [--out result.jpg]

import argparse

import cv2
import numpy as np

from models.demos.blackhole.pp_ocrv6.common import DET_L1_SMALL_SIZE, synthetic_gate_image
from models.demos.blackhole.pp_ocrv6.demo.pipeline import PPOCRv6Pipeline, TorchBackend, TtnnBackend


def draw(bgr, lines):
    out = bgr.copy()
    for line in lines:
        pts = line.box.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 200, 0), 2)
        cv2.putText(
            out,
            f"{line.text} ({line.confidence:.2f})",
            (int(pts[0, 0]), max(20, int(pts[0, 1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="BGR image (a YOLO text-region crop or a full frame); synthetic if omitted")
    parser.add_argument("--backend", choices=["ttnn", "torch"], default="ttnn")
    parser.add_argument("--out", default="pp_ocrv6_result.jpg")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    bgr = cv2.imread(args.image) if args.image else synthetic_gate_image()
    assert bgr is not None, f"could not read {args.image}"

    device = None
    if args.backend == "ttnn":
        import ttnn

        device = ttnn.open_device(device_id=0, l1_small_size=DET_L1_SMALL_SIZE)
        backend = TtnnBackend(device)
    else:
        backend = TorchBackend()

    try:
        pipeline = PPOCRv6Pipeline(backend)
        pipeline(bgr)
        results = [pipeline(bgr) for _ in range(args.runs)]
        result = results[-1]
        for line in result.lines:
            print(f"{line.text!r}  confidence {line.confidence:.3f}")
        keys = results[0].timings_ms.keys()
        median = {k: sorted(r.timings_ms[k] for r in results)[len(results) // 2] for k in keys}
        print(
            f"[{backend.name}] median ms over {args.runs} runs: " + ", ".join(f"{k} {v:.2f}" for k, v in median.items())
        )
        cv2.imwrite(args.out, draw(bgr, result.lines))
        print(f"wrote {args.out}")
    finally:
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
