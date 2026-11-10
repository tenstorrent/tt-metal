# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Gate OCR demo on 4096x4096 frames.

python models/demos/blackhole/gate_ocr/demo/demo.py --images DIR --out DIR [--make-4k SRC_DIR]

--make-4k builds the 4096x4096 inputs from smaller photos: each photo upscaled onto its own
frame, plus one 2x2 collage so a frame holds several vehicles. Per frame it writes an annotated
image, one JSON event (what would go to the yard management system) and the stage timings.
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import ttnn
from models.demos.blackhole.gate_ocr.pipeline import DEVICE_PARAMS, GateOCR

FRAME = 4096
PAD = (114, 114, 114)
RUNS = 10
THROUGHPUT_FRAMES = 200


def fit_on_frame(bgr, size):
    scale = min(size / bgr.shape[0], size / bgr.shape[1])
    r = cv2.resize(bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    out = np.full((size, size, 3), PAD, np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y : y + r.shape[0], x : x + r.shape[1]] = r
    return out


def make_4k_inputs(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    photos = sorted(p for p in glob.glob(os.path.join(src_dir, "*.jpg")) if "_result" not in p)
    for p in photos:
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(
            os.path.join(dst_dir, f"{name}_4096.jpg"),
            fit_on_frame(cv2.imread(p), FRAME),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    tiles = [cv2.imread(p) for p in photos if "lithuania" not in p][:4]
    half = FRAME // 2
    collage = np.full((FRAME, FRAME, 3), PAD, np.uint8)
    for i, t in enumerate(tiles):
        y, x = (i // 2) * half, (i % 2) * half
        collage[y : y + half, x : x + half] = fit_on_frame(t, half)
    cv2.imwrite(os.path.join(dst_dir, "collage_4vehicles_4096.jpg"), collage, [cv2.IMWRITE_JPEG_QUALITY, 95])


def draw(frame, result):
    out = frame.copy()
    for v in result.vehicles:
        x1, y1, x2, y2 = map(int, v.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 12)
        cv2.putText(
            out, f"{v.label} {v.confidence:.2f}", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 8
        )
        for line in v.lines:
            pts = line.box.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, (0, 200, 0), 8)
            tx, ty = int(line.box[:, 0].min()), int(line.box[:, 1].min()) - 15
            cv2.putText(
                out,
                f"{line.text} {line.confidence:.2f}",
                (tx, max(60, ty)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 0, 255),
                7,
            )
    return cv2.resize(out, (FRAME // 2, FRAME // 2), interpolation=cv2.INTER_AREA)


def throughput(gate, frames, workers, count):
    """Frames per second and per-frame latency (p50, p99 ms) with `workers` frames in flight."""

    def one(frame):
        start = time.perf_counter()
        gate(frame)
        return (time.perf_counter() - start) * 1000

    work = [frames[i % len(frames)] for i in range(count)]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, work[: 2 * workers]))
        start = time.perf_counter()
        latencies = list(pool.map(one, work))
        wall = time.perf_counter() - start
    return count / wall, float(np.percentile(latencies, 50)), float(np.percentile(latencies, 99))


def median_timings(samples):
    return {k: float(np.median([s[k] for s in samples])) for k in samples[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--make-4k", dest="make_4k")
    ap.add_argument("--threads", type=int, nargs="*", help="also measure throughput with these worker counts")
    args = ap.parse_args()
    if args.make_4k:
        make_4k_inputs(args.make_4k, args.images)
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))

    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        t0 = time.perf_counter()
        gate = GateOCR(device)
        print(f"setup (weights + traces for YOLO, det, rec): {time.perf_counter() - t0:.1f} s", flush=True)

        rows, with_vehicle = [], []
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            t_load = time.perf_counter()
            frame = cv2.imread(path)
            load_ms = (time.perf_counter() - t_load) * 1000

            result = gate(frame)  # warm-up for this frame's host-side shapes
            frames = [gate(frame) for _ in range(RUNS)]
            result = frames[-1]
            t = median_timings([f.timings_ms for f in frames])

            event = {
                "frame": os.path.basename(path),
                "frame_size": [frame.shape[1], frame.shape[0]],
                "reads": [
                    {
                        "vehicle": v.label,
                        "vehicle_confidence": round(v.confidence, 3),
                        "vehicle_box": [round(c) for c in v.box],
                        "text": [{"text": l.text, "confidence": round(l.confidence, 3)} for l in v.lines],
                    }
                    for v in result.vehicles
                ],
                "latency_ms": {k: round(val, 2) for k, val in t.items()},
                "jpeg_decode_ms": round(load_ms, 1),
            }
            with open(os.path.join(args.out, f"{name}.json"), "w") as f:
                json.dump(event, f, indent=2)
            cv2.imwrite(
                os.path.join(args.out, f"{name}_result.jpg"), draw(frame, result), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            crops = len(result.vehicles)
            print(f"\n== {name} ({frame.shape[1]}x{frame.shape[0]}) ==", flush=True)
            for v in result.vehicles:
                texts = ", ".join(f"{l.text!r} {l.confidence:.2f}" for l in v.lines) or "-"
                print(f"  {v.label} {v.confidence:.2f} box {[round(c) for c in v.box]}: {texts}", flush=True)
            print(f"  frame ms (median of {RUNS}): " + ", ".join(f"{k} {val:.2f}" for k, val in t.items()), flush=True)
            rows.append((name, crops, t, load_ms))
            if not any(v.label.startswith("no vehicle") for v in result.vehicles):
                with_vehicle.append((name, path))

        print("\n| frame | crops | YOLO pre | YOLO | NMS | OCR all crops | total | JPEG decode |", flush=True)
        print("|---|---|---|---|---|---|---|---|", flush=True)
        for name, crops, t, load_ms in rows:
            print(
                f"| {name} | {crops} | {t['yolo_pre']:.1f} | {t['yolo']:.1f} | {t['yolo_nms']:.1f} | "
                f"{t['ocr_total']:.1f} | {t['total']:.1f} | {load_ms:.0f} |",
                flush=True,
            )

        if args.threads:
            # Frames with a vehicle only: an idle gate sends nothing.
            loaded = [(n, cv2.imread(p)) for n, p in with_vehicle]
            print(f"\nthroughput over {[n for n, _ in loaded]}, {THROUGHPUT_FRAMES} frames per run", flush=True)
            print("| worker threads | frames/s | ms per frame p50 | p99 |", flush=True)
            print("|---|---|---|---|", flush=True)
            for n in args.threads:
                fps, p50, p99 = throughput(gate, [f for _, f in loaded], n, THROUGHPUT_FRAMES)
                print(f"| {n} | {fps:.1f} | {p50:.1f} | {p99:.1f} |", flush=True)
        gate.release()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
