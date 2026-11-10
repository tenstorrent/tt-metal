# Gate OCR on Blackhole (P150): YOLO11-small + PP-OCRv6 small

Camera frame -> YOLO11-small at 640x640 (vehicles) -> each vehicle cropped from the
full-resolution frame -> PP-OCRv6 small detector + recognizer -> text, one JSON event per frame.
A frame's vehicle crops go through the detector in pairs (batch 2) and all their text lines
through the recognizer together (batches 1/2/4/8), not one call per vehicle.
All three models are traced on one device with two command queues; pre/post-processing is on the host.

- YOLO11-small: `models/demos/yolov11s` (COCO weights; vehicle classes car, motorcycle, bus, truck)
- PP-OCRv6 small: `models/demos/blackhole/pp_ocrv6`

## How to run

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD YOLO_OFFLINE=1
# yolo11s.pt in the working directory; PP-OCRv6 weights from the Hugging Face cache or PPOCRV6_WEIGHTS_DIR
python models/demos/blackhole/gate_ocr/demo/demo.py --images IN_DIR --out OUT_DIR [--make-4k PHOTO_DIR] [--threads 1 2 3]
```

`--make-4k` builds 4096x4096 frames from smaller photos (upscaled, plus a 2x2 collage).
Each frame gets `<name>_result.jpg`, `<name>.json` and a line in the timing table.
`--threads` also measures throughput with that many frames in flight.

## Device setup

Both models share one device. `l1_small_size` is the sum of the two (48 KB + 24 KB, plus 12 KB
for the detector's batch-2 trace), since every traced shape's conv configs stay resident; the
max of the two runs out. Every model's weights are
prepared untraced before any trace is captured (`TtnnBackend.prepare()`, then the YOLO runner,
then `TtnnBackend.capture()`), so no persistent buffer lands in another trace's scratch memory.

The detector is not traced at batch 4: its output goes wrong (PCC 0.1-0.86 against torch, no
error raised) once any L1 buffer is allocated after its first run, which YOLO's resident L1 input
is. Batches 1 and 2 are unaffected. Repro without YOLO: run the detector at batch 4, allocate a
640x640x16 bf16 height-sharded L1 tensor, run it again.

`GateOCR` is thread-safe. One lock is held around each device call and its readback; host
work (resize, NMS, box post-processing, line crops) runs outside it, so with two or more worker
threads one frame's host work overlaps another frame's device work. Readbacks wait on an event
first, which releases the GIL (`from_device` holds it while blocking).

## Measured on 4096x4096 frames (p150a)

Frames are public photos upscaled to 4096x4096, not camera frames. JPEG decode of a frame
(70-105 ms on the host CPU) is not counted; the customer decodes video on the host.

### Throughput

| Frames | Worker threads | Frames/s | ms per frame p50 | p99 |
|---|---|---|---|---|
| One vehicle each (3 frames) | 1 | 45.4 | 21.7 | 25.6 |
| | 2 | 72.2 | 27.6 | 32.9 |
| | 3 | 76.4 | 39.1 | 41.8 |
| | 4 | 77.5 | 51.2 | 56.2 |
| Mix, 1.6 vehicles on average (5 frames) | 1 | 34.5 | 23.8 | 47.3 |
| | 2 | 50.2 | 37.5 | 59.9 |
| | 3 | 55.6 | 47.6 | 86.0 |

Target: ~60 frames/s worst case (3 gates, 12 views, 5 frames/s each), under 500 ms. One vehicle
per frame meets it with two workers; at three or more the device is the limit (~12 ms of device
time per one-vehicle frame: YOLO ~3, detector ~4.3, recognizer ~2-4.6). Frames with several
vehicles are below it: each extra vehicle adds ~4 ms of detector and ~2-3 ms of recognizer.

### One frame at a time (median of 10)

| Frame | Vehicles | YOLO pre | YOLO | NMS | OCR, all crops | Total |
|---|---|---|---|---|---|---|
| car_zaz | 1 | 2.1 ms | 3.9 ms | 0.4 ms | 14.2 ms | 20.8 ms |
| container_CMAU1011386 | 1 | 1.8 ms | 3.9 ms | 0.4 ms | 14.0 ms | 20.0 ms |
| truck_jbhunt | 1 | 2.0 ms | 3.8 ms | 0.4 ms | 16.7 ms | 23.1 ms |
| container_MRKU3158293 | 2 | 2.0 ms | 3.8 ms | 0.4 ms | 26.7 ms | 33.0 ms |
| collage, 4 vehicles | 3 | 2.0 ms | 3.8 ms | 0.5 ms | 35.0 ms | 41.3 ms |

OCR per call, including host input and readback: detector ~6.2 ms at batch 1 and ~11.2 ms at
batch 2; recognizer 2.8-7.9 ms (batch bucket by line count); host post-processing and line
crops 2.8-5.3 ms.

From the first version of this pipeline: YOLO 5.5 -> 3.9 ms (bf16 cast in torch before
`from_torch`, event wait before readback); line crops area-downscaled to twice the recognizer
height before the perspective warp (full-frame close-up 155 -> 19 ms of post-processing,
reads unchanged on plates and numbers); no copy of the vehicle crop.

Reads: the plate `AE 6133 CT` (0.97) in its own frame and in the collage, `MRKU` `315829`
and `MAERSK`, `CA388-632`, `Intermodal`. Missed: vertical container numbers, the rail wagon
(COCO `train` is not a vehicle class here), close-ups with no vehicle (full-frame fallback reads
`777` but not `ZPK`).

## Next steps

- [ ] Customer-tuned YOLO with a text-region class, so PP-OCR gets tight crops (smaller detector input, ~3.5 ms saved per crop)
- [ ] Device time for frames with several vehicles: detector batch 4 (see Device setup), neck reshards and the 160x160 depthwise conv in the detector
- [x] Batch a frame's crops: detector in pairs, recognizer over all lines of the frame. Mix 47.4 -> 50.2 frames/s at 2 workers, 49.5 -> 55.6 at 3 (p99 105 -> 86 ms); also includes the detector op fixes (5.1 -> 4.3 ms)
- [x] Lower precision (bfp8 weights, HiFi2 / LoFi): at most 0.35 ms saved and some reads changed, so kept at HiFi4 / bf16
- [ ] Vertical text
- [ ] Load test with arrival timing: 8-12 cameras at 2-5 frames/s per view, p50 / p99 from frame arrival against 500 ms
- [x] Overlap host and device work across frames (worker threads, one device lock): 31 -> 47 frames/s on the mix, 44 -> 68 on one-vehicle frames
- [x] YOLO input conversion and readback: 5.5 -> 3.9 ms
- [x] Line crops at recognizer scale instead of full resolution
