# Gate OCR on Blackhole (P150): YOLO11-small + PP-OCRv6 small

Camera frame -> YOLO11-small at 640x640 (vehicles) -> each vehicle cropped from the
full-resolution frame -> PP-OCRv6 small detector + recognizer -> text, one JSON event per frame.
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

Both models share one device. `l1_small_size` is the sum of the two (32 KB + 24 KB), since every
model's conv configs stay resident; the max of the two runs out. Every model's weights are
prepared untraced before any trace is captured (`TtnnBackend.prepare()`, then the YOLO runner,
then `TtnnBackend.capture()`), so no persistent buffer lands in another trace's scratch memory.

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
| One vehicle each (3 frames) | 1 | 43.8 | 22.2 | 26.5 |
| | 2 | 68.2 | 29.6 | 35.5 |
| | 3 | 71.8 | 41.3 | 45.5 |
| Mix, 1.6 vehicles on average (5 frames) | 1 | 31.1 | 25.2 | 55.4 |
| | 2 | 47.4 | 35.6 | 71.5 |
| | 3 | 49.5 | 47.0 | 105.0 |

Target: ~60 frames/s worst case (3 gates, 12 views, 5 frames/s each), under 500 ms. One vehicle
per frame meets it with two workers; at three or more the device is the limit (~13 ms of device
time per one-vehicle frame: YOLO ~3, detector ~5, recognizer ~2.5-5).

### One frame at a time (median of 10)

| Frame | Vehicles | YOLO pre | YOLO | NMS | OCR, all crops | Total |
|---|---|---|---|---|---|---|
| car_zaz | 1 | 1.5 ms | 3.9 ms | 0.4 ms | 13.9 ms | 19.7 ms |
| container_CMAU1011386 | 1 | 1.7 ms | 3.9 ms | 0.5 ms | 15.9 ms | 22.0 ms |
| truck_jbhunt | 1 | 1.6 ms | 3.9 ms | 0.4 ms | 18.3 ms | 24.3 ms |
| container_MRKU3158293 | 2 | 1.7 ms | 3.9 ms | 0.4 ms | 32.4 ms | 38.3 ms |
| collage, 4 vehicles | 3 | 1.6 ms | 3.9 ms | 0.5 ms | 44.2 ms | 50.3 ms |

Per crop: detector ~7 ms, recognizer 3-5.5 ms (batch bucket by line count), host
post-processing and line crops 1-5.5 ms.

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
- [ ] Device time for frames with several vehicles: batch the detector across crops, bfp8 / lower fidelity in the detector
- [ ] Vertical text
- [ ] Load test with arrival timing: 8-12 cameras at 2-5 frames/s per view, p50 / p99 from frame arrival against 500 ms
- [x] Overlap host and device work across frames (worker threads, one device lock): 31 -> 47 frames/s on the mix, 44 -> 68 on one-vehicle frames
- [x] YOLO input conversion and readback: 5.5 -> 3.9 ms
- [x] Line crops at recognizer scale instead of full resolution
