# Gate OCR on Blackhole (P150): YOLO11-small + PP-OCRv6 small

Camera frame -> YOLO11-small at 640x640 (vehicles) -> each vehicle cropped from the
full-resolution frame -> PP-OCRv6 small detector + recognizer -> text, one JSON event per frame.
A frame's vehicle crops go through the detector together (batches 1/2/4, split without padding)
and all their text lines through the recognizer together (batches 1/2/4/8, padded).
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
per extra detector batch), since every traced shape's conv configs stay resident; the max of
the two runs out. Every model's weights are
prepared untraced before any trace is captured (`TtnnBackend.prepare()`, then the YOLO runner,
then `TtnnBackend.capture()`), so no persistent buffer lands in another trace's scratch memory.

Each PP-OCR conv fixes its slicing on its first run (see `Conv` in `pp_ocrv6/tt/common.py`).
conv2d's own choice for a DRAM input follows the L1 free at each call; once YOLO's L1 input
was allocated, the detector at batch 4 switched slicing but kept weights prepared for the old
one, and read wrong with no error. The fixed choice is made while a stand-in for YOLO's L1
input is allocated, so it holds after YOLO starts.

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
| One vehicle each (3 frames) | 1 | 47.7 | 20.6 | 25.6 |
| | 2 | 76.0 | 26.1 | 32.9 |
| | 3 | 81.5 | 36.4 | 40.1 |
| | 4 | 81.6 | 48.7 | 53.9 |
| Mix, 1.6 vehicles on average (5 frames) | 1 | 35.8 | 23.2 | 44.3 |
| | 2 | 52.9 | 34.7 | 60.0 |
| | 3 | 60.0 | 44.7 | 78.0 |

Target: ~60 frames/s worst case (3 gates, 12 views, 5 frames/s each), under 500 ms. One vehicle
per frame meets it with two workers; at three or more the device is the limit (~12 ms of device
time per one-vehicle frame: YOLO ~3, detector ~3.9, recognizer ~1.9-4). The mix reaches it with
three workers; each extra vehicle adds ~3.5 ms of detector and ~2 ms of recognizer.

### One frame at a time (median of 10)

| Frame | Vehicles | YOLO pre | YOLO | NMS | OCR, all crops | Total |
|---|---|---|---|---|---|---|
| car_zaz | 1 | 2.0 ms | 3.8 ms | 0.4 ms | 12.7 ms | 18.8 ms |
| container_CMAU1011386 | 1 | 2.1 ms | 4.0 ms | 0.5 ms | 14.4 ms | 21.3 ms |
| truck_jbhunt | 1 | 2.2 ms | 3.9 ms | 0.5 ms | 16.0 ms | 22.9 ms |
| container_MRKU3158293 | 2 | 2.3 ms | 4.0 ms | 0.5 ms | 27.9 ms | 35.1 ms |
| collage, 4 vehicles | 3 | 2.0 ms | 3.8 ms | 0.5 ms | 33.6 ms | 39.9 ms |

PP-OCR device replay: detector 3.9 ms at batch 1; recognizer 1.85 ms at batch 1, 4.0 ms at
batch 8. Per call, host input and readback add ~2 ms to the detector.

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
- [ ] Device time for frames with several vehicles: neck reshards and the 160x160 depthwise conv in the detector
- [x] Detector batch 4, after fixing each conv's slicing (it read wrong once YOLO's L1 input existed). With preparation now on device inputs: detector 4.27 -> 3.91 ms, recognizer batch 8 4.62 -> 4.03 ms; mix 50.2 -> 52.9 frames/s at 2 workers, 55.6 -> 60.0 at 3; one vehicle 72 -> 76 at 2
- [x] Batch a frame's crops: detector in pairs, recognizer over all lines of the frame. Mix 47.4 -> 50.2 frames/s at 2 workers, 49.5 -> 55.6 at 3 (p99 105 -> 86 ms); also includes the detector op fixes (5.1 -> 4.3 ms)
- [x] Lower precision (bfp8 weights, HiFi2 / LoFi): at most 0.35 ms saved and some reads changed, so kept at HiFi4 / bf16
- [ ] Vertical text
- [ ] Load test with arrival timing: 8-12 cameras at 2-5 frames/s per view, p50 / p99 from frame arrival against 500 ms
- [x] Overlap host and device work across frames (worker threads, one device lock): 31 -> 47 frames/s on the mix, 44 -> 68 on one-vehicle frames
- [x] YOLO input conversion and readback: 5.5 -> 3.9 ms
- [x] Line crops at recognizer scale instead of full resolution
