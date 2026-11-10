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
python models/demos/blackhole/gate_ocr/demo/demo.py --images IN_DIR --out OUT_DIR [--make-4k PHOTO_DIR]
```

`--make-4k` builds 4096x4096 frames from smaller photos (upscaled, plus a 2x2 collage).
Each frame gets `<name>_result.jpg`, `<name>.json` and a line in the timing table.

## Device setup

Both models share one device. `l1_small_size` is the sum of the two (32 KB + 24 KB), since every
model's conv configs stay resident; the max of the two runs out. Every model's weights are
prepared untraced before any trace is captured (`TtnnBackend.prepare()`, then the YOLO runner,
then `TtnnBackend.capture()`), so no persistent buffer lands in another trace's scratch memory.

## Measured on 4096x4096 frames (p150a, median of 10 runs per frame)

Frames are public photos upscaled to 4096x4096, not camera frames.

| Frame | Vehicles | YOLO pre | YOLO | NMS | OCR, all crops | Total |
|---|---|---|---|---|---|---|
| car_zaz | 1 | 1.7 ms | 5.5 ms | 0.4 ms | 16.9 ms | 24.6 ms |
| container_CMAU1011386 | 1 | 2.0 ms | 5.7 ms | 0.4 ms | 21.6 ms | 29.8 ms |
| truck_jbhunt | 1 | 1.8 ms | 5.7 ms | 0.4 ms | 19.6 ms | 27.5 ms |
| container_MRKU3158293 | 2 | 1.8 ms | 5.5 ms | 0.4 ms | 41.9 ms | 49.7 ms |
| collage, 4 vehicles | 3 | 1.8 ms | 5.6 ms | 0.5 ms | 45.0 ms | 52.7 ms |

Per crop: detector 6.9-7.3 ms, recognizer 2.9-6.3 ms (batch bucket by line count), host
post-processing and line crops 1-12 ms. YOLO here is 5.5 ms against 3.05 ms in its own perf
test: the extra is the per-frame torch -> device input conversion and output readback.
JPEG decode of a 4096x4096 frame is 70-85 ms on the host CPU and is not counted.

Reads: the plate `AE 6133 CT` (0.97) in its own frame and in the collage, `MRKU` `315829`
and `MAERSK`, `CA388-632`, `Intermodal`. Missed: vertical container numbers, the rail wagon
(COCO `train` is not a vehicle class here), close-ups with no vehicle (full-frame fallback reads
`777` but not `ZPK`).

## Next steps

- [ ] Overlap host and device work across frames (host pre/post ~5-15 ms per frame runs serially now)
- [ ] YOLO input: host conversion and readback (5.5 -> ~3 ms)
- [ ] Line crops: warp straight to recognizer height instead of full resolution (host post-processing up to 12 ms per crop, 155 ms on a full-frame close-up)
- [ ] Customer-tuned YOLO with a text-region class, so PP-OCR gets tight crops (smaller detector input, ~3.5 ms saved per crop)
- [ ] Vertical text
- [ ] Load test: 8-12 cameras at 2-5 frames/s per view, p50 / p99 latency against 500 ms
