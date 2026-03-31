# SAHI + Ultralytics quick test

This adds a lightweight test harness to evaluate whether sliced inference with SAHI helps your use case (for example, large images with small objects).

## 1) Install dependencies

From repo root:

```bash
python -m pip install -U ultralytics sahi
```

If your environment already uses `tt-metal/tt_metal/python_env/requirements-dev.txt`, Ultralytics is pinned there and you can install only SAHI:

```bash
python -m pip install -U sahi
```

## 2) Run baseline vs sliced inference

Single image:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --input /absolute/path/to/image.jpg \
  --model yolo11n.pt \
  --pre-resize-to 640 640 \
  --slice-height 512 \
  --slice-width 512 \
  --overlap-height-ratio 0.2 \
  --overlap-width-ratio 0.2 \
  --postprocess-type NMS \
  --postprocess-match-metric IOU \
  --postprocess-match-threshold 0.5 \
  --save-visuals \
  --save-slice-grid-overlay
```

Optional quick sanity image:

```bash
python -c "from PIL import Image; Image.new('RGB',(1024,768),(120,80,200)).save('/tmp/sahi_test_image.png')"
```

Then run:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py --input /tmp/sahi_test_image.png --model yolov8x.pt --slice-height 320 --slice-width 320 --overlap-height-ratio 0.2 --overlap-width-ratio 0.2
```

Directory of images:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --input /absolute/path/to/images_dir \
  --model yolo11n.pt \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0.25 \
  --overlap-width-ratio 0.25 \
  --device cuda:0 \
  --save-visuals
```

TT backend (SAHI on host + TT inference per slice, sequential):

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --backend tt \
  --tt-model yolov8x \
  --input /absolute/path/to/large_image_or_dir \
  --pre-resize-to 1280 1280 \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0 \
  --overlap-width-ratio 0 \
  --postprocess-type GREEDYNMM \
  --postprocess-match-metric IOS \
  --postprocess-match-threshold 0.1 \
  --confidence-threshold 0.55 \
  --save-visuals \
  --save-slice-grid-overlay
```

## 3) Outputs

- Printed per-image comparison:
  - full-image detection count + latency
  - sliced detection count + latency
  - detection delta (sliced - full)
- JSON summary:
  - `models/demos/yolo_eval/sahi_outputs/summary.json`
- Optional visual outputs:
  - `<image>_full.*`
  - `<image>_sliced.*`
  - `<image>_slice_grid.png` (original image with SAHI tile boundaries)

## Notes for tuning

- Start with `slice-height/width` in the range `512-768`.
- Increase overlap (`0.2 -> 0.3`) when objects are often cut at tile boundaries.
- Expect sliced inference to be slower; the tradeoff is often better recall on small objects.
- If you see many duplicate boxes, use stricter merge settings:
  - `--postprocess-type NMS --postprocess-match-metric IOU --postprocess-match-threshold 0.5`
  - optionally raise `--confidence-threshold` to `0.35` or `0.4`
- If you need an exact fixed-size tiling setup, use `--pre-resize-to WIDTH HEIGHT`.
  - Example: `--pre-resize-to 640 640 --slice-height 320 --slice-width 320 --overlap-height-ratio 0 --overlap-width-ratio 0`
  - This yields exactly `4` tiles per image.
