# SAHI + Ultralytics quick test

This adds a lightweight test harness to evaluate whether sliced inference with SAHI helps your use case (for example, large images with small objects).

## 1) Install dependencies

From repo root, use the same Python interpreter you use for TT Metal / demos (for example `tt-metal/python_env/bin/python`).

### If `pip` is missing (`No module named pip`)

Bootstrap and upgrade `pip`:

```bash
python -m ensurepip --upgrade && python -m pip install -U pip
```

### Core packages

```bash
python -m pip install -U ultralytics sahi
```

If your environment already uses `tt-metal/tt_metal/python_env/requirements-dev.txt`, Ultralytics is pinned there and you can install only SAHI:

```bash
python -m pip install -U sahi
```

### Headless OpenCV (servers / no GUI)

Prefer the headless OpenCV wheel. After installing SAHI, remove the full `opencv-python` package if it is present, then install `opencv-python-headless`:

```bash
python -m pip install -U sahi && python -m pip uninstall -y opencv-python || true
python -m pip install -U opencv-python-headless
```

### ttnn / NumPy / OpenCV pins

Upgrading `opencv-python-headless` without a cap can pull **NumPy 2.x**, which conflicts with **ttnn** (`numpy<2`) and common **Ultralytics** pins. If you hit that, constrain NumPy and OpenCV:

```bash
python -m pip install -U "numpy>=1.24.4,<2" "opencv-python-headless<=4.11.0.86"
```

### Example: explicit `python_env` path

Replace `/path/to/tt-metal` with your checkout (for example `/home/you/tt-metal`):

```bash
"/path/to/tt-metal/python_env/bin/python" -m ensurepip --upgrade && \
"/path/to/tt-metal/python_env/bin/python" -m pip install -U pip

"/path/to/tt-metal/python_env/bin/python" -m pip install -U sahi && \
"/path/to/tt-metal/python_env/bin/python" -m pip uninstall -y opencv-python || true
"/path/to/tt-metal/python_env/bin/python" -m pip install -U opencv-python-headless

"/path/to/tt-metal/python_env/bin/python" -m pip install -U "numpy>=1.24.4,<2" "opencv-python-headless<=4.11.0.86"
```

Pip may warn that some packages list a dependency on the `opencv-python` *distribution*; `opencv-python-headless` still provides `cv2` and is the right choice for headless machines.

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

**Multi-chip / Ethernet dispatch:** On systems such as T3K (1×8), you may want dispatch on Ethernet cores (same idea as metal unit tests). Either export before Python:

```bash
export TT_METAL_GTEST_ETH_DISPATCH=1
```

or pass the script flag (sets the variable before `ttnn` loads):

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py --tt-eth-dispatch --backend tt ...
```

**One slice per wormhole (SAHI merge):** To run up to **N different** 640×640 SAHI tiles in **one** TT forward—batch dim 0 sharded so **each chip gets one slice**—set **`--tt-slice-parallel-devices N`**. The script opens a dedicated **1×N** mesh (first N devices in row-major order), not a submesh off a larger parent mesh—avoiding Metal teardown errors from shared command queues between parent and child meshes. SAHI still slices and merges predictions on the host. Example for four 640×640 quadrants on four chips:

```bash
python models/demos/yolo_eval/sahi_ultralytics_eval.py \
  --backend tt \
  --tt-model yolov8s \
  --tt-slice-parallel-devices 4 \
  --tt-eth-dispatch \
  --input /path/to/image.jpg \
  --pre-resize-to 1280 1280 \
  --slice-height 640 \
  --slice-width 640 \
  --overlap-height-ratio 0 \
  --overlap-width-ratio 0 \
  --postprocess-type GREEDYNMM \
  --postprocess-match-metric IOS \
  --postprocess-match-threshold 0.1 \
  --confidence-threshold 0.55 \
  --save-visuals
```

If the last batch has fewer than N slices, the remainder is padded with black images; padded slots are not merged into SAHI output. Omit **`--tt-slice-parallel-devices`** to keep the previous behavior (each slice replicated across the full mesh per forward).

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
